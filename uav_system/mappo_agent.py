"""
BA-MAPPO Agent 模块

基于 MAPPO (Multi-Agent PPO, Yu et al. 2022 NeurIPS) 的多智能体强化学习智能体，
并引入两项改进：

1. Business-Aware Actor (BA Actor):
   共享 RNN 特征提取层 + 3 个独立的业务类型输出头，
   使不同业务类型的 UAV 学习差异化的策略。

2. Attention-Enhanced Critic:
   使用 Multi-Head Attention 动态聚合各 UAV 的观测信息，
   替代简单拼接，提升 Critic 对复杂交互关系的建模能力。

核心组件:
- RolloutBuffer: On-policy 经验缓冲区，支持 GAE 计算
- ActorNetwork: 策略网络 (shared GRU + optional biz_type heads)
- CriticNetwork: 价值网络 (global state + optional attention)
- MAPPOAgent: 整合训练循环，PPO Clip Loss 更新

核心公式:
  Actor:  pi(a|o) = softmax(h_biz(o)),  h_biz = Linear_biz(GRU(o))
  Critic: V(s) = MLP(Attn(obs_1, ..., obs_N; s_global))
  GAE:    delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
          A_t = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_{t+l}
  PPO:    L_clip = E[min(r(theta)*A, clip(r(theta), 1+/-eps)*A)]

配置开关:
  use_biz_heads=True/False      — 是否启用 BA Actor
  use_attention_critic=True/False — 是否启用 Attention Critic
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
from typing import Dict, Tuple, Optional


# ==============================================================================
# Observation Normalizer (Running mean/std)
# ==============================================================================

class ObsNormalizer:
    """对观测值做 running z-score 归一化，加速 PPO 收敛"""

    def __init__(self, obs_dim: int, decay: float = 0.999, clip_val: float = 5.0):
        self.obs_dim = obs_dim
        self.decay = decay
        self.clip_val = clip_val
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.var = np.ones(obs_dim, dtype=np.float64)
        self.count = 0

    def reset(self, new_obs_dim=None):
        """Reset normalizer for new observation dimension (for transfer learning)"""
        if new_obs_dim is not None:
            self.obs_dim = new_obs_dim
        self.mean = np.zeros(self.obs_dim, dtype=np.float64)
        self.var = np.ones(self.obs_dim, dtype=np.float64)
        self.count = 0

    def update(self, obs: np.ndarray):
        """更新 running stats (仅训练时调用)"""
        self.count += 1
        batch_mean = obs.mean(axis=0)
        batch_var = obs.var(axis=0) if len(obs) > 1 else np.ones(self.obs_dim)
        # 使用更稳定的更新方式
        alpha = 1.0 / max(self.count, 1)
        self.mean = (1 - alpha) * self.mean + alpha * batch_mean
        self.var = (1 - alpha) * self.var + alpha * batch_var

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """归一化 obs 到 ~N(0,1)，clip 到 [-clip_val, clip_val]"""
        std = np.sqrt(np.maximum(self.var, 1e-8))
        normed = (obs - self.mean) / std
        # 更激进的clip，减少极端值的影响
        return np.clip(normed, -self.clip_val, self.clip_val)

    def state_dict(self):
        return {'mean': self.mean.copy(), 'var': self.var.copy(), 'count': self.count}

    def load_state_dict(self, state):
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']


# ==============================================================================
# Rollout Buffer (On-policy 经验缓冲区 + GAE)
# ==============================================================================

class RolloutBuffer:
    """
    On-policy 经验缓冲区

    存储一个 rollout 周期的经验，并计算 GAE 优势估计。
    每个 agent 独立记录轨迹。
    """

    def __init__(self, num_agents: int, obs_dim: int, state_dim: int,
                 action_dim: int, rollout_length: int, gamma: float = 0.99,
                 gae_lambda: float = 0.95, hidden_dim: int = 64,
                 device: torch.device = None):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.hidden_dim = hidden_dim
        self.device = device or torch.device('cpu')

        # 数据存储 (预分配张量以提高效率)
        self.obs = torch.zeros(rollout_length, num_agents, obs_dim, device=self.device)
        self.states = torch.zeros(rollout_length, state_dim, device=self.device)
        self.actions = torch.zeros(rollout_length, num_agents, dtype=torch.long, device=self.device)
        self.rewards = torch.zeros(rollout_length, num_agents, device=self.device)
        self.team_rewards = torch.zeros(rollout_length, device=self.device)
        self.dones = torch.zeros(rollout_length, dtype=torch.bool, device=self.device)
        self.log_probs = torch.zeros(rollout_length, num_agents, device=self.device)
        self.values = torch.zeros(rollout_length, num_agents, device=self.device)
        self.biz_types = torch.zeros(rollout_length, num_agents, dtype=torch.long, device=self.device)
        # 存储 Actor RNN 的 hidden state，保证训练时采样分布与收集时一致
        self.hiddens = torch.zeros(rollout_length, num_agents, hidden_dim, device=self.device)

        self.ptr = 0  # 当前写入指针

    def insert(self, step: int, obs: np.ndarray, state: np.ndarray,
               actions: Dict[int, int], rewards: Dict[int, float],
               team_reward: float, done: bool, log_probs: Dict[int, float],
               values: Dict[int, float], biz_types: Dict[int, int] = None,
               hiddens: np.ndarray = None):
        """
        插入一条经验

        Args:
            log_probs: {agent_id: float} — 对应采样动作的 log probability
            values: {agent_id: float} — Critic 对该 agent 的价值估计
            biz_types: {agent_id: int} — 业务类型索引 (0/1/2)
            hiddens: (num_agents, hidden_dim) — Actor RNN 隐藏状态 (采样时的 pre-step hidden)
        """
        self.obs[step] = torch.FloatTensor(obs)
        self.states[step] = torch.FloatTensor(state)
        self.team_rewards[step] = float(team_reward)
        self.dones[step] = done
        for uid in range(self.num_agents):
            self.actions[step, uid] = int(actions[uid])
            self.rewards[step, uid] = float(rewards[uid])
            self.log_probs[step, uid] = float(log_probs[uid])
            self.values[step, uid] = float(values[uid])
            if biz_types is not None:
                self.biz_types[step, uid] = int(biz_types[uid])
        if hiddens is not None:
            self.hiddens[step] = torch.FloatTensor(hiddens)
        self.ptr = max(self.ptr, step + 1)

    def compute_gae(self, next_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算广义优势估计 (GAE)

        Args:
            next_values: (num_agents,) 下一步的价值估计

        Returns:
            advantages: (ptr, num_agents)
            returns: (ptr, num_agents)  (V-targets for value loss)
        """
        advantages = np.zeros((self.ptr, self.num_agents), dtype=np.float32)
        last_gae = np.zeros(self.num_agents, dtype=np.float32)

        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_val = next_values
            else:
                next_val = self.values[t + 1].cpu().numpy()
            next_non_terminal = 1.0 - float(self.dones[t])

            # 仅使用归一化后的个体 reward (team_reward 已在个体 reward 中体现)
            # 旧版 mixed_reward 存在尺度不一致: 个体 reward 已归一化但 team_reward 未归一化
            reward = self.rewards[t].cpu().numpy()

            delta = reward + self.gamma * next_val * next_non_terminal - self.values[t].cpu().numpy()
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values[:self.ptr].cpu().numpy()
        return advantages, returns

    def get_batches(self, batch_size: int, advantages: np.ndarray,
                    returns: np.ndarray, num_epochs: int = 5,
                    burn_in: int = 0):
        """
        生成 mini-batch 数据（带随机打乱）

        Args:
            burn_in: 跳过前 burn_in 步（这些步的 hidden 是"冷启动"，不用于 PPO 更新）

        Yields:
            (obs_batch, obs_all_batch, state_batch, actions_batch,
             old_log_probs_batch, advantages_batch, returns_batch,
             old_values_batch, biz_types_batch, hidden_batch)
        """
        ptr = self.ptr
        # burn-in: 跳过轨迹开头"冷"步骤
        start_idx = min(burn_in, ptr)
        N = self.num_agents
        # obs/actions/log_probs: (ptr, N, ...) -> (ptr*N, ...)
        obs_flat = self.obs[start_idx:ptr].reshape(-1, self.obs_dim)
        actions_flat = self.actions[start_idx:ptr].reshape(-1)
        log_probs_flat = self.log_probs[start_idx:ptr].reshape(-1)
        # states: (ptr, state_dim) -> 重复到 (ptr*N, state_dim)
        actual_len = ptr - start_idx
        state_repeated = self.states[start_idx:ptr].unsqueeze(1).expand(actual_len, N, self.state_dim)
        states_flat = state_repeated.reshape(-1, self.state_dim)
        # obs_all: (ptr, N, obs_dim) -> 每个 step 重复 N 次 -> (ptr*N, N, obs_dim)
        obs_all_repeated = self.obs[start_idx:ptr].unsqueeze(2).expand(actual_len, N, N, self.obs_dim)
        obs_all_flat = obs_all_repeated.reshape(-1, N, self.obs_dim)
        # biz_types: (ptr, N) -> (ptr*N,)
        biz_flat = self.biz_types[start_idx:ptr].reshape(-1)
        # hiddens: (ptr, N, hidden_dim) -> (ptr*N, hidden_dim)
        hidden_flat = self.hiddens[start_idx:ptr].reshape(-1, self.hidden_dim)
        # old values: (ptr, N) -> (ptr*N,) — 用于 value clipping
        values_flat = self.values[start_idx:ptr].reshape(-1)

        adv_flat = torch.FloatTensor(advantages[start_idx:].reshape(-1), device=self.device)
        ret_flat = torch.FloatTensor(returns[start_idx:].reshape(-1), device=self.device)

        # 归一化优势 (标准 PPO 实践，降低方差)
        if adv_flat.std() > 1e-8:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        dataset_size = obs_flat.shape[0]

        for _ in range(num_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                idx = indices[start:end]
                yield (obs_flat[idx], obs_all_flat[idx], states_flat[idx],
                       actions_flat[idx], log_probs_flat[idx],
                       adv_flat[idx], ret_flat[idx], values_flat[idx],
                       biz_flat[idx], hidden_flat[idx])

    def clear(self):
        """清空缓冲区"""
        self.ptr = 0


# ==============================================================================
# Actor Network (策略网络)
# ==============================================================================

class ActorNetwork(nn.Module):
    """
    MAPPO Actor 策略网络

    结构:
      - 共享层: Linear -> GRU (处理部分可观测性)
      - 业务类型嵌入层: 增强业务感知能力
      - BA 模式 (use_biz_heads=True):
          3 个独立的 Linear 输出头，按业务类型选择
      - 标准 MAPPO (use_biz_heads=False):
          1 个共享的 Linear 输出头
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64,
                 num_biz_types: int = 3, use_biz_heads: bool = True):
        super().__init__()
        self.use_biz_heads = use_biz_heads
        self.action_dim = action_dim
        self.num_biz_types = num_biz_types

        # 业务类型嵌入层
        self.biz_embedding = nn.Embedding(num_biz_types, hidden_dim)
        
        # 共享特征提取 — 正交初始化
        # 注意：输入维度是obs_dim，因为业务嵌入是在forward方法中拼接的
        self.fc = nn.Linear(obs_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

        if use_biz_heads:
            # BA Actor: 每种业务类型一个独立输出头
            self.biz_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim)
                ) for _ in range(num_biz_types)
            ])
        else:
            # 标准 MAPPO: 共享输出头
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )

        # 正交初始化 (标准 PPO 实践，保证初始策略接近均匀)
        self._init_weights()

    def _init_weights(self):
        """正交初始化 - 均衡探索与利用"""
        for module in [self.fc, self.rnn]:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
                elif 'bias' in name:
                    nn.init.zeros_(param)

        def get_last_linear(module):
            if isinstance(module, nn.Sequential):
                return module[-1]
            return module

        heads = self.biz_heads if self.use_biz_heads else [self.output_head]
        for head in heads:
            last_linear = get_last_linear(head)
            nn.init.orthogonal_(last_linear.weight, gain=0.5)
            nn.init.zeros_(last_linear.bias)
            last_linear.bias.data[0] = 0.1

    def _get_logits(self, h: torch.Tensor,
                    biz_types: torch.Tensor = None) -> torch.Tensor:
        """根据隐藏状态 h 计算 logits，不进行采样"""
        batch_size = h.shape[0]
        if self.use_biz_heads and biz_types is not None:
            logits = torch.zeros(batch_size, self.action_dim, device=h.device)
            for bt_idx in range(len(self.biz_heads)):
                mask = (biz_types == bt_idx)
                if mask.any():
                    logits[mask] = self.biz_heads[bt_idx](h[mask])
        else:
            logits = self.output_head(h)
        return logits

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None,
                biz_types: torch.Tensor = None):
        """
        前向传播 (仅提取特征，不采样)

        Args:
            obs: (batch, obs_dim)
            hidden: (batch, hidden_dim) GRU 隐藏状态
            biz_types: (batch,) 业务类型索引 (0/1/2)

        Returns:
            logits: (batch, action_dim)
            new_hidden: (batch, hidden_dim)
        """
        # 先通过fc层处理观测值
        x = torch.relu(self.fc(obs))
        
        # 添加业务类型嵌入
        if biz_types is not None:
            biz_emb = self.biz_embedding(biz_types)
            # 拼接fc输出和业务嵌入
            x = x + biz_emb  # 使用加法融合，而不是拼接
        
        if hidden is None:
            hidden = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        h = self.rnn(x, hidden)
        logits = self._get_logits(h, biz_types)
        return logits, h

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                         hidden: torch.Tensor = None,
                         biz_types: torch.Tensor = None):
        """
        评估给定动作的 log_prob 和 entropy

        Returns:
            log_probs: (batch,)
            entropy: (batch,)
        """
        # 先通过fc层处理观测值
        x = torch.relu(self.fc(obs))
        
        # 添加业务类型嵌入
        if biz_types is not None:
            biz_emb = self.biz_embedding(biz_types)
            # 拼接fc输出和业务嵌入
            x = x + biz_emb  # 使用加法融合，而不是拼接
        
        if hidden is None:
            hidden = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        h = self.rnn(x, hidden)
        logits = self._get_logits(h, biz_types)
        # logits clipping 防止 NaN (极端 logits 会导致 log_prob = inf/nan)
        logits = torch.clamp(logits, -10.0, 10.0)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """初始化隐藏状态"""
        return torch.zeros(batch_size, self.rnn.hidden_size)


class HierarchicalActorNetwork(nn.Module):
    """
    分层 Actor 策略网络

    结构:
      - 共享特征提取层: Linear -> GRU
      - 高层策略网络: 决定是否切换 (2个动作: stay, switch)
      - 底层策略网络: 选择目标基站 (action_dim-1个动作)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64,
                 num_biz_types: int = 3, use_biz_heads: bool = True):
        super().__init__()
        self.use_biz_heads = use_biz_heads
        self.action_dim = action_dim
        self.num_biz_types = num_biz_types

        # 业务类型嵌入层
        self.biz_embedding = nn.Embedding(num_biz_types, hidden_dim)
        
        # 共享特征提取
        self.fc = nn.Linear(obs_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

        # 高层策略网络：决定是否切换
        if use_biz_heads:
            self.high_level_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 2)
                ) for _ in range(num_biz_types)
            ])
        else:
            self.high_level_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )

        # 底层策略网络：选择目标基站
        if use_biz_heads:
            self.low_level_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim-1)
                ) for _ in range(num_biz_types)
            ])
        else:
            self.low_level_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim-1)
            )

        # 正交初始化
        self._init_weights()

    def _init_weights(self):
        """正交初始化"""
        for module in [self.fc, self.rnn]:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
                elif 'bias' in name:
                    nn.init.zeros_(param)

        # 辅助函数：获取Sequential中的最后一个线性层
        def get_last_linear(module):
            if isinstance(module, nn.Sequential):
                return module[-1]
            return module

        # 高层策略网络：移除 stay bias，增加探索性
        if self.use_biz_heads:
            for head in self.high_level_heads:
                last_linear = get_last_linear(head)
                nn.init.orthogonal_(last_linear.weight, gain=1.0)
                nn.init.zeros_(last_linear.bias)
                # 移除 stay bias，让模型自由学习
        else:
            last_linear = get_last_linear(self.high_level_head)
            nn.init.orthogonal_(last_linear.weight, gain=1.0)
            nn.init.zeros_(last_linear.bias)
            # 移除 stay bias，让模型自由学习

        # 底层策略网络：增加探索性
        if self.use_biz_heads:
            for head in self.low_level_heads:
                last_linear = get_last_linear(head)
                nn.init.orthogonal_(last_linear.weight, gain=1.0)
                nn.init.zeros_(last_linear.bias)
        else:
            last_linear = get_last_linear(self.low_level_head)
            nn.init.orthogonal_(last_linear.weight, gain=1.0)
            nn.init.zeros_(last_linear.bias)

    def _get_high_level_logits(self, h: torch.Tensor, biz_types: torch.Tensor = None) -> torch.Tensor:
        """获取高层策略 logits"""
        batch_size = h.shape[0]
        logits = torch.zeros(batch_size, 2, device=h.device)
        
        if self.use_biz_heads and biz_types is not None:
            # 修复掩码处理
            for bt_idx in range(len(self.high_level_heads)):
                mask = (biz_types == bt_idx)
                if mask.any():
                    # 确保h[mask]的形状正确
                    h_masked = h[mask]
                    if h_masked.shape[0] > 0:
                        logits[mask] = self.high_level_heads[bt_idx](h_masked)
        else:
            if self.use_biz_heads:
                # 当biz_types为None时，使用第一个业务类型的头部
                logits = self.high_level_heads[0](h)
            else:
                logits = self.high_level_head(h)
        return logits

    def _get_low_level_logits(self, h: torch.Tensor, biz_types: torch.Tensor = None) -> torch.Tensor:
        """获取底层策略 logits"""
        batch_size = h.shape[0]
        logits = torch.zeros(batch_size, self.action_dim-1, device=h.device)
        
        if self.use_biz_heads and biz_types is not None:
            # 修复掩码处理
            for bt_idx in range(len(self.low_level_heads)):
                mask = (biz_types == bt_idx)
                if mask.any():
                    # 确保h[mask]的形状正确
                    h_masked = h[mask]
                    if h_masked.shape[0] > 0:
                        logits[mask] = self.low_level_heads[bt_idx](h_masked)
        else:
            if self.use_biz_heads:
                # 当biz_types为None时，使用第一个业务类型的头部
                logits = self.low_level_heads[0](h)
            else:
                logits = self.low_level_head(h)
        return logits

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None,
                biz_types: torch.Tensor = None):
        """
        前向传播

        Args:
            obs: (batch, obs_dim)
            hidden: (batch, hidden_dim)
            biz_types: (batch,) 业务类型索引

        Returns:
            high_level_logits: (batch, 2)
            low_level_logits: (batch, action_dim-1)
            new_hidden: (batch, hidden_dim)
        """
        # 先通过fc层处理观测值
        x = torch.relu(self.fc(obs))
        
        # 添加业务类型嵌入
        if biz_types is not None:
            biz_emb = self.biz_embedding(biz_types)
            x = x + biz_emb  # 使用加法融合
        
        if hidden is None:
            hidden = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        h = self.rnn(x, hidden)
        
        high_level_logits = self._get_high_level_logits(h, biz_types)
        low_level_logits = self._get_low_level_logits(h, biz_types)
        
        return high_level_logits, low_level_logits, h

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                         hidden: torch.Tensor = None,
                         biz_types: torch.Tensor = None):
        """
        评估给定动作的 log_prob 和 entropy

        Args:
            obs: (batch, obs_dim)
            actions: (batch,)
            hidden: (batch, hidden_dim)
            biz_types: (batch,) 业务类型索引

        Returns:
            log_probs: (batch,)
            entropy: (batch,)
        """
        # 先通过fc层处理观测值
        x = torch.relu(self.fc(obs))
        
        # 添加业务类型嵌入
        if biz_types is not None:
            biz_emb = self.biz_embedding(biz_types)
            x = x + biz_emb  # 使用加法融合
        
        if hidden is None:
            hidden = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        h = self.rnn(x, hidden)
        
        high_level_logits = self._get_high_level_logits(h, biz_types)
        low_level_logits = self._get_low_level_logits(h, biz_types)
        
        # 计算 log probability
        log_probs = torch.zeros(obs.shape[0], device=obs.device)
        entropy = torch.zeros(obs.shape[0], device=obs.device)
        
        for i in range(obs.shape[0]):
            action = actions[i].item()
            if action == 0:  # stay
                dist = Categorical(logits=high_level_logits[i].unsqueeze(0))
                log_probs[i] = dist.log_prob(torch.tensor(0, device=obs.device))
                entropy[i] = dist.entropy()
            else:  # switch
                # 高层策略概率
                high_dist = Categorical(logits=high_level_logits[i].unsqueeze(0))
                high_log_prob = high_dist.log_prob(torch.tensor(1, device=obs.device))
                high_entropy = high_dist.entropy()
                
                # 底层策略概率
                low_dist = Categorical(logits=low_level_logits[i].unsqueeze(0))
                low_log_prob = low_dist.log_prob(torch.tensor(action-1, device=obs.device))
                low_entropy = low_dist.entropy()
                
                # 总概率
                log_probs[i] = high_log_prob + low_log_prob
                entropy[i] = high_entropy + low_entropy

        return log_probs, entropy

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """初始化隐藏状态"""
        return torch.zeros(batch_size, self.rnn.hidden_size)


class TransformerActorNetwork(nn.Module):
    """
    基于 Transformer 的 Actor 策略网络

    结构:
      - 业务类型嵌入层
      - 位置编码
      - Transformer Encoder
      - 业务类型特定的输出头
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64,
                 num_biz_types: int = 3, use_biz_heads: bool = True,
                 num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.use_biz_heads = use_biz_heads
        self.action_dim = action_dim
        self.num_biz_types = num_biz_types
        self.hidden_dim = hidden_dim

        # 业务类型嵌入层
        self.biz_embedding = nn.Embedding(num_biz_types, hidden_dim)
        
        # 输入嵌入
        self.input_embedding = nn.Linear(obs_dim, hidden_dim)
        
        # 位置编码
        self.position_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 业务类型特定的输出头
        if use_biz_heads:
            self.biz_heads = nn.ModuleList([
                nn.Linear(hidden_dim, action_dim) for _ in range(num_biz_types)
            ])
        else:
            self.output_head = nn.Linear(hidden_dim, action_dim)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in [self.input_embedding]:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
                elif 'bias' in name:
                    nn.init.zeros_(param)

        # 输出头: 均衡初始化，轻微stay偏好
        if self.use_biz_heads:
            for head in self.biz_heads:
                nn.init.orthogonal_(head.weight, gain=0.5)
                nn.init.zeros_(head.bias)
                head.bias.data[0] = 0.1
        else:
            nn.init.orthogonal_(self.output_head.weight, gain=0.5)
            nn.init.zeros_(self.output_head.bias)
            self.output_head.bias.data[0] = 0.1

    def _get_logits(self, h: torch.Tensor, biz_types: torch.Tensor = None) -> torch.Tensor:
        """根据隐藏状态 h 计算 logits"""
        batch_size = h.shape[0]
        if self.use_biz_heads and biz_types is not None:
            logits = torch.zeros(batch_size, self.action_dim, device=h.device)
            for bt_idx in range(len(self.biz_heads)):
                mask = (biz_types == bt_idx)
                if mask.any():
                    logits[mask] = self.biz_heads[bt_idx](h[mask])
        else:
            logits = self.output_head(h)
        return logits

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None,
                biz_types: torch.Tensor = None):
        """
        前向传播

        Args:
            obs: (batch, obs_dim)
            hidden: 未使用，为了保持接口一致
            biz_types: (batch,) 业务类型索引

        Returns:
            logits: (batch, action_dim)
            hidden: 未使用，为了保持接口一致
        """
        # 输入嵌入
        x = self.input_embedding(obs)
        
        # 添加位置编码
        x = x + self.position_encoding.expand(x.shape[0], -1, -1)
        
        # 添加业务类型嵌入
        if biz_types is not None:
            biz_emb = self.biz_embedding(biz_types).unsqueeze(1)
            x = x + biz_emb
        
        # Transformer 编码
        h = self.transformer(x)
        h = h.squeeze(1)  # (batch, 1, hidden_dim) -> (batch, hidden_dim)
        
        # 生成 logits
        logits = self._get_logits(h, biz_types)
        
        return logits, h

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                         hidden: torch.Tensor = None,
                         biz_types: torch.Tensor = None):
        """
        评估给定动作的 log_prob 和 entropy

        Args:
            obs: (batch, obs_dim)
            actions: (batch,)
            hidden: 未使用，为了保持接口一致
            biz_types: (batch,) 业务类型索引

        Returns:
            log_probs: (batch,)
            entropy: (batch,)
        """
        logits, _ = self.forward(obs, hidden, biz_types)
        # logits clipping 防止 NaN
        logits = torch.clamp(logits, -10.0, 10.0)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """初始化隐藏状态"""
        return torch.zeros(batch_size, self.hidden_dim)


# ==============================================================================
# Critic Network (价值网络)
# ==============================================================================

class AttentionCritic(nn.Module):
    """
    Multi-Head Attention 聚合模块

    将各 agent 的观测通过 self-attention 聚合为统一表示，
    再与全局状态拼接后输入 MLP 得到价值估计。
    """

    def __init__(self, obs_dim: int, embed_dim: int = 64, num_heads: int = 2):
        super().__init__()
        self.obs_embed = nn.Linear(obs_dim, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, obs_all: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_all: (batch, num_agents, obs_dim)

        Returns:
            aggregated: (batch, num_agents * embed_dim)
        """
        B, N, _ = obs_all.shape
        embedded = self.obs_embed(obs_all)  # (B, N, embed_dim)
        attn_out, _ = self.attention(embedded, embedded, embedded)
        attn_out = self.norm(attn_out + embedded)  # 残差连接
        return attn_out.reshape(B, -1)  # 展平


class CriticNetwork(nn.Module):
    """
    MAPPO Critic 价值网络

    输入全局状态（+ 可选的各 agent 观测聚合），输出 V(s) 标量。

    结构:
      - Attention 模式 (use_attention=True):
          AttentionCritic(obs_all) -> concat(state) -> MLP -> V(s)
      - 标准模式 (use_attention=False):
          MLP(state) -> V(s)
    """

    def __init__(self, state_dim: int, obs_dim: int, num_agents: int,
                 hidden_dim: int = 128, use_attention: bool = True,
                 attn_embed_dim: int = 64, attn_num_heads: int = 2):
        super().__init__()
        self.use_attention = use_attention

        if use_attention:
            self.att_critic = AttentionCritic(
                obs_dim=obs_dim, embed_dim=attn_embed_dim, num_heads=attn_num_heads
            )
            input_dim = state_dim + num_agents * attn_embed_dim
        else:
            input_dim = state_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 正交初始化
        self._init_weights()

    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                gain = nn.init.calculate_gain('relu')
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
        # 最终输出层用更小的 gain (value 初始接近 0)
        last = self.net[-1]
        nn.init.orthogonal_(last.weight, gain=1.0)
        nn.init.zeros_(last.bias)

    def forward(self, global_state: torch.Tensor,
                obs_all: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            global_state: (batch, state_dim) — 始终为 2D
            obs_all: (batch, num_agents, obs_dim) — 仅 Attention 模式

        Returns:
            value: (batch,) 标量价值
        """
        if self.use_attention and obs_all is not None:
            attn_out = self.att_critic(obs_all)
            x = torch.cat([global_state, attn_out], dim=-1)
        else:
            x = global_state

        value = self.net(x).squeeze(-1)  # (batch, 1) -> (batch,)
        return value


# ==============================================================================
# MAPPO Agent (整合训练循环)
# ==============================================================================

class MAPPOAgent:
    """
    BA-MAPPO 多智能体强化学习智能体

    CTDE 架构:
    - 训练时: Actor 使用局部观测，Critic 使用全局状态 + 所有 agent 观测
    - 执行时: 仅使用 Actor，根据局部观测选择动作

    改进开关:
    - use_biz_heads: 启用业务感知 Actor 头
    - use_attention_critic: 启用注意力增强 Critic
    - use_enhanced_algorithm: 启用与增强算法的联动
    """

    def __init__(self, num_agents: int, obs_dim: int, state_dim: int,
                 action_dim: int = 5, hidden_dim: int = 64,
                 critic_hidden_dim: int = 128,
                 actor_lr: float = 1e-4, critic_lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, entropy_coef: float = 0.02,
                 value_coef: float = 0.5, max_grad_norm: float = 2.0,
                 rollout_length: int = 150, num_epochs: int = 5,
                 batch_size: int = 32,
                 use_biz_heads: bool = True,
                 use_attention_critic: bool = True,
                 use_enhanced_algorithm: bool = False,
                 use_distillation: bool = True,
                 distillation_weight: float = 0.1,
                 use_pretrain: bool = True,
                 use_hierarchical: bool = False,
                 use_transformer: bool = False,
                 transformer_layers: int = 2,
                 transformer_heads: int = 2,
                 use_data_augmentation: bool = True,
                 augmentation_noise: float = 0.01,
                 device: Optional[str] = None):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_length = rollout_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.use_biz_heads = use_biz_heads
        self.use_attention_critic = use_attention_critic
        self.use_enhanced_algorithm = use_enhanced_algorithm
        self.use_distillation = use_distillation  # 策略蒸馏开关
        self.distillation_weight = distillation_weight  # 蒸馏损失权重
        self.use_pretrain = use_pretrain  # 模仿学习预训练开关
        self.use_hierarchical = use_hierarchical  # 分层强化学习开关
        self.use_transformer = use_transformer  # Transformer 网络开关
        self.transformer_layers = transformer_layers  # Transformer 层数
        self.transformer_heads = transformer_heads  # Transformer 头数
        self.use_data_augmentation = use_data_augmentation  # 数据增强开关
        self.augmentation_noise = augmentation_noise  # 数据增强噪声强度
        self.enhanced_algorithm = None
        self.enhanced_algorithm_prob = 1.0  # 增强算法的使用概率，随训练进程递减

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 网络初始化
        if self.use_transformer:
            self.actor = TransformerActorNetwork(
                obs_dim, action_dim, hidden_dim,
                num_biz_types=3, use_biz_heads=use_biz_heads,
                num_layers=self.transformer_layers,
                num_heads=self.transformer_heads
            ).to(self.device)
        elif self.use_hierarchical:
            self.actor = HierarchicalActorNetwork(
                obs_dim, action_dim, hidden_dim,
                num_biz_types=3, use_biz_heads=use_biz_heads
            ).to(self.device)
        else:
            self.actor = ActorNetwork(
                obs_dim, action_dim, hidden_dim,
                num_biz_types=3, use_biz_heads=use_biz_heads
            ).to(self.device)

        self.critic = CriticNetwork(
            state_dim, obs_dim, num_agents,
            hidden_dim=critic_hidden_dim,
            use_attention=use_attention_critic
        ).to(self.device)

        # 优化器 (Actor 和 Critic 独立优化)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # LR scheduler: warmup + cosine decay (标准 PPO 实践，稳定训练)
        self.actor_lr_init = actor_lr
        self.critic_lr_init = critic_lr
        self._total_train_steps = 0  # 由 experiments_mappo 设置
        self._current_train_step = 0
        self._warmup_steps = 50  # 前 50 个 PPO update 做 linear warmup

        # Rollout Buffer (传入 hidden_dim 用于存储 RNN hidden state)
        self.buffer = RolloutBuffer(
            num_agents, obs_dim, state_dim, action_dim,
            rollout_length, gamma, gae_lambda, hidden_dim=hidden_dim, device=self.device
        )

        # RNN 隐藏状态
        self.actor_hidden = None

        # 切换冷却机制：防止过度频繁切换（减少断连率）
        self.switch_cooldown_steps = 5  # 切换后 N 步内强制留守
        self._switch_cooldown = None    # per-agent 冷却计数器

        # Observation Normalizer (running mean/std)
        self.obs_normalizer = ObsNormalizer(obs_dim, decay=0.999, clip_val=5.0)

        # 训练统计
        self.train_step_count = 0
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_history = []
        self.total_loss_history = []

    def select_actions(self, obs_dict: Dict[int, np.ndarray],
                       global_state: np.ndarray,
                       biz_types: Dict[int, int] = None,
                       training: bool = True,
                       env=None):
        """
        为所有 agent 选择动作，同时返回 log_probs 和 values

        一次前向传播中完成：Actor采样 + Critic估值，确保 log_prob 与动作一致。

        Args:
            obs_dict: {agent_id: obs_array}
            global_state: 全局状态数组 (state_dim,)
            biz_types: {agent_id: business_type_index (0/1/2)}
            training: 是否训练模式（训练时采样，评估时取 greedy）
            env: 环境实例，用于增强算法

        Returns:
            actions: {agent_id: int}
            log_probs_dict: {agent_id: float}
            values_dict: {agent_id: float}
            pre_hidden: (num_agents, hidden_dim) — 本次 GRU 输入的 hidden state (numpy)
        """
        actions = {}
        log_probs_dict = {}
        values_dict = {}

        # 混合专家模型：根据情况选择增强算法或MAPPO策略
        if self.use_enhanced_algorithm and self.enhanced_algorithm and training:
            # 基于环境状态和业务类型动态选择专家
            use_enhanced = self._select_expert(env, biz_types)
        else:
            use_enhanced = False

        if use_enhanced and env:
            # 使用增强算法选择动作，但仍通过 Actor 计算有效的 log_prob（保证 PPO 一致性）
            self.enhanced_algorithm.run_step(enable_load_balancing=True)

            with torch.no_grad():
                pre_hidden = self.actor_hidden
                obs_batch = np.array([obs_dict[i] for i in range(self.num_agents)])
                if training:
                    self.obs_normalizer.update(obs_batch)
                obs_batch_norm = self.obs_normalizer.normalize(obs_batch)
                obs_t = torch.FloatTensor(obs_batch_norm).to(self.device)
                state_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
                obs_all_t = obs_t.unsqueeze(0)

                if biz_types is not None:
                    biz_batch = torch.LongTensor(
                        [biz_types[i] for i in range(self.num_agents)]
                    ).to(self.device)
                else:
                    biz_batch = None

                if self.use_hierarchical:
                    high_level_logits, low_level_logits, new_hidden = self.actor(obs_t, self.actor_hidden, biz_batch)
                    self.actor_hidden = new_hidden.detach()
                else:
                    logits, new_hidden = self.actor(obs_t, self.actor_hidden, biz_batch)
                    self.actor_hidden = new_hidden.detach()

                state_expanded = state_t.expand(self.num_agents, -1)
                obs_all_expanded = obs_t.unsqueeze(1).expand(self.num_agents, self.num_agents, self.obs_dim)
                per_agent_values = self.critic(state_expanded, obs_all_expanded)

            # 将增强算法的决策转换为动作索引，同时计算 Actor 的 log_prob
            for uid in range(self.num_agents):
                uav = env.env.uavs[uid]
                sinr_row = env.env.sinr_matrix[uid]
                capacities = []
                num_base_stations = len(env.env.base_stations)
                for bs_id in range(num_base_stations):
                    if isinstance(env.env.base_stations, dict):
                        bs = env.env.base_stations[bs_id]
                    else:
                        bs = env.env.base_stations[bs_id]
                    if hasattr(bs, 'available_capacity'):
                        capacities.append(bs.available_capacity)
                    else:
                        capacities.append(0)

                best_sinr_bs = np.argmax(sinr_row)
                best_cap_bs = np.argmax(capacities)

                # 增强算法的动作映射
                if uav.connected_bs_id == best_sinr_bs:
                    action = 1
                elif uav.connected_bs_id == best_cap_bs:
                    action = 2
                else:
                    action = 3
                actions[uid] = action

                # 用 Actor 网络计算该动作的有效 log_prob（关键修复！）
                if self.use_hierarchical:
                    high_dist = Categorical(logits=high_level_logits[uid].unsqueeze(0))
                    action_tensor = torch.tensor([action], device=self.device, dtype=torch.long)
                    if action == 0:
                        log_probs_dict[uid] = high_dist.log_prob(action_tensor)[0].item()
                    else:
                        low_dist = Categorical(logits=low_level_logits[uid].unsqueeze(0))
                        low_action_tensor = torch.tensor([action - 1], device=self.device, dtype=torch.long)
                        high_act_t = torch.tensor([1], device=self.device, dtype=torch.long)
                        log_probs_dict[uid] = (high_dist.log_prob(high_act_t)[0].item() +
                                                low_dist.log_prob(low_action_tensor)[0].item())
                else:
                    dist = Categorical(logits=logits[uid])
                    action_tensor = torch.tensor([action], device=self.device, dtype=torch.long)
                    log_probs_dict[uid] = dist.log_prob(action_tensor).item()

                values_dict[uid] = per_agent_values[uid].item()

            pre_hidden_np = pre_hidden.cpu().numpy() if pre_hidden is not None else np.zeros((self.num_agents, self.hidden_dim))
        else:
            with torch.no_grad():
                # 保存 pre-step hidden (传给 GRU 的 hidden，即上一步的输出)
                pre_hidden = self.actor_hidden

                obs_batch = np.array([obs_dict[i] for i in range(self.num_agents)])
                # 训练时更新 running stats；eval 时仅使用
                if training:
                    self.obs_normalizer.update(obs_batch)
                obs_batch_norm = self.obs_normalizer.normalize(obs_batch)
                
                # 数据增强：在训练时添加随机噪声
                if training and self.use_data_augmentation:
                    noise = np.random.normal(0, self.augmentation_noise, obs_batch_norm.shape)
                    obs_batch_norm = obs_batch_norm + noise

                obs_t = torch.FloatTensor(obs_batch_norm).to(self.device)  # (N, obs_dim)
                state_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)  # (1, state_dim)
                obs_all_t = obs_t.unsqueeze(0)  # (1, N, obs_dim)

                if biz_types is not None:
                    biz_batch = torch.LongTensor(
                        [biz_types[i] for i in range(self.num_agents)]
                    ).to(self.device)
                else:
                    biz_batch = None

                # Actor: logits + hidden
                if self.use_hierarchical:
                    high_level_logits, low_level_logits, new_hidden = self.actor(obs_t, self.actor_hidden, biz_batch)
                    self.actor_hidden = new_hidden.detach()
                else:
                    logits, new_hidden = self.actor(obs_t, self.actor_hidden, biz_batch)
                    self.actor_hidden = new_hidden.detach()

                # Critic: per-agent value — 将 obs_all 中每个 agent 的信息独立传入
                # 通过扩展 global_state 为 (N, state_dim) 实现差异化估值
                state_expanded = state_t.expand(self.num_agents, -1)  # (N, state_dim)
                obs_all_expanded = obs_t.unsqueeze(1).expand(self.num_agents, self.num_agents, self.obs_dim)  # (N, N, obs_dim)
                per_agent_values = self.critic(state_expanded, obs_all_expanded)  # (N,)

                for uid in range(self.num_agents):
                    if self.use_hierarchical:
                        high_dist = Categorical(logits=high_level_logits[uid].unsqueeze(0))
                        if training:
                            high_action_tensor = high_dist.sample()
                            high_action = high_action_tensor.item()
                        else:
                            high_action = high_level_logits[uid].argmax().item()
                            high_action_tensor = torch.tensor([high_action], device=self.device)
                        
                        if high_action == 0:
                            action = 0
                            if training:
                                log_probs_dict[uid] = high_dist.log_prob(high_action_tensor)[0].item()
                            else:
                                log_probs_dict[uid] = 0.0
                        else:
                            low_dist = Categorical(logits=low_level_logits[uid].unsqueeze(0))
                            if training:
                                low_action_tensor = low_dist.sample()
                                low_action = low_action_tensor.item()
                                log_probs_dict[uid] = high_dist.log_prob(high_action_tensor)[0].item() + low_dist.log_prob(low_action_tensor)[0].item()
                            else:
                                low_action = low_level_logits[uid].argmax().item()
                                log_probs_dict[uid] = 0.0
                            action = low_action + 1
                    else:
                        dist = Categorical(logits=logits[uid])
                        if training:
                            action_tensor = dist.sample()
                            log_probs_dict[uid] = dist.log_prob(action_tensor).item()
                            action = action_tensor.item()
                        else:
                            action = logits[uid].argmax().item()
                            log_probs_dict[uid] = 0.0
                    actions[uid] = action
                    values_dict[uid] = per_agent_values[uid].item()

            pre_hidden_np = pre_hidden.cpu().numpy() if pre_hidden is not None else np.zeros((self.num_agents, self.hidden_dim))

        # 切换冷却机制：减少过度频繁切换导致的断连
        if self._switch_cooldown is not None:
            for uid in range(self.num_agents):
                if self._switch_cooldown[uid] > 0:
                    if actions.get(uid, 0) != 0:
                        actions[uid] = 0  # 强制留守
                        log_probs_dict[uid] = 0.0  # 冷却覆盖的动作不参与策略梯度
                    self._switch_cooldown[uid] -= 1
                elif actions.get(uid, 0) != 0:
                    self._switch_cooldown[uid] = self.switch_cooldown_steps

        return actions, log_probs_dict, values_dict, pre_hidden_np

    def _select_expert(self, env, biz_types):
        """
        混合专家模型：根据环境状态和业务类型选择合适的专家

        Args:
            env: 环境实例
            biz_types: 业务类型字典

        Returns:
            bool: 是否使用增强算法
        """
        # 1. 基于环境状态的启发式规则
        # 计算网络负载
        total_load = 0
        total_capacity = 0
        # 检查base_stations的结构
        if hasattr(env.env, 'base_stations'):
            base_stations = env.env.base_stations
            if isinstance(base_stations, dict):
                for bs_id, bs in base_stations.items():
                    if hasattr(bs, 'current_load') and hasattr(bs, 'total_capacity'):
                        total_load += bs.current_load
                        total_capacity += bs.total_capacity
            elif isinstance(base_stations, list):
                for bs in base_stations:
                    if hasattr(bs, 'current_load') and hasattr(bs, 'total_capacity'):
                        total_load += bs.current_load
                        total_capacity += bs.total_capacity
        load_ratio = total_load / total_capacity if total_capacity > 0 else 0
        
        # 2. 基于业务类型的启发式规则
        # 统计不同业务类型的数量
        biz_counts = {0: 0, 1: 0, 2: 0}
        for uid in biz_types:
            biz_counts[biz_types[uid]] += 1
        
        # 3. 综合决策
        # 高负载时使用增强算法
        if load_ratio > 0.7:
            return True
        # 可靠性敏感业务占比高时使用增强算法
        if biz_counts[2] / sum(biz_counts.values()) > 0.5:
            return True
        # 训练初期使用增强算法
        if self.enhanced_algorithm_prob > 0.5:
            return np.random.rand() < self.enhanced_algorithm_prob
        # 否则使用MAPPO
        return False

    def reset_hidden(self):
        """重置 RNN 隐藏状态"""
        self.actor_hidden = self.actor.init_hidden(batch_size=self.num_agents).to(self.device)
        self._switch_cooldown = np.zeros(self.num_agents, dtype=np.int32)

    def _update_lr(self):
        """根据当前训练进度更新学习率: linear warmup + cosine decay"""
        if self._total_train_steps <= 0:
            return
        step = self._current_train_step
        total = self._total_train_steps
        warmup = self._warmup_steps

        if step < warmup:
            # Linear warmup
            frac = (step + 1) / warmup
        else:
            # Cosine decay
            progress = (step - warmup) / max(total - warmup, 1)
            frac = 0.5 * (1.0 + np.cos(np.pi * progress))

        for opt, lr_init in [(self.actor_optimizer, self.actor_lr_init),
                              (self.critic_optimizer, self.critic_lr_init)]:
            for pg in opt.param_groups:
                pg['lr'] = lr_init * frac

    def insert_experience(self, step: int, obs_dict: Dict[int, np.ndarray],
                          global_state: np.ndarray, actions: Dict[int, int],
                          rewards: Dict[int, float], team_reward: float,
                          done: bool, log_probs: Dict[int, float],
                          values: Dict[int, float],
                          biz_types: Dict[int, int] = None,
                          pre_hidden: np.ndarray = None):
        """插入一条经验到 buffer

        Args:
            pre_hidden: (num_agents, hidden_dim) — 采样时 GRU 的输入 hidden state
        """
        obs_batch = np.array([obs_dict[i] for i in range(self.num_agents)])
        # 在存储时归一化 obs，确保与 select_actions 中计算 value 时使用相同的归一化参数
        obs_batch = self.obs_normalizer.normalize(obs_batch)
        self.buffer.insert(step, obs_batch, global_state, actions, rewards,
                           team_reward, done, log_probs, values, biz_types,
                           hiddens=pre_hidden)

    def train(self) -> Dict[str, float]:
        """
        执行一轮 PPO 更新

        使用存储的 hidden state 保证训练时 log_prob 与采样时一致 (burn-in 方案)。

        Returns:
            训练统计信息
        """
        print(f"[DEBUG-TRAIN] train() called, buffer.ptr={self.buffer.ptr}")

        if self.buffer.ptr == 0:
            return {}

        # LR schedule
        self._update_lr()
        self._current_train_step += 1

        # 获取 next_values (episode 结束时为 0)
        next_values = np.zeros(self.num_agents, dtype=np.float32)

        # 计算 GAE
        advantages, returns = self.buffer.compute_gae(next_values)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        approx_kl = 0.0
        total_actor_grad = 0.0
        total_critic_grad = 0.0
        total_value_error = 0.0
        total_ratio_mean = 0.0
        total_adv_mean = 0.0
        total_ret_mean = 0.0

        # burn-in: 跳过前几步 (hidden 是零向量冷启动，信息不可靠)
        burn_in = min(5, self.buffer.ptr // 3)

        try:
            batch_iterator = list(self.buffer.get_batches(
                self.batch_size, advantages, returns, self.num_epochs, burn_in=burn_in
            ))
        except Exception as e:
            print(f"[ERROR] Exception in get_batches(): {e}")
            import traceback
            traceback.print_exc()
            return {}

        if len(batch_iterator) == 0:
            print(f"[WARN] No batches generated! ptr={self.buffer.ptr}, burn_in={burn_in}")
            return {}

        for obs_batch, obs_all_batch, states_batch, actions_batch, old_log_probs_batch, adv_batch, ret_batch, old_values_batch, biz_types_batch, hidden_batch in batch_iterator:

            # obs 已在 insert_experience 中归一化，无需重复归一化

            # Actor 更新 — 传入采样时的 hidden state，保证 log_prob 分布一致
            old_log_probs = old_log_probs_batch.detach()
            new_log_probs, entropy = self.actor.evaluate_actions(
                obs_batch, actions_batch, hidden=hidden_batch, biz_types=biz_types_batch
            )
            ratio = torch.exp(new_log_probs - old_log_probs)

            with torch.no_grad():
                total_ratio_mean += ratio.mean().item()
                total_adv_mean += adv_batch.mean().item()
                total_ret_mean += ret_batch.mean().item()

            # KL Divergence Early Stop (标准 PPO 实践)
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - (new_log_probs - old_log_probs)).mean().item()
            if approx_kl > 0.2:  # 大幅放宽阈值以适应hidden state不一致
                if num_updates == 0:
                    print(f"[DEBUG] KL early stop! kl={approx_kl:.4f}")
                break

            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
            actor_loss = -torch.min(surr1, surr2).mean()

            entropy_loss = -entropy.mean()
            ppo_loss = actor_loss + self.entropy_coef * entropy_loss

            # 策略蒸馏：从增强算法中学习
            if self.use_distillation and self.enhanced_algorithm:
                with torch.no_grad():
                    distillation_loss = 0.0
                    for i in range(obs_batch.shape[0]):
                        biz_type = biz_types_batch[i].item()
                        if biz_type == 0:
                            enhanced_probs = torch.tensor([0.3, 0.5, 0.1, 0.1, 0.0, 0.0], device=self.device)
                        elif biz_type == 1:
                            enhanced_probs = torch.tensor([0.3, 0.1, 0.5, 0.1, 0.0, 0.0], device=self.device)
                        else:
                            enhanced_probs = torch.tensor([0.2, 0.2, 0.2, 0.4, 0.0, 0.0], device=self.device)

                        if self.use_hierarchical:
                            high_level_logits, low_level_logits, _ = self.actor(obs_batch[i].unsqueeze(0), hidden_batch[i].unsqueeze(0), biz_types_batch[i].unsqueeze(0))
                            high_probs = torch.softmax(high_level_logits[0], dim=0)
                            low_probs = torch.softmax(low_level_logits[0], dim=0)
                            mappo_probs = torch.zeros_like(enhanced_probs)
                            mappo_probs[0] = high_probs[0]
                            for j in range(len(low_probs)):
                                mappo_probs[j+1] = high_probs[1] * low_probs[j]
                        else:
                            actor_logits, _ = self.actor(obs_batch[i].unsqueeze(0), hidden_batch[i].unsqueeze(0), biz_types_batch[i].unsqueeze(0))
                            mappo_probs = torch.softmax(actor_logits[0], dim=0)

                        distillation_loss += torch.sum(enhanced_probs * torch.log(enhanced_probs / (mappo_probs + 1e-8)))

                    distillation_loss = distillation_loss / obs_batch.shape[0]
                    ppo_loss += self.distillation_weight * distillation_loss

            self.actor_optimizer.zero_grad()
            ppo_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Critic 更新: value clipping (标准 PPO 实践，防止 critic 震荡)
            value_pred = self.critic(states_batch, obs_all_batch)
            value_pred_clipped = old_values_batch + torch.clamp(
                value_pred - old_values_batch, -self.clip_epsilon, self.clip_epsilon
            )
            critic_loss_unclipped = (value_pred - ret_batch).pow(2).mean()
            critic_loss_clipped = (value_pred_clipped - ret_batch).pow(2).mean()
            critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
            total_actor_grad += actor_grad_norm
            total_critic_grad += critic_grad_norm
            total_value_error += ((value_pred.detach() - ret_batch) ** 2).mean().item()
            num_updates += 1

        avg_stats = {
            'actor_loss': total_actor_loss / max(num_updates, 1),
            'critic_loss': total_critic_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'total_loss': (total_actor_loss + self.value_coef * total_critic_loss) / max(num_updates, 1),
            'approx_kl': approx_kl,
            'actor_grad_norm': total_actor_grad / max(num_updates, 1),
            'critic_grad_norm': total_critic_grad / max(num_updates, 1),
            'value_mse': total_value_error / max(num_updates, 1),
            'ratio_mean': total_ratio_mean / max(num_updates, 1),
            'advantage_mean': total_adv_mean / max(num_updates, 1),
            'return_mean': total_ret_mean / max(num_updates, 1),
            'num_updates': num_updates,
        }

        self.actor_loss_history.append(avg_stats['actor_loss'])
        self.critic_loss_history.append(avg_stats['critic_loss'])
        self.entropy_history.append(avg_stats['entropy'])
        self.total_loss_history.append(avg_stats['total_loss'])
        self.train_step_count += 1

        # 清空 buffer
        self.buffer.clear()

        return avg_stats

    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        state = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'train_step_count': self.train_step_count,
            'obs_normalizer': self.obs_normalizer.state_dict(),
            'config': {
                'use_biz_heads': self.use_biz_heads,
                'use_attention_critic': self.use_attention_critic,
                'use_hierarchical': getattr(self, 'use_hierarchical', False),
                'use_transformer': getattr(self, 'use_transformer', False),
                'use_data_augmentation': getattr(self, 'use_data_augmentation', False),
                'num_agents': self.num_agents,
                'obs_dim': self.obs_dim,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
                'critic_hidden_dim': self.critic_hidden_dim,
            },
        }
        torch.save(state, path)
        print(f"  BA-MAPPO 模型已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.train_step_count = state['train_step_count']
        if 'obs_normalizer' in state:
            self.obs_normalizer.load_state_dict(state['obs_normalizer'])
        print(f"  BA-MAPPO 模型已加载: {path}")

    def set_enhanced_algorithm(self, enhanced_algorithm):
        """设置增强算法实例
        
        Args:
            enhanced_algorithm: 增强算法实例
        """
        self.enhanced_algorithm = enhanced_algorithm

    def update_enhanced_algorithm_prob(self, step, total_steps):
        """更新增强算法的使用概率
        
        Args:
            step: 当前训练步数
            total_steps: 总训练步数
        """
        if not self.use_enhanced_algorithm:
            return
        # 线性递减策略
        self.enhanced_algorithm_prob = max(0.0, 1.0 - step / total_steps)

    def get_stats(self) -> dict:
        """获取训练统计"""
        def _avg(lst, window=200):
            if len(lst) < window:
                return 0.0
            return np.mean(lst[-window:])

        return {
            'train_steps': self.train_step_count,
            'avg_actor_loss': _avg(self.actor_loss_history),
            'avg_critic_loss': _avg(self.critic_loss_history),
            'avg_entropy': _avg(self.entropy_history),
            'avg_total_loss': _avg(self.total_loss_history),
            'enhanced_algorithm_prob': self.enhanced_algorithm_prob if self.use_enhanced_algorithm else 0.0,
        }

    def collect_demonstrations(self, env, num_demos=1000):
        """
        收集增强算法的决策作为示范数据

        Args:
            env: 环境实例
            num_demos: 收集的示范数量

        Returns:
            demonstrations: 示范数据列表，每个元素为 (obs, action)
        """
        demonstrations = []
        print(f"\n  开始收集增强算法的示范数据...")
        
        for i in range(num_demos):
            # 重置环境
            obs_dict, global_state = env.reset()
            
            # 使用增强算法选择动作
            self.enhanced_algorithm.run_step(enable_load_balancing=True)
            
            # 收集每个智能体的观测和动作
            for uid in range(self.num_agents):
                uav = env.env.uavs[uid]
                obs = obs_dict[uid]
                
                # 分析增强算法的决策，映射到对应的动作
                sinr_row = env.env.sinr_matrix[uid]
                capacities = []
                num_base_stations = len(env.env.base_stations)
                for bs_id in range(num_base_stations):
                    if isinstance(env.env.base_stations, dict):
                        bs = env.env.base_stations[bs_id]
                    else:
                        bs = env.env.base_stations[bs_id]
                    if hasattr(bs, 'available_capacity'):
                        capacities.append(bs.available_capacity)
                    else:
                        capacities.append(0)
                
                best_sinr_bs = np.argmax(sinr_row)
                best_cap_bs = np.argmax(capacities)
                
                if uav.connected_bs_id == best_sinr_bs:
                    action = 1  # best_sinr
                elif uav.connected_bs_id == best_cap_bs:
                    action = 2  # best_capacity
                else:
                    action = 3  # sinr_capacity
                
                demonstrations.append((obs, action))
            
            if (i + 1) % 100 == 0:
                print(f"  已收集 {i + 1}/{num_demos} 个示范")
        
        print(f"  示范数据收集完成，共 {len(demonstrations)} 个样本")
        return demonstrations

    def pretrain(self, demonstrations, epochs=100, batch_size=32):
        """
        使用增强算法的示范进行预训练

        Args:
            demonstrations: 示范数据列表，每个元素为 (obs, action)
            epochs: 预训练轮数
            batch_size: 批量大小
        """
        print(f"\n  开始模仿学习预训练...")
        
        # 准备数据集
        obs_list = []
        action_list = []
        for obs, action in demonstrations:
            obs_list.append(obs)
            action_list.append(action)
        
        obs_array = np.array(obs_list)
        action_array = np.array(action_list)
        
        # 归一化观测值
        self.obs_normalizer.update(obs_array)
        obs_array_norm = self.obs_normalizer.normalize(obs_array)
        
        # 转换为张量
        obs_tensor = torch.FloatTensor(obs_array_norm).to(self.device)
        action_tensor = torch.LongTensor(action_array).to(self.device)
        
        # 预训练循环
        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(len(obs_tensor))
            obs_tensor_shuffled = obs_tensor[indices]
            action_tensor_shuffled = action_tensor[indices]
            
            total_loss = 0
            num_batches = len(obs_tensor) // batch_size
            
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                
                obs_batch = obs_tensor_shuffled[start:end]
                action_batch = action_tensor_shuffled[start:end]
                
                # 前向传播
                output = self.actor(obs_batch, None, None)
                if isinstance(output, tuple):
                    if len(output) == 2:
                        # 标准Actor网络
                        logits, _ = output
                    else:
                        # 分层Actor网络，需要合并高层和底层的logits
                        high_level_logits, low_level_logits, _ = output
                        # 合并logits: [stay, switch + 底层动作]
                        # 注意：这里我们使用简单的合并方式，因为pretrain是模仿学习
                        # 对于分层策略，我们需要构建完整的动作分布
                        batch_size = obs_batch.shape[0]
                        logits = torch.zeros(batch_size, self.action_dim, device=self.device)
                        logits[:, 0] = high_level_logits[:, 0]  # stay的logits
                        for i in range(low_level_logits.shape[1]):
                            logits[:, i+1] = high_level_logits[:, 1] + low_level_logits[:, i]  # switch + 底层动作的logits
                else:
                    logits = output
                loss = torch.nn.functional.cross_entropy(logits, action_batch)
                
                # 反向传播
                self.actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            if (epoch + 1) % 10 == 0:
                print(f"  预训练 epoch {epoch + 1}/{epochs}, 损失: {avg_loss:.4f}")
        
        print("  模仿学习预训练完成")
