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
                 gae_lambda: float = 0.95, device: torch.device = None):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
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

        self.ptr = 0  # 当前写入指针

    def insert(self, step: int, obs: np.ndarray, state: np.ndarray,
               actions: Dict[int, int], rewards: Dict[int, float],
               team_reward: float, done: bool, log_probs: Dict[int, float],
               values: Dict[int, float]):
        """
        插入一条经验

        Args:
            log_probs: {agent_id: float} — 对应采样动作的 log probability
            values: {agent_id: float} — Critic 对该 agent 的价值估计
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

            # 团队奖励均分 + 个体奖励
            mixed_reward = (self.rewards[t].cpu().numpy()
                            + self.team_rewards[t].item() / max(self.num_agents, 1))

            delta = mixed_reward + self.gamma * next_val * next_non_terminal - self.values[t].cpu().numpy()
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values[:self.ptr].cpu().numpy()
        return advantages, returns

    def get_batches(self, batch_size: int, advantages: np.ndarray,
                    returns: np.ndarray, num_epochs: int = 5):
        """
        生成 mini-batch 数据（带随机打乱）

        Yields:
            (obs_batch, obs_all_batch, state_batch, actions_batch,
             old_log_probs_batch, advantages_batch, returns_batch) 的元组

        其中 obs_all_batch: (batch/num_agents, num_agents, obs_dim)
        用于 Critic 的 attention 聚合。
        """
        ptr = self.ptr
        N = self.num_agents
        # obs/actions/log_probs: (ptr, N, ...) -> (ptr*N, ...)
        obs_flat = self.obs[:ptr].reshape(-1, self.obs_dim)
        actions_flat = self.actions[:ptr].reshape(-1)
        log_probs_flat = self.log_probs[:ptr].reshape(-1)
        # states: (ptr, state_dim) -> 重复到 (ptr*N, state_dim)
        state_repeated = self.states[:ptr].unsqueeze(1).expand(ptr, N, self.state_dim)
        states_flat = state_repeated.reshape(-1, self.state_dim)
        # obs_all: (ptr, N, obs_dim) -> 每个 step 重复 N 次 -> (ptr*N, N, obs_dim)
        obs_all_repeated = self.obs[:ptr].unsqueeze(2).expand(ptr, N, N, self.obs_dim)
        obs_all_flat = obs_all_repeated.reshape(-1, N, self.obs_dim)

        adv_flat = torch.FloatTensor(advantages.reshape(-1), device=self.device)
        ret_flat = torch.FloatTensor(returns.reshape(-1), device=self.device)

        # 归一化优势
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
                       adv_flat[idx], ret_flat[idx])

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

        # 共享特征提取
        self.fc = nn.Linear(obs_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

        if use_biz_heads:
            # BA Actor: 每种业务类型一个独立输出头
            self.biz_heads = nn.ModuleList([
                nn.Linear(hidden_dim, action_dim) for _ in range(num_biz_types)
            ])
        else:
            # 标准 MAPPO: 共享输出头
            self.output_head = nn.Linear(hidden_dim, action_dim)

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
        x = torch.relu(self.fc(obs))
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
        x = torch.relu(self.fc(obs))
        if hidden is None:
            hidden = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        h = self.rnn(x, hidden)
        logits = self._get_logits(h, biz_types)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """初始化隐藏状态"""
        return torch.zeros(batch_size, self.rnn.hidden_size)


# ==============================================================================
# Critic Network (价值网络)
# ==============================================================================

class AttentionCritic(nn.Module):
    """
    Multi-Head Attention 聚合模块

    将各 agent 的观测通过 self-attention 聚合为统一表示，
    再与全局状态拼接后输入 MLP 得到价值估计。
    """

    def __init__(self, obs_dim: int, embed_dim: int = 32, num_heads: int = 4):
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
                 attn_embed_dim: int = 32, attn_num_heads: int = 4):
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
    """

    def __init__(self, num_agents: int, obs_dim: int, state_dim: int,
                 action_dim: int = 5, hidden_dim: int = 64,
                 critic_hidden_dim: int = 128,
                 actor_lr: float = 3e-4, critic_lr: float = 5e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 10.0,
                 rollout_length: int = 100, num_epochs: int = 5,
                 batch_size: int = 32,
                 use_biz_heads: bool = True,
                 use_attention_critic: bool = True,
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
        self.use_biz_heads = use_biz_heads
        self.use_attention_critic = use_attention_critic

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 网络初始化
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

        # Rollout Buffer
        self.buffer = RolloutBuffer(
            num_agents, obs_dim, state_dim, action_dim,
            rollout_length, gamma, gae_lambda, self.device
        )

        # RNN 隐藏状态
        self.actor_hidden = None

        # 训练统计
        self.train_step_count = 0
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_history = []
        self.total_loss_history = []

    def select_actions(self, obs_dict: Dict[int, np.ndarray],
                       global_state: np.ndarray,
                       biz_types: Dict[int, int] = None,
                       training: bool = True):
        """
        为所有 agent 选择动作，同时返回 log_probs 和 values

        一次前向传播中完成：Actor采样 + Critic估值，确保 log_prob 与动作一致。

        Args:
            obs_dict: {agent_id: obs_array}
            global_state: 全局状态数组 (state_dim,)
            biz_types: {agent_id: business_type_index (0/1/2)}
            training: 是否训练模式（训练时采样，评估时取 greedy）

        Returns:
            actions: {agent_id: int}
            log_probs_dict: {agent_id: float}
            values_dict: {agent_id: float}
        """
        actions = {}
        log_probs_dict = {}
        values_dict = {}

        with torch.no_grad():
            obs_batch = np.array([obs_dict[i] for i in range(self.num_agents)])
            obs_t = torch.FloatTensor(obs_batch).to(self.device)  # (N, obs_dim)
            state_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)  # (1, state_dim)
            obs_all_t = obs_t.unsqueeze(0)  # (1, N, obs_dim)

            if biz_types is not None:
                biz_batch = torch.LongTensor(
                    [biz_types[i] for i in range(self.num_agents)]
                ).to(self.device)
            else:
                biz_batch = None

            # Actor: logits + hidden
            logits, new_hidden = self.actor(obs_t, self.actor_hidden, biz_batch)
            self.actor_hidden = new_hidden.detach()

            # Critic: centralized value V(s), broadcast to all agents
            team_value = self.critic(state_t, obs_all_t).cpu().item()  # scalar

            # 采样动作 + 计算 log_prob
            for uid in range(self.num_agents):
                dist = Categorical(logits=logits[uid])
                if training:
                    action = dist.sample()
                    log_probs_dict[uid] = dist.log_prob(action).item()
                else:
                    action = logits[uid].argmax()
                    log_probs_dict[uid] = 0.0
                actions[uid] = action.item()
                values_dict[uid] = team_value

        return actions, log_probs_dict, values_dict

    def reset_hidden(self):
        """重置 RNN 隐藏状态"""
        self.actor_hidden = self.actor.init_hidden(batch_size=self.num_agents).to(self.device)

    def insert_experience(self, step: int, obs_dict: Dict[int, np.ndarray],
                          global_state: np.ndarray, actions: Dict[int, int],
                          rewards: Dict[int, float], team_reward: float,
                          done: bool, log_probs: Dict[int, float],
                          values: Dict[int, float]):
        """插入一条经验到 buffer"""
        obs_batch = np.array([obs_dict[i] for i in range(self.num_agents)])
        self.buffer.insert(step, obs_batch, global_state, actions, rewards,
                           team_reward, done, log_probs, values)

    def train(self) -> Dict[str, float]:
        """
        执行一轮 PPO 更新

        Returns:
            训练统计信息
        """
        if self.buffer.ptr == 0:
            return {}

        # 获取 next_values (episode 结束时为 0)
        next_values = np.zeros(self.num_agents, dtype=np.float32)

        # 计算 GAE
        advantages, returns = self.buffer.compute_gae(next_values)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for obs_batch, obs_all_batch, states_batch, actions_batch, old_log_probs_batch, adv_batch, ret_batch in \
                self.buffer.get_batches(self.batch_size, advantages, returns, self.num_epochs):

            # 从 obs 中提取 biz_types
            # obs 结构: [sinr(4*num_bs), biz_onehot(3), sat(1), conn(1), vel(1), last_action(action_dim), sat_trend(1), peer_avg(1)]
            # biz_onehot 起始位置 = 4 * num_bs
            # 需要先算 num_bs: (obs_dim - 6 - action_dim - 2) / 4
            biz_start = (self.obs_dim - 6 - self.action_dim - 2) // 4 * 4
            # 但更简单: biz_start = 4 * num_bs，先通过 action_dim 反推
            # obs_dim = 4*num_bs + 6 + action_dim + 2 => num_bs = (obs_dim - 8 - action_dim) / 4
            num_bs = (self.obs_dim - 8 - self.action_dim) // 4
            biz_start = 4 * num_bs
            biz_onehot = obs_batch[:, biz_start:biz_start + 3]
            biz_types_batch = biz_onehot.argmax(dim=1)

            # Actor 更新
            old_log_probs = old_log_probs_batch.detach()
            new_log_probs, entropy = self.actor.evaluate_actions(
                obs_batch, actions_batch, biz_types=biz_types_batch
            )
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
            actor_loss = -torch.min(surr1, surr2).mean()

            entropy_loss = -entropy.mean()
            ppo_loss = actor_loss + self.entropy_coef * entropy_loss

            self.actor_optimizer.zero_grad()
            ppo_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Critic 更新: global_state + obs_all (attention)
            value_pred = self.critic(states_batch, obs_all_batch)  # (batch,)
            critic_loss = nn.MSELoss()(value_pred, ret_batch)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
            num_updates += 1

        avg_stats = {
            'actor_loss': total_actor_loss / max(num_updates, 1),
            'critic_loss': total_critic_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'total_loss': (total_actor_loss + self.value_coef * total_critic_loss) / max(num_updates, 1),
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
            'config': {
                'use_biz_heads': self.use_biz_heads,
                'use_attention_critic': self.use_attention_critic,
                'num_agents': self.num_agents,
                'obs_dim': self.obs_dim,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
            },
        }
        torch.save(state, path)
        print(f"  BA-MAPPO 模型已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.train_step_count = state['train_step_count']
        print(f"  BA-MAPPO 模型已加载: {path}")

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
        }
