"""
QMIX Agent 模块

实现 QMIX (Q-value Mixing Network) 多智能体强化学习智能体。

组件:
- IndividualRNNAgent: 每个 agent 的独立 RNN Q 网络
- MixingNetwork: 值分解混合网络（单调约束保证)
- QMIXAgent: 整合训练循环，管理经验回放和模型更新

核心公式:
  Q_tot = Σ_i w_i · Q_i + b
  w_i = softmax(hyper_w(s_global))_i / sum(softmax(...))
  b = hyper_b(s_global)

  L(θ) = E[(r_team + γ max_a' Q_tot(s', a', θ⁻) - Q_tot(s, a, θ))²]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random
import os
from typing import Dict, List, Tuple, Optional


# ==============================================================================
# Individual RNN Agent (每个 UAV 独立的 Q 网络)
# ==============================================================================

class IndividualRNNAgent(nn.Module):
    """
    单个 agent 的 RNN Q 网络

    输入: 局部观测 (obs_dim)
    输出: 每个动作的 Q 值 (action_dim)

    使用 GRU 隐藏状态处理部分可观测性。
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            obs: (batch, obs_dim) 或 (obs_dim,)
            hidden: (batch, hidden_dim) 或 (hidden_dim,)

        Returns:
            q_values: (batch, action_dim) 或 (action_dim,)
            new_hidden: (batch, hidden_dim) 或 (hidden_dim,)
        """
        squeeze = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze = True

        x = torch.relu(self.fc1(obs))

        if hidden is None:
            batch_size = x.shape[0]
            hidden = torch.zeros(batch_size, x.shape[1], device=x.device)
        elif hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)

        h = self.rnn(x, hidden)
        q = self.fc2(h)

        if squeeze:
            q = q.squeeze(0)
            h = h.squeeze(0)

        return q, h

    def init_hidden(self) -> torch.Tensor:
        """初始化隐藏状态"""
        return torch.zeros(1, self.fc1.out_features)


# ==============================================================================
# Mixing Network (值分解混合网络)
# ==============================================================================

class MixingNetwork(nn.Module):
    """
    QMIX 混合网络

    将各 agent 的个体 Q 值通过单调约束混合为团队 Q_tot:
      Q_tot = Σ_i w_i · Q_i + b

    其中 w_i = |softmax(hyper_w(s_global))_i|, b = hyper_b(s_global)
    绝对值保证 ∂Q_tot/∂Q_i ≥ 0 (单调性约束)。
    """

    def __init__(self, num_agents: int, state_dim: int,
                 hidden_dim: int = 64, hyper_hidden_dim: int = 32):
        super().__init__()
        self.num_agents = num_agents

        # 超网络: 从全局状态生成混合权重
        self.hyper_w = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, num_agents),
        )
        # 超网络: 从全局状态生成偏置
        self.hyper_b = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            agent_qs: (batch, num_agents) 各 agent 的 Q 值
            global_state: (batch, state_dim) 全局状态

        Returns:
            q_tot: (batch,) 团队 Q 值
        """
        # 生成权重并取绝对值保证单调性
        w = self.hyper_w(global_state)  # (batch, num_agents)
        w = torch.abs(w)
        w_sum = w.sum(dim=-1, keepdim=True) + 1e-10
        w = w / w_sum  # 归一化

        # 生成偏置
        b = self.hyper_b(global_state).squeeze(-1)  # (batch,)

        # 混合
        q_tot = (w * agent_qs).sum(dim=-1) + b  # (batch,)
        return q_tot


# ==============================================================================
# QMIX Agent (整合训练循环)
# ==============================================================================

class QMIXAgent:
    """
    QMIX 多智能体强化学习智能体

    管理:
    - N 个 IndividualRNNAgent (每个 UAV 一个)
    - 1 个 MixingNetwork (值分解)
    - 经验回放缓冲区
    - Epsilon-Greedy 探索策略
    """

    def __init__(self, num_agents: int, obs_dim: int, state_dim: int,
                 action_dim: int = 5, hidden_dim: int = 64,
                 lr: float = 5e-4, gamma: float = 0.99,
                 buffer_size: int = 50000, batch_size: int = 32,
                 target_update_freq: int = 200, tau: float = 0.005,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 device: Optional[str] = None):
        """
        Args:
            num_agents: agent 数量
            obs_dim: 每个 agent 的观测维度
            state_dim: 全局状态维度
            action_dim: 每个 agent 的动作维度
            hidden_dim: 隐藏层维度
            lr: 学习率
            gamma: 折扣因子
            buffer_size: 回放缓冲区大小
            batch_size: 训练批次大小
            target_update_freq: 目标网络更新频率
            tau: 软更新系数
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            epsilon_decay: 探索率衰减
            device: 计算设备
        """
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Agent 网络 (每个 agent 独立)
        self.agent_networks = [
            IndividualRNNAgent(obs_dim, action_dim, hidden_dim).to(self.device)
            for _ in range(num_agents)
        ]
        self.target_networks = [
            IndividualRNNAgent(obs_dim, action_dim, hidden_dim).to(self.device)
            for _ in range(num_agents)
        ]
        # 初始化目标网络
        for i in range(num_agents):
            self.target_networks[i].load_state_dict(self.agent_networks[i].state_dict())
            self.target_networks[i].eval()

        # Mixing Network
        self.mixing_net = MixingNetwork(num_agents, state_dim, hidden_dim).to(self.device)
        self.target_mixing_net = MixingNetwork(num_agents, state_dim, hidden_dim).to(self.device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        self.target_mixing_net.eval()

        # 优化器 (所有参数共用一个优化器)
        all_params = list(self.mixing_net.parameters())
        for net in self.agent_networks:
            all_params += list(net.parameters())
        self.optimizer = optim.Adam(all_params, lr=lr)

        # 经验回放
        self.memory = deque(maxlen=buffer_size)

        # Epsilon-Greedy
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 训练统计
        self.train_step_count = 0
        self.loss_history = deque(maxlen=2000)
        self.q_tot_history = deque(maxlen=2000)

    def select_actions(self, obs_dict: Dict[int, np.ndarray],
                       training: bool = True) -> Dict[int, int]:
        """
        为所有 agent 选择动作

        Args:
            obs_dict: {agent_id: obs}
            training: 是否训练模式

        Returns:
            actions: {agent_id: action}
        """
        actions = {}
        with torch.no_grad():
            for uid, obs in obs_dict.items():
                if training and random.random() < self.epsilon:
                    actions[uid] = random.randint(0, self.action_dim - 1)
                else:
                    obs_t = torch.FloatTensor(obs).to(self.device)
                    q_values, _ = self.agent_networks[uid](obs_t)
                    actions[uid] = q_values.argmax().item()
        return actions

    def store_transition(self, obs_dict: Dict[int, np.ndarray],
                         actions: Dict[int, int],
                         rewards: Dict[int, float],
                         next_obs_dict: Dict[int, np.ndarray],
                         global_state: np.ndarray,
                         next_global_state: np.ndarray,
                         team_reward: float,
                         done: bool):
        """存储一条经验到回放缓冲区"""
        # 将 obs 和 actions 转换为固定顺序的数组
        obs_list = np.array([obs_dict[i] for i in range(self.num_agents)])
        next_obs_list = np.array([next_obs_dict[i] for i in range(self.num_agents)])
        actions_list = np.array([actions[i] for i in range(self.num_agents)])
        rewards_list = np.array([rewards[i] for i in range(self.num_agents)])

        self.memory.append((
            obs_list, actions_list, rewards_list,
            next_obs_list, global_state, next_global_state,
            team_reward, done
        ))

    def train_step(self) -> Optional[float]:
        """
        执行一次训练步骤

        Returns:
            loss: 训练损失，缓冲区不足时返回 None
        """
        if len(self.memory) < self.batch_size:
            return None

        # 采样批次
        batch = random.sample(self.memory, self.batch_size)

        obs_batch = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        act_batch = torch.LongTensor(np.array([b[1] for b in batch])).to(self.device)
        rew_batch = torch.FloatTensor(np.array([b[2] for b in batch])).to(self.device)
        next_obs_batch = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        state_batch = torch.FloatTensor(np.array([b[4] for b in batch])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([b[5] for b in batch])).to(self.device)
        team_rew_batch = torch.FloatTensor(np.array([b[6] for b in batch])).to(self.device)
        done_batch = torch.FloatTensor(np.array([b[7] for b in batch])).to(self.device)

        # batch_size x num_agents x obs_dim -> 逐 agent 处理
        bs = self.batch_size

        # ====== 计算 Q_tot(s, a) ======
        agent_qs = []
        for i in range(self.num_agents):
            obs_i = obs_batch[:, i, :]  # (batch, obs_dim)
            q_i, _ = self.agent_networks[i](obs_i)  # (batch, action_dim)
            # 选取对应动作的 Q 值
            q_i_a = q_i.gather(1, act_batch[:, i].unsqueeze(1)).squeeze(1)  # (batch,)
            agent_qs.append(q_i_a)
        agent_qs = torch.stack(agent_qs, dim=1)  # (batch, num_agents)

        q_tot = self.mixing_net(agent_qs, state_batch)  # (batch,)

        # ====== 计算 target Q_tot(s', a') ======
        with torch.no_grad():
            next_agent_qs = []
            next_max_qs = []
            for i in range(self.num_agents):
                next_obs_i = next_obs_batch[:, i, :]  # (batch, obs_dim)
                next_q_i, _ = self.target_networks[i](next_obs_i)  # (batch, action_dim)
                next_max_q_i = next_q_i.max(dim=1).values  # (batch,)
                next_agent_qs.append(next_max_q_i)
            next_agent_qs = torch.stack(next_agent_qs, dim=1)  # (batch, num_agents)

            target_q_tot = self.target_mixing_net(next_agent_qs, next_state_batch)  # (batch,)
            target = team_rew_batch + self.gamma * target_q_tot * (1 - done_batch)

        # ====== 计算损失 ======
        loss = nn.SmoothL1Loss()(q_tot, target)

        # ====== 反向传播 ======
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.mixing_net.parameters()) +
            [p for net in self.agent_networks for p in net.parameters()],
            max_norm=10.0
        )
        self.optimizer.step()

        # ====== 软更新目标网络 ======
        if self.train_step_count % self.target_update_freq == 0:
            for i in range(self.num_agents):
                for tp, sp in zip(self.target_networks[i].parameters(),
                                  self.agent_networks[i].parameters()):
                    tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
            for tp, sp in zip(self.target_mixing_net.parameters(),
                              self.mixing_net.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        # 记录统计
        self.train_step_count += 1
        self.loss_history.append(loss.item())
        self.q_tot_history.append(q_tot.mean().item())

        return loss.item()

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        state = {
            'agent_networks': [net.state_dict() for net in self.agent_networks],
            'target_networks': [net.state_dict() for net in self.target_networks],
            'mixing_net': self.mixing_net.state_dict(),
            'target_mixing_net': self.target_mixing_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step_count': self.train_step_count,
        }
        torch.save(state, path)
        print(f"  QMIX 模型已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        state = torch.load(path, map_location=self.device, weights_only=True)
        for i in range(self.num_agents):
            self.agent_networks[i].load_state_dict(state['agent_networks'][i])
            self.target_networks[i].load_state_dict(state['target_networks'][i])
        self.mixing_net.load_state_dict(state['mixing_net'])
        self.target_mixing_net.load_state_dict(state['target_mixing_net'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.epsilon = state['epsilon']
        self.train_step_count = state['train_step_count']
        print(f"  QMIX 模型已加载: {path}")

    def get_avg_loss(self, window: int = 200) -> float:
        """获取最近 window 步的平均 loss"""
        if len(self.loss_history) < window:
            return 0.0
        return np.mean(list(self.loss_history)[-window:])

    def get_avg_q_tot(self, window: int = 200) -> float:
        """获取最近 window 步的平均 Q_tot"""
        if len(self.q_tot_history) < window:
            return 0.0
        return np.mean(list(self.q_tot_history)[-window:])

    def get_stats(self) -> dict:
        """获取训练统计"""
        return {
            'epsilon': self.epsilon,
            'train_steps': self.train_step_count,
            'buffer_size': len(self.memory),
            'avg_loss': self.get_avg_loss(),
            'avg_q_tot': self.get_avg_q_tot(),
        }
