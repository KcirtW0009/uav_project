"""
DQN Agent 模块

实现 Deep Q-Network 强化学习智能体，用于 UAV 切换决策。
包含 Q 网络、经验回放、目标网络等标准 DQN 组件。
支持动作掩码（Action Masking）以屏蔽无效动作。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from typing import Tuple, Optional, List


class DQNNetwork(nn.Module):
    """
    Q 网络：输入状态，输出每个动作的 Q 值

    结构: state_dim -> hidden -> hidden -> action_dim
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """
    DQN 智能体

    组件:
    - Q-Network: 主网络，用于选择动作和训练
    - Target Network: 目标网络，用于计算 TD 目标（定期同步）
    - Experience Replay: 经验回放缓冲区
    - Epsilon-Greedy: 探索-利用策略
    - Action Masking: 动作掩码，屏蔽无效动作

    Args:
        state_dim: 状态空间维度
        action_dim: 动作空间维度
        lr: 学习率
        gamma: 折扣因子
        hidden_dim: 隐藏层维度
        buffer_size: 经验回放缓冲区大小
        batch_size: 训练批次大小
        epsilon_start: 初始探索率
        epsilon_end: 最终探索率
        epsilon_decay: 探索率衰减系数
        target_update_freq: 目标网络更新频率(步数)
        device: 计算设备 ('cpu' 或 'cuda')
    """

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 1e-3, gamma: float = 0.99,
                 hidden_dim: int = 128, buffer_size: int = 10000,
                 batch_size: int = 64, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
                 target_update_freq: int = 100, tau: float = 0.005,
                 device: Optional[str] = None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau  # 软更新系数 (Polyak averaging)
        self.train_step_count = 0

        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 网络
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 经验回放
        self.memory = deque(maxlen=buffer_size)

        # Epsilon-Greedy
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 训练统计
        self.loss_history = deque(maxlen=1000)

    def select_action(self, state: np.ndarray, training: bool = True,
                      invalid_actions: Optional[List[int]] = None) -> int:
        """
        选择动作 (epsilon-greedy + 动作掩码)

        Args:
            state: 状态向量
            training: 是否为训练模式（训练时探索，评估时利用）
            invalid_actions: 无效动作索引列表，会被屏蔽（如切换到当前基站）

        Returns:
            动作索引
        """
        valid_actions = self._get_valid_actions(invalid_actions)

        # 探索：偏向 stay 动作（提高"保持连接"的探索概率）
        if training and random.random() < self.epsilon:
            if 0 in valid_actions and random.random() < 0.1:
                return 0  # 10% 概率直接选 stay（从30%降低，鼓励更多切换探索）
            return random.choice(valid_actions)

        # 利用：在有效动作中选 Q 值最大的
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0)
            # 屏蔽无效动作（设为极小值）
            for invalid_idx in (invalid_actions or []):
                q_values[invalid_idx] = -1e9
            return q_values.argmax().item()

    def _get_valid_actions(self, invalid_actions: Optional[List[int]] = None) -> List[int]:
        """获取有效动作列表"""
        if invalid_actions is None:
            return list(range(self.action_dim))
        return [a for a in range(self.action_dim) if a not in invalid_actions]

    def get_invalid_actions(self, connected_bs_id: Optional[int],
                            num_bs: int, num_ratios: int = 3) -> List[int]:
        """
        计算当前状态下应该被屏蔽的无效动作

        Args:
            connected_bs_id: 当前连接的基站 ID（None 表示未连接）
            num_bs: 基站数量
            num_ratios: 每个基站的降级档位数

        Returns:
            无效动作索引列表
        """
        invalid = []
        if connected_bs_id is not None:
            # 屏蔽"切换到当前基站"的所有降级档位
            # 动作映射: 0=stay, 1~num_bs*num_ratios = switch(bs, ratio)
            # bs_i 的动作起始索引: 1 + bs_i * num_ratios
            for r in range(num_ratios):
                action_idx = 1 + connected_bs_id * num_ratios + r
                invalid.append(action_idx)
        return invalid

    def store_transition(self, state: np.ndarray, action: int,
                         reward: float, next_state: np.ndarray, done: bool,
                         next_invalid_actions: Optional[List[int]] = None):
        """存储一条经验到回放缓冲区（含 next_state 的无效动作掩码）"""
        self.memory.append((state, action, reward, next_state, done, next_invalid_actions))

    def train_step(self) -> Optional[float]:
        """
        执行一次训练步骤

        Returns:
            本次训练的 loss 值，如果缓冲区不足返回 None
        """
        if len(self.memory) < self.batch_size:
            return None

        # 采样批次
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, next_invalid_actions = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 当前 Q 值: Q(s, a)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 目标 Q 值: DQN（用目标网络评估）
        # replay buffer 中存有 next_invalid_actions，用于屏蔽无效动作
        with torch.no_grad():
            next_q_all = self.target_network(next_states)
            # 屏蔽 next_state 中的无效动作（设为极小值）
            for i, inv_list in enumerate(next_invalid_actions):
                if inv_list:
                    for idx in inv_list:
                        next_q_all[i, idx] = -1e9
            max_next_q = next_q_all.max(dim=1).values
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # Huber Loss (比 MSE 更鲁棒，防止 Q 值发散时 loss 爆炸)
        loss = nn.SmoothL1Loss()(q_values, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 更新统计
        self.train_step_count += 1
        loss_val = loss.item()
        self.loss_history.append(loss_val)

        # 软更新目标网络 (Polyak averaging): θ_target = τ·θ + (1-τ)·θ_target
        # 比硬拷贝更平滑，防止 TD target 突变导致策略跳变
        for target_param, param in zip(self.target_network.parameters(),
                                       self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return loss_val

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """手动同步目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step_count': self.train_step_count,
        }, path)
        print(f"  模型已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step_count = checkpoint['train_step_count']
        print(f"  模型已加载: {path}")

    def get_avg_loss(self, window: int = 100) -> float:
        """获取最近 window 步的平均 loss"""
        if len(self.loss_history) < window:
            return 0.0
        return np.mean(list(self.loss_history)[-window:])

    def get_stats(self) -> dict:
        """获取 Agent 统计信息"""
        return {
            'epsilon': self.epsilon,
            'train_steps': self.train_step_count,
            'buffer_size': len(self.memory),
            'avg_loss': self.get_avg_loss(),
        }
