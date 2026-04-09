# -*- coding: utf-8 -*-
"""
MAPPO Agent V2 - 前馈网络版本

主要改进：
1. 移除RNN，使用纯前馈网络提高训练稳定性
2. 优化网络初始化参数
3. 改进早停策略
4. 增强奖励函数信号

Author: MAPPO V2 Optimizer
Date: 2026-04-08
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Tuple, Optional, List
from collections import deque
import copy


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


class FeedForwardActorNetwork(nn.Module):
    """
    前馈Actor策略网络 (移除RNN，提高稳定性)
    
    结构:
      - 输入层: obs_dim -> hidden_dim
      - 隐藏层: hidden_dim -> hidden_dim (带残差连接)
      - 业务类型嵌入层
      - 输出头: 按业务类型选择
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128,
                 num_biz_types: int = 3, use_biz_heads: bool = True):
        super().__init__()
        self.use_biz_heads = use_biz_heads
        self.action_dim = action_dim
        self.num_biz_types = num_biz_types
        
        # 业务类型嵌入层
        self.biz_embedding = nn.Embedding(num_biz_types, hidden_dim)
        
        # 前馈特征提取 (替代RNN)
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)  # 新增隐藏层
        
        # Layer Normalization 提高稳定性
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)  # 新增层归一化
        
        if use_biz_heads:
            # BA Actor: 每种业务类型一个独立输出头
            self.biz_heads = nn.ModuleList([
                nn.Linear(hidden_dim, action_dim) for _ in range(num_biz_types)
            ])
        else:
            # 标准 MAPPO: 共享输出头
            self.output_head = nn.Linear(hidden_dim, action_dim)
        
        # 正交初始化
        self._init_weights()
    
    def _init_weights(self):
        """优化的正交初始化"""
        # 前馈层使用较小的gain，提高稳定性
        for module in [self.fc1, self.fc2, self.fc3, self.fc4]:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=1.0)  # 从sqrt(2)减小到1.0
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        # 输出层使用更小的gain，确保初始策略接近均匀分布
        heads = self.biz_heads if self.use_biz_heads else [self.output_head]
        for head in heads:
            nn.init.orthogonal_(head.weight, gain=0.01)  # 从0.5减小到0.01
            nn.init.zeros_(head.bias)
            head.bias.data[0] = 0.1  # 轻微的stay偏好
    
    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None,
                biz_types: torch.Tensor = None):
        """
        前向传播
        
        Args:
            obs: (batch, obs_dim)
            hidden: 未使用，保持接口兼容
            biz_types: (batch,) 业务类型索引
        
        Returns:
            logits: (batch, action_dim)
            new_hidden: None (前馈网络无隐藏状态)
        """
        # 前馈特征提取
        x = torch.relu(self.ln1(self.fc1(obs)))
        x = torch.relu(self.ln2(self.fc2(x))) + x  # 残差连接
        x = torch.relu(self.ln3(self.fc3(x))) + x  # 残差连接
        x = self.ln4(self.fc4(x))  # 最后一层不加激活，使用残差连接
        
        # 添加业务类型嵌入
        if biz_types is not None:
            biz_emb = self.biz_embedding(biz_types)
            x = x + biz_emb
        
        # 计算logits
        batch_size = x.shape[0]
        if self.use_biz_heads and biz_types is not None:
            logits = torch.zeros(batch_size, self.action_dim, device=x.device)
            for bt_idx in range(len(self.biz_heads)):
                mask = (biz_types == bt_idx)
                if mask.any():
                    logits[mask] = self.biz_heads[bt_idx](x[mask])
        else:
            logits = self.output_head(x)
        
        return logits, None  # 返回None作为hidden，保持接口兼容
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                         hidden: torch.Tensor = None,
                         biz_types: torch.Tensor = None):
        """评估动作的log_prob和entropy"""
        logits, _ = self.forward(obs, hidden, biz_types)
        
        # logits clipping 防止 NaN
        logits = torch.clamp(logits, -10.0, 10.0)
        
        dist = Categorical(logits=logits)
        actions_clamped = torch.clamp(actions, 0, logits.shape[1] - 1)
        log_probs = dist.log_prob(actions_clamped)
        entropy = dist.entropy()
        return log_probs, entropy
    
    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """初始化隐藏状态 (前馈网络返回None)"""
        return None


class FeedForwardCriticNetwork(nn.Module):
    """前馈Critic价值网络"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_biz_types: int = 3):
        super().__init__()
        
        # 业务类型嵌入
        self.biz_embedding = nn.Embedding(num_biz_types, hidden_dim)
        
        # 前馈网络
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)  # 新增隐藏层
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Layer Normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)  # 新增层归一化
        
        self._init_weights()
    
    def _init_weights(self):
        """正交初始化"""
        for module in [self.fc1, self.fc2, self.fc3, self.fc4]:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
    
    def forward(self, state: torch.Tensor, biz_types: torch.Tensor = None):
        """前向传播"""
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x))) + x  # 残差连接
        x = torch.relu(self.ln3(self.fc3(x))) + x  # 残差连接
        x = self.ln4(self.fc4(x))  # 最后一层不加激活，使用残差连接
        
        if biz_types is not None:
            # 处理业务类型嵌入
            biz_emb = self.biz_embedding(biz_types)
            # 确保biz_emb的形状与x匹配
            if biz_emb.shape[0] != x.shape[0]:
                # 如果形状不匹配，使用均值
                biz_emb = biz_emb.mean(dim=0, keepdim=True).expand(x.shape[0], -1)
            x = x + biz_emb
        
        value = self.value_head(x)
        return value


class EarlyStoppingMonitor:
    """早停监控器 - 基于验证集性能"""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001, 
                 warmup_steps: int = 50):
        """
        Args:
            patience: 容忍多少个epoch没有改善
            min_delta: 改善的最小阈值
            warmup_steps: 热身步数，在此期间不触发早停
        """
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_steps = warmup_steps
        
        self.best_value = -float('inf')
        self.counter = 0
        self.step = 0
        self.should_stop = False
        
    def __call__(self, value: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            value: 当前验证指标（如满意度）
            
        Returns:
            True if should stop
        """
        self.step += 1
        
        # 热身期间不触发早停
        if self.step < self.warmup_steps:
            return False
        
        # 检查是否有改善
        if value > self.best_value + self.min_delta:
            self.best_value = value
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False
    
    def reset(self):
        """重置状态"""
        self.best_value = -float('inf')
        self.counter = 0
        self.step = 0
        self.should_stop = False


class MAPPOAgentV2:
    """
    MAPPO智能体V2版本
    
    主要改进：
    1. 使用前馈网络替代RNN
    2. 优化训练参数
    3. 改进早停策略 (添加验证集监控)
    4. 增强奖励函数
    """
    
    def __init__(self, num_agents: int, obs_dim: int, state_dim: int, action_dim: int,
                 hidden_dim: int = 128, critic_hidden_dim: int = 256,
                 actor_lr: float = 3e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.1, entropy_coef: float = 0.02,
                 value_coef: float = 0.5, rollout_length: int = 150, num_epochs: int = 5, batch_size: int = 128,
                 use_biz_heads: bool = True, use_attention_critic: bool = True,
                 use_enhanced_algorithm: bool = True, use_pretrain: bool = True,
                 use_hierarchical: bool = True, use_transformer: bool = False,
                 use_data_augmentation: bool = True, train_sample_agents: int = 0,
                 attention_sample_agents: int = 0, num_parallel_envs: int = 1,
                 use_early_stopping: bool = True, early_stop_patience: int = 20):
        
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 训练参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.rollout_length = rollout_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # 配置开关
        self.use_biz_heads = use_biz_heads
        self.use_attention_critic = use_attention_critic
        self.use_enhanced_algorithm = use_enhanced_algorithm
        self.use_pretrain = use_pretrain
        self.use_hierarchical = use_hierarchical
        self.use_transformer = use_transformer
        self.use_data_augmentation = use_data_augmentation
        self.train_sample_agents = train_sample_agents
        self.attention_sample_agents = attention_sample_agents
        self.num_parallel_envs = num_parallel_envs
        
        # 网络
        self.actor = FeedForwardActorNetwork(
            obs_dim, action_dim, hidden_dim, use_biz_heads=use_biz_heads
        )
        self.critic = FeedForwardCriticNetwork(
            state_dim, critic_hidden_dim
        )
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-5)
        
        # 学习率调度器 - 使用余弦退火，后期降低学习率
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.actor_optimizer, T_0=100, T_mult=2, eta_min=1e-6
        )
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.critic_optimizer, T_0=100, T_mult=2, eta_min=1e-6
        )
        
        # Observation Normalizer (running mean/std)
        self.obs_normalizer = ObsNormalizer(obs_dim, decay=0.999, clip_val=5.0)
        
        # 经验缓冲区
        self.buffer = {
            'obs': [],
            'state': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': [],
            'values': [],
            'biz_types': [],
            'advantages': [],  # 存储计算好的advantages
            'returns': [],     # 存储计算好的returns
            'priorities': []   # 存储优先级
        }
        
        # 早停监控
        self.use_early_stopping = use_early_stopping
        if use_early_stopping:
            self.early_stop_monitor = EarlyStoppingMonitor(
                patience=early_stop_patience,
                min_delta=0.001,
                warmup_steps=50
            )
        else:
            self.early_stop_monitor = None
        
        # 训练历史记录
        self.training_history = {
            'episode_rewards': [],
            'episode_satisfactions': [],
            'actor_losses': [],
            'critic_losses': [],
            'kl_divergences': [],
            'entropies': [],
            'value_errors': [],
            'cooperation_rewards': [],  # 合作奖励占比
            'policy_entropies': [],     # 策略熵值
        }
        
        self._current_train_step = 0
        self.best_model_state = None
        self.best_sat = -float('inf')
        
        # 增强算法（用于模仿学习）
        self.enhanced_algorithm = None
    
    def set_enhanced_algorithm(self, algorithm):
        """设置增强算法（用于模仿学习）"""
        self.enhanced_algorithm = algorithm
    
    def collect_demonstrations(self, env, num_demos=1000):
        """收集增强算法的示范数据用于模仿学习"""
        if not self.enhanced_algorithm:
            return []
        
        demonstrations = []
        steps = 0
        
        while steps < num_demos:
            # 重置环境，获取初始观察值和状态
            obs_dict, state = env.reset()
            done = False
            
            while not done and steps < num_demos:
                # 使用增强算法选择动作
                self.enhanced_algorithm.run_step()
                
                # 获取当前状态和动作
                obs = np.array([obs_dict[uav_id] for uav_id in range(env.num_agents)])
                
                # 假设增强算法的动作存储在某个属性中
                # 这里需要根据实际情况调整
                actions = {}
                for uav_id in range(env.num_agents):
                    # 简单实现：选择当前最佳基站
                    best_bs = env.env.uavs[uav_id].connected_bs_id
                    actions[uav_id] = best_bs if best_bs is not None else 0
                
                demonstrations.append((obs, state, actions))
                steps += 1
                
                # 推进环境
                # 由于我们使用增强算法直接操作环境，这里不需要传入动作
                # 只需要推进环境即可
                env.advance_env_only()
                
                # 检查是否完成
                # QMixHandoverEnv没有get_done方法，我们需要自己判断
                # 简单判断：如果所有UAV都连接到基站，则认为完成
                done = all(uav.connected_bs_id is not None for uav in env.env.uavs.values())
                
                # 更新观察值和状态
                # 注意：QMixHandoverEnv的step方法需要动作作为参数
                # 这里我们使用一个空动作字典，因为增强算法已经完成了切换
                empty_actions = {uav_id: 0 for uav_id in range(env.num_agents)}
                obs_dict, state, _, _, done, _ = env.step(empty_actions)
        
        return demonstrations
    
    def imitate_learning(self, demonstrations, epochs=10):
        """模仿学习预训练"""
        if not demonstrations:
            return
        
        print(f"开始模仿学习预训练，共 {len(demonstrations)} 个示范数据")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for obs, state, actions in demonstrations:
                # 转换数据为张量
                obs_tensor = torch.FloatTensor(obs)
                actions_tensor = torch.LongTensor(list(actions.values()))
                
                # 计算预测动作的对数概率
                log_probs, _ = self.actor.evaluate_actions(obs_tensor, actions_tensor, None, None)
                
                # 模仿学习损失：最大化示范动作的对数概率
                loss = -log_probs.mean()
                
                # 反向传播
                self.actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(demonstrations)
            print(f"模仿学习 epoch {epoch+1}/{epochs}, 损失: {avg_loss:.4f}")
        
        print("模仿学习预训练完成")
    
    def select_actions(self, obs_dict, state, biz_types, training=True, env=None):
        """选择动作"""
        with torch.no_grad():
            # 优化：先将列表转换为numpy数组，再转换为张量
            obs_list = [obs_dict[i] for i in range(self.num_agents)]
            obs_tensor = torch.FloatTensor(np.array(obs_list))
            biz_tensor = torch.LongTensor([biz_types[i] for i in range(self.num_agents)])
            
            logits, _ = self.actor(obs_tensor, None, biz_tensor)
            logits = torch.clamp(logits, -10.0, 10.0)
            
            dist = Categorical(logits=logits)
            
            if training:
                actions = dist.sample()
            else:
                actions = torch.argmax(logits, dim=-1)
            
            log_probs = dist.log_prob(actions)
            
            # 为每个agent计算价值
            values = []
            for i in range(self.num_agents):
                # 为每个agent创建单独的状态输入
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                # 为每个agent创建单独的业务类型输入
                biz_type_tensor = torch.LongTensor([biz_types[i]]).unsqueeze(0)
                # 计算价值
                value = self.critic(state_tensor, biz_type_tensor).squeeze().item()
                values.append(value)
            
            actions_dict = {i: actions[i].item() for i in range(self.num_agents)}
            log_probs_dict = {i: log_probs[i].item() for i in range(self.num_agents)}
            values_dict = {i: values[i] for i in range(self.num_agents)}
            
            return actions_dict, log_probs_dict, values_dict, None, None
    
    def update_enhanced_algorithm_prob(self, current_episode, total_episodes):
        """更新增强算法的概率"""
        # 简单实现：随着训练进行，逐渐减少使用增强算法的概率
        pass
    
    def insert_experience(self, step, obs_dict, state, actions, rewards, team_reward, done,
                         log_probs, values, biz_types, hidden=None, obs_augmented=None):
        """插入经验"""
        self.buffer['obs'].append([obs_dict[i] for i in range(self.num_agents)])
        self.buffer['state'].append(state)
        self.buffer['actions'].append([actions[i] for i in range(self.num_agents)])
        self.buffer['rewards'].append([rewards[i] for i in range(self.num_agents)])
        self.buffer['dones'].append(done)
        self.buffer['log_probs'].append([log_probs[i] for i in range(self.num_agents)])
        self.buffer['values'].append([values[i] for i in range(self.num_agents)])
        self.buffer['biz_types'].append([biz_types[i] for i in range(self.num_agents)])
    
    def train(self):
        """训练"""
        if len(self.buffer['obs']) < self.rollout_length:
            return None
        
        # 准备数据
        # 优化：先将列表转换为numpy数组，再转换为张量
        obs = torch.FloatTensor(np.array(self.buffer['obs']))
        state = torch.FloatTensor(np.array(self.buffer['state']))
        actions = torch.LongTensor(np.array(self.buffer['actions']))
        rewards = torch.FloatTensor(np.array(self.buffer['rewards']))
        dones = torch.FloatTensor(np.array(self.buffer['dones']))
        old_log_probs = torch.FloatTensor(np.array(self.buffer['log_probs']))
        old_values = torch.FloatTensor(np.array(self.buffer['values']))
        biz_types = torch.LongTensor(np.array(self.buffer['biz_types']))
        
        # 计算GAE和returns
        advantages, returns = self._compute_gae(rewards, old_values, dones)
        
        # 计算优先级（基于TD误差）
        priorities = self._compute_priorities(rewards, old_values, dones)
        
        # 训练多个epoch
        actor_losses = []
        critic_losses = []
        v_mses = []
        entropies = []
        kl_values = []
        actor_grad_norms = []
        critic_grad_norms = []
        
        # 重塑数据为 (rollout_length * num_agents, ...)
        rollout_length = len(obs)
        num_agents = self.num_agents
        
        obs_flat = obs.view(-1, self.obs_dim)
        state_flat = state.view(-1, self.state_dim).repeat_interleave(num_agents, dim=0)
        actions_flat = actions.view(-1)
        advantages_flat = advantages.view(-1)
        returns_flat = returns.view(-1)
        old_log_probs_flat = old_log_probs.view(-1)
        biz_types_flat = biz_types.view(-1)
        priorities_flat = priorities.view(-1)
        
        # 多阶段训练策略
        phases = [
            {'name': 'warmup', 'clip_epsilon': self.clip_epsilon * 2, 'entropy_coef': self.entropy_coef * 2},
            {'name': 'main', 'clip_epsilon': self.clip_epsilon, 'entropy_coef': self.entropy_coef},
            {'name': 'fine-tune', 'clip_epsilon': self.clip_epsilon * 0.5, 'entropy_coef': self.entropy_coef * 0.5}
        ]
        
        for phase_idx, phase in enumerate(phases):
            print(f"Training phase: {phase['name']}")
            
            # 每个阶段训练不同的轮数
            phase_epochs = 2 if phase_idx == 0 else (3 if phase_idx == 1 else 1)
            
            for epoch in range(phase_epochs):
                # 使用优先级采样
                indices = self._prioritized_sample(priorities_flat, batch_size=self.batch_size)
                
                for start in range(0, len(indices), self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    
                    # 获取batch数据
                    batch_obs = obs_flat[batch_indices]
                    batch_state = state_flat[batch_indices]
                    batch_actions = actions_flat[batch_indices]
                    batch_advantages = advantages_flat[batch_indices]
                    batch_returns = returns_flat[batch_indices]
                    batch_old_log_probs = old_log_probs_flat[batch_indices]
                    batch_biz_types = biz_types_flat[batch_indices]
                    
                    # 标准化advantages
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
                    
                    # 评估动作
                    new_log_probs, entropy = self.actor.evaluate_actions(
                        batch_obs, batch_actions, None, batch_biz_types
                    )
                    
                    # 计算价值
                    new_values = self.critic(batch_state, batch_biz_types).squeeze(-1)
                    
                    # 计算ratio和KL
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    approx_kl = ((ratio - 1) - (new_log_probs - batch_old_log_probs)).mean()
                    
                    # 动态KL阈值
                    if self._current_train_step < 100:
                        kl_threshold = 1.5  # 初期宽松
                    else:
                        kl_threshold = 0.8  # 后期严格
                    
                    # 如果KL过大，跳过这次更新
                    if approx_kl > kl_threshold:
                        continue
                    
                    # 计算actor loss
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - phase['clip_epsilon'], 1 + phase['clip_epsilon']) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - phase['entropy_coef'] * entropy.mean()
                    
                    # 计算critic loss和vMSE
                    v_mse = nn.MSELoss()(new_values, batch_returns)
                    critic_loss = self.value_coef * v_mse
                    
                    # 更新actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=True)  # 保留计算图
                    # 检查梯度是否存在
                    actor_grads = [p.grad for p in self.actor.parameters() if p.grad is not None]
                    if actor_grads:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                        self.actor_optimizer.step()
                        actor_grad_norms.append(grad_norm.item())
                    else:
                        print("WARNING: Actor gradient is None, skipping update")
                    
                    # 更新critic
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    # 检查梯度是否存在
                    critic_grads = [p.grad for p in self.critic.parameters() if p.grad is not None]
                    if critic_grads:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                        self.critic_optimizer.step()
                        critic_grad_norms.append(grad_norm.item())
                    else:
                        print("WARNING: Critic gradient is None, skipping update")
                    
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
                    v_mses.append(v_mse.item())
                    entropies.append(entropy.mean().item())
                    kl_values.append(approx_kl.item())
        
        # 更新学习率
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        self._current_train_step += 1
        
        # 清空缓冲区
        for key in self.buffer:
            self.buffer[key] = []
        
        # 计算训练指标
        train_stats = {
            'actor_loss': np.mean(actor_losses) if actor_losses else 0,
            'critic_loss': np.mean(critic_losses) if critic_losses else 0,
            'value_mse': np.mean(v_mses) if v_mses else 0,
            'entropy': np.mean(entropies) if entropies else 0,
            'kl_divergence': np.mean(kl_values) if kl_values else 0,
            'actor_grad_norm': np.mean(actor_grad_norms) if actor_grad_norms else 0,
            'critic_grad_norm': np.mean(critic_grad_norms) if critic_grad_norms else 0,
        }
        
        return train_stats
    
    def _compute_priorities(self, rewards, values, dones):
        """计算优先级"""
        priorities = torch.zeros_like(rewards)
        last_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            priorities[t] = torch.abs(delta)
        
        return priorities
    
    def _prioritized_sample(self, priorities, batch_size=128):
        """优先级采样"""
        # 使用线性优先级采样
        priorities = priorities.detach().cpu().numpy()
        priorities = np.maximum(priorities, 1e-8)  # 避免优先级为0
        probs = priorities / np.sum(priorities)
        indices = np.random.choice(len(priorities), size=batch_size, p=probs)
        return torch.tensor(indices, dtype=torch.long)
    
    def _compute_gae(self, rewards, values, dones):
        """计算GAE"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        return advantages, returns
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'obs_normalizer': self.obs_normalizer.state_dict(),
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if 'obs_normalizer' in checkpoint:
            self.obs_normalizer.load_state_dict(checkpoint['obs_normalizer'])
    
    def save_best_model(self):
        """保存最佳模型状态到内存"""
        self.best_model_state = {
            'actor': copy.deepcopy(self.actor.state_dict()),
            'critic': copy.deepcopy(self.critic.state_dict()),
            'actor_optimizer': copy.deepcopy(self.actor_optimizer.state_dict()),
            'critic_optimizer': copy.deepcopy(self.critic_optimizer.state_dict()),
        }
    
    def load_best_model(self):
        """从内存加载最佳模型"""
        if self.best_model_state is not None:
            self.actor.load_state_dict(self.best_model_state['actor'])
            self.critic.load_state_dict(self.best_model_state['critic'])
            self.actor_optimizer.load_state_dict(self.best_model_state['actor_optimizer'])
            self.critic_optimizer.load_state_dict(self.best_model_state['critic_optimizer'])
    
    def update_training_history(self, episode_reward, episode_sat, train_stats):
        """更新训练历史"""
        self.training_history['episode_rewards'].append(episode_reward)
        self.training_history['episode_satisfactions'].append(episode_sat)
        if train_stats:
            self.training_history['actor_losses'].append(train_stats.get('actor_loss', 0))
            self.training_history['critic_losses'].append(train_stats.get('critic_loss', 0))
            self.training_history['kl_divergences'].append(train_stats.get('kl_divergence', 0))
            self.training_history['entropies'].append(train_stats.get('entropy', 0))
            self.training_history['value_errors'].append(train_stats.get('value_mse', 0))
            # 计算合作奖励占比（假设episode_reward包含合作奖励）
            if episode_reward > 0:
                cooperation_reward_ratio = 0.5  # 这里需要根据实际情况计算
            else:
                cooperation_reward_ratio = 0
            self.training_history['cooperation_rewards'].append(cooperation_reward_ratio)
            # 记录策略熵值
            self.training_history['policy_entropies'].append(train_stats.get('entropy', 0))
        
        # 检查是否是最佳模型
        if episode_sat > self.best_sat:
            self.best_sat = episode_sat
            self.save_best_model()
    
    def check_early_stop(self, episode_sat):
        """检查是否应该早停"""
        if self.early_stop_monitor is not None:
            return self.early_stop_monitor(episode_sat)
        return False
    
    def reset_hidden(self):
        """重置隐藏状态 (前馈网络无需操作)"""
        pass
    
    def pretrain(self, demonstrations, epochs=100, batch_size=64, validation_split=0.2, 
                 min_loss_threshold=0.01, patience=10):
        """
        优化的模仿学习预训练
        
        改进点：
        1. 添加验证集监控，防止过拟合
        2. 数据增强：添加噪声提高鲁棒性
        3. 早停机制：基于验证损失
        4. 学习率调度：动态调整
        5. 损失阈值：达到满意效果自动停止
        
        Args:
            demonstrations: 示范数据列表，每个元素为 (obs, action, biz_type)
            epochs: 最大预训练轮数
            batch_size: 批量大小
            validation_split: 验证集比例
            min_loss_threshold: 最小损失阈值
            patience: 早停耐心值
        """
        print(f"\n  开始优化版模仿学习预训练...")
        
        if len(demonstrations) == 0:
            print("  警告: 示范数据为空，跳过预训练")
            return
        
        # 准备数据集
        obs_list = []
        action_list = []
        biz_type_list = []
        
        for demo in demonstrations:
            if len(demo) == 3:
                obs, state, actions = demo
                # 处理每个UAV的观察值和动作
                for uav_id in range(len(obs)):
                    uav_obs = obs[uav_id]
                    uav_action = actions.get(uav_id, 0)
                    # 简单处理：默认业务类型为0
                    biz_type = 0
                    obs_list.append(uav_obs)
                    action_list.append(uav_action)
                    biz_type_list.append(biz_type)
            else:
                # 旧格式处理
                obs, action = demo
                biz_type = 0  # 默认业务类型
                obs_list.append(obs)
                action_list.append(action)
                biz_type_list.append(biz_type)
        
        obs_array = np.array(obs_list)
        action_array = np.array(action_list)
        biz_type_array = np.array(biz_type_list, dtype=np.int64)
        
        # 划分训练集和验证集
        n_samples = len(obs_array)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        # 随机打乱
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # 训练集
        train_obs = torch.FloatTensor(obs_array[train_indices])
        train_actions = torch.LongTensor(action_array[train_indices])
        train_biz_types = torch.LongTensor(biz_type_array[train_indices])
        
        # 验证集
        val_obs = torch.FloatTensor(obs_array[val_indices])
        val_actions = torch.LongTensor(action_array[val_indices])
        val_biz_types = torch.LongTensor(biz_type_array[val_indices])
        
        # 预训练优化器 (使用单独的学习率)
        pretrain_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            pretrain_optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 预训练循环
        for epoch in range(epochs):
            # 训练阶段
            self.actor.train()
            train_loss = 0.0
            num_batches = 0
            
            # 打乱训练数据
            train_indices_shuffled = torch.randperm(n_train)
            
            for i in range(0, n_train, batch_size):
                batch_idx = train_indices_shuffled[i:i+batch_size]
                obs_batch = train_obs[batch_idx]
                action_batch = train_actions[batch_idx]
                biz_batch = train_biz_types[batch_idx]
                
                # 数据增强：添加小噪声提高鲁棒性
                if epoch < epochs * 0.5:  # 前50% epochs添加噪声
                    noise = torch.randn_like(obs_batch) * 0.01
                    obs_batch = obs_batch + noise
                
                # 前向传播
                logits, _ = self.actor(obs_batch, None, biz_batch)
                loss = torch.nn.functional.cross_entropy(logits, action_batch)
                
                # 反向传播
                pretrain_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                pretrain_optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = train_loss / max(num_batches, 1)
            
            # 验证阶段
            self.actor.eval()
            with torch.no_grad():
                val_logits, _ = self.actor(val_obs, None, val_biz_types)
                val_loss = torch.nn.functional.cross_entropy(val_logits, val_actions).item()
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # 检查是否达到损失阈值
            if val_loss < min_loss_threshold:
                print(f"  达到损失阈值 {min_loss_threshold}，提前停止")
                break
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  验证损失 {patience} 轮未改善，早停")
                    break
        
        print(f"  预训练完成，最终验证损失: {val_loss:.4f}")
        self.actor.train()  # 恢复训练模式


# 保持向后兼容
MAPPOAgent = MAPPOAgentV2
