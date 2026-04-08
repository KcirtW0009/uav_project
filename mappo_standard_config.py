"""
MAPPO标准环境优化配置

基于系统分析报告的优化参数配置。
包含学习率调整、探索策略优化、奖励函数修复等配置。
"""

# =============================================================================
# 标准环境配置（适合CPU训练）
# =============================================================================

MAPPO_STANDARD_CONFIG = {
    # ===== 环境参数 =====
    # 标准规模配置（与 main.py --exp mappo 保持一致）
    # 负载率: 200UAV/4BS → ~103%, 280UAV/5BS → ~116%
    "num_uav_list": (200, 280),
    "num_bs_list": (4, 5),
    "num_steps": 100,
    "pos_range": 1000.0,               # 地图范围
    
    # ===== 容量配置 =====
    "bs_capacity_range": (500, 1000),    # 与实验2/3/4保持一致
    
    # ===== 训练参数 =====
    "train_episodes": 500,              # 标准训练轮次
    "eval_episodes": 5,                  # 评估轮次
    "rollout_length": 100,              # PPO rollout长度
    
    # ===== 网络结构 =====
    "hidden_dim": 64,                   # Actor隐藏层（CPU训练用较小网络）
    "critic_hidden_dim": 128,           # Critic隐藏层
    "use_biz_heads": True,              # Business-Aware Actor
    "use_attention_critic": True,       # Attention-Enhanced Critic
    
    # ===== 学习率（优化）=====
    "actor_lr": 1e-4,                  # 提高至1e-4
    "critic_lr": 3e-4,                 # 提高至3e-4
    
    # ===== PPO超参数 =====
    "gamma": 0.95,                      # 折扣因子
    "gae_lambda": 0.95,                 # GAE参数
    "clip_epsilon": 0.2,                # PPO clip范围
    "entropy_coef": 0.05,               # 熵系数（提高到0.05增加探索）
    "value_loss_coef": 0.5,             # 价值损失权重
    
    # ===== 批次参数（CPU优化）=====
    "batch_size": 32,                   # 批次大小（CPU训练用较小批次）
    "num_epochs": 3,                    # 更新轮次（CPU训练减少）
    "num_parallel_envs": 1,              # 并行环境数（CPU训练用1）
    
    # ===== 探索策略（优化）=====
    "initial_epsilon": 0.5,             # 初始探索率
    "min_epsilon": 0.15,                # 最小探索率
    
    # ===== 模仿学习预训练 =====
    "use_pretrain": True,               # 使用预训练
    "pretrain_epochs": 30,              # 预训练轮次
    
    # ===== 监控频率 =====
    "log_interval": 20,                  # 日志间隔
    "eval_interval": 50,                 # 评估间隔
    
    # ===== 早停参数 =====
    "early_stop_patience": 150,         # 早停耐心值
    "early_stop_min_delta": 0.001,      # 最小改善幅度
}


# =============================================================================
# 小规模快速测试配置（用于验证代码正确性）
# =============================================================================

MAPPO_SMALL_TEST_CONFIG = {
    # ===== 环境参数 =====
    "num_uav_list": (20,),              # 小规模测试
    "num_bs_list": (4,),
    "num_steps": 50,                    # 快速测试
    "pos_range": 1000.0,
    
    # ===== 容量配置 =====
    "bs_capacity_range": (500, 1000),
    
    # ===== 训练参数 =====
    "train_episodes": 100,               # 快速测试
    "eval_episodes": 3,
    "rollout_length": 50,
    
    # ===== 网络结构 =====
    "hidden_dim": 64,
    "critic_hidden_dim": 128,
    "use_biz_heads": True,
    "use_attention_critic": True,
    
    # ===== 学习率 ======
    "actor_lr": 1e-4,
    "critic_lr": 3e-4,
    
    # ===== PPO超参数 ======
    "gamma": 0.95,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.05,
    "value_loss_coef": 0.5,
    
    # ===== 批次参数 ======
    "batch_size": 16,
    "num_epochs": 2,
    "num_parallel_envs": 1,
    
    # ===== 探索策略 ======
    "initial_epsilon": 0.5,
    "min_epsilon": 0.15,
    
    # ===== 预训练 ======
    "use_pretrain": True,
    "pretrain_epochs": 20,
    
    # ===== 监控 ======
    "log_interval": 10,
    "eval_interval": 20,
    
    # ===== 早停 ======
    "early_stop_patience": 30,
    "early_stop_min_delta": 0.002,
}


def get_config(config_type="standard"):
    """
    获取配置字典
    
    Args:
        config_type: "standard" | "small_test"
    
    Returns:
        dict: 配置字典
    """
    if config_type == "small_test":
        return MAPPO_SMALL_TEST_CONFIG.copy()
    else:
        return MAPPO_STANDARD_CONFIG.copy()


def print_config(config):
    """打印配置信息"""
    print("\n" + "=" * 60)
    print("MAPPO 配置信息")
    print("=" * 60)
    print(f"  UAV/BS 配置: {list(zip(config['num_uav_list'], config['num_bs_list']))}")
    print(f"  每轮步数: {config['num_steps']}")
    print(f"  训练轮次: {config['train_episodes']}")
    print(f"  容量范围: {config['bs_capacity_range']}")
    print(f"  地图范围: {config['pos_range']}m")
    print(f"  Actor LR: {config['actor_lr']}, Critic LR: {config['critic_lr']}")
    print(f"  Clip Epsilon: {config['clip_epsilon']}, Entropy: {config['entropy_coef']}")
    print(f"  Batch Size: {config['batch_size']}, Epochs: {config['num_epochs']}")
    print(f"  BA Actor: {config['use_biz_heads']}, Attention Critic: {config['use_attention_critic']}")
    print(f"  预训练: {config['use_pretrain']} ({config['pretrain_epochs']} epochs)")
    print("=" * 60)


if __name__ == "__main__":
    print("标准配置:")
    print_config(MAPPO_STANDARD_CONFIG)
    
    print("\n小规模测试配置:")
    print_config(MAPPO_SMALL_TEST_CONFIG)
