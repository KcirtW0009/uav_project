"""
MAPPO环境快速验证脚本
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 强制使用CPU

print("=" * 60)
print("MAPPO环境验证")
print("=" * 60)

# 1. 检查PyTorch
print("\n[1] PyTorch检查:")
import torch
print(f"    PyTorch版本: {torch.__version__}")
print(f"    CUDA可用: {torch.cuda.is_available()}")
print(f"    设备: CPU")

# 2. 检查numpy
print("\n[2] NumPy检查:")
import numpy as np
print(f"    NumPy版本: {np.__version__}")

# 3. 检查模块导入
print("\n[3] 模块导入检查:")
from uav_system.experiments_mappo import ExperimentBAMAPPO
print("    [OK] ExperimentBAMAPPO导入成功")

# 4. 检查参数签名
print("\n[4] 参数签名检查:")
import inspect
sig = inspect.signature(ExperimentBAMAPPO.run)
params = list(sig.parameters.keys())
ppo_params = ['gamma', 'gae_lambda', 'clip_epsilon', 'entropy_coef', 'value_loss_coef', 'batch_size', 'num_epochs']
for p in ppo_params:
    if p in params:
        print(f"    [OK] {p} = {sig.parameters[p].default}")
    else:
        print(f"    [MISSING] {p}")

# 5. 检查环境创建
print("\n[5] 环境创建检查:")
from uav_system.qmix_environment import QMixHandoverEnv
env = QMixHandoverEnv(num_bs=2, num_uav=10, max_steps=50)
print(f"    [OK] 环境创建成功")
print(f"         Agent数量: {env.num_agents}")
print(f"         观测维度: {env.obs_dim}")
print(f"         状态维度: {env.state_dim}")
print(f"         动作维度: {env.action_dim}")

# 6. 检查MAPPO Agent创建
print("\n[6] MAPPO Agent创建检查:")
from uav_system.mappo_agent import MAPPOAgent
agent = MAPPOAgent(
    num_agents=env.num_agents,
    obs_dim=env.obs_dim,
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    hidden_dim=64,
    critic_hidden_dim=128,
    actor_lr=1e-4,
    critic_lr=3e-4,
    gamma=0.95,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coef=0.05,
    value_coef=0.5,
    rollout_length=50,
    num_epochs=3,
    batch_size=32,
    use_biz_heads=True,
    use_attention_critic=True,
    use_enhanced_algorithm=True,
    use_pretrain=False,
    use_hierarchical=True,
    use_transformer=False,
    use_data_augmentation=False,
    train_sample_agents=0,
    attention_sample_agents=0,
    num_parallel_envs=1,
)
print(f"    [OK] MAPPO Agent创建成功")
print(f"         Actor设备: {agent.device}")
print(f"         Critic设备: {agent.critic_device if hasattr(agent, 'critic_device') else agent.device}")

# 7. 测试一步交互
print("\n[7] 环境交互测试:")
obs_dict, state = env.reset()
agent.reset_hidden()
actions = agent.select_actions(obs_dict, state, training=False)
print(f"    [OK] 动作选择成功")
print(f"         动作形状: {len(actions)}")

# 清理
del agent, env

print("\n" + "=" * 60)
print("环境验证通过！可以运行MAPPO实验。")
print("=" * 60)
