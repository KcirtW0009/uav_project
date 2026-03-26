"""
DQN 训练脚本

训练 DQN Agent 进行 UAV 切换决策，并可视化训练过程。
快速验证模式：200 episodes，约 5 分钟完成。
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_system.config import set_global_seed
from uav_system.rl_environment import create_rl_env
from uav_system.rl_agent import DQNAgent


def train_dqn(num_episodes: int = 200, num_bs: int = 8, num_uav: int = 15,
               max_steps: int = 150, seed: int = 42,
               target_uav_id: int = 0, eval_interval: int = 20,
               save_model: bool = True):
    """
    训练 DQN Agent

    Args:
        num_episodes: 训练 episode 数
        num_bs: 基站数量
        num_uav: UAV 总数
        max_steps: 每个 episode 最大步数
        seed: 随机种子
        target_uav_id: RL 控制的目标 UAV
        eval_interval: 评估间隔
        save_model: 是否保存模型
    """
    set_global_seed(seed)
    print("=" * 60)
    print("  DQN 训练 - UAV 切换决策")
    print("=" * 60)
    print(f"  Episodes: {num_episodes}")
    print(f"  配置: {num_bs} 基站 × {num_uav} UAV × {max_steps} 步")
    print(f"  目标 UAV: {target_uav_id}")
    print(f"  评估间隔: 每 {eval_interval} episodes")
    print("=" * 60 + "\n")

    # 创建环境
    rl_env = create_rl_env(
        num_bs=num_bs, num_uav=num_uav, target_uav_id=target_uav_id,
        max_steps=max_steps, seed=seed, skip_recognition=True
    )

    # 创建 Agent
    agent = DQNAgent(
        state_dim=rl_env.state_dim,
        action_dim=rl_env.action_dim,
        lr=5e-4,
        gamma=0.95,
        hidden_dim=128,
        buffer_size=50000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_freq=500,
    )

    # 训练记录
    episode_rewards = []
    episode_satisfactions = []
    episode_handovers = []
    episode_losses = []
    eval_rewards = []
    eval_satisfactions = []
    epsilon_history = []

    t_start_total = time.time()
    best_eval_reward = -float('inf')  # 追踪最佳评估奖励

    for episode in range(num_episodes):
        # 固定拓扑训练：让 DQN 充分收敛，评估时再用不同种子测泛化
        t_start_ep = time.time()
        state = rl_env.reset()
        total_reward = 0.0
        sat_sum = 0.0
        ep_losses = []

        for step_i in range(max_steps):
            # 选择动作（带动作掩码）
            invalid = rl_env.get_invalid_actions()
            action = agent.select_action(state, training=True, invalid_actions=invalid)

            # 执行动作
            next_state, reward, done, info = rl_env.step(action)

            # 存储经验（含 next_state 的无效动作掩码）
            next_invalid = rl_env.get_invalid_actions()
            agent.store_transition(state, action, reward, next_state, float(done),
                                   next_invalid_actions=next_invalid)

            # 训练
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            total_reward += reward
            sat_sum += info['satisfaction']
            state = next_state

            if done:
                break

        # 衰减 epsilon
        agent.decay_epsilon()

        # 记录
        episode_rewards.append(total_reward)
        episode_satisfactions.append(sat_sum / max_steps)
        episode_handovers.append(info.get('total_handovers', 0))
        episode_losses.append(np.mean(ep_losses) if ep_losses else 0.0)
        epsilon_history.append(agent.epsilon)

        ep_time = time.time() - t_start_ep

        # 定期评估（关闭探索，跑 3 次取平均）
        if (episode + 1) % eval_interval == 0 or episode == num_episodes - 1:
            eval_reward, eval_sat = _evaluate(agent, rl_env, num_eval=3, max_steps=max_steps)
            eval_rewards.append((episode + 1, eval_reward))
            eval_satisfactions.append((episode + 1, eval_sat))

            # 保存最佳模型（基于 eval reward，防止过拟合退化）
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_path = os.path.join('experiment_results', 'dqn_model_best.pt')
                agent.save(best_path)

            elapsed = time.time() - t_start_total
            eta = elapsed / (episode + 1) * (num_episodes - episode - 1)
            print(f"  Ep {episode+1:4d}/{num_episodes} | "
                  f"Train R: {total_reward:7.2f} | "
                  f"Eval R: {eval_reward:7.2f} | "
                  f"Eval Sat: {eval_sat:.3f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Loss: {np.mean(ep_losses[-50:]):.4f} | "
                  f"Time: {ep_time:.1f}s | "
                  f"ETA: {eta/60:.1f}min")
        elif (episode + 1) % 10 == 0:
            elapsed = time.time() - t_start_total
            eta = elapsed / (episode + 1) * (num_episodes - episode - 1)
            print(f"  Ep {episode+1:4d}/{num_episodes} | "
                  f"R: {total_reward:7.2f} | "
                  f"Sat: {sat_sum/max_steps:.3f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Time: {ep_time:.1f}s | "
                  f"ETA: {eta/60:.1f}min")

    total_time = time.time() - t_start_total
    print(f"\n  训练完成! 总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")

    # 保存模型
    if save_model:
        model_path = os.path.join('experiment_results', 'dqn_model.pt')
        agent.save(model_path)

    # 保存训练数据
    np.savez(os.path.join('experiment_results', 'dqn_training_data.npz'),
             episode_rewards=episode_rewards,
             episode_satisfactions=episode_satisfactions,
             episode_handovers=episode_handovers,
             episode_losses=episode_losses,
             epsilon_history=epsilon_history,
             eval_episodes=[e[0] for e in eval_rewards],
             eval_rewards=[e[1] for e in eval_rewards],
             eval_satisfactions=[e[1] for e in eval_satisfactions])

    # 绘制训练曲线
    _plot_training_curves(episode_rewards, episode_satisfactions,
                          episode_losses, epsilon_history,
                          eval_rewards, eval_satisfactions,
                          episode_handovers, num_episodes)

    return agent, rl_env


def _evaluate(agent: DQNAgent, rl_env, num_eval: int = 3, max_steps: int = 150):
    """评估 Agent 性能（关闭探索）"""
    eval_rewards = []
    eval_sats = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(num_eval):
        state = rl_env.reset()
        total_reward = 0.0
        sat_sum = 0.0
        for step_i in range(max_steps):
            invalid = rl_env.get_invalid_actions()
            action = agent.select_action(state, training=False, invalid_actions=invalid)
            next_state, reward, done, info = rl_env.step(action)
            total_reward += reward
            sat_sum += info['satisfaction']
            state = next_state
            if done:
                break
        eval_rewards.append(total_reward)
        eval_sats.append(sat_sum / max_steps)

    agent.epsilon = old_epsilon
    return np.mean(eval_rewards), np.mean(eval_sats)


def _plot_training_curves(episode_rewards, episode_satisfactions,
                          episode_losses, epsilon_history,
                          eval_rewards, eval_satisfactions,
                          episode_handovers, num_episodes):
    """绘制并保存训练曲线"""
    os.makedirs('experiment_results', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 奖励曲线（带滑动平均）
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.3, color='#667eea', linewidth=0.8, label='原始奖励')
    if len(episode_rewards) >= 10:
        window = min(10, len(episode_rewards))
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg,
                 color='#667eea', linewidth=2, label=f'滑动平均(w={window})')
    if eval_rewards:
        ax1.scatter([e[0] for e in eval_rewards], [e[1] for e in eval_rewards],
                    color='#f093fb', s=60, zorder=5, marker='D', label='评估奖励')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('总奖励')
    ax1.set_title('DQN 训练 - 奖励曲线')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. 满意度曲线
    ax2 = axes[0, 1]
    ax2.plot(episode_satisfactions, alpha=0.3, color='#4ECDC4', linewidth=0.8, label='训练满意度')
    if len(episode_satisfactions) >= 10:
        window = min(10, len(episode_satisfactions))
        moving_avg = np.convolve(episode_satisfactions, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_satisfactions)), moving_avg,
                 color='#4ECDC4', linewidth=2, label=f'滑动平均(w={window})')
    if eval_satisfactions:
        ax2.scatter([e[0] for e in eval_satisfactions], [e[1] for e in eval_satisfactions],
                    color='#f093fb', s=60, zorder=5, marker='D', label='评估满意度')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('平均满意度')
    ax2.set_title('DQN 训练 - 满意度曲线')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Loss 曲线
    ax3 = axes[1, 0]
    valid_losses = [l for l in episode_losses if l > 0]
    if valid_losses:
        ax3.plot(range(len(valid_losses)), valid_losses, alpha=0.5, color='#FF6B6B', linewidth=0.8)
        if len(valid_losses) >= 10:
            window = min(20, len(valid_losses))
            moving_avg = np.convolve(valid_losses, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(valid_losses)), moving_avg,
                     color='#FF6B6B', linewidth=2, label=f'滑动平均(w={window})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.set_title('DQN 训练 - Loss 曲线')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Epsilon + 切换次数
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    ax4.plot(epsilon_history, color='#764ba2', linewidth=1.5, label='ε (探索率)')
    ax4_twin.plot(episode_handovers, alpha=0.4, color='#fbbf24', linewidth=0.8, label='切换次数')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('ε', color='#764ba2')
    ax4_twin.set_ylabel('切换次数', color='#fbbf24')
    ax4.set_title('DQN 训练 - 探索率 & 切换次数')
    ax4.grid(True, alpha=0.3)

    fig.suptitle('DQN UAV 切换决策训练报告', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('experiment_results/dqn_training_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  训练曲线已保存: experiment_results/dqn_training_curves.png")


if __name__ == '__main__':
    train_dqn(num_episodes=200)
