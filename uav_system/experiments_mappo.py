"""
BA-MAPPO 多智能体强化学习实验

基于 MAPPO (Multi-Agent PPO, Yu et al. 2022 NeurIPS) 的 UAV 切换决策实验，
并引入 Business-Aware Actor 和 Attention-Enhanced Critic 两项改进。

研究脉络:
  实验1: 识别准确率对系统性能的影响
  实验2/2b/2c: 切换算法设计（传统 3GPP A3 → 增强算法）
  实验3: 识别与切换联动系统
  实验4 (本实验): BA-MAPPO 多智能体协同决策
    → 目标: 通过多智能体协同超越增强算法的单步贪心策略

基线设计逻辑:
  - 传统算法 (3GPP A3): 论文核心对比对象，代表工业界基线
  - 增强算法: 本文前序工作，BA-MAPPO 的直接改进对象
  - stay/best_sinr: 仅作参考，不作为核心对比对象
  - 容量设计: 控制负载率在 65%~85%，确保资源竞争真实存在

改进:
  1. Business-Aware Actor: 业务类型特定的独立输出头，差异化策略
  2. Attention-Enhanced Critic: 多头注意力聚合全局信息
  3. 分层策略网络: 高层(是否切换) + 底层(目标基站)
  4. 模仿学习预训练: 从增强算法示范中冷启动
  5. Domain Randomization: 训练时随机化环境参数提升泛化

实验内容:
1. Phase 1 — 训练收敛分析
2. Phase 2 — 算法对比评估 (传统 → 增强 → BA-MAPPO，分业务类型统计)
3. Phase 3 — 多场景泛化验证
4. 可视化 — 训练曲线 / 算法对比 / 动作分布 / 场景泛化
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

from .config import GLOBAL_SEED, set_global_seed, RESULT_DIR, COLORS
from .qmix_environment import QMixHandoverEnv
from .mappo_agent_v2 import MAPPOAgentV2 as MAPPOAgent
from .algorithms import EnhancedHandoverAlgorithm, IntegratedHandoverAlgorithm
from .business import BusinessType

# Import data validation system
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data_validation_system import integrate_with_experiment
except ImportError:
    print("[WARNING] Data validation system not found, continuing without validation")
    integrate_with_experiment = None

# 业务类型中文名称（用于统计输出）
BIZ_TYPE_NAMES = {
    0: '控制信令', 1: '视频回传', 2: '环境监测'
}
BIZ_TYPE_KEYS = [BusinessType.CONTROL_SIGNAL, BusinessType.VIDEO_STREAMING, BusinessType.ENVIRONMENT_MONITORING]


class VectorizedEnvs:
    """并行运行多个环境以加速数据收集

    每个 env 独立 reset/step，收集到所有 env 完成一个 episode 后汇总经验。
    """

    def __init__(self, env_fn, num_envs: int, seed: int = 0):
        """
        Args:
            env_fn: 返回新 QMixHandoverEnv 实例的 callable
            num_envs: 并行环境数
            seed: 基础随机种子
        """
        self.num_envs = num_envs
        self.envs = [env_fn(seed + i) for i in range(num_envs)]
        self.num_agents = self.envs[0].num_agents
        self.obs_dim = self.envs[0].obs_dim
        self.state_dim = self.envs[0].state_dim
        self.action_dim = self.envs[0].action_dim

    def reset(self, bs_capacity_range=None):
        """重置所有环境，返回合并的 obs_dict 和 global_state

        Args:
            bs_capacity_range: 如果提供，每个环境随机化 BS 容量 (min, max)
        """
        all_obs = []
        all_states = []
        for env in self.envs:
            obs, state = env.reset(bs_capacity_range=bs_capacity_range)
            all_obs.append(obs)
            all_states.append(state)
        return all_obs, all_states

    def step(self, actions_list):
        """
        对所有环境执行动作

        Args:
            actions_list: list of dict, 每个元素是一个环境的动作 dict

        Returns:
            all_next_obs, all_next_states, all_rewards, all_team_rewards,
            all_dones, all_infos
        """
        results = []
        for env, actions in zip(self.envs, actions_list):
            results.append(env.step(actions))
        return list(zip(*results))


def _collect_biz_satisfaction(env_or_env_wrapper):
    """
    按业务类型统计满意度（统一接口）。

    Args:
        env_or_env_wrapper: QMixHandoverEnv 实例（env.env 为底层网络环境）

    Returns:
        dict: {
            'avg': float,  # 全局平均满意度
            'per_biz': {0: (mean, std), 1: (mean, std), 2: (mean, std)},
            'critical_sat': float,  # 关键业务(控制信令+视频)平均满意度
        }
    """
    inner_env = env_or_env_wrapper.env if hasattr(env_or_env_wrapper, 'env') else env_or_env_wrapper
    biz_sats = {}
    for bt_val in [0, 1, 2]:
        sats = [uav.current_satisfaction for uav in inner_env.uavs.values()
                if uav.true_business_type.value == bt_val]
        biz_sats[bt_val] = (np.mean(sats), np.std(sats)) if sats else (0.0, 0.0)

    all_sats = [uav.current_satisfaction for uav in inner_env.uavs.values()]
    avg_sat = np.mean(all_sats) if all_sats else 0.0

    critical_sats = [uav.current_satisfaction for uav in inner_env.uavs.values()
                     if uav.true_business_type.value in (0, 1)]
    critical_sat = np.mean(critical_sats) if critical_sats else 0.0

    return {
        'avg': avg_sat,
        'per_biz': biz_sats,
        'critical_sat': critical_sat,
    }


class TrainingHealthMonitor:
    """
    训练健康监控器：自动检测不健康的训练走势并输出警告。

    检测项:
      1. 切换 100% 回滚 (ok=0 持续 N 个 episode)
      2. Reward/Satisfaction 停滞 (连续 N 个 episode 无改善)
      3. Advantage 信号消失 (adv ≈ 0 持续 N 个 episode)
      4. KL 散度爆炸 (kl > 10 或剧烈波动)
      5. Entropy 过低 (探索不足)
      6. Reward 组分恒定 (Δs, biz, act, conn 无变化)
      7. 切换成功率过低 (< 5%)
    """

    def __init__(self, check_interval=10, verbose=True):
        self.check_interval = check_interval
        self.verbose = verbose
        # 历史记录
        self.episode_rewards = []
        self.episode_sats = []
        self.episode_switch_ok = []
        self.episode_switch_rb = []
        self.episode_switch_attempts = []
        self.episode_adv_std = []  # 原始 advantage std (归一化前)
        self.episode_adv = []  # advantage 均值
        self.episode_kl = []
        self.episode_entropy = []
        self.episode_delta_sat = []
        self.episode_biz_reward = []
        self.episode_act_reward = []
        self.episode_conn_reward = []
        self.episode_num = []
        # 告警计数（避免同一问题反复刷屏）
        self._alert_cooldown = {}  # alert_key -> episode_number

    def update(self, ep, reward, sat, train_stats=None,
               sw_attempts=0, sw_ok=0, sw_rb=0,
               delta_sat=0.0, biz_reward=0.0, act_reward=0.0, conn_reward=0.0):
        """每个 episode 结束后调用，记录数据。"""
        self.episode_num.append(ep)
        self.episode_rewards.append(reward)
        self.episode_sats.append(sat)
        self.episode_switch_attempts.append(sw_attempts)
        self.episode_switch_ok.append(sw_ok)
        self.episode_switch_rb.append(sw_rb)
        self.episode_delta_sat.append(delta_sat)
        self.episode_biz_reward.append(biz_reward)
        self.episode_act_reward.append(act_reward)
        self.episode_conn_reward.append(conn_reward)

        if train_stats:
            self.episode_adv.append(train_stats.get('advantage_mean', None))
            # 记录原始 advantage std (归一化前)，供 Health Check 判断信号强度
            self.episode_adv_std.append(train_stats.get('raw_adv_std', None))
            self.episode_kl.append(train_stats.get('approx_kl', None))
            self.episode_entropy.append(train_stats.get('entropy', None))

    def check(self, ep):
        """
        定期健康检查。返回 (is_critical, warnings)。
        is_critical=True 表示检测到严重问题，建议人工介入。
        """
        if not self.verbose:
            return False, []
        n = len(self.episode_rewards)
        window = min(20, n)
        if n < 5:
            return False, []

        warnings = []
        is_critical = False

        # ===== 1. 切换 100% 回滚 =====
        recent_attempts = self.episode_switch_attempts[-window:]
        recent_ok = self.episode_switch_ok[-window:]
        recent_rb = self.episode_switch_rb[-window:]
        total_attempts = sum(recent_attempts)
        total_ok = sum(recent_ok)
        if total_attempts > 50 and total_ok == 0:
            msg = (f"[CRITICAL] 最近 {window} 个 episode 共 {total_attempts} 次切换尝试，"
                   f"成功次数=0 (100% 回滚)。策略无法获得正向切换学习信号。"
                   f"可能原因: 预检查容量门槛过严 / 预训练示范有误。")
            warnings.append(msg)
            is_critical = True

        # ===== 2. 切换成功率过低 =====
        elif total_attempts > 50 and total_ok / total_attempts < 0.05:
            msg = (f"[WARNING] 切换成功率仅 {total_ok/total_attempts:.1%} "
                   f"({total_ok}/{total_attempts})，低于 5%。策略难以学到有效切换。")
            warnings.append(msg)

        # ===== 3. Reward 停滞 =====
        if n >= 20:
            recent_rews = self.episode_rewards[-window:]
            rew_std = np.std(recent_rews)
            rew_mean = np.mean(recent_rews)
            # 变异系数 < 1% 视为停滞
            if rew_std / max(abs(rew_mean), 0.01) < 0.01:
                msg = (f"[WARNING] Reward 完全停滞: 均值={rew_mean:.1f}, 标准差={rew_std:.2f} "
                       f"(变异系数 {rew_std/max(abs(rew_mean),0.01)*100:.2f}%)。"
                       f"PPO 无法从恒定 reward 中学习。")
                warnings.append(msg)

        # ===== 4. Satisfaction 停滞 =====
        if n >= 20:
            recent_sats = self.episode_sats[-window:]
            sat_std = np.std(recent_sats)
            sat_mean = np.mean(recent_sats)
            if sat_std < 0.005:
                msg = (f"[WARNING] Satisfaction 完全停滞: 均值={sat_mean:.4f}, 标准差={sat_std:.4f}。"
                       f"策略未在改善满意度。")
                warnings.append(msg)

        # ===== 5. Advantage 信号消失 =====
        # 使用归一化前的原始 advantage std 判断（归一化后的均值永远≈0，是误报）
        valid_adv_stds = [s for s in self.episode_adv_std[-window:] if s is not None]
        if len(valid_adv_stds) >= 10:
            adv_std_mean = np.mean(valid_adv_stds)
            if adv_std_mean < 0.3:  # V16优化: 0.5→0.3，放宽避免误报
                msg = (f"[WARNING] Advantage 信号消失: 原始 advantage std 均值={adv_std_mean:.4f} (应 > 0.3)。"
                       f"Critic 认为所有动作价值相同 → 策略无法区分好坏动作。"
                       f"建议: 检查 reward 设计，或增大 action 差异化。")
                warnings.append(msg)
                if n >= 30 and adv_std_mean < 0.1:
                    is_critical = True  # 长期 adv std≈0 + reward 不涨 = 致命

        # ===== 6. KL 散度爆炸 =====
        valid_kls = [k for k in self.episode_kl[-window:] if k is not None]
        if len(valid_kls) >= 10:
            kl_max = max(valid_kls)
            kl_std = np.std(valid_kls)
            if kl_max > 1.5:  # V16优化: 1.0→1.5，适应新的KL计算方式
                msg = (f"[WARNING] KL 散度爆炸: max={kl_max:.4f}, std={kl_std:.4f}。"
                       f"策略更新不稳定，部分 epoch 被早停截断。"
                       f"建议: 降低 actor_lr 或增大 clip_epsilon。")
                warnings.append(msg)
            elif kl_std > 1.0:  # V16优化: 0.5→1.0，放宽避免误报
                msg = (f"[INFO] KL 散度波动: std={kl_std:.4f} (range={min(valid_kls):.4f}~{max(valid_kls):.4f})。"
                       f"策略更新幅度不均匀。")
                warnings.append(msg)

        # ===== 7. Entropy 过低 =====
        valid_ents = [e for e in self.episode_entropy[-window:] if e is not None]
        if len(valid_ents) >= 10:
            ent_mean = np.mean(valid_ents)
            if ent_mean < 0.01:
                msg = (f"[WARNING] Entropy 过低: {ent_mean:.4f}，策略严重坍缩。"
                       f"探索几乎为零。建议: 增大 entropy_coef (当前可能为 0.02 → 建议 0.05~0.1)。")
                warnings.append(msg)
                is_critical = True

        # ===== 8. Reward 组分恒定 =====
        if n >= 15:
            recent_ds = self.episode_delta_sat[-min(15, n):]
            recent_biz = self.episode_biz_reward[-min(15, n):]
            recent_act = self.episode_act_reward[-min(15, n):]
            recent_conn = self.episode_conn_reward[-min(15, n):]
            # 所有组分标准差都极小 → reward 信号完全消失
            if (np.std(recent_ds) < 0.001 and np.std(recent_biz) < 0.001
                    and np.std(recent_act) < 0.001 and np.std(recent_conn) < 0.001):
                msg = (f"[CRITICAL] 所有 Reward 组分恒定: Δs={np.mean(recent_ds):.4f}, "
                       f"biz={np.mean(recent_biz):.4f}, act={np.mean(recent_act):.4f}, "
                       f"conn={np.mean(recent_conn):.4f}。"
                       f"学习信号完全消失，继续训练无意义。"
                       f"建议: 检查 reward 归一化 / 修改 reward 设计。")
                warnings.append(msg)
                is_critical = True

        # 去重：同一个 alert_key 在 30 个 episode 内不重复
        filtered = []
        for w in warnings:
            key = w[:60]  # 用前 60 字符做 key
            last_ep = self._alert_cooldown.get(key, -100)
            if ep - last_ep >= 30:
                filtered.append(w)
                self._alert_cooldown[key] = ep
        warnings = filtered

        # 输出
        if warnings:
            print(f"\n  {'!'*60}")
            print(f"  HEALTH CHECK [Episode {ep+1}]")
            print(f"  {'!'*60}")
            for w in warnings:
                print(f"  {w}")
            if is_critical:
                print(f"  >>> 检测到严重问题，建议暂停训练并人工排查 <<<")
            print(f"  {'!'*60}\n")

        return is_critical, warnings


def _run_fixed_action_baseline(env, num_steps, action=0):
    """运行固定动作基线 (action=0: stay, action=1: best_sinr 等)"""
    for step in range(num_steps):
        actions = {uid: action for uid in range(env.num_agents)}
        env.step(actions)


def _run_algo_baseline(env, num_steps, algo_class, enable_lb=False):
    """运行启发式算法基线 (传统/增强算法)

    注意: algo.run_step() 直接操作环境状态执行切换，
    然后 advance_env_only() 推进环境。
    为确保通信指标可被收集，需显式调用 collect_step_metrics()。
    """
    if algo_class == EnhancedHandoverAlgorithm:
        # 在MAPPO实验中使用优化的权重配置
        algo = algo_class(env.env, weight_config='optimized')
    else:
        algo = algo_class(env.env)
    for step in range(num_steps):
        kwargs = {}
        if enable_lb and algo_class == EnhancedHandoverAlgorithm:
            kwargs['enable_load_balancing'] = True
        algo.run_step(**kwargs)
        env.advance_env_only()
        # 收集通信质量指标（Ping抖动、丢包率、QoS违规率）
        env.collect_step_metrics()
        # 注入算法内部记录的切换延迟
        if hasattr(algo, 'switching_latency_history') and algo.switching_latency_history:
            env._communication_metrics['handover_latencies'].extend(algo.switching_latency_history[-10:])


class ExperimentBAMAPPO:
    """BA-MAPPO 多智能体强化学习实验"""

    MODEL_DIR = os.path.join(RESULT_DIR, 'mappo_models')
    RESULT_FILE = os.path.join(RESULT_DIR, 'mappo_experiment_data.pkl')

    @staticmethod
    def run(num_uav_list=(30, 80, 150),
            num_bs_list=(4, 6, 8),
            num_steps=150,
            train_episodes=100, eval_episodes=5,
            bs_capacity_range=(500, 1000),
            pos_range=1000,
            load_models=False, phase='both',
            verbose=True,
            # BA-MAPPO 配置开关
            use_biz_heads=True,
            use_attention_critic=True,
            rollout_length=150,
            # V16优化: 降低 actor_lr 解决 KL 散度爆炸 (1e-4 → 5e-5)
            actor_lr=5e-5, critic_lr=3e-4,
            hidden_dim=64, critic_hidden_dim=128,
            # 训练效率优化
            train_sample_agents=0,
            attention_sample_agents=0,
            num_parallel_envs=1,
            # PPO超参数（与mappo_standard_config对齐）
            gamma=0.95,
            gae_lambda=0.95,
            clip_epsilon=0.3,  # V16优化: 0.2→0.3，允许更大策略更新,
            entropy_coef=0.05,
            value_loss_coef=0.5,
            batch_size=32,
            num_epochs=3):
        """
        运行 BA-MAPPO 实验

        Args:
            num_uav_list: 要测试的 UAV 数量列表
            num_bs_list: 基站数量列表（与 num_uav_list 一一对应）
            num_steps: 每个 episode 的步数
            train_episodes: 训练 episodes 数
            eval_episodes: 评估重复次数
            bs_capacity_range: 基站容量范围（与主实验 default 场景对齐: (500, 1000)）
            pos_range: 地图范围（与主实验一致: 1000m）
            load_models: 是否加载已有模型
            phase: 运行阶段 'both' / 'phase1' / 'phase2'
            verbose: 是否打印详细信息
            use_biz_heads: 是否启用 BA Actor
            use_attention_critic: 是否启用 Attention Critic
            rollout_length: rollout 长度
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            hidden_dim: Actor 隐藏层维度
            critic_hidden_dim: Critic 隐藏层维度
        """
        # 确保 num_bs_list 与 num_uav_list 长度一致
        if len(num_bs_list) != len(num_uav_list):
            if len(num_bs_list) == 1:
                num_bs_list = list(num_bs_list) * len(num_uav_list)
            else:
                raise ValueError(f"num_bs_list 长度({len(num_bs_list)}) 必须与 num_uav_list 长度({len(num_uav_list)}) 一致")

        # 计算各规模的负载率（用于打印信息）
        avg_demand_per_uav = 0.4 * 0.5 + 0.3 * 50 + 0.3 * 1.0  # 15.5 Mbps
        avg_bs_cap = sum(bs_capacity_range) / 2
        load_info = []
        for nu, nb in zip(num_uav_list, num_bs_list):
            load_rate = nu * avg_demand_per_uav / (nb * avg_bs_cap) * 100
            load_info.append(f"UAV={nu}/BS={nb} → 负载率~{load_rate:.0f}%")

        mode_name = "BA-MAPPO" if use_biz_heads and use_attention_critic else \
                    "MAPPO+BA" if use_biz_heads else \
                    "MAPPO+Attn" if use_attention_critic else "MAPPO"

        print("\n" + "=" * 80)
        print(f"{mode_name} 多智能体强化学习实验")
        print("=" * 80)
        print(f"  UAV/BS 配置: {list(zip(num_uav_list, num_bs_list))}")
        print(f"  负载率估算: {', '.join(load_info)}")
        print(f"  训练 episodes: {train_episodes}")
        print(f"  评估重复次数: {eval_episodes}")
        print(f"  容量范围: {bs_capacity_range} (与主实验 default 场景对齐)")
        print(f"  地图范围: {pos_range}m")
        print(f"  BA Actor: {use_biz_heads}")
        print(f"  Attention Critic: {use_attention_critic}")
        print(f"  Rollout 长度: {rollout_length}")
        print(f"  Actor LR: {actor_lr}, Critic LR: {critic_lr}")
        print("=" * 80)

        os.makedirs(ExperimentBAMAPPO.MODEL_DIR, exist_ok=True)

        all_results = {
            'config': {
                'mode_name': mode_name,
                'use_biz_heads': use_biz_heads,
                'use_attention_critic': use_attention_critic,
                'rollout_length': rollout_length,
                'actor_lr': actor_lr,
                'critic_lr': critic_lr,
            }
        }

        if phase in ('both', 'phase1'):
            training_results = ExperimentBAMAPPO._phase1_training(
                num_uav_list, num_bs_list, num_steps, train_episodes,
                bs_capacity_range, pos_range, load_models, verbose,
                use_biz_heads, use_attention_critic,
                rollout_length, actor_lr, critic_lr,
                hidden_dim, critic_hidden_dim,
                train_sample_agents, attention_sample_agents, num_parallel_envs,
                gamma, gae_lambda, clip_epsilon, entropy_coef, value_loss_coef,
                batch_size, num_epochs,
            )
            all_results['training'] = training_results

        if phase in ('both', 'phase2'):
            eval_results = ExperimentBAMAPPO._phase2_evaluation(
                num_uav_list, num_bs_list, num_steps, eval_episodes,
                bs_capacity_range, pos_range, load_models, verbose,
                use_biz_heads, use_attention_critic,
                hidden_dim, critic_hidden_dim,
                attention_sample_agents=attention_sample_agents,
            )
            all_results['evaluation'] = eval_results

            # 构建 trained_uav_bs_map 供后续使用
            trained_uav_bs_map = dict(zip(num_uav_list, num_bs_list))

            # 跳过Phase 3模块，简化实验流程

        # 可视化
        ExperimentBAMAPPO._plot_all(all_results, num_uav_list)

        # 保存结果
        with open(ExperimentBAMAPPO.RESULT_FILE, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\n  实验数据已保存: {ExperimentBAMAPPO.RESULT_FILE}")

        return all_results

    # ==================== Phase 1: 训练 ====================

    @staticmethod
    def _phase1_training(num_uav_list, num_bs_list, num_steps, train_episodes,
                         bs_capacity_range, pos_range, load_models, verbose,
                         use_biz_heads, use_attention_critic,
                         rollout_length, actor_lr, critic_lr,
                         hidden_dim, critic_hidden_dim,
                         train_sample_agents, attention_sample_agents, num_parallel_envs,
                         gamma, gae_lambda, clip_epsilon, entropy_coef, value_loss_coef,
                         batch_size, num_epochs):
        """Phase 1: 训练收敛分析"""
        print("\n" + "-" * 60)
        print("Phase 1: BA-MAPPO 训练收敛分析")
        print("-" * 60)
        if train_sample_agents > 0:
            print(f"  [优化] Agent采样: PPO更新时随机采样 {train_sample_agents} 个agent")
        if attention_sample_agents > 0:
            print(f"  [优化] Attention采样: Critic注意力计算采样 {attention_sample_agents} 个agent")
        if num_parallel_envs > 1:
            print(f"  [优化] 并行环境: {num_parallel_envs} 个环境并行收集数据")
        print("-" * 60)

        training_results = {}

        for num_uav, num_bs in zip(num_uav_list, num_bs_list):
            avg_demand = 0.4 * 0.5 + 0.3 * 50 + 0.3 * 1.0
            avg_cap = sum(bs_capacity_range) / 2
            load_rate = num_uav * avg_demand / (num_bs * avg_cap) * 100
            print(f"\n>>> 训练 UAV={num_uav}/BS={num_bs} (预估负载率~{load_rate:.0f}%) <<<")

            set_global_seed(GLOBAL_SEED + num_uav * 100)
            model_path = os.path.join(ExperimentBAMAPPO.MODEL_DIR,
                                      f'mappo_{num_bs}bs_{num_uav}uav.pt')

            # 直接在标准环境中训练，确保模型能够适应真实场景
            # 移除简单环境，避免环境变化过于剧烈导致模型不适应
            env = QMixHandoverEnv(
                num_bs=num_bs, num_uav=num_uav,
                max_steps=num_steps, seed=GLOBAL_SEED + num_uav * 100,
                bs_capacity_range=bs_capacity_range,
                pos_range=pos_range,
            )
            env.reset_normalizer()  # 训练前重置 normalizer
            
            # 为了保持代码兼容性，将 env 赋值给 simple_env
            simple_env = env

            # 初始化智能体（使用传入的PPO超参数，不再硬编码）
            agent = MAPPOAgent(
                num_agents=env.num_agents,
                obs_dim=env.obs_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=hidden_dim,
                critic_hidden_dim=critic_hidden_dim,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_epsilon=clip_epsilon,
                entropy_coef=entropy_coef,
                value_coef=value_loss_coef,
                rollout_length=max(rollout_length, num_steps),
                num_epochs=num_epochs,
                batch_size=batch_size,
                use_biz_heads=use_biz_heads,
                use_attention_critic=use_attention_critic,
                use_enhanced_algorithm=True,
                use_pretrain=True,
                use_hierarchical=True,
                use_transformer=False,
                use_data_augmentation=True,
                train_sample_agents=train_sample_agents,
                attention_sample_agents=attention_sample_agents,
                num_parallel_envs=num_parallel_envs,
            )
            
            # 初始化增强算法
            enhanced_algorithm = EnhancedHandoverAlgorithm(env.env, weight_config='optimized')
            agent.set_enhanced_algorithm(enhanced_algorithm)
            
            # 执行模仿学习预训练
            if agent.use_pretrain and not (load_models and os.path.exists(model_path)):
                # 收集示范数据
                demonstrations = agent.collect_demonstrations(simple_env, num_demos=1000)
                # 预训练
                agent.pretrain(demonstrations, epochs=50, batch_size=64)

            if load_models and os.path.exists(model_path):
                agent.load(model_path)
                training_results[num_uav] = {'loaded': True}
                print(f"  已加载模型: {model_path}")
                continue
            
            # 方案2: 添加obs_normalizer预热
            if verbose:
                print(f"\n  [WARMUP] 开始obs_normalizer预热...")
            
            warmup_episodes = 10
            for warmup_ep in range(warmup_episodes):
                obs_dict, global_state = env.reset()
                agent.reset_hidden()
                
                for step in range(num_steps):
                    # 获取业务类型
                    biz_types = {}
                    for uid in range(env.num_agents):
                        uav = env.env.uavs[uid]
                        biz_types[uid] = uav.true_business_type.value
                    
                    # 全部使用stay动作，避免影响模型
                    actions = {uid: 0 for uid in range(env.num_agents)}
                    
                    # 执行动作
                    next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
                    
                    # 选择动作（仅用于获取log_probs和values，不执行）
                    _, log_probs, values, pre_hidden, _ = agent.select_actions(
                        obs_dict, global_state, biz_types, training=True, env=env
                    )
                    
                    # 存储经验（仅用于更新normalizer）
                    agent.insert_experience(
                        step, obs_dict, global_state, actions,
                        rewards, team_reward, done, log_probs, values,
                        biz_types, pre_hidden
                    )
                    
                    obs_dict = next_obs
                    global_state = next_state
                
                # 不进行训练，只更新normalizer
                if verbose and (warmup_ep + 1) % 5 == 0:
                    print(f"  预热进度: {warmup_ep + 1}/{warmup_episodes}")
            
            if verbose:
                print(f"  [WARMUP] 预热完成! Normalizer已适应数据分布")
                # 打印normalizer状态
                if hasattr(agent, 'obs_normalizer'):
                    mean_sample = agent.obs_normalizer.mean[:5] if hasattr(agent.obs_normalizer, 'mean') else []
                    var_sample = agent.obs_normalizer.var[:5] if hasattr(agent.obs_normalizer, 'var') else []
                    print(f"  Normalizer mean (前5维): {mean_sample}")
                    print(f"  Normalizer var (前5维): {var_sample}")

            # 清空 warmup 积累的经验，避免污染正式训练
            for key in agent.buffer:
                agent.buffer[key] = []

            episode_rewards = []
            episode_satisfactions = []
            episode_actor_losses = []
            episode_critic_losses = []
            episode_entropies = []
            episode_actor_grads = []
            episode_value_mses = []
            best_reward = float('-inf')
            best_sat = float('-inf')  # satisfaction-based 模型选择
            save_interval = 50
            health_monitor = TrainingHealthMonitor(check_interval=10, verbose=True)
            # 分离 best 和 latest 模型路径
            best_model_path = model_path.replace('.pt', '_best.pt')
            latest_model_path = model_path.replace('.pt', '_latest.pt')
            # ---- Early stopping 参数 (V2: 放宽限制，延长训练) ----
            early_stop_patience = train_episodes // 2       # ~100轮无改善才停止 (原~66)
            early_stop_min_delta = 0.001                    # 更敏感的检测 (原0.002)
            early_stop_warmup = train_episodes // 4         # 前25%不计入 (原20%)
            no_improve_count = 0
            early_stopped = False
            # 早期健康检查: 在 10% 训练进度时检查 reward 是否在正增长
            health_check_ep = max(10, train_episodes // 10)
            mid_check_eps = [2 * health_check_ep, 3 * health_check_ep]  # Ep200, Ep300

            # 设置 LR schedule 的总步数
            agent._total_train_steps = train_episodes
            agent._current_train_step = 0

            # 直接在标准环境中训练，确保模型能够适应真实场景
            print("\n  开始训练：标准环境 + Domain Randomization")
            simple_episodes = 0  # 移除简单环境训练
            for ep in range(simple_episodes):
                obs_dict, global_state = simple_env.reset()
                agent.reset_hidden()
                episode_reward = 0.0
                episode_sat = []
                # ---- 诊断: reward 组分分解 ----
                ep_action_counts = {'stay': 0, 'switch': 0}
                ep_per_action = np.zeros(simple_env.action_dim)
                ep_switch_success = 0
                ep_switch_fail = 0
                ep_disconnected_steps = 0
                ep_reward_diag = {'delta_sum': 0, 'value_reward': 0, 'biz_reward': 0, 'action_reward': 0,
                                  'connect_reward': 0, 'good_switch': 0, 'bad_switch': 0,
                                  'raw_mean': 0, 'norm_mean': 0, 'count': 0,
                                  'switch_attempts': 0, 'switch_success': 0, 'switch_rollback': 0, 'switch_disconnect': 0}
                # 业务类型统计
                ep_biz_stats = {}
                for bt in range(3):
                    ep_biz_stats[bt] = {'stay': 0, 'switch': 0, 'satisfaction': [], 'reward': []}

                for step in range(num_steps):
                    # 获取 biz_types
                    biz_types = {}
                    for uid in range(simple_env.num_agents):
                        uav = simple_env.env.uavs[uid]
                        biz_types[uid] = uav.true_business_type.value

                    # 更新增强算法的使用概率
                    agent.update_enhanced_algorithm_prob(ep, simple_episodes)

                    # 选择动作 + 获取 log_probs, values, pre-step hidden
                    actions, log_probs, values, pre_hidden, obs_aug = agent.select_actions(
                        obs_dict, global_state, biz_types, training=True, env=simple_env
                    )

                    # 诊断: 统计 action 分布
                    for uid, a in actions.items():
                        # 确保动作索引在有效范围内
                        if a < len(ep_per_action):
                            ep_per_action[a] += 1
                        else:
                            # 对于超出范围的动作，统计到最后一个类别
                            ep_per_action[-1] += 1
                        if a == 0:
                            ep_action_counts['stay'] += 1
                        else:
                            ep_action_counts['switch'] += 1
                        # 更新业务类型统计
                        biz_type = biz_types[uid]
                        if a == 0:
                            ep_biz_stats[biz_type]['stay'] += 1
                        else:
                            ep_biz_stats[biz_type]['switch'] += 1

                    # 执行动作
                    next_obs, next_state, rewards, team_reward, done, info = simple_env.step(actions)

                    # 诊断: 断连率 + reward 组分
                    if info['connected_rate'] < 1.0:
                        ep_disconnected_steps += 1
                    if 'reward_diag' in info:
                        rd = info['reward_diag']
                        for k in ep_reward_diag:
                            if k == 'count':
                                ep_reward_diag[k] += 1
                            elif k in ('good_switch', 'bad_switch'):
                                ep_reward_diag[k] += rd.get(k, 0)
                            else:
                                ep_reward_diag[k] += rd.get(k, 0)

                    # 存储经验 (传入 biz_types + hidden state 供训练时使用)
                    agent.insert_experience(
                        step, obs_dict, global_state, actions,
                        rewards, team_reward, done, log_probs, values,
                        biz_types, pre_hidden, obs_augmented=obs_aug
                    )

                    obs_dict = next_obs
                    global_state = next_state
                    episode_reward += team_reward
                    episode_sat.append(info['avg_satisfaction'])
                    
                    # 更新业务类型满意度和奖励统计
                    for uid in range(simple_env.num_agents):
                        biz_type = biz_types[uid]
                        uav = simple_env.env.uavs[uid]
                        ep_biz_stats[biz_type]['satisfaction'].append(uav.current_satisfaction)
                        ep_biz_stats[biz_type]['reward'].append(rewards.get(uid, 0))

                episode_rewards.append(episode_reward)
                episode_satisfactions.append(np.mean(episode_sat))

                # PPO 更新 (每个 episode 结束后，包括第一个 episode)
                train_stats = agent.train()
                if train_stats:
                    episode_actor_losses.append(train_stats['actor_loss'])
                    episode_critic_losses.append(train_stats['critic_loss'])
                    episode_entropies.append(train_stats['entropy'])
                    if 'actor_grad_norm' in train_stats:
                        episode_actor_grads.append(train_stats['actor_grad_norm'])
                    if 'value_mse' in train_stats:
                        episode_value_mses.append(train_stats['value_mse'])

                if verbose and (ep + 1) % 30 == 0:
                    avg_al = np.mean(episode_actor_losses[-20:]) if episode_actor_losses else 0
                    avg_cl = np.mean(episode_critic_losses[-20:]) if episode_critic_losses else 0
                    avg_ent = np.mean(episode_entropies[-20:]) if episode_entropies else 0
                    avg_ag = np.mean(episode_actor_grads[-20:]) if episode_actor_grads else None
                    avg_vmse = np.mean(episode_value_mses[-20:]) if episode_value_mses else None
                    recent_rews = episode_rewards[-30:]
                    stay_pct = ep_action_counts['stay'] / max(sum(ep_action_counts.values()), 1) * 100
                    # reward 组分 + 切换诊断摘要
                    n = max(ep_reward_diag['count'], 1)
                    rd_str = (f"Δs={ep_reward_diag['delta_sum']/n:.2f} "
                              f"biz={ep_reward_diag['biz_reward']/n:.2f} "
                              f"act={ep_reward_diag['action_reward']/n:.2f} "
                              f"conn={ep_reward_diag['connect_reward']/n:.2f}")
                    sa = ep_reward_diag.get('switch_attempts', 0)
                    ss = ep_reward_diag.get('switch_success', 0)
                    sr = ep_reward_diag.get('switch_rollback', 0)
                    sd = ep_reward_diag.get('switch_disconnect', 0)
                    sw_str = f"sw={sa}(ok={ss},rb={sr},dc={sd})" if sa > 0 else "sw=0"
                    grad_str = f"{avg_ag:.2f}" if avg_ag is not None else "N/A"
                    vmse_str = f"{avg_vmse:.1f}" if avg_vmse is not None else "N/A"
                    print(f"  简单环境 Episode {ep+1}/{simple_episodes}: "
                          f"reward={episode_reward:.1f}(μ={np.mean(recent_rews):.1f},σ={np.std(recent_rews):.1f}), "
                          f"sat={np.mean(episode_sat):.3f}, "
                          f"stay={stay_pct:.0f}%, {sw_str}, "
                          f"a_loss={avg_al:.4f}, c_loss={avg_cl:.2f}, "
                          f"H={avg_ent:.3f} | {rd_str}")

            # 直接在标准环境中训练 + Domain Randomization
            # ---- Vectorized 环境: 并行收集数据提高效率 ----
            num_parallel = agent.num_parallel_envs
            use_vectorized = num_parallel > 1

            if use_vectorized:
                def _make_env(seed_val):
                    return QMixHandoverEnv(
                        num_bs=num_bs, num_uav=num_uav,
                        max_steps=num_steps, seed=seed_val,
                        bs_capacity_range=bs_capacity_range,
                        pos_range=pos_range,
                    )
                vec_envs = VectorizedEnvs(
                    _make_env, num_envs=num_parallel, seed=GLOBAL_SEED + num_uav * 100
                )
                print(f"  [VECTORIZED] 使用 {num_parallel} 个并行环境收集数据")

            for ep in range(train_episodes):
                # Domain Randomization: 随机化环境参数 (±20%，避免低负载场景)
                random_capacity_range = (
                    int(bs_capacity_range[0] * (0.9 + 0.2 * np.random.rand())),
                    int(bs_capacity_range[1] * (0.9 + 0.2 * np.random.rand()))
                )
                
                if use_vectorized:
                    # 重置所有并行环境（传入 DR 随机化容量）
                    all_obs, all_states = vec_envs.reset(bs_capacity_range=random_capacity_range)
                    # 使用第一个环境的参数做 reset hidden 和后续 select_actions
                    obs_dict = all_obs[0]
                    global_state = all_states[0]
                else:
                    obs_dict, global_state = env.reset(bs_capacity_range=random_capacity_range)
                agent.reset_hidden()
                episode_reward = 0.0
                episode_sat = []
                # ---- 诊断: reward 组分分解 ----
                ep_action_counts = {'stay': 0, 'switch': 0}
                ep_per_action = np.zeros(env.action_dim)
                ep_switch_success = 0
                ep_switch_fail = 0
                ep_disconnected_steps = 0
                ep_reward_diag = {'delta_sum': 0, 'value_reward': 0, 'biz_reward': 0, 'action_reward': 0,
                                  'connect_reward': 0, 'good_switch': 0, 'bad_switch': 0,
                                  'raw_mean': 0, 'norm_mean': 0, 'count': 0,
                                  'switch_attempts': 0, 'switch_success': 0, 'switch_rollback': 0, 'switch_disconnect': 0}
                # 业务类型统计
                ep_biz_stats = {}
                for bt in range(3):
                    ep_biz_stats[bt] = {'stay': 0, 'switch': 0, 'satisfaction': [], 'reward': []}

                for step in range(num_steps):
                    # 获取 biz_types
                    biz_types = {}
                    current_env = vec_envs.envs[0] if use_vectorized else env
                    for uid in range(current_env.num_agents):
                        uav = current_env.env.uavs[uid]
                        biz_types[uid] = uav.true_business_type.value

                    # 更新增强算法的使用概率
                    agent.update_enhanced_algorithm_prob(ep - simple_episodes, train_episodes - simple_episodes)

                    # 选择动作 + 获取 log_probs, values, pre-step hidden
                    actions, log_probs, values, pre_hidden, obs_aug = agent.select_actions(
                        obs_dict, global_state, biz_types, training=True, env=current_env
                    )

                    # 诊断: 统计 action 分布
                    for uid, a in actions.items():
                        if a < len(ep_per_action):
                            ep_per_action[a] += 1
                        else:
                            ep_per_action[-1] += 1
                        if a == 0:
                            ep_action_counts['stay'] += 1
                        else:
                            ep_action_counts['switch'] += 1
                        biz_type = biz_types[uid]
                        if a == 0:
                            ep_biz_stats[biz_type]['stay'] += 1
                        else:
                            ep_biz_stats[biz_type]['switch'] += 1

                    if use_vectorized:
                        # 向量化: 所有环境执行同一组动作
                        actions_list = [actions] * num_parallel
                        results = vec_envs.step(actions_list)
                        all_next_obs, all_next_states, all_rewards, all_team_rewards, all_dones, all_infos = results
                        # 使用主环境的结果
                        next_obs, next_state, rewards, team_reward, done, info = (
                            all_next_obs[0], all_next_states[0], all_rewards[0],
                            all_team_rewards[0], all_dones[0], all_infos[0]
                        )
                        # 累加所有环境的 reward 和满意度
                        for ei in range(num_parallel):
                            episode_reward += all_team_rewards[ei]
                            episode_sat.append(all_infos[ei]['avg_satisfaction'])
                            if 'reward_diag' in all_infos[ei]:
                                rd = all_infos[ei]['reward_diag']
                                for k in ep_reward_diag:
                                    if k == 'count':
                                        ep_reward_diag[k] += 1
                                    elif k in ('good_switch', 'bad_switch'):
                                        ep_reward_diag[k] += rd.get(k, 0)
                                    else:
                                        ep_reward_diag[k] += rd.get(k, 0)
                    else:
                        next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
                        episode_reward += team_reward
                        episode_sat.append(info['avg_satisfaction'])

                    # 诊断: 断连率
                    if info['connected_rate'] < 1.0:
                        ep_disconnected_steps += 1
                    if not use_vectorized and 'reward_diag' in info:
                        rd = info['reward_diag']
                        for k in ep_reward_diag:
                            if k == 'count':
                                ep_reward_diag[k] += 1
                            elif k in ('good_switch', 'bad_switch'):
                                ep_reward_diag[k] += rd.get(k, 0)
                            else:
                                ep_reward_diag[k] += rd.get(k, 0)

                    # 存储经验 (仅主环境)
                    agent.insert_experience(
                        step, obs_dict, global_state, actions,
                        rewards, team_reward, done, log_probs, values,
                        biz_types, pre_hidden, obs_augmented=obs_aug
                    )

                    obs_dict = next_obs
                    global_state = next_state

                    # 更新业务类型满意度和奖励统计 (主环境)
                    for uid in range(current_env.num_agents):
                        biz_type = biz_types[uid]
                        uav = current_env.env.uavs[uid]
                        ep_biz_stats[biz_type]['satisfaction'].append(uav.current_satisfaction)
                        ep_biz_stats[biz_type]['reward'].append(rewards.get(uid, 0))

                episode_rewards.append(episode_reward)
                episode_satisfactions.append(np.mean(episode_sat))

                # PPO 更新 (每个 episode 结束后，包括第一个 episode)
                train_stats = agent.train()
                if train_stats:
                    episode_actor_losses.append(train_stats['actor_loss'])
                    episode_critic_losses.append(train_stats['critic_loss'])
                    episode_entropies.append(train_stats['entropy'])
                    if 'actor_grad_norm' in train_stats:
                        episode_actor_grads.append(train_stats['actor_grad_norm'])
                    if 'value_mse' in train_stats:
                        episode_value_mses.append(train_stats['value_mse'])

                # ---- 持续健康监控 (每 10 个 episode) ----
                n_diag = max(ep_reward_diag['count'], 1)
                health_monitor.update(
                    ep=ep,
                    reward=episode_reward,
                    sat=np.mean(episode_sat) if episode_sat else 0,
                    train_stats=train_stats,
                    sw_attempts=ep_reward_diag.get('switch_attempts', 0),
                    sw_ok=ep_reward_diag.get('switch_success', 0),
                    sw_rb=ep_reward_diag.get('switch_rollback', 0),
                    delta_sat=ep_reward_diag['delta_sum'] / n_diag,
                    biz_reward=ep_reward_diag['biz_reward'] / n_diag,
                    act_reward=ep_reward_diag['action_reward'] / n_diag,
                    conn_reward=ep_reward_diag['connect_reward'] / n_diag,
                )
                if (ep + 1) % health_monitor.check_interval == 0:
                    is_critical, _ = health_monitor.check(ep)
                    if is_critical:
                        print(f"  [HEALTH] 严重问题已检测到。如需暂停，请手动 Ctrl+C。")

                # ---- 早期/中期健康检查 ----
                if (ep == health_check_ep or ep + 1 in mid_check_eps) and verbose:
                    recent_avg = np.mean(episode_rewards[-health_check_ep:])
                    early_avg = np.mean(episode_rewards[:max(1, health_check_ep // 2)])
                    stay_pct = ep_action_counts['stay'] / max(sum(ep_action_counts.values()), 1) * 100
                    reward_std = np.std(episode_rewards[-health_check_ep:])
                    # per-action 分布
                    action_total = max(ep_per_action.sum(), 1)
                    action_str = ', '.join(
                        [f'stay={ep_per_action[0]/action_total:.0%}',
                         f'best_sinr={ep_per_action[1]/action_total:.0%}',
                         f'best_cap={ep_per_action[2]/action_total:.0%}',
                         f'sinr_cap={ep_per_action[3]/action_total:.0%}',
                         f'predict={ep_per_action[4]/action_total:.0%}',
                         f'biz_spec={ep_per_action[5]/action_total:.0%}']
                    )
                    print(f"\n  {'='*60}")
                    print(f"  诊断报告 [Episode {ep+1}]")
                    print(f"  {'='*60}")
                    print(f"  Reward: 均值={np.mean(episode_rewards):.2f}, "
                          f"标准差={reward_std:.2f}, "
                          f"变异系数={reward_std/max(abs(np.mean(episode_rewards)),0.01)*100:.1f}%")
                    print(f"  Satisfaction: {np.mean(episode_satisfactions):.3f}")
                    print(f"  Action 分布: {action_str}")

                    # [增强] 基站容量利用率分布
                    if hasattr(env, 'env') and hasattr(env.env, 'base_stations'):
                        bs_list = env.env.base_stations
                        cap_utils = []
                        for bs_idx, bs in enumerate(bs_list):
                            if hasattr(bs, 'available_capacity') and hasattr(bs, 'capacity'):
                                total_cap = bs.capacity
                                avail = bs.available_capacity
                                if total_cap and total_cap > 0:
                                    used_pct = (total_cap - avail) / total_cap * 100
                                    cap_utils.append((bs_idx, used_pct, avail, total_cap))
                        if cap_utils:
                            avg_load = np.mean([c[1] for c in cap_utils])
                            max_load = max(c[1] for c in cap_utils)
                            max_bs = max(cap_utils, key=lambda x: x[1])
                            print(f"  容量利用率: 平均={avg_load:.0f}%, 最高={max_load:.0f}% "
                                  f"(BS#{max_bs[0]}: 已用{max_bs[3]-max_bs[2]}/{max_bs[3]})")
                            if avg_load > 90:
                                print(f"    [OVERLOAD] 系统过载! 平均负载>90%，断连率高是预期行为")

                    # [增强] 增强算法使用概率
                    if hasattr(agent, 'enhanced_algorithm_prob'):
                        print(f"  增强算法概率: {agent.enhanced_algorithm_prob:.2%} "
                              f"(ep {ep+1}/{train_episodes})")
                    if recent_avg <= early_avg and len(episode_rewards) > health_check_ep // 2:
                        print(f"  [WARN] Reward 无上升趋势 (前期={early_avg:.2f} -> 近期={recent_avg:.2f})")
                    else:
                        print(f"  [OK] Reward 趋势正常 (前期={early_avg:.2f} -> 近期={recent_avg:.2f})")
                    if reward_std / max(abs(np.mean(episode_rewards)), 0.01) > 0.3:
                        print(f"  [WARN] Reward 变异系数 > 30%，PPO 难以收敛 -- 考虑 reward 归一化")
                    # Reward 组分
                    if ep_reward_diag['count'] > 0:
                        n = ep_reward_diag['count']
                        print(f"  Reward 组分 (avg/step): "
                              f"delta={ep_reward_diag['delta_sum']/n:.3f}, "
                              f"value={ep_reward_diag['value_reward']/n:.3f}, "
                              f"biz={ep_reward_diag['biz_reward']/n:.3f}, "
                              f"action={ep_reward_diag['action_reward']/n:.3f}, "
                              f"connect={ep_reward_diag['connect_reward']/n:.3f}")
                        print(f"  Switch 质量: good={ep_reward_diag['good_switch']}, "
                              f"bad={ep_reward_diag['bad_switch']}, "
                              f"ratio={ep_reward_diag['good_switch']/max(ep_reward_diag['good_switch']+ep_reward_diag['bad_switch'],1):.1%}")
                    # 切换尝试/成功率
                    sa = ep_reward_diag.get('switch_attempts', 0)
                    ss = ep_reward_diag.get('switch_success', 0)
                    sr = ep_reward_diag.get('switch_rollback', 0)
                    sd = ep_reward_diag.get('switch_disconnect', 0)
                    if sa > 0:
                        print(f"  切换统计: 尝试={sa}, 成功={ss}({ss/sa:.0%}), "
                              f"回滚={sr}({sr/sa:.0%}), 断连={sd}({sd/sa:.0%})")
                    else:
                        print(f"  切换统计: 无切换尝试")
                    # 业务类型统计
                    print(f"  业务类型统计:")
                    for bt in range(3):
                        stats = ep_biz_stats[bt]
                        total = stats['stay'] + stats['switch']
                        if total > 0:
                            stay_pct = stats['stay'] / total * 100
                            switch_pct = stats['switch'] / total * 100
                            avg_sat = np.mean(stats['satisfaction']) if stats['satisfaction'] else 0
                            avg_reward = np.mean(stats['reward']) if stats['reward'] else 0
                            print(f"    业务类型 {bt}: stay={stay_pct:.1f}%, switch={switch_pct:.1f}%, sat={avg_sat:.3f}, reward={avg_reward:.3f}")
                    print(f"  {'='*60}\n")

                if verbose and (ep + 1) % 30 == 0:
                    avg_al = np.mean(episode_actor_losses[-20:]) if episode_actor_losses else 0
                    avg_cl = np.mean(episode_critic_losses[-20:]) if episode_critic_losses else 0
                    avg_ent = np.mean(episode_entropies[-20:]) if episode_entropies else 0
                    avg_ag = np.mean(episode_actor_grads[-20:]) if episode_actor_grads else None
                    avg_vmse = np.mean(episode_value_mses[-20:]) if episode_value_mses else None
                    recent_rews = episode_rewards[-30:]
                    stay_pct = ep_action_counts['stay'] / max(sum(ep_action_counts.values()), 1) * 100
                    n = max(ep_reward_diag['count'], 1)
                    rd_str = (f"Δs={ep_reward_diag['delta_sum']/n:.2f} "
                              f"biz={ep_reward_diag['biz_reward']/n:.2f} "
                              f"act={ep_reward_diag['action_reward']/n:.2f} "
                              f"conn={ep_reward_diag['connect_reward']/n:.2f}")
                    sa = ep_reward_diag.get('switch_attempts', 0)
                    ss = ep_reward_diag.get('switch_success', 0)
                    sr = ep_reward_diag.get('switch_rollback', 0)
                    sd = ep_reward_diag.get('switch_disconnect', 0)
                    sw_str = f"sw={sa}(ok={ss},rb={sr},dc={sd})" if sa > 0 else "sw=0"
                    total_steps = (ep + 1) * num_steps
                    dc_rate = ep_disconnected_steps / max(total_steps, 1) * 100
                    dc_trend = "[BAD]" if dc_rate > 30 else ("[WARN]" if dc_rate > 15 else "[OK]")
                    grad_str = f"{avg_ag:.2f}" if avg_ag is not None else "N/A"
                    vmse_str = f"{avg_vmse:.1f}" if avg_vmse is not None else "N/A"
                    print(f"  标准环境 Episode {ep+1}/{train_episodes}: "
                          f"reward={episode_reward:.1f}(mu={np.mean(recent_rews):.1f},sigma={np.std(recent_rews):.1f}), "
                          f"sat={np.mean(episode_sat):.3f}, "
                          f"stay={stay_pct:.0f}%, {sw_str}, "
                          f"dc={dc_rate:.0f}%{dc_trend}, "
                          f"a_loss={avg_al:.4f}, c_loss={avg_cl:.2f}, "
                          f"H={avg_ent:.3f}, grad={grad_str}, vMSE={vmse_str} | {rd_str}")
                    if avg_al < 1e-6 and avg_cl < 1e-6:
                        print(f"    [WARN] WARNING: Loss values near zero! Possible causes:")
                        print(f"           - Policy not updating (ratio~=1, insufficient exploration)")
                        print(f"           - Advantage values too small (weak reward signal)")
                        print(f"           - Suggestion: increase entropy_coef or check reward design")

                # ---- Early stopping 判断 (基于 satisfaction) ----
                ep_sat = np.mean(episode_sat)
                is_best_sat = ep_sat > best_sat + early_stop_min_delta
                is_best_reward = episode_reward > best_reward + early_stop_min_delta
                if is_best_sat:
                    best_sat = ep_sat
                if is_best_reward:
                    best_reward = episode_reward

                # warmup 期间: 记录 best_sat 但不计入 no_improve
                if ep < early_stop_warmup:
                    no_improve_count = 0
                elif is_best_sat:
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # 模型保存: best 模型仅在 satisfaction 新高时保存, latest 每 save_interval 保存
                if is_best_sat:
                    agent.save(best_model_path)
                if (ep + 1) % save_interval == 0:
                    agent.save(latest_model_path)

                if no_improve_count >= early_stop_patience and ep >= health_check_ep * 2:
                    if verbose:
                        print(f"\n  [STOP] Early stopping [Episode {ep+1}]: "
                              f"连续 {early_stop_patience} 轮 satisfaction 无改善 "
                              f"(best_sat={best_sat:.4f}, best_reward={best_reward:.3f})")
                    early_stopped = True
                    break

            # 训练结束: 确保 model_path 指向 best 模型
            import shutil
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, model_path)
                if verbose:
                    print(f"  [OK] 最终模型已更新为 best_sat={best_sat:.4f} 的版本")

                training_results[num_uav] = {
                    'rewards': episode_rewards,
                    'satisfactions': episode_satisfactions,
                    'actor_losses': episode_actor_losses,
                    'critic_losses': episode_critic_losses,
                    'entropies': episode_entropies,
                    'final_avg_sat': np.mean(episode_satisfactions[-50:]),
                    'early_stopped': early_stopped,
                }
                
                # 运行数据验证
                if integrate_with_experiment:
                    print(f"\n  [VALIDATION] 开始数据验证...")
                    env_config = {
                        'num_bs': num_bs,
                        'num_uav': num_uav,
                        'max_steps': num_steps,
                    }
                    hp_config = {
                        'actor_lr': actor_lr,
                        'critic_lr': critic_lr,
                        'clip_epsilon': 0.1,  # 方案1: 降低到0.1
                    }
                    validation_results = integrate_with_experiment(
                        training_results[num_uav],
                        env_config,
                        hp_config,
                        GLOBAL_SEED
                    )
                    training_results[num_uav]['validation'] = validation_results

        return training_results

    # ==================== Phase 2: 对比评估 ====================

    @staticmethod
    def _phase2_evaluation(num_uav_list, num_bs_list, num_steps, eval_episodes,
                           bs_capacity_range, pos_range, load_models, verbose,
                           use_biz_heads, use_attention_critic,
                           hidden_dim, critic_hidden_dim,
                           attention_sample_agents=0):
        """
        Phase 2: 算法对比评估

        研究脉络:
          实验1→2→2b→3 验证了增强算法相对于传统算法(3GPP A3)的优势；
          BA-MAPPO 的核心目标是: 通过多智能体协同决策，超越增强算法（单步贪心）。
          因此基线优先级: 传统算法(3GPP) < 增强算法(本文) < BA-MAPPO(本文)。
          简单基线(stay/best_sinr)仅作参考，不作为核心对比对象。

        统计维度:
          - 整体平均满意度
          - 关键业务满意度（控制信令+视频回传）
          - 分业务类型满意度
        """
        print("\n" + "-" * 60)
        print("Phase 2: 算法对比评估 (传统 → 增强 → BA-MAPPO)")
        print("-" * 60)

        eval_results = {}

        for num_uav, num_bs in zip(num_uav_list, num_bs_list):
            print(f"\n>>> 评估 UAV={num_uav}/BS={num_bs} <<<")

            set_global_seed(GLOBAL_SEED + num_uav * 200)
            model_path = os.path.join(ExperimentBAMAPPO.MODEL_DIR,
                                      f'mappo_{num_bs}bs_{num_uav}uav.pt')

            env = QMixHandoverEnv(
                num_bs=num_bs, num_uav=num_uav,
                max_steps=num_steps, seed=GLOBAL_SEED + num_uav * 200,
                bs_capacity_range=bs_capacity_range,
                pos_range=pos_range,
            )

            agent = MAPPOAgent(
                num_agents=env.num_agents,
                obs_dim=env.obs_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=hidden_dim,
                critic_hidden_dim=critic_hidden_dim,
                use_biz_heads=use_biz_heads,
                use_attention_critic=use_attention_critic,
                use_hierarchical=True,
                use_transformer=False,
                use_data_augmentation=True,
                attention_sample_agents=attention_sample_agents,
            )

            if os.path.exists(model_path):
                agent.load(model_path)
                print(f"  已加载模型: {model_path}")
            elif load_models:
                print(f"  警告: 未找到模型 {model_path}, 使用随机策略")

            # ---- 评估 BA-MAPPO ----
            mappo_avg_sats = []
            mappo_biz_sats = {0: [], 1: [], 2: []}
            mappo_critical_sats = []
            strategy_counts = {}
            # 通信指标收集
            mappo_handover_latencies = []
            mappo_ping_jitters = []
            mappo_packet_losses = []
            mappo_qos_violations = []

            for rep in range(eval_episodes):
                obs_dict, global_state = env.reset()
                agent.reset_hidden()
                for step in range(num_steps):
                    biz_types = {}
                    for uid in range(env.num_agents):
                        uav = env.env.uavs[uid]
                        biz_types[uid] = uav.true_business_type.value
                    actions, _, _, _, _ = agent.select_actions(obs_dict, global_state, biz_types, training=False)
                    next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
                    obs_dict = next_obs
                    global_state = next_state
                    for sname, count in info['strategy_distribution'].items():
                        strategy_counts[sname] = strategy_counts.get(sname, 0) + count
                    
                    # 收集通信指标
                    if 'communication_metrics' in info:
                        comm_metrics = info['communication_metrics']
                        mappo_handover_latencies.append(comm_metrics.get('handover_latency', 0.0))
                        mappo_ping_jitters.append(comm_metrics.get('ping_jitter', 0.0))
                        mappo_packet_losses.append(comm_metrics.get('packet_loss_rate', 0.0))
                        mappo_qos_violations.append(comm_metrics.get('qos_violation_rate', 0.0))

                stats = _collect_biz_satisfaction(env)
                mappo_avg_sats.append(stats['avg'])
                mappo_critical_sats.append(stats['critical_sat'])
                for bt in range(3):
                    mappo_biz_sats[bt].append(stats['per_biz'][bt][0])

            # ---- 评估基线算法 ----
            def _eval_baseline(baseline_name, run_fn):
                """通用基线评估函数，返回 {avg, critical, per_biz} 的多次重复均值"""
                avg_list, critical_list = [], []
                biz_lists = {0: [], 1: [], 2: []}
                # 通信指标收集
                handover_latencies, ping_jitters = [], []
                packet_losses, qos_violations = [], []
                for rep in range(eval_episodes):
                    seed = GLOBAL_SEED + num_uav * 300 + rep * 1000 + hash(baseline_name) % 10000
                    set_global_seed(seed)
                    eval_env = QMixHandoverEnv(
                        num_bs=num_bs, num_uav=num_uav,
                        max_steps=num_steps, seed=seed,
                        bs_capacity_range=bs_capacity_range,
                        pos_range=pos_range,
                    )
                    eval_env.reset()
                    run_fn(eval_env, num_steps)
                    stats = _collect_biz_satisfaction(eval_env)
                    avg_list.append(stats['avg'])
                    critical_list.append(stats['critical_sat'])
                    for bt in range(3):
                        biz_lists[bt].append(stats['per_biz'][bt][0])
                    # 收集通信指标（从环境的通信指标记录中获取）
                    if hasattr(eval_env, '_communication_metrics'):
                        comm_metrics = eval_env._communication_metrics
                        if comm_metrics['handover_latencies']:
                            handover_latencies.append(np.mean(comm_metrics['handover_latencies']))
                        if comm_metrics['ping_jitters']:
                            ping_jitters.append(np.mean(comm_metrics['ping_jitters']))
                        if comm_metrics['packet_losses']:
                            packet_losses.append(np.mean(comm_metrics['packet_losses']))
                        if comm_metrics['qos_violations']:
                            qos_violations.append(np.mean(comm_metrics['qos_violations']))
                return {
                    'avg': (np.mean(avg_list), np.std(avg_list)),
                    'critical': (np.mean(critical_list), np.std(critical_list)),
                    'per_biz': {bt: (np.mean(biz_lists[bt]), np.std(biz_lists[bt])) for bt in range(3)},
                    'communication_metrics': {
                        'handover_latency': (np.mean(handover_latencies) if handover_latencies else 0.0, 
                                             np.std(handover_latencies) if handover_latencies else 0.0),
                        'ping_jitter': (np.mean(ping_jitters) if ping_jitters else 0.0, 
                                        np.std(ping_jitters) if ping_jitters else 0.0),
                        'packet_loss_rate': (np.mean(packet_losses) if packet_losses else 0.0, 
                                            np.std(packet_losses) if packet_losses else 0.0),
                        'qos_violation_rate': (np.mean(qos_violations) if qos_violations else 0.0, 
                                               np.std(qos_violations) if qos_violations else 0.0),
                    },
                }

            # 传统算法基线 (3GPP A3 — 论文核心对比对象)
            traditional_results = _eval_baseline('traditional',
                lambda e, steps: _run_algo_baseline(e, steps, IntegratedHandoverAlgorithm))

            # 增强算法基线 (本文前序工作 — BA-MAPPO 的直接改进对象)
            enhanced_results = _eval_baseline('enhanced',
                lambda e, steps: _run_algo_baseline(e, steps, EnhancedHandoverAlgorithm, enable_lb=True))

            # 简单参考基线 (仅作参考，不作为核心对比)
            stay_results = _eval_baseline('stay',
                lambda e, steps: _run_fixed_action_baseline(e, steps, action=0))
            best_sinr_results = _eval_baseline('best_sinr',
                lambda e, steps: _run_fixed_action_baseline(e, steps, action=1))

            # ---- 结果汇总 ----
            eval_results[num_uav] = {
                'mappo': {
                    'avg': (np.mean(mappo_avg_sats), np.std(mappo_avg_sats)),
                    'critical': (np.mean(mappo_critical_sats), np.std(mappo_critical_sats)),
                    'per_biz': {bt: (np.mean(mappo_biz_sats[bt]), np.std(mappo_biz_sats[bt]))
                                for bt in range(3)},
                    'communication_metrics': {
                        'handover_latency': (np.mean(mappo_handover_latencies) if mappo_handover_latencies else 0.0, 
                                             np.std(mappo_handover_latencies) if mappo_handover_latencies else 0.0),
                        'ping_jitter': (np.mean(mappo_ping_jitters) if mappo_ping_jitters else 0.0, 
                                        np.std(mappo_ping_jitters) if mappo_ping_jitters else 0.0),
                        'packet_loss_rate': (np.mean(mappo_packet_losses) if mappo_packet_losses else 0.0, 
                                            np.std(mappo_packet_losses) if mappo_packet_losses else 0.0),
                        'qos_violation_rate': (np.mean(mappo_qos_violations) if mappo_qos_violations else 0.0, 
                                               np.std(mappo_qos_violations) if mappo_qos_violations else 0.0),
                    },
                },
                'enhanced': enhanced_results,
                'traditional': traditional_results,
                'ref_baselines': {
                    'stay': stay_results['avg'],
                    'best_sinr': best_sinr_results['avg'],
                },
                'strategy_distribution': strategy_counts,
            }

            if verbose:
                print(f"\n  UAV={num_uav} 对比结果:")
                print(f"  {'算法':<16} {'平均满意度':>12} {'关键业务':>12} {'控制信令':>10} {'视频回传':>10} {'环境监测':>10}")
                print(f"  {'-'*70}")
                for name, data in [
                    ('传统算法(3GPP)', traditional_results),
                    ('增强算法(本文)', enhanced_results),
                    ('BA-MAPPO(本文)', eval_results[num_uav]['mappo']),
                ]:
                    avg = data['avg']
                    cri = data['critical']
                    biz0 = data['per_biz'][0]
                    biz1 = data['per_biz'][1]
                    biz2 = data['per_biz'][2]
                    print(f"  {name:<16} {avg[0]:>8.4f}+/-{avg[1]:.3f} {cri[0]:>8.4f}+/-{cri[1]:.3f} "
                          f"{biz0[0]:>8.4f} {biz1[0]:>8.4f} {biz2[0]:>8.4f}")
                print(f"\n  [参考基线] stay={stay_results['avg'][0]:.4f}, "
                      f"best_sinr={best_sinr_results['avg'][0]:.4f}")

                # 打印通信指标
                print(f"\n  通信指标对比:")
                print(f"  {'算法':<16} {'切换延迟':>10} {'Ping抖动':>10} {'丢包率':>10} {'QoS违规率':>12}")
                print(f"  {'-'*65}")
                for name, data in [
                    ('传统算法(3GPP)', traditional_results),
                    ('增强算法(本文)', enhanced_results),
                    ('BA-MAPPO(本文)', eval_results[num_uav]['mappo']),
                ]:
                    if 'communication_metrics' in data:
                        comm = data['communication_metrics']
                        print(f"  {name:<16} {comm['handover_latency'][0]:>8.2f}ms "
                              f"{comm['ping_jitter'][0]:>8.2f}ms "
                              f"{comm['packet_loss_rate'][0]:>8.2f}% "
                              f"{comm['qos_violation_rate'][0]:>10.2f}%")
                    else:
                        print(f"  {name:<16} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>12}")

                # 计算相对提升
                trad_avg = traditional_results['avg'][0]
                enh_avg = enhanced_results['avg'][0]
                mappo_avg = eval_results[num_uav]['mappo']['avg'][0]
                if trad_avg > 0.001:
                    print(f"\n  BA-MAPPO 相对提升:")
                    print(f"    vs 传统算法: {(mappo_avg - trad_avg)/trad_avg*100:+.1f}%")
                    print(f"    vs 增强算法: {(mappo_avg - enh_avg)/max(enh_avg,0.001)*100:+.1f}%")

        return eval_results

    # ==================== 可视化 ====================

    @staticmethod
    def _plot_all(all_results, num_uav_list):
        """生成所有可视化图表"""
        mode_name = all_results.get('config', {}).get('mode_name', 'BA-MAPPO')
        print(f"\n生成 {mode_name} 实验可视化...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'{mode_name} 实验结果', fontsize=16, fontweight='bold')

        # 图1: 训练收敛曲线
        ax = axes[0, 0]
        if 'training' in all_results:
            for num_uav in num_uav_list:
                if num_uav in all_results['training']:
                    tr = all_results['training'][num_uav]
                    if 'rewards' in tr:
                        rewards = np.array(tr['rewards'])
                        window = max(1, len(rewards) // 20)
                        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                        ax.plot(smoothed, label=f'UAV={num_uav}', alpha=0.8)
            ax.set_xlabel('Episode')
            ax.set_ylabel('团队奖励')
            ax.set_title('(a) 训练收敛曲线')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 图2: 训练期间满意度变化
        ax = axes[0, 1]
        if 'training' in all_results:
            for num_uav in num_uav_list:
                if num_uav in all_results['training']:
                    tr = all_results['training'][num_uav]
                    if 'satisfactions' in tr:
                        sats = np.array(tr['satisfactions'])
                        window = max(1, len(sats) // 20)
                        smoothed = np.convolve(sats, np.ones(window)/window, mode='valid')
                        ax.plot(smoothed, label=f'UAV={num_uav}', alpha=0.8)
            ax.set_xlabel('Episode')
            ax.set_ylabel('平均满意度')
            ax.set_title('(b) 训练期间满意度变化')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 图3: 对比评估柱状图 (核心对比: 传统 → 增强 → BA-MAPPO)
        ax = axes[0, 2]
        if 'evaluation' in all_results:
            uav_to_show = [u for u in num_uav_list if u in all_results.get('evaluation', {})]
            if uav_to_show:
                x = np.arange(len(uav_to_show))
                width = 0.25
                mappo_vals = [all_results['evaluation'][u]['mappo']['avg'][0] for u in uav_to_show]
                enh_vals = [all_results['evaluation'][u]['enhanced']['avg'][0] for u in uav_to_show]
                trad_vals = [all_results['evaluation'][u]['traditional']['avg'][0] for u in uav_to_show]
                ax.bar(x - width, trad_vals, width, label='传统算法(3GPP)', color=COLORS['danger'], alpha=0.8)
                ax.bar(x, enh_vals, width, label='增强算法(本文)', color=COLORS['primary'], alpha=0.8)
                ax.bar(x + width, mappo_vals, width, label=mode_name, color=COLORS['warning'], alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels([f'UAV={u}' for u in uav_to_show])
                ax.set_ylabel('平均满意度')
                ax.set_title('(c) 算法对比: 传统→增强→BA-MAPPO')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3, axis='y')

        # 图4: 策略分布热力图
        ax = axes[1, 0]
        if 'evaluation' in all_results:
            uav_to_show = [u for u in num_uav_list if u in all_results.get('evaluation', {})]
            if uav_to_show:
                sample_dist = all_results['evaluation'][uav_to_show[0]].get('strategy_distribution', {})
                strat_names = list(sample_dist.keys()) if sample_dist else ['stay']
                matrix = np.zeros((len(uav_to_show), len(strat_names)))
                for i, u in enumerate(uav_to_show):
                    total = sum(all_results['evaluation'][u].get('strategy_distribution', {}).values()) or 1
                    for j, sn in enumerate(strat_names):
                        matrix[i, j] = all_results['evaluation'][u].get('strategy_distribution', {}).get(sn, 0) / total
                im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
                ax.set_xticks(range(len(strat_names)))
                ax.set_xticklabels(strat_names, rotation=30, ha='right', fontsize=8)
                ax.set_yticks(range(len(uav_to_show)))
                ax.set_yticklabels([f'UAV={u}' for u in uav_to_show])
                ax.set_title(f'(d) {mode_name} 动作分布')
                plt.colorbar(im, ax=ax)
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)

        # 图5: 多场景泛化对比 (仅核心算法: 传统+增强+BA-MAPPO)
        ax = axes[1, 1]
        ax.axis('off')
        if 'scenarios' in all_results and all_results['scenarios']:
            ax2 = fig.add_subplot(2, 3, 6)
            scenarios_data = all_results['scenarios']
            s_names = list(scenarios_data.keys())
            s_names_cn = [scenarios_data[s].get('name_cn', s) for s in s_names]

            trad_sats = [scenarios_data[s]['traditional'][0] for s in s_names]
            enh_sats = [scenarios_data[s]['enhanced'][0] for s in s_names]
            mappo_scenario_sats = [scenarios_data[s].get('mappo', (0, 0))[0] if 'mappo' in scenarios_data[s] else 0 for s in s_names]

            x = np.arange(len(s_names))
            width = 0.25
            has_mappo = any('mappo' in scenarios_data[s] for s in s_names)
            if has_mappo:
                offsets = [-width, 0, width]
                ax2.bar(x + offsets[0], trad_sats, width, label='传统算法', color=COLORS['danger'], alpha=0.8)
                ax2.bar(x + offsets[1], enh_sats, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
                ax2.bar(x + offsets[2], mappo_scenario_sats, width, label=mode_name, color=COLORS['warning'], alpha=0.8)
            else:
                ax2.bar(x - width/2, trad_sats, width, label='传统算法', color=COLORS['danger'], alpha=0.8)
                ax2.bar(x + width/2, enh_sats, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
            ax2.set_xticks(x)
            ax2.set_xticklabels(s_names_cn, rotation=20, ha='right', fontsize=8)
            ax2.set_ylabel('平均满意度')
            ax2.set_title('(f) 多场景泛化对比')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')

        # 图6: 关键结果文本
        ax = axes[1, 2]
        ax.axis('off')
        text_lines = [f'【{mode_name} 关键发现】\n']
        if 'evaluation' in all_results:
            for num_uav in num_uav_list:
                if num_uav in all_results['evaluation']:
                    ev = all_results['evaluation'][num_uav]
                    mp = ev['mappo']['avg'][0]
                    en = ev['enhanced']['avg'][0]
                    tr = ev['traditional']['avg'][0]
                    improvement_vs_trad = (mp - tr) / max(tr, 0.001) * 100
                    improvement_vs_enh = (mp - en) / max(en, 0.001) * 100
                    text_lines.append(f'UAV={num_uav}:')
                    text_lines.append(f'  {mode_name} vs 传统: {improvement_vs_trad:+.1f}%')
                    text_lines.append(f'  {mode_name} vs 增强: {improvement_vs_enh:+.1f}%')
                    text_lines.append(f'  {mode_name}={mp:.4f}, 增强={en:.4f}, 传统={tr:.4f}')
                    text_lines.append('')
        if 'scenarios' in all_results:
            text_lines.append('【场景泛化】\n')
            for sn, sd in all_results['scenarios'].items():
                line = f'  {sd["name_cn"]}: 传统={sd["traditional"][0]:.4f}, 增强={sd["enhanced"][0]:.4f}'
                if 'mappo' in sd:
                    line += f', MAPPO={sd["mappo"][0]:.4f}'
                text_lines.append(line)

        ax.text(0.05, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.tight_layout()
        save_path = os.path.join(RESULT_DIR, 'mappo_results.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  可视化已保存: {save_path}")
