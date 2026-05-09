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
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import signal
import threading
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

from .config import GLOBAL_SEED, set_global_seed, RESULT_DIR, COLORS, MAPPOConfig  # V21: 添加MAPPOConfig
from .mappo_environment import MultiAgentHandoverEnv
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
            env_fn: 返回新 MultiAgentHandoverEnv 实例的 callable
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
        env_or_env_wrapper: MultiAgentHandoverEnv 实例（env.env 为底层网络环境）

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


class OverfittingMonitor:
    """
    过拟合实时监控器 (V22: 多维度风险检测与预警)
    
    监控维度:
      1. Satisfaction稳定性 - 检测方差突变和趋势偏移
      2. Reward分布健康度 - 检测均值/方差异常
      3. 动作分布漂移 - 检测策略突然变化
      4. Loss趋势分析 - 检测过拟合典型模式
      5. 学习信号强度 - 验证是否在真实学习
      6. 泛化能力指标 - 基于统计的泛化评估
    
    输出:
      - 综合风险评分 (0.0-1.0, 越低越安全)
      - 各维度详细评分
      - 趋势图和警报
    
    使用方式:
      monitor = OverfittingMonitor(window_size=20)
      risk_report = monitor.check(episode_data)
      print(monitor.format_report(risk_report))
    """
    
    def __init__(self, window_size=20, alert_threshold=0.65):
        """
        初始化监控器
        
        Args:
            window_size: 滑动窗口大小（用于计算趋势）
            alert_threshold: 风险警报阈值 (0-1)
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        # 历史数据存储
        self.history = {
            'satisfaction': [],
            'reward': [],
            'reward_std': [],
            'stay_pct': [],
            'switch_success': [],
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'grad_norm': [],
            'value_mse': [],
            'rollback_rate': [],
            'composite_score': [],
        }
        
        # 风险历史
        self.risk_history = []
        
        # 初始基准值（用于检测漂移）
        self.baseline = {
            'stay_pct': None,
            'satisfaction_mean': None,
            'reward_cv': None,
        }
        
        # 统计计数器
        self.total_checks = 0
        self.alert_count = 0
        
    def update(self, episode_data):
        """
        更新历史数据
        
        Args:
            episode_data: 当前episode的数据字典
        """
        sat = episode_data.get('satisfaction', 0)
        reward = episode_data.get('reward', 0)
        reward_std = episode_data.get('reward_std', 0)
        stay_pct = episode_data.get('stay_percentage', 0)
        switch_success = episode_data.get('switch_success_rate', 0)
        actor_loss = abs(episode_data.get('actor_loss', 0))
        critic_loss = episode_data.get('critic_loss', 0)
        entropy = episode_data.get('entropy', 0)
        grad_norm = episode_data.get('grad_norm', 0)
        value_mse = episode_data.get('value_mse', 0)
        rollback_rate = 0
        if episode_data.get('switch_attempts', 0) > 0:
            rollback_rate = episode_data.get('switch_rollback', 0) / episode_data.get('switch_attempts', 1)
        composite_score = episode_data.get('composite_score', 0)
        
        # 更新各维度历史
        for key, value in [
            ('satisfaction', sat),
            ('reward', reward),
            ('reward_std', reward_std),
            ('stay_pct', stay_pct),
            ('switch_success', switch_success),
            ('actor_loss', actor_loss),
            ('critic_loss', critic_loss),
            ('entropy', entropy),
            ('grad_norm', grad_norm),
            ('value_mse', value_mse),
            ('rollback_rate', rollback_rate),
            ('composite_score', composite_score),
        ]:
            self.history[key].append(value)
            if len(self.history[key]) > self.window_size * 3:
                self.history[key].pop(0)
        
        # 设置初始基准值（前5个episode的平均）
        if len(self.history['satisfaction']) == 5 and self.baseline['satisfaction_mean'] is None:
            self.baseline['satisfaction_mean'] = np.mean(self.history['satisfaction'])
            self.baseline['stay_pct'] = np.mean(self.history['stay_pct'])
            if np.mean(self.history['reward']) != 0:
                self.baseline['reward_cv'] = np.std(self.history['reward']) / abs(np.mean(self.history['reward']))
    
    def _check_satisfaction_stability(self):
        """
        检查1: Satisfaction稳定性
        
        指标:
          - 短期标准差 (最近5个episode)
          - 长期趋势 (线性回归斜率)
          - 方差变化率 (短期vs长期)
        
        返回: (risk_score, details_dict)
        """
        sats = self.history['satisfaction']
        if len(sats) < 5:
            return 0.0, {'status': 'INSUFFICIENT_DATA'}
        
        recent_5 = sats[-5:]
        recent_10 = sats[-min(10, len(sats)):]
        
        # 短期标准差
        short_std = np.std(recent_5)
        long_std = np.std(recent_10) if len(recent_10) >= 5 else short_std
        
        # 趋势分析 (线性回归斜率)
        if len(sats) >= 10:
            x = np.arange(len(sats[-10:]))
            y = np.array(sats[-10:])
            slope, _ = np.polyfit(x, y, 1)
            slope_normalized = slope * 1000  # 放大以便比较
        else:
            slope_normalized = 0
        
        # 方差变化率 (短期/长期)
        variance_ratio = short_std / (long_std + 1e-8)
        
        # 计算风险分数
        risk = 0.0
        
        # 标准差风险 (σ > 0.01 开始有风险)
        if short_std > 0.015:
            risk += min(0.4, (short_std - 0.015) * 20)
        elif short_std > 0.008:
            risk += (short_std - 0.008) * 15
        
        # 趋势风险 (快速下降是危险信号)
        if slope_normalized < -0.5:
            risk += min(0.35, abs(slope_normalized) * 0.5)
        elif slope_normalized < -0.2:
            risk += abs(slope_normalized) * 0.3
        
        # 方差增大风险 (短期方差 >> 长期方差)
        if variance_ratio > 1.5:
            risk += min(0.25, (variance_ratio - 1.5) * 0.5)
        
        risk = min(1.0, risk)
        
        status = '[OK] HEALTHY' if risk < 0.3 else ('[!] WARNING' if risk < 0.6 else '[X] DANGER')
        
        details = {
            'short_std': short_std,
            'long_std': long_std,
            'slope': slope_normalized,
            'variance_ratio': variance_ratio,
            'risk': risk,
            'status': status,
        }
        
        return risk, details
    
    def _check_reward_health(self):
        """
        检查2: Reward分布健康度
        
        指标:
          - 变异系数稳定性 (CV = σ/μ)
          - 均值漂移程度
          - 极值频率 (超出2σ的频率)
        
        返回: (risk_score, details_dict)
        """
        rewards = self.history['reward']
        if len(rewards) < 5:
            return 0.0, {'status': 'INSUFFICIENT_DATA'}
        
        recent_10 = rewards[-min(10, len(rewards)):]
        mean_r = np.mean(recent_10)
        std_r = np.std(recent_10)
        
        # 变异系数
        cv = std_r / (abs(mean_r) + 1e-8)
        
        # CV稳定性 (比较近期CV vs 历史CV)
        if len(rewards) >= 15:
            old_rewards = rewards[:-10]
            old_cv = np.std(old_rewards) / (abs(np.mean(old_rewards)) + 1e-8)
            cv_change = abs(cv - old_cv) / (old_cv + 1e-8)
        else:
            cv_change = 0
        
        # 极值检测 (超出2σ的频率)
        if len(rewards) >= 10:
            threshold_high = mean_r + 2 * std_r
            threshold_low = mean_r - 2 * std_r
            extremes = sum(1 for r in recent_10 if r > threshold_high or r < threshold_low)
            extreme_rate = extremes / len(recent_10)
        else:
            extreme_rate = 0
        
        # 计算风险
        risk = 0.0
        
        # CV异常风险 (CV > 8% 或 CV变化>50%)
        if cv > 0.08:
            risk += min(0.3, (cv - 0.08) * 5)
        if cv_change > 0.5:
            risk += min(0.3, (cv_change - 0.5) * 0.4)
        
        # 极值频率过高 (>30% 是危险的)
        if extreme_rate > 0.3:
            risk += min(0.4, (extreme_rate - 0.3) * 1.0)
        
        risk = min(1.0, risk)
        
        status = '[OK] HEALTHY' if risk < 0.3 else ('[!] WARNING' if risk < 0.6 else '[X] DANGER')
        
        details = {
            'cv': cv,
            'cv_change': cv_change,
            'extreme_rate': extreme_rate,
            'mean': mean_r,
            'std': std_r,
            'risk': risk,
            'status': status,
        }
        
        return risk, details
    
    def _check_action_drift(self):
        """
        检查3: 动作分布漂移
        
        指标:
          - Stay比例变化幅度 (vs基线)
          - 变化速率 (每episode变化量)
          - 方向一致性 (持续单向变化)
        
        返回: (risk_score, details_dict)
        """
        stays = self.history['stay_pct']
        if len(stays) < 5 or self.baseline['stay_pct'] is None:
            return 0.0, {'status': 'INSUFFICIENT_DATA'}
        
        current_stay = stays[-1]
        baseline_stay = self.baseline['stay_pct']
        
        # 绝对漂移量
        drift_abs = abs(current_stay - baseline_stay)
        
        # 近期变化速率 (最近5个episode的平均变化)
        if len(stays) >= 6:
            recent_changes = [abs(stays[i] - stays[i-1]) for i in range(-5, 0)]
            drift_rate = np.mean(recent_changes)
        else:
            drift_rate = 0
        
        # 方向性检查 (连续同向变化的次数)
        if len(stays) >= 5:
            directions = [np.sign(stays[i] - stays[i-1]) for i in range(-5, 0) if stays[i] - stays[i-1] != 0]
            consistent_direction = len(directions) >= 4 and len(set(directions)) == 1
        else:
            consistent_direction = False
        
        # 计算风险
        risk = 0.0
        
        # 漂移幅度风险 (>10% 是危险的)
        if drift_abs > 0.10:
            risk += min(0.4, (drift_abs - 0.10) * 3)
        elif drift_abs > 0.05:
            risk += (drift_abs - 0.05) * 2
        
        # 变化速率风险 (>2%/ep 是危险的)
        if drift_rate > 0.02:
            risk += min(0.3, (drift_rate - 0.02) * 5)
        
        # 单向漂移风险 (可能表示策略坍缩)
        if consistent_direction and drift_abs > 0.03:
            risk += 0.3
        
        risk = min(1.0, risk)
        
        status = '[OK] HEALTHY' if risk < 0.3 else ('[!] WARNING' if risk < 0.6 else '[X] DANGER')
        
        details = {
            'current': current_stay,
            'baseline': baseline_stay,
            'drift_abs': drift_abs,
            'drift_rate': drift_rate,
            'consistent_direction': consistent_direction,
            'risk': risk,
            'status': status,
        }
        
        return risk, details
    
    def _check_loss_trends(self):
        """
        检查4: Loss趋势分析
        
        典型过拟合模式:
          - Actor Loss持续下降且接近零
          - Critic Loss持续上升 (Value函数过拟合)
          - Entropy快速下降 (探索能力丧失)
          - Value MSE上升 (泛化误差增大)
        
        返回: (risk_score, details_dict)
        """
        actor_losses = self.history['actor_loss']
        critic_losses = self.history['critic_loss']
        entropies = self.history['entropy']
        value_mses = self.history['value_mse']
        
        if len(actor_losses) < 10:
            return 0.0, {'status': 'INSUFFICIENT_DATA'}
        
        risk = 0.0
        
        # Actor Loss趋势 (持续下降到接近0是危险的)
        if len(actor_losses) >= 10:
            recent_actor = actor_losses[-5:]
            early_actor = actor_losses[-10:-5]
            
            actor_trend = np.mean(recent_actor) - np.mean(early_actor)
            actor_current = np.mean(recent_actor)
            
            # Actor Loss过低 (<0.05) 且仍在下降
            if actor_current < 0.05 and actor_trend < -0.005:
                risk += min(0.25, (0.05 - actor_current) * 3 + abs(actor_trend) * 10)
            elif actor_current < 0.1 and actor_trend < -0.01:
                risk += min(0.15, abs(actor_trend) * 5)
        
        # Critic Loss上升趋势 (Value函数不匹配训练状态)
        if len(critic_losses) >= 10:
            recent_critic = critic_losses[-5:]
            early_critic = critic_losses[-10:-5]
            
            critic_trend = (np.mean(recent_critic) - np.mean(early_critic)) / (np.mean(early_critic) + 1e-8)
            
            # Critic Loss上升超过10%
            if critic_trend > 0.1:
                risk += min(0.25, (critic_trend - 0.1) * 1.5)
            elif critic_trend > 0.05:
                risk += (critic_trend - 0.05) * 1.0
        
        # Entropy下降 (探索能力丧失)
        if len(entropies) >= 10:
            recent_ent = entropies[-5:]
            early_ent = entropies[-10:-5]
            
            ent_trend = np.mean(recent_ent) - np.mean(early_ent)
            ent_current = np.mean(recent_ent)
            
            # Entropy < 0.3 且下降速度快
            if ent_current < 0.3 and ent_trend < -0.02:
                risk += min(0.2, (0.3 - ent_current) * 0.5 + abs(ent_trend) * 3)
            elif ent_current < 0.5 and ent_trend < -0.03:
                risk += min(0.1, abs(ent_trend) * 2)
        
        # Value MSE上升 (泛化误差增大)
        if len(value_mses) >= 10:
            recent_vmse = value_mses[-5:]
            early_vmse = value_mses[-10:-5]
            
            vmse_trend = (np.mean(recent_vmse) - np.mean(early_vmse)) / (np.mean(early_vmse) + 1e-8)
            
            if vmse_trend > 0.1:
                risk += min(0.15, (vmse_trend - 0.1) * 1.0)
            elif vmse_trend > 0.05:
                risk += (vmse_trend - 0.05) * 0.5
        
        risk = min(1.0, risk)
        
        status = '[OK] HEALTHY' if risk < 0.3 else ('[!] WARNING' if risk < 0.6 else '[X] DANGER')
        
        details = {
            'actor_trend': 'v LOW' if len(actor_losses) >= 10 else 'N/A',
            'critic_trend': f'^ {critic_trend*100:.1f}%' if len(critic_losses) >= 10 else 'N/A',
            'entropy_trend': f'{ent_trend:+.3f}' if len(entropies) >= 10 else 'N/A',
            'vmse_trend': f'^ {vmse_trend*100:.1f}%' if len(value_mses) >= 10 else 'N/A',
            'risk': risk,
            'status': status,
        }
        
        return risk, details
    
    def _check_learning_signal(self):
        """
        检查5: 学习信号强度
        
        正学习信号:
          - 回滚率持续下降 (切换质量提升)
          - Switch Success提升
          - Composite Score改善
          - Satisfaction稳定或微升
        
        负信号 (可能的过拟合):
          - 所有指标停止改善但Loss继续变化
          - 高方差但无明确趋势
          - 指标震荡加剧
        
        返回: (risk_score, details_dict)
        """
        rollbacks = self.history['rollback_rate']
        switch_success = self.history['switch_success']
        composite = self.history['composite_score']
        
        if len(rollbacks) < 10:
            return 0.0, {'status': 'INSUFFICIENT_DATA', 'learning_strength': 'N/A'}
        
        risk = 0.0
        
        # 回滚率趋势 (应该下降)
        if len(rollbacks) >= 10:
            recent_rb = np.mean(rollbacks[-5:])
            early_rb = np.mean(rollbacks[-10:-5])
            
            rb_improvement = (early_rb - recent_rb) / (early_rb + 1e-8)
            
            # 回滚率改善是好信号 (降低风险)
            if rb_improvement > 0.3:
                risk -= 0.1  # 强学习信号，降低风险
            elif rb_improvement < -0.1:
                risk += 0.15  # 回滚率恶化，增加风险
        
        # Composite Score趋势
        if len(composite) >= 10:
            recent_comp = composite[-5:]
            early_comp = composite[-10:-5]
            
            comp_improvement = np.mean(recent_comp) - np.mean(early_comp)
            
            # 综合评分持续改善是好信号
            if comp_improvement > 0.002:
                risk -= 0.05
            elif comp_improvement < -0.001:
                risk += 0.15
        
        # Switch Success稳定性
        if len(switch_success) >= 10:
            sw_std = np.std(switch_success[-5:])
            
            # Switch成功率方差过大 (>2%)
            if sw_std > 0.02:
                risk += min(0.15, (sw_std - 0.02) * 3)
        
        risk = max(0.0, min(1.0, risk))
        
        # 判断学习强度
        if risk <= 0:
            learning_strength = '[STRONG] STRONG'
        elif risk <= 0.15:
            learning_strength = '[MODERATE] MODERATE'
        elif risk <= 0.3:
            learning_strength = '[WEAK] WEAK'
        else:
            learning_strength = '[!] NO SIGNAL'
        
        status = '[OK] LEARNING' if risk < 0.3 else ('[!] STAGNANT' if risk < 0.6 else '[X] REGRESSING')
        
        details = {
            'rb_improvement': rb_improvement if len(rollbacks) >= 10 else 0,
            'comp_trend': comp_improvement if len(composite) >= 10 else 0,
            'learning_strength': learning_strength,
            'risk': risk,
            'status': status,
        }
        
        return risk, details
    
    def check(self, episode_data):
        """
        执行完整过拟合检查
        
        Args:
            episode_data: 当前episode数据
            
        Returns:
            dict: 包含所有维度风险评估的完整报告
        """
        self.update(episode_data)
        self.total_checks += 1
        
        # 执行各项检查
        sat_risk, sat_details = self._check_satisfaction_stability()
        reward_risk, reward_details = self._check_reward_health()
        action_risk, action_details = self._check_action_drift()
        loss_risk, loss_details = self._check_loss_trends()
        learning_risk, learning_details = self._check_learning_signal()
        
        # 加权综合评分 (根据重要性调整权重)
        weights = {
            'satisfaction': 0.25,      # 最重要：核心指标稳定性
            'reward': 0.20,            # 重要：训练健康度
            'action_drift': 0.20,      # 重要：策略变化检测
            'loss_trends': 0.15,       # 中等：辅助诊断
            'learning_signal': 0.20,   # 重要：验证真实学习
        }
        
        total_risk = (
            weights['satisfaction'] * sat_risk +
            weights['reward'] * reward_risk +
            weights['action_drift'] * action_risk +
            weights['loss_trends'] * loss_risk +
            weights['learning_signal'] * learning_risk
        )
        
        # 记录历史
        self.risk_history.append(total_risk)
        if len(self.risk_history) > self.window_size * 2:
            self.risk_history.pop(0)
        
        # 警报检查
        is_alert = total_risk >= self.alert_threshold
        if is_alert:
            self.alert_count += 1
        
        # 组装完整报告
        report = {
            'total_risk': total_risk,
            'is_alert': is_alert,
            'alert_level': '[HIGH] HIGH RISK' if total_risk >= 0.7 else ('[ELEVATED] ELEVATED' if total_risk >= self.alert_threshold else ('[MODERATE] MODERATE' if total_risk >= 0.4 else '[LOW] LOW')),
            'dimensions': {
                'satisfaction': {'weight': weights['satisfaction'], 'risk': sat_risk, **sat_details},
                'reward': {'weight': weights['reward'], 'risk': reward_risk, **reward_details},
                'action_drift': {'weight': weights['action_drift'], 'risk': action_risk, **action_details},
                'loss_trends': {'weight': weights['loss_trends'], 'risk': loss_risk, **loss_details},
                'learning_signal': {'weight': weights['learning_signal'], 'risk': learning_risk, **learning_details},
            },
            'summary': {
                'checks_performed': self.total_checks,
                'total_alerts': self.alert_count,
                'alert_rate': self.alert_count / max(self.total_checks, 1),
                'data_points': len(self.history['satisfaction']),
            },
            'recommendation': self._generate_recommendation(total_risk, sat_risk, action_risk, loss_risk),
        }
        
        return report
    
    def _generate_recommendation(self, total_risk, sat_risk, action_risk, loss_risk):
        """生成建议"""
        if total_risk < 0.3:
            return "[OK] Training healthy - continue monitoring"
        elif total_risk < 0.5:
            if action_risk > 0.4:
                return "[!] Monitor action distribution closely"
            elif loss_risk > 0.4:
                return "[!] Watch loss trends - possible overfitting starting"
            else:
                return "[!] Slight risk detected - increase monitoring frequency"
        elif total_risk < 0.7:
            if sat_risk > 0.5:
                return "[X] Satisfaction instability - consider reducing LR"
            elif action_risk > 0.5:
                return "[X] Action drift detected - increase entropy coefficient"
            else:
                return "[X] Elevated risk - consider early stopping or regularization"
        else:
            return "[CRITICAL] CRITICAL: Strong overfitting signal - STOP training immediately"
    
    def format_report(self, report):
        """
        格式化输出报告
        
        Args:
            report: check()方法返回的报告字典
            
        Returns:
            str: 格式化的报告字符串
        """
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append(f"[MONITOR] OVERFITTING DETECTION - Episode {self.total_checks}")
        lines.append(f"{'='*80}")
        
        # 总体风险
        total_risk = report['total_risk']
        alert_level = report['alert_level']
        lines.append(f"\n  [SCORE] Overall Risk Score: {total_risk:.3f} / 1.000 [{alert_level}]")
        
        if report['is_alert']:
            lines.append(f"  [!! ALERT !!] Risk threshold ({self.alert_threshold:.2f}) EXCEEDED!")
        
        # 各维度详情
        lines.append(f"\n  +-------------------------------------------------------------------+")
        lines.append(f"  | Dimension           | Weight | Risk  | Status     | Key Metrics |")
        lines.append(f"  +-------------------------------------------------------------------+")
        
        dim_names = {
            'satisfaction': 'Sat Stability',
            'reward': 'Reward Health',
            'action_drift': 'Action Drift',
            'loss_trends': 'Loss Trends',
            'learning_signal': 'Learning Signal',
        }
        
        for dim_key, dim_data in report['dimensions'].items():
            name = dim_names.get(dim_key, dim_key)
            weight = dim_data['weight']
            risk = dim_data['risk']
            status = dim_data.get('status', 'N/A')
            
            # 提取关键指标
            if dim_key == 'satisfaction':
                metrics = f"std={dim_data.get('short_std', 0):.4f}"
            elif dim_key == 'reward':
                metrics = f"CV={dim_data.get('cv', 0):.3f}"
            elif dim_key == 'action_drift':
                metrics=f"delta={dim_data.get('drift_abs', 0)*100:.1f}%"
            elif dim_key == 'loss_trends':
                metrics = f"{dim_data.get('actor_trend', 'N/A')}"
            elif dim_key == 'learning_signal':
                metrics = dim_data.get('learning_strength', 'N/A')
            else:
                metrics = "-"
            
            bar = self._risk_bar(risk)
            lines.append(f"  | {name:<19} | {weight:>5.0%} | {risk:>4.2f} | {status:<11} | {metrics:<12} | {bar}")
        
        lines.append(f"  +-------------------------------------------------------------------+")
        
        # 学习信号详情
        ls = report['dimensions']['learning_signal']
        if ls.get('rb_improvement') is not None:
            rb_imp = ls['rb_improvement'] * 100
            lines.append(f"\n  [EVIDENCE] Learning Signal:")
            rb_status = '[OK]' if rb_imp > 0 else '[WARN]'
            lines.append(f"     * Rollback rate improvement: {rb_imp:+.1f}% {rb_status}")
            if ls.get('comp_trend') is not None:
                comp_trend_str = '^' if ls['comp_trend'] > 0 else 'v'
                lines.append(f"     * Composite score trend: {ls['comp_trend']:+.4f}/ep {comp_trend_str}")
        
        # 建议
        lines.append(f"\n  [RECOMMENDATION] {report['recommendation']}")
        
        # 统计摘要
        summary = report['summary']
        lines.append(f"\n  [STATISTICS] Monitor Summary:")
        lines.append(f"     * Total checks: {summary['checks_performed']}")
        lines.append(f"     * Alerts triggered: {summary['total_alerts']} ({summary['alert_rate']*100:.1f}%)")
        lines.append(f"     * Data points analyzed: {summary['data_points']}")
        
        # 风险趋势
        if len(self.risk_history) >= 5:
            recent_risks = self.risk_history[-5:]
            trend = np.polyfit(range(len(recent_risks)), recent_risks, 1)[0]
            trend_str = '^ Worsening' if trend > 0.01 else ('v Improving' if trend < -0.01 else '-> Stable')
            lines.append(f"     * Risk trend (last 5): {trend_str} ({trend:+.4f}/check)")
        
        lines.append(f"{'='*80}\n")
        
        return '\n'.join(lines)
    
    def _risk_bar(self, risk):
        """生成风险条形图"""
        filled = int(risk * 20)
        empty = 20 - filled
        bar = '#' * filled + '-' * empty
        return bar


class EpisodeDetailedReporter:
    """
    Episode详细指标报告器 (V21: 全面的可观测指标输出)
    
    功能:
      1. 收集并打印每个Episode的所有关键性能指标
      2. 与历史最佳和基线算法对比
      3. 业务类型细分统计
      4. 负载均衡与连接状态分析
      5. Reward组分深度分解
    
    目标: 确保MAPPO在至少2-3个核心指标上超越增强算法
    """
    
    def __init__(self):
        self.episode_history = []
        self.best_metrics = {}
        self.baseline_metrics = {}  # 增强算法基线 (从实验3获取)
        
    def generate_report(self, ep_num, total_eps, episode_data, env=None):
        """
        生成详细的Episode报告
        
        Args:
            ep_num: 当前episode编号
            total_eps: 总episodes数
            episode_data: 包含所有指标的字典
            env: 环境实例(可选,用于提取额外指标)
        """
        print(f"\n{'='*80}")
        print(f"📊 EPISODE {ep_num}/{total_eps} 详细性能报告")
        print(f"{'='*80}")
        
        # ====== 第一部分: 核心性能指标 ======
        print(f"\n【第一部分】核心性能指标 (Core Performance Metrics)")
        print(f"{'-'*60}")
        
        sat = episode_data.get('satisfaction', 0)
        reward = episode_data.get('reward', 0)
        connected_ratio = episode_data.get('connected_ratio', 0)
        switch_success_rate = episode_data.get('switch_success_rate', 0)
        load_variance = episode_data.get('load_variance', 0)
        composite_score = episode_data.get('composite_score', 0)
        
        # 格式化输出
        metrics_display = [
            ('用户满意度 (Satisfaction)', f'{sat:.4f}', '↑'),
            ('平均Reward (Avg Reward)', f'{reward:.2f}', '↑'),
            ('连接保持率 (Connected Ratio)', f'{connected_ratio:.2%}', '↑'),
            ('切换成功率 (Switch Success)', f'{switch_success_rate:.2%}', '↑'),
            ('负载均衡度 (Load Balance)', f'{1.0-min(load_variance,1.0):.3f}', '↑'),
            ('综合评分 (Composite Score)', f'{composite_score:.4f}', '↑'),
        ]
        
        for name, value, trend in metrics_display:
            print(f"  {trend} {name}: {value}")
        
        # ====== 第二部分: 切换行为分析 ======
        print(f"\n【第二部分】切换行为分析 (Handover Behavior Analysis)")
        print(f"{'-'*60}")
        
        stay_pct = episode_data.get('stay_percentage', 0)
        switch_attempts = episode_data.get('switch_attempts', 0)
        switch_success = episode_data.get('switch_success', 0)
        switch_rollback = episode_data.get('switch_rollback', 0)
        switch_disconnect = episode_data.get('switch_disconnect', 0)
        
        print(f"  📌 动作分布: Stay={stay_pct:.1f}% | Switch={100-stay_pct:.1f}%")
        if switch_attempts > 0:
            print(f"  📌 切换统计:")
            print(f"     - 总尝试: {switch_attempts}")
            print(f"     - 成功率: {switch_success/switch_attempts:.1%} ({switch_success}/{switch_attempts})")
            print(f"     - 回滚率: {switch_rollback/switch_attempts:.1%} ({switch_rollback})")
            print(f"     - 断连率: {switch_disconnect/switch_attempts:.1%} ({switch_disconnect})")
            
            # 切换质量评级
            success_ratio = switch_success / max(switch_attempts, 1)
            if success_ratio >= 0.9:
                quality = "[EXCELLENT] ⭐⭐⭐"
            elif success_ratio >= 0.7:
                quality = "[GOOD] ⭐⭐"
            elif success_ratio >= 0.5:
                quality = "[ACCEPTABLE] ⭐"
            else:
                quality = "[POOR] ❌"
            print(f"  📌 切换质量评级: {quality}")
        else:
            print(f"  📌 无切换尝试 (纯Stay策略)")
        
        # ====== 第三部分: 业务类型细分 ======
        print(f"\n【第三部分】业务类型细分统计 (Business Type Breakdown)")
        print(f"{'-'*60}")
        
        biz_stats = episode_data.get('biz_statistics', {})
        biz_names = {0: '控制信令', 1: '视频回传', 2: '环境监测'}
        
        for bt in range(3):
            stats = biz_stats.get(bt, {})
            if stats:
                avg_sat_bt = stats.get('avg_satisfaction', 0)
                stay_bt = stats.get('stay_count', 0)
                switch_bt = stats.get('switch_count', 0)
                total_bt = stay_bt + switch_bt
                if total_bt > 0:
                    print(f"  🔹 业务{bt} ({biz_names[bt]}): "
                          f"sat={avg_sat_bt:.3f}, "
                          f"stay={stay_bt/total_bt:.0%}, "
                          f"switch={switch_bt/total_bt:.0%}")
        
        # ====== 第四部分: Reward组分分解 ======
        print(f"\n【第四部分】Reward组分分解 (Reward Decomposition)")
        print(f"{'-'*60}")
        
        n = max(episode_data.get('sample_count', 1), 1)
        reward_components = {
            '速率比增量 (Δrate)': episode_data.get('delta_sum', 0) / n,
            '反事实比较 (Counterfactual)': episode_data.get('value_reward', 0) / n,
            '业务权重奖励 (Business)': episode_data.get('biz_reward', 0) / n,
            '动作奖励 (Action)': episode_data.get('action_reward', 0) / n,
            '连接状态奖励 (Connection)': episode_data.get('connect_reward', 0) / n,
            '负载自适应 (Load Adaptive)': episode_data.get('load_adaptive', 0) / n,
        }
        
        for name, value in sorted(reward_components.items(), key=lambda x: x[1], reverse=True):
            bar_len = min(int(abs(value) * 20), 30)
            bar = '#' * bar_len if value >= 0 else '-' * bar_len
            sign = '+' if value >= 0 else ''
            print(f"  {name}: {sign}{value:.3f} {bar}")
        
        # ====== 第五部分: 训练健康指标 ======
        print(f"\n【第五部分】训练健康指标 (Training Health)")
        print(f"{'-'*60}")
        
        actor_loss = episode_data.get('actor_loss', 0)
        critic_loss = episode_data.get('critic_loss', 0)
        entropy = episode_data.get('entropy', 0)
        grad_norm = episode_data.get('grad_norm', 0)
        value_mse = episode_data.get('value_mse', 0)
        
        health_metrics = [
            ('Actor Loss', actor_loss, 1e-4),
            ('Critic Loss', critic_loss, 0.5),
            ('Entropy', entropy, 0.01),  # 最小阈值
            ('Grad Norm', grad_norm, None),
            ('Value MSE', value_mse, 1.0),
        ]
        
        for name, value, threshold in health_metrics:
            if threshold is not None:
                status = '[OK]' if value > threshold else '[LOW]'
            else:
                status = ''
            print(f"  {name}: {value:.6f} {status}")
        
        # ====== 第六部分: 历史对比与趋势 ======
        print(f"\n【第六部分】历史最佳对比 (Historical Best Comparison)")
        print(f"{'-'*60}")
        
        # 更新并显示历史最佳
        for metric_name in ['satisfaction', 'composite_score', 'switch_success_rate']:
            current_val = episode_data.get(metric_name, 0)
            best_val = self.best_metrics.get(metric_name, -float('inf'))
            
            if current_val > best_val:
                self.best_metrics[metric_name] = current_val
                # V21: 修复NaN问题 - 当best_val为-inf时显示为首次记录
                if best_val == -float('inf') or abs(best_val) < 1e-10:
                    improvement = 0
                    print(f"  🏆 NEW BEST! {metric_name}: {current_val:.4f} (首次记录)")
                else:
                    improvement = ((current_val - best_val) / abs(best_val) * 100)
                    print(f"  🏆 NEW BEST! {metric_name}: {current_val:.4f} (提升 {improvement:+.2f}%)")
            else:
                gap = best_val - current_val
                gap_pct = (gap / best_val * 100) if abs(best_val) > 1e-10 else 0
                print(f"  📈 {metric_name}: {current_val:.4f} (距最佳 {-gap_pct:.1f}%)")
        
        # 保存到历史
        self.episode_history.append({
            'episode': ep_num,
            **episode_data
        })
        
        print(f"\n{'='*80}")
        return self.best_metrics.copy()


# 创建全局reporter实例
episode_reporter = EpisodeDetailedReporter()

# 创建全局过拟合监控器实例 (V22)
overfitting_monitor = OverfittingMonitor(window_size=20, alert_threshold=0.65)


class TrainingTimer:
    """
    训练计时系统 (V21: 全面的时间追踪与预估)
    
    功能:
      1. 记录训练开始/结束时间
      2. 实时显示已用时间、预计剩余时间(ETA)
      3. Episode级别的耗时统计
      4. 训练速度监控 (episodes/hour, steps/second)
      5. 早停影响的时间节省预估
    
    输出格式:
      [TIME] Episode 25/500 | Elapsed: 00:15:32 | ETA: 01:23:45 | Speed: 98.5 ep/h
    """
    
    def __init__(self):
        self.start_time = None
        self.episode_start_time = None
        self.episode_times = []
        self.total_episodes = 0
        self.completed_episodes = 0
        
    def start_training(self, total_episodes):
        """训练开始时调用"""
        import time
        self.start_time = time.time()
        self.total_episodes = total_episodes
        self.completed_episodes = 0
        self.episode_times = []
        
        elapsed = self._format_time(0)
        print(f"\n{'='*80}")
        print(f"⏱️  TRAINING TIMER STARTED")
        print(f"{'='*80}")
        print(f"  📊 Total Episodes: {total_episodes}")
        print(f"  🎯 Target: Train MAPPO model with new V21 config system")
        print(f"  ⏰ Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  ⏱️  Elapsed: {elapsed}")
        print(f"{'='*80}\n")
        
    def start_episode(self, ep_num):
        """每个episode开始时调用"""
        import time
        self.episode_start_time = time.time()
        
    def end_episode(self, ep_num, verbose=True):
        """每个episode结束时调用"""
        import time
        if self.episode_start_time is None:
            return
            
        ep_duration = time.time() - self.episode_start_time
        self.episode_times.append(ep_duration)
        self.completed_episodes += 1
        
        if verbose and (self.completed_episodes % 10 == 0 or self.completed_episodes == 1):
            self._print_progress(ep_num)
            
    def _print_progress(self, current_ep):
        """打印进度信息"""
        import time
        
        # 计算已用时间
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)
        
        # 计算平均episode时间
        if len(self.episode_times) > 0:
            avg_ep_time = sum(self.episode_times[-20:]) / min(len(self.episode_times), 20)  # 最近20个平均
        else:
            avg_ep_time = 0
            
        # 计算预计剩余时间 (ETA)
        remaining_eps = self.total_episodes - current_ep - 1
        if avg_ep_time > 0 and remaining_eps > 0:
            eta_seconds = avg_ep_time * remaining_eps
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "N/A"
            
        # 计算速度
        if elapsed > 0:
            speed_eph = self.completed_episodes / (elapsed / 3600)  # episodes per hour
        else:
            speed_eph = 0
            
        # 进度百分比
        progress = (current_ep + 1) / self.total_episodes * 100
        
        # 创建进度条
        bar_length = 40
        filled = int(bar_length * current_ep / self.total_epochs if hasattr(self, 'total_epochs') else bar_length * current_ep / self.total_episodes)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\n⏱️  [TIME PROGRESS] Episode {current_ep+1}/{self.total_episodes} [{bar}] {progress:.1f}%")
        print(f"   ⏱️  Elapsed: {elapsed_str} | ETA: {eta_str} | Speed: {speed_eph:.1f} ep/h")
        print(f"   📈 Avg Episode Time: {avg_ep_time:.2f}s | Last Episode: {self.episode_times[-1]:.2f}s")
        
    def end_training(self, early_stopped=False, stopped_at_ep=None):
        """训练结束时调用"""
        import time
        
        total_time = time.time() - self.start_time
        total_str = self._format_time(total_time)
        
        # 统计信息
        if len(self.episode_times) > 0:
            avg_time = sum(self.episode_times) / len(self.episode_times)
            min_time = min(self.episode_times)
            max_time = max(self.episode_times)
        else:
            avg_time = min_time = max_time = 0
            
        print(f"\n{'='*80}")
        print(f"⏱️  TRAINING TIMER ENDED")
        print(f"{'='*80}")
        print(f"  ⏰ End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  ⏱️  Total Duration: {total_str}")
        print(f"  📊 Episodes Completed: {self.completed_episodes}/{self.total_episodes}")
        
        if early_stopped and stopped_at_ep is not None:
            saved_eps = self.total_episodes - stopped_at_ep
            saved_time_est = avg_time * saved_eps
            saved_str = self._format_time(saved_time_est)
            print(f"  ✅ Early Stopped at Episode: {stopped_at_ep}")
            print(f"  💰 Time Saved (vs full training): ~{saved_str} ({saved_eps} episodes)")
        
        print(f"  ⚡ Performance Stats:")
        print(f"     - Average Episode Time: {avg_time:.2f}s")
        print(f"     - Fastest Episode: {min_time:.2f}s")
        print(f"     - Slowest Episode: {max_time:.2f}s")
        
        if total_time > 0:
            speed = self.completed_episodes / (total_time / 3600)
            print(f"     - Overall Speed: {speed:.1f} episodes/hour")
            
        print(f"{'='*80}\n")
        
        return {
            'total_time': total_time,
            'completed_episodes': self.completed_episodes,
            'avg_episode_time': avg_time,
            'early_stopped': early_stopped,
        }
        
    def _format_time(self, seconds):
        """格式化时间为 HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# 创建全局timer实例
training_timer = TrainingTimer()


class TrainingSafetyManager:
    """
    训练安全管理器 (V21: 中断安全 + 预训练缓存)
    
    功能:
      1. 信号处理: 捕获 Ctrl+C (SIGINT) 和终止信号 (SIGTERM)
      2. 紧急保存: 中断时自动保存当前模型和训练状态
      3. 预训练缓存: 缓存预训练结果，避免重复收集示范数据
      4. 断点续训: 支持从中断点恢复训练
    
    使用方式:
      safety_mgr = TrainingSafetyManager(agent, model_path)
      safety_mgr.enable_signal_handlers()
      # ... 训练过程中 ...
      safety_mgr.disable_signal_handlers()  # 训练结束时禁用
    """
    
    def __init__(self):
        self.agent = None
        self.latest_model_path = None
        self.best_model_path = None
        self.checkpoint_dir = None
        self.current_episode = 0
        self.training_state = {
            'episode_rewards': [],
            'episode_satisfactions': [],
            'best_composite_score': float('-inf'),
            'best_sat': float('-inf'),
            'composite_window': [],
            'satisfaction_window': [],
        }
        self._original_sigint = None
        self._original_sigterm = None
        self._interrupted = False
        self._lock = threading.Lock()
        
    def setup(self, agent, latest_model_path, best_model_path, checkpoint_dir=None):
        """初始化安全管理器"""
        self.agent = agent
        self.latest_model_path = latest_model_path
        self.best_model_path = best_model_path
        self.checkpoint_dir = checkpoint_dir or os.path.dirname(latest_model_path)
        
        # 创建checkpoint目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"\n[SAFETY] TrainingSafetyManager initialized")
        print(f"         Latest model: {latest_model_path}")
        print(f"         Best model: {best_model_path}")
        print(f"         Checkpoint dir: {self.checkpoint_dir}")
        
    def update_state(self, episode_num, reward, satisfaction, composite_score,
                     episode_rewards=None, episode_sats=None,
                     composite_window=None, sat_window=None,
                     best_composite=None, best_sat=None):
        """更新当前训练状态（用于断点续训）"""
        with self._lock:
            self.current_episode = episode_num
            if episode_rewards is not None:
                self.training_state['episode_rewards'] = episode_rewards.copy()
            if episode_sats is not None:
                self.training_state['episode_satisfactions'] = episode_sats.copy()
            if composite_window is not None:
                self.training_state['composite_window'] = composite_window.copy()
            if sat_window is not None:
                self.training_state['satisfaction_window'] = sat_window.copy()
            if best_composite is not None:
                self.training_state['best_composite_score'] = best_composite
            if best_sat is not None:
                self.training_state['best_sat'] = best_sat
                
    def enable_signal_handlers(self):
        """启用信号处理器（捕获Ctrl+C等）"""
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Windows可能不支持SIGTERM，使用try-except
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
        except Exception as e:
            print(f"[WARN] Cannot set SIGINT handler: {e}")
            
        print(f"[SAFETY] Signal handlers ENABLED (Press Ctrl+C for safe shutdown)")
        
    def disable_signal_handlers(self):
        """禁用信号处理器"""
        try:
            if self._original_sigint is not None:
                signal.signal(signal.SIGINT, self._original_sigint)
            if self._original_sigterm is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm)
        except Exception:
            pass
            
        print(f"[SAFETY] Signal handlers DISABLED")
        
    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        print(f"\n\n{'='*80}")
        print(f"[INTERRUPT] Signal {signum} received - Initiating SAFE SHUTDOWN")
        print(f"{'='*80}")
        
        self._interrupted = True
        
        # 执行紧急保存
        self._emergency_save()
        
        # 恢复原始信号处理并退出
        self.disable_signal_handlers()
        
        print(f"\n[SAFE EXIT] Training interrupted safely at Episode {self.current_episode}")
        print(f"         Model saved to: {self.latest_model_path}")
        print(f"         You can resume training later with the saved model.")
        sys.exit(0)
        
    def _emergency_save(self):
        """紧急保存当前状态"""
        if self.agent is None:
            print("[ERROR] No agent to save!")
            return
            
        try:
            import time
            
            print(f"\n[EMERGENCY SAVE] Saving current state...")
            start_time = time.time()
            
            # 1. 保存最新模型
            if self.latest_model_path:
                self.agent.save(self.latest_model_path)
                print(f"  [OK] Latest model saved: {self.latest_model_path}")
            
            # 2. 保存训练状态checkpoint
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'training_checkpoint_ep{self.current_episode}.pkl'
            )
            
            checkpoint_data = {
                'episode': self.current_episode,
                'timestamp': datetime.now().isoformat(),
                'training_state': self.training_state.copy(),
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
                
            elapsed = time.time() - start_time
            print(f"  [OK] Checkpoint saved: {checkpoint_path} ({elapsed:.2f}s)")
            print(f"  [OK] Emergency save COMPLETED successfully")
            
        except Exception as e:
            print(f"[ERROR] Emergency save failed: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def get_pretrain_cache_path(num_uav, num_bs):
        """获取预训练缓存路径"""
        cache_dir = os.path.join(RESULT_DIR, 'mappo_models', 'pretrain_cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f'pretrain_{num_uav}uav_{num_bs}bs.pkl')
    
    @staticmethod
    def save_pretrain_cache(agent, demos, num_uav, num_bs, 
                           pretrain_epochs=50, pretrain_loss=0.0):
        """保存预训练缓存"""
        cache_path = TrainingSafetyManager.get_pretrain_cache_path(num_uav, num_bs)
        
        cache_data = {
            'agent_state_dict': agent.actor.state_dict(),
            'demos': demos[:500],  # 只保存500条作为样本
            'num_uav': num_uav,
            'num_bs': num_bs,
            'pretrain_epochs': pretrain_epochs,
            'final_loss': pretrain_loss,
            'created_at': datetime.now().isoformat(),
            'version': 'V21',
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
            
        print(f"[CACHE] Pretrain cache saved: {cache_path}")
        print(f"        Demos cached: {len(demos)} -> 500 samples")
        return cache_path
        
    @staticmethod
    def load_pretrain_cache(num_uav, num_bs):
        """加载预训练缓存"""
        cache_path = TrainingSafetyManager.get_pretrain_cache_path(num_uav, num_bs)
        
        if not os.path.exists(cache_path):
            return None
            
        if os.environ.get('FORCE_RETRAIN'):
            print(f"[CACHE] FORCE_RETRAIN set - ignoring cache")
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            print(f"[CACHE] Pretrain cache loaded: {cache_path}")
            print(f"        Created: {cache_data.get('created_at', 'unknown')}")
            print(f"        Pretrain epochs: {cache_data.get('pretrain_epochs', 'unknown')}")
            print(f"        Final loss: {cache_data.get('final_loss', 'unknown'):.4f}")
            
            return cache_data
            
        except Exception as e:
            print(f"[WARN] Failed to load pretrain cache: {e}")
            return None
    
    @staticmethod
    def apply_pretrain_cache(agent, cache_data):
        """应用预训练缓存到agent"""
        try:
            agent.actor.load_state_dict(cache_data['agent_state_dict'])
            print(f"[CACHE] Pretrained weights applied to agent")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to apply pretrained weights: {e}")
            return False


# 创建全局safety manager实例
training_safety = TrainingSafetyManager()


class ExperimentBAMAPPO:
    """BA-MAPPO 多智能体强化学习实验"""

    MODEL_DIR = os.path.join(RESULT_DIR, 'mappo_models')
    RESULT_FILE = os.path.join(RESULT_DIR, 'mappo_experiment_data.pkl')

    @staticmethod
    def run(num_uav_list=(300,),       # [Step3对齐] 300UAV/8BS ≈ 77%负载率 (与实验3一致)
            num_bs_list=(8,),
            num_steps=350,              # [Step3对齐] 与实验3步数一致 (原150)
            train_episodes=500,         # [Step3增加] 增加训练轮次确保低负载下充分收敛 (原300)
            eval_episodes=10,           # [Step3增加] 增加评估重复次数提高统计可靠性 (原5)
            bs_capacity_range=(500, 1000),  # 与实验3一致
            pos_range=1000,
            load_models=False, phase='both',
            verbose=True,
            # BA-MAPPO 配置开关
            use_biz_heads=True,
            use_attention_critic=True,
            rollout_length=200,          # [Step3增加] 增长rollout以适应更长episode (原150)
            # V23优化: 调整学习率以加速收敛（针对实验3场景优化）
            actor_lr=8e-5, critic_lr=5e-4,       # 原(5e-5, 3e-4) → 提升60%加速收敛
            hidden_dim=64, critic_hidden_dim=128,
            # 训练效率优化
            train_sample_agents=50,      # [Step3启用] Agent采样加速训练 (原0)
            attention_sample_agents=50,  # [Step3启用] Attention采样加速训练 (原0)
            num_parallel_envs=2,         # [Step3增加] 并行环境加速数据收集 (原1)
            # PPO超参数（与mappo_standard_config对齐）
            gamma=0.99,                  # [Step3调整] 增大折扣因子适应长episode (原0.95)
            gae_lambda=0.95,
            clip_epsilon=0.3,
            entropy_coef=0.15,           # [V17] 进一步增大熵系数，保持训练全程探索 (原0.08)
            value_loss_coef=0.5,
            batch_size=64,               # [Step3调整] 增大批次大小适应更多agents (原32)
            num_epochs=4,                # [Step3调整] 增加PPO epoch数 (原3)
            # [V17] 业务识别模型（评估阶段接入，训练阶段不用）
            recognition_model=None, scaler=None):
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
                # V17: 评估阶段接入业务识别模型（带噪声，更接近真实场景）
                recognition_model=recognition_model, scaler=scaler,
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

        # V23: 训练容量与实验3完全对齐（消除训练/评估分布偏移）
        #   训练: (500, 1000) → 平均750 → 负载率~78%（与实验3一致）
        #   评估: (500, 1000) → 平均750 → 负载率~78%
        #   原因: V22的高负载训练(129%)导致模型学到过度保守策略，
        #         在标准场景(78%负载)下表现次优（sat仅0.927 vs 增强0.995）
        train_capacity_range = bs_capacity_range  # 直接使用评估容量，完全对齐

        training_results = {}

        for num_uav, num_bs in zip(num_uav_list, num_bs_list):
            avg_demand = 0.4 * 0.5 + 0.3 * 50 + 0.3 * 1.0
            avg_cap_train = sum(train_capacity_range) / 2
            avg_cap_eval = sum(bs_capacity_range) / 2
            load_rate_train = num_uav * avg_demand / (num_bs * avg_cap_train) * 100
            load_rate_eval = num_uav * avg_demand / (num_bs * avg_cap_eval) * 100
            print(f"\n>>> 训练 UAV={num_uav}/BS={num_bs}")
            print(f"    训练负载率~{load_rate_train:.0f}% (容量{train_capacity_range}) | 评估负载率~{load_rate_eval:.0f}% (容量{bs_capacity_range}) <<<")

            set_global_seed(GLOBAL_SEED + num_uav * 100)
            model_path = os.path.join(ExperimentBAMAPPO.MODEL_DIR,
                                      f'mappo_{num_bs}bs_{num_uav}uav.pt')

            # 直接在标准环境中训练，确保模型能够适应真实场景
            # V20: 使用高负载容量范围进行训练
            env = MultiAgentHandoverEnv(
                num_bs=num_bs, num_uav=num_uav,
                max_steps=num_steps, seed=GLOBAL_SEED + num_uav * 100,
                bs_capacity_range=train_capacity_range,
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
            
            # V22: 启用增强算法行为克隆预训练
            # 原因: 随机初始化下 Actor 完全不学（a_loss 恒定，H=1.79 纯随机）
            #       Critic 在学但策略梯度为零 → 训练停滞
            # 方案: 用增强算法收集示范数据，行为克隆初始化策略网络
            #       让 RL 从合理策略出发微调，而非从纯随机开始探索

            if agent.use_pretrain and not (load_models and os.path.exists(model_path)):
                # V21: 检查预训练缓存
                pretrain_cache = TrainingSafetyManager.load_pretrain_cache(num_uav, num_bs)
                
                if pretrain_cache is not None:
                    # 使用缓存的预训练权重
                    print("  [V22] Loading PRETRAINED weights from CACHE...")
                    success = TrainingSafetyManager.apply_pretrain_cache(agent, pretrain_cache)
                    if success:
                        print(f"         Pretrained model loaded (cache created: {pretrain_cache.get('created_at', 'unknown')})")
                        print(f"         → 预期: sat 从 ~0.92 起步，逐步上升到 ~0.96+")
                    else:
                        print("         [WARN] Failed to load cache, will retrain...")
                        pretrain_cache = None
                
                if pretrain_cache is None:
                    # 无缓存或加载失败，执行完整预训练
                    print("  [V22] 启用增强算法行为克隆预训练...")
                    print("         收集增强算法决策作为示范数据...")

                    demos = agent.collect_demonstrations(env, num_demos=2000)
                    print(f"         收集到 {len(demos)} 条示范数据")

                    # 执行预训练并获取最终loss
                    pretrain_result = agent.pretrain(demos, epochs=50, batch_size=64, min_loss_threshold=0.05, patience=5)
                    final_loss = pretrain_result.get('final_loss', 0.0) if isinstance(pretrain_result, dict) else 0.0

                    # V21: 保存预训练缓存
                    TrainingSafetyManager.save_pretrain_cache(
                        agent, demos, num_uav, num_bs,
                        pretrain_epochs=50,
                        pretrain_loss=final_loss
                    )

                    print("         预训练完成，策略已初始化为近似增强算法水平")
                    print(f"         Final loss: {final_loss:.4f}")
                    print("         → 预期: sat 从 ~0.92 起步，逐步上升到 ~0.96+")
                    print("         → Cache saved for future use (skip collection next time)")

            _bs = num_bs_list[0] if isinstance(num_bs_list, (list, tuple)) else num_bs_list

            if load_models and os.path.exists(model_path):
                agent.load(model_path)
                training_results[num_uav] = {'loaded': True}
                print(f"  已加载模型: {model_path}")
                continue
            
            # 方案2: 添加obs_normalizer预热（含BS数以匹配obs_dim）
            warmup_cache_path = os.path.join(ExperimentBAMAPPO.MODEL_DIR, f'warmup_normalizer_{num_uav}uav_{_bs}bs.pkl')
            
            if os.path.exists(warmup_cache_path) and not os.environ.get('FORCE_RETRAIN'):
                print(f"  加载预热状态缓存: {warmup_cache_path}")
                with open(warmup_cache_path, 'rb') as f:
                    warmup_state = pickle.load(f)
                    agent.obs_normalizer.mean = warmup_state['mean']
                    agent.obs_normalizer.var = warmup_state['var']
                    agent.obs_normalizer.count = warmup_state['count']
            else:
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
                
                # 保存预热状态
                warmup_state = {
                    'mean': agent.obs_normalizer.mean,
                    'var': agent.obs_normalizer.var,
                    'count': agent.obs_normalizer.count
                }
                with open(warmup_cache_path, 'wb') as f:
                    pickle.dump(warmup_state, f)
                print(f"  预热状态已保存: {warmup_cache_path}")

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
            save_interval = 10  # V21: 减少保存间隔到10个episode（原50，提高中断安全性）
            health_monitor = TrainingHealthMonitor(check_interval=10, verbose=True)
            # 分离 best 和 latest 模型路径
            best_model_path = model_path.replace('.pt', '_best.pt')
            latest_model_path = model_path.replace('.pt', '_latest.pt')
            # ---- Early stopping 参数 (V4/V21: 从配置读取) ----
            tc = MAPPOConfig.TrainingConfig  # V21: 简化引用
            early_stop_average_window = tc.early_stop_window                  # 平均40轮无改善即停止 (原120, 减少67%)
            early_stop_min_delta = tc.early_stop_min_delta                   # 改善阈值 (原0.0005, 稍微放宽)
            early_stop_warmup = max(20, int(train_episodes * tc.warmup_ratio))  # warmup缩短到10% (原25%)
            satisfaction_window = []                        # 存储最近N轮综合评分
            composite_window = []                           # 存储最近N轮综合评分用于平均
            best_composite_score = float('-inf')            # 最佳综合评分
            early_stopped = False
            
            # 综合评分权重配置 (V21: 从配置读取)
            COMPOSITE_WEIGHTS = tc.composite_weights.copy()  # V21: 使用配置中的权重
            # 早期健康检查: 在 10% 训练进度时检查 reward 是否在正增长
            health_check_ep = max(10, train_episodes // 10)
            mid_check_eps = [2 * health_check_ep, 3 * health_check_ep]  # Ep200, Ep300

            # 设置 LR schedule 的总步数
            agent._total_train_steps = train_episodes
            agent._current_train_step = 0

            # 直接在标准环境中训练，确保模型能够适应真实场景
            print("\n  开始训练：标准环境 + Domain Randomization")
            
            # V21: 启动训练计时系统
            training_timer.start_training(train_episodes)
            
            # V21: 初始化训练安全管理器（中断安全 + 预训练缓存）
            training_safety.setup(agent, latest_model_path, best_model_path)
            training_safety.enable_signal_handlers()
            
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

                # V17: simple环境进度打印也改为每episode
                if verbose and (ep + 1) % 1 == 0:
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
                    return MultiAgentHandoverEnv(
                        num_bs=num_bs, num_uav=num_uav,
                        max_steps=num_steps, seed=seed_val,
                        bs_capacity_range=train_capacity_range,   # V20: 训练用高负载
                        pos_range=pos_range,
                    )
                vec_envs = VectorizedEnvs(
                    _make_env, num_envs=num_parallel, seed=GLOBAL_SEED + num_uav * 100
                )
                print(f"  [VECTORIZED] 使用 {num_parallel} 个并行环境收集数据")

            for ep in range(train_episodes):
                # V21: 开始episode计时
                training_timer.start_episode(ep)
                
                # ====== Seed Randomization (V14: 提升泛化能力) ======
                # 核心思想: 每个episode使用不同的随机种子，避免模型过拟合到特定seed
                # 实现方式: 
                #   base_seed = GLOBAL_SEED (保证可重现性)
                #   ep_seed = base_seed + ep * prime_offset + random_jitter
                #   其中 prime_offset = 1009 (大质数, 减少周期性)
                #         random_jitter = [0, 100) 范围内随机值
                #
                # 泛化效果预期:
                #   - 训练时: 模型看到更多样的初始状态分布
                #   - 测试时: 对不同seed的适应性提升3-5%
                #   - 过拟合风险: 降低 (不再记忆特定seed的模式)
                PRIME_OFFSET = tc.prime_offset  # V21: 大质数，避免简单的线性关系
                MAX_JITTER = tc.max_jitter       # V21: 最大随机偏移量
                ep_seed = GLOBAL_SEED + ep * PRIME_OFFSET + np.random.randint(0, MAX_JITTER)
                set_global_seed(ep_seed)
                
                # V20/V21: 基于训练容量范围进行温和DR（从配置读取）
                # 训练用高负载(500,680)，评估保持(500,1000)
                dr_cfg = MAPPOConfig.TrainingConfig  # V21
                random_capacity_range = (
                    int(train_capacity_range[0] * (dr_cfg.dr_capacity_low_scale + (dr_cfg.dr_capacity_high_scale - dr_cfg.dr_capacity_low_scale) * np.random.rand())),
                    int(train_capacity_range[1] * (dr_cfg.dr_capacity_low_scale + (dr_cfg.dr_capacity_high_scale - dr_cfg.dr_capacity_low_scale) * np.random.rand()))
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

                # V17: 进度打印从每30个episode改为每1个episode（修复长时间无输出的问题）
                if verbose and (ep + 1) % 1 == 0:
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
                          f"seed={ep_seed}, "
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

                # ---- Early stopping 判断 (V4: 基于综合评分) ----
                ep_sat = np.mean(episode_sat)
                
                # 计算当前episode的综合指标
                ep_connected_ratio = 1.0 - (ep_disconnected_steps / max(env.num_agents * num_steps, 1))
                ep_load_variance = np.var([bs.load_ratio for bs in env.env.base_stations.values()]) if hasattr(env, 'env') else 0.0
                ep_switch_success_rate = ss / max(sa, 1) if sa > 0 else 1.0
                ep_critical_sat = np.mean([s for s in episode_sat if True])  # 简化: 使用整体满意度作为代理
                
                # 计算综合评分 (加权组合, 所有指标归一化到[0,1])
                composite_score = (
                    COMPOSITE_WEIGHTS['satisfaction'] * ep_sat +
                    COMPOSITE_WEIGHTS['connected_ratio'] * ep_connected_ratio +
                    COMPOSITE_WEIGHTS['load_balance'] * (1.0 - min(ep_load_variance, 1.0)) +  # 负载方差越小越好
                    COMPOSITE_WEIGHTS['switch_success'] * ep_switch_success_rate +
                    COMPOSITE_WEIGHTS['critical_sat'] * ep_critical_sat
                )
                
                # V21: 生成详细Episode报告 (每5个episode输出一次详细报告)
                if verbose and (ep + 1) % 5 == 0:
                    episode_data = {
                        'satisfaction': ep_sat,
                        'reward': episode_reward,
                        'reward_std': np.std(episode_rewards[-min(10, len(episode_rewards)):]) if len(episode_rewards) > 1 else 0.0,  # V22: 添加reward标准差
                        'connected_ratio': ep_connected_ratio,
                        'switch_success_rate': ep_switch_success_rate,
                        'load_variance': ep_load_variance,
                        'composite_score': composite_score,
                        'stay_percentage': stay_pct,
                        'switch_attempts': sa,
                        'switch_success': ss,
                        'switch_rollback': sr,
                        'switch_disconnect': sd,
                        'biz_statistics': {
                            bt: {
                                'avg_satisfaction': np.mean(ep_biz_stats[bt]['satisfaction']) if ep_biz_stats[bt]['satisfaction'] else 0,
                                'stay_count': ep_biz_stats[bt]['stay'],
                                'switch_count': ep_biz_stats[bt]['switch'],
                            } for bt in range(3)
                        },
                        'delta_sum': ep_reward_diag.get('delta_sum', 0),
                        'value_reward': ep_reward_diag.get('value_reward_sum', 0),
                        'biz_reward': ep_reward_diag.get('biz_reward_sum', 0),
                        'action_reward': ep_reward_diag.get('action_reward_sum', 0),
                        'connect_reward': ep_reward_diag.get('connect_reward_sum', 0),
                        'load_adaptive': ep_reward_diag.get('load_adaptive_sum', 0),
                        'sample_count': max(ep_reward_diag.get('count', 1), 1),
                        'actor_loss': avg_al,
                        'critic_loss': avg_cl,
                        'entropy': avg_ent,
                        'grad_norm': avg_ag if avg_ag is not None else 0,
                        'value_mse': avg_vmse if avg_vmse is not None else 0,
                    }
                    
                    best_metrics = episode_reporter.generate_report(
                        ep_num=ep+1,
                        total_eps=train_episodes,
                        episode_data=episode_data,
                        env=env
                    )
                    
                    # V22: 过拟合监控检查 (在详细报告后输出)
                    risk_report = overfitting_monitor.check(episode_data)
                    print(overfitting_monitor.format_report(risk_report))
                    
                    # V22: 高风险警报处理
                    if risk_report['is_alert']:
                        print(f"  [!! OVERFITTING ALERT !!] Risk score {risk_report['total_risk']:.3f} exceeds threshold!")
                        print(f"     -> Recommendation: {risk_report['recommendation']}")
                        # 可选: 自动保存当前状态以便分析
                        # if risk_report['total_risk'] > 0.8:
                        #     print("  🚨 CRITICAL: Consider stopping training immediately")
                
                is_best_composite = composite_score > best_composite_score + early_stop_min_delta
                is_best_sat = ep_sat > best_sat + early_stop_min_delta
                is_best_reward = episode_reward > best_reward + early_stop_min_delta
                
                if is_best_composite:
                    best_composite_score = composite_score
                    print(f"    [NEW_BEST] 综合评分新高: {composite_score:.4f} "
                          f"(sat={ep_sat:.3f}, conn={ep_connected_ratio:.3f}, "
                          f"lb={ep_load_variance:.4f}, sw={ep_switch_success_rate:.2%})")
                
                if is_best_sat:
                    best_sat = ep_sat
                if is_best_reward:
                    best_reward = episode_reward

                # 更新滑动窗口 (使用综合评分)
                composite_window.append(composite_score)
                satisfaction_window.append(ep_sat)
                if len(composite_window) > early_stop_average_window:
                    composite_window.pop(0)
                if len(satisfaction_window) > early_stop_average_window:
                    satisfaction_window.pop(0)

                # 模型保存: best 模型基于综合评分, latest 定期保存
                if is_best_composite:
                    agent.save(best_model_path)
                if (ep + 1) % save_interval == 0:
                    agent.save(latest_model_path)
                
                # V21: 更新训练安全管理器状态（用于中断恢复）
                training_safety.update_state(
                    episode_num=ep,
                    reward=episode_reward,
                    satisfaction=np.mean(episode_sat) if episode_sat else 0,
                    composite_score=composite_score,
                    episode_rewards=episode_rewards,
                    episode_sats=episode_satisfactions,
                    composite_window=composite_window,
                    sat_window=satisfaction_window,
                    best_composite=best_composite_score,
                    best_sat=best_sat
                )

                # V21: 结束episode计时
                training_timer.end_episode(ep, verbose=verbose)

                # 平均早停判断: 基于综合评分 (V4核心改进)
                if (ep >= early_stop_warmup and 
                    ep >= health_check_ep * 2 and
                    len(composite_window) >= early_stop_average_window):
                    window_avg_composite = np.mean(composite_window)
                    if window_avg_composite <= best_composite_score + early_stop_min_delta:
                        if verbose:
                            print(f"\n  [STOP] Early stopping [Episode {ep+1}] (V4-Composite): "
                                  f"最近 {early_stop_average_window} 轮平均综合评分 {window_avg_composite:.4f} 无显著改善 "
                                  f"(best_composite={best_composite_score:.4f}, "
                                  f"best_sat={best_sat:.4f}, episodes_saved={500-(ep+1)})")
                        early_stopped = True
                        break

            # V21: 结束训练计时
            timer_results = training_timer.end_training(
                early_stopped=early_stopped,
                stopped_at_ep=ep if early_stopped else None
            )

            # V21: 禁用信号处理器（训练正常结束）
            training_safety.disable_signal_handlers()

            # 训练结束: 确保 model_path 指向 best 模型
            import shutil
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, model_path)
                if verbose:
                    print(f"  [OK] 最终模型已更新为 best_composite={best_composite_score:.4f} 的版本 "
                          f"(best_sat={best_sat:.4f})")

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
                           attention_sample_agents=0,
                           recognition_model=None, scaler=None):
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

            # V17: 评估阶段传入业务识别模型（训练时用ground truth）
            _has_recog = recognition_model is not None and scaler is not None
            if _has_recog:
                print(f"  [评估] 已接入业务识别模型 (带预测噪声)")

            env = MultiAgentHandoverEnv(
                num_bs=num_bs, num_uav=num_uav,
                max_steps=num_steps, seed=GLOBAL_SEED + num_uav * 200,
                bs_capacity_range=bs_capacity_range,
                pos_range=pos_range,
                recognition_model=recognition_model if _has_recog else None,
                scaler=scaler if _has_recog else None,
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
                    eval_env = MultiAgentHandoverEnv(
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
        """生成所有可视化图表（V17优化：适配高负载率下接近满分的场景）"""
        mode_name = all_results.get('config', {}).get('mode_name', 'BA-MAPPO')
        print(f"\n生成 {mode_name} 实验可视化...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'{mode_name} 实验结果', fontsize=16, fontweight='bold')

        # ===== 图1(a): 训练收敛曲线（带压力测试标注） =====
        ax = axes[0, 0]
        if 'training' in all_results:
            for num_uav in num_uav_list:
                if num_uav in all_results['training']:
                    tr = all_results['training'][num_uav]
                    if 'rewards' in tr and len(tr['rewards']) > 0:
                        rewards = np.array(tr['rewards'])
                        episodes = np.arange(1, len(rewards) + 1)
                        
                        # 绘制原始reward（浅色）
                        ax.plot(episodes, rewards, alpha=0.2, color='steelblue', linewidth=0.5)
                        
                        # 移动平均
                        window = max(3, min(len(rewards) // 15, 15))
                        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                        ax.plot(range(window, len(rewards)+1), smoothed,
                                label=f'UAV={num_uav}', alpha=0.9, linewidth=1.5, color='steelblue')
                        
                        # V17: 压力测试标注已移除（V17取消压力测试）
                        
            ax.set_xlabel('Episode')
            ax.set_ylabel('团队奖励')
            ax.set_title('(a) 训练收敛曲线')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # ===== 图2(b): 训练损失曲线（替代无变化的满意度） =====
        ax = axes[0, 1]
        has_loss_data = False
        if 'training' in all_results:
            for num_uav in num_uav_list:
                if num_uav in all_results['training']:
                    tr = all_results['training'][num_uav]
                    if 'actor_losses' in tr and len(tr['actor_losses']) > 0:
                        has_loss_data = True
                        episodes_al = np.arange(1, len(tr['actor_losses'])+1)
                        episodes_cl = np.arange(1, len(tr['critic_losses'])+1)
                        ax.plot(episodes_al, tr['actor_losses'], alpha=0.8,
                                linewidth=1, label=f'Actor Loss ({num_uav}UAV)', color='#e74c3c')
                        ax.plot(episodes_cl, [cl*0.01 for cl in tr['critic_losses']], alpha=0.8,
                                linewidth=1, label=f'Critic Loss×0.01 ({num_uav}UAV)', color='#3498db')
        
        if has_loss_data:
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss值')
            ax.set_title('(b) 训练损失变化')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '暂无损失数据', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('(b) 训练损失变化')

        # ===== 图3(c): 算法多维度对比柱状图（放大Y轴显示差异） =====
        ax = axes[0, 2]
        if 'evaluation' in all_results:
            uav_to_show = [u for u in num_uav_list if u in all_results.get('evaluation', {})]
            if uav_to_show:
                u = uav_to_show[0]
                ev = all_results['evaluation'][u]
                
                metrics = ['avg', 'critical']
                metric_labels = ['平均满意度', '关键业务满意度']
                
                x = np.arange(len(metrics))
                width = 0.25
                
                mappo_vals = [ev['mappo'].get(m, (0,0))[0] for m in metrics]
                enh_vals = [ev['enhanced'].get(m, (0,0))[0] for m in metrics]
                trad_vals = [ev['traditional'].get(m, (0,0))[0] for m in metrics]
                
                bars1 = ax.bar(x - width, trad_vals, width, label='传统算法(3GPP)',
                              color=COLORS.get('danger', '#e74c3c'), alpha=0.85)
                bars2 = ax.bar(x, enh_vals, width, label='增强算法(本文)',
                              color=COLORS.get('primary', '#3498db'), alpha=0.85)
                bars3 = ax.bar(x + width, mappo_vals, width, label=mode_name,
                              color=COLORS.get('warning', '#f39c12'), alpha=0.85)
                
                # V17: Y轴范围缩小到[0.93, 1.0]以放大差异可见性
                all_vals = trad_vals + enh_vals + mappo_vals
                y_min = max(0.93, min(all_vals) - 0.015)
                y_max = min(1.001, max(all_vals) + 0.005)
                ax.set_ylim(y_min, y_max)
                
                # 在柱子上标数值
                for bars in [bars1, bars2, bars3]:
                    for bar in bars:
                        h = bar.get_height()
                        if h > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., h + 0.0008,
                                   f'{h:.4f}', ha='center', va='bottom', fontsize=6.5, rotation=0)
                
                ax.set_xticks(x)
                ax.set_xticklabels(metric_labels, fontsize=9)
                ax.set_ylabel('满意度')
                ax.set_title(f'(c) 算法对比 (UAV={u}, Y轴放大)')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3, axis='y')

        # ===== 图4(d): 策略分布热力图 =====
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

        # ===== 图5(e): 通信指标对比（替代空位置） =====
        ax = axes[1, 1]
        if 'evaluation' in all_results:
            uav_to_show = [u for u in num_uav_list if u in all_results.get('evaluation', {})]
            if uav_to_show:
                u = uav_to_show[0]
                ev = all_results['evaluation'][u]
                
                comm_metrics = ['handover_latency', 'ping_jitter', 'packet_loss_rate', 'qos_violation_rate']
                comm_labels = ['切换延迟(ms)', 'Ping抖动(ms)', '丢包率(%)', 'QoS违规(%)']
                
                has_comm_data = any(
                    ev.get(algo, {}).get('communication_metrics', {}).get(cm, (0,0))[0] > 0.001
                    for algo in ['mappo', 'enhanced', 'traditional']
                    for cm in comm_metrics
                )
                
                if has_comm_data:
                    x = np.arange(len(comm_metrics))
                    width = 0.25
                    
                    def get_comm(algo):
                        return [ev.get(algo, {}).get('communication_metrics', {}).get(cm, (0,0))[0]
                               for cm in comm_metrics]
                    
                    mappo_comm = get_comm('mappo')
                    enh_comm = get_comm('enhanced')
                    trad_comm = get_comm('traditional')
                    
                    ax.bar(x - width, trad_comm, width, label='传统算法',
                          color=COLORS.get('danger', '#e74c3c'), alpha=0.85)
                    ax.bar(x, enh_comm, width, label='增强算法',
                          color=COLORS.get('primary', '#3498db'), alpha=0.85)
                    ax.bar(x + width, mappo_comm, width, label=mode_name,
                          color=COLORS.get('warning', '#f39c12'), alpha=0.85)
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(comm_labels, fontsize=7, rotation=15, ha='right')
                    ax.set_ylabel('指标值')
                    ax.set_title('(e) 通信质量指标对比')
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3, axis='y')
                else:
                    ax.text(0.5, 0.5, '暂无通信指标数据\n(训练完成后Phase2评估生成)',
                           ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
                    ax.set_title('(e) 通信质量指标对比')
        
        # V17: 多场景数据整合到图(f)文字区中（不再单独占一个子图）
        scenario_text_lines = []
        if 'scenarios' in all_results and all_results['scenarios']:
            scenarios_data = all_results['scenarios']
            s_names = list(scenarios_data.keys())
            scenario_text_lines.append('\n【场景泛化】')
            for sn in s_names:
                sd = scenarios_data[sn]
                line = f"  {sd.get('name_cn', sn)}: 传统={sd['traditional'][0]:.4f}, 增强={sd['enhanced'][0]:.4f}"
                if 'mappo' in sd:
                    line += f", MAPPO={sd['mappo'][0]:.4f}"
                scenario_text_lines.append(line)

        # ===== 图6(f): 关键结果文本（增强版：含通信+分业务） =====
        ax = axes[1, 2]
        ax.axis('off')
        text_lines = [f'【{mode_name} 关键发现】', '─' * 28]
        
        if 'evaluation' in all_results:
            for num_uav in num_uav_list:
                if num_uav in all_results['evaluation']:
                    ev = all_results['evaluation'][num_uav]
                    mp = ev['mappo']['avg'][0]
                    en = ev['enhanced']['avg'][0]
                    tr = ev['traditional']['avg'][0]
                    improvement_vs_trad = (mp - tr) / max(tr, 0.001) * 100
                    improvement_vs_enh = (mp - en) / max(en, 0.001) * 100
                    
                    text_lines.append(f'\nUAV={num_uav}:')
                    text_lines.append(f'  满意度: {mode_name}={mp:.4f}')
                    text_lines.append(f'           增强={en:.4f} ({improvement_vs_enh:+.2f}%)')
                    text_lines.append(f'           传统={tr:.4f} ({improvement_vs_trad:+.2f}%)')
                    
                    # 分业务类型满意度
                    if 'per_biz' in ev['mappo']:
                        biz_names = ['控制信令', '视频回传', '环境监测']
                        text_lines.append(f'\n  分业务满意度:')
                        for bt in range(3):
                            mp_b = ev['mappo']['per_biz'][bt][0] if bt in ev['mappo']['per_biz'] else 0
                            en_b = ev['enhanced']['per_biz'][bt][0] if bt in ev['enhanced']['per_biz'] else 0
                            text_lines.append(f'    {biz_names[bt]}: {mp_b:.4f} vs {en_b:.4f}')
                    
                    # 通信指标摘要
                    m_comm = ev.get('mappo', {}).get('communication_metrics', {})
                    e_comm = ev.get('enhanced', {}).get('communication_metrics', {})
                    if m_comm:
                        text_lines.append(f'\n  通信指标:')
                        for cm_key, cm_label in [('handover_latency', '切换延迟'),
                                                   ('ping_jitter', 'Ping抖动'),
                                                   ('packet_loss_rate', '丢包率')]:
                            mv = m_comm.get(cm_key, (0,0))[0]
                            ev_ = e_comm.get(cm_key, (0,0))[0]
                            if mv > 0.001 or ev_ > 0.001:
                                unit = 'ms' if 'rate' not in cm_key else '%'
                                delta = mv - ev_
                                text_lines.append(f'    {cm_label}: {mv:.1f}{unit} ({"↓" if delta < 0 else "↑"}{abs(delta):.1f})')
                    text_lines.append('')
        
        # 整合场景泛化数据
        if 'scenario_text_lines' in dir() and scenario_text_lines:
            text_lines.extend(scenario_text_lines)
        
        # 训练摘要
        if 'training' in all_results:
            for num_uav in num_uav_list:
                if num_uav in all_results['training']:
                    tr = all_results['training'][num_uav]
                    n_ep = len(tr.get('rewards', []))
                    final_sat = np.mean(tr.get('satisfactions', [])[-20:]) if len(tr.get('satisfactions', [])) > 20 else 0
                    early_stopped = tr.get('early_stopped', False)
                    status = f"早停@Ep{n_ep}" if early_stopped else f"完成{n_ep}ep"
                    text_lines.append(f'训练状态: {status}, 最终sat≈{final_sat:.4f}')

        ax.text(0.05, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
                fontsize=7.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.tight_layout()
        save_path = os.path.join(RESULT_DIR, 'mappo_results.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  可视化已保存: {save_path}")
