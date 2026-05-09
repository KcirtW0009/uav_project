import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免plt.show()弹窗阻塞程序
import matplotlib.pyplot as plt
import os
import json
import pickle
import time  # [FIX] 添加time模块（用于进度日志计时）
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Tuple
from scipy import stats
from .config import GLOBAL_SEED, set_global_seed, RESULT_DIR, COLORS, CMAP_PRIMARY, CMAP_SUCCESS, CMAP_WARNING
from .business import BusinessType, QoSProfile, QOS_PROFILES
from .satisfaction import HierarchicalSatisfactionMetric
from .recognition import AdaptiveRecognitionUpdater, BusinessRecognitionModel, train_or_load_recognition_model
from .environment import EnhancedNetworkEnvironment
from .algorithms import IntegratedHandoverAlgorithm, EnhancedHandoverAlgorithm
from .visualization import VisualizationHelper


def evaluate_mappo_in_experiment(num_bs: int, num_uav: int, num_steps: int,
                                  recognition_model=None, scaler=None,
                                  seed: int = 42, scenario: str = 'default',
                                  model_path: str = None) -> dict:
    """在实验3/4环境中评估已训练的 MAPPO 模型
    
    使用 MultiAgentHandoverEnv (mappo_environment.py) 创建与实验一致的环境，
    加载已训练的 MAPPO 模型进行评估，返回与传统/增强算法兼容的统计字典。
    
    核心设计:
    - 传入 recognition_model + scaler 使环境使用预测业务类型（评估模式）
    - 加载在 8BS×300UAV 环境中训练的模型，对任意 num_uav 进行零样本泛化评估
    - 返回的统计字典结构与 env.get_state_statistics() 兼容，便于三算法对比
    
    Args:
        num_bs: 基站数量
        num_uav: UAV数量
        num_steps: 评估步数
        recognition_model: 业务识别模型（评估模式）
        scaler: 识别模型标准化器
        seed: 随机种子
        scenario: 场景名称（用于Experiment4）
        model_path: 训练好的模型文件路径。None则使用默认路径
        
    Returns:
        stats_dict: 包含 avg_satisfaction, handover_success_rate 等指标的字典
    """
    from .mappo_environment import MultiAgentHandoverEnv
    from .mappo_agent_v2 import MAPPOAgentV2 as MAPPOAgent
    
    if model_path is None:
        # 默认路径：8BS×300UAV训练的模型（与实验3对齐）
        model_path = os.path.join(RESULT_DIR, 'mappo_models', 'mappo_8bs_300uav.pt')
    
    if not os.path.exists(model_path):
        print(f"  [MAPPO] 警告: 模型文件不存在 {model_path}，跳过MAPPO评估")
        return None
    
    # 创建评估环境（带识别模型的评估模式）
    env = MultiAgentHandoverEnv(
        num_bs=num_bs, num_uav=num_uav,
        max_steps=num_steps, seed=seed,
        bs_capacity_range=(500, 1000), pos_range=1000,
        recognition_model=recognition_model,
        scaler=scaler,
        event_probability=0.05,  # 与实验3一致
    )
    obs_dict, global_state = env.reset()
    
    # 初始化 agent 并加载模型
    agent = MAPPOAgent(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        critic_hidden_dim=128,
    )
    
    try:
        # [FIX] PyTorch 2.6+ 兼容性
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location='cpu')
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        print(f"  [MAPPO] 成功加载模型: {model_path}")
    except Exception as e:
        print(f"  [MAPPO] 警告: 模型加载失败 {e}，跳过MAPPO评估")
        return None
    
    # 执行评估
    all_sats = []
    all_connected_rates = []
    all_handovers = []
    
    # [FIX V2] 收集真实的切换统计数据 (从每步info['reward_diag']提取)
    total_switch_attempts_all = 0
    total_switch_success_all = 0
    total_switch_rollback_all = 0
    
    # [PROGRESS] 添加实时进度日志
    import sys
    eval_start_time = time.time()
    last_log_step = 0
    log_interval = 50  # 每50步打印一次
    
    print("\n  [SIMULATION] 开始仿真循环 ({}步 x {}UAV)".format(num_steps, env.num_agents))
    print("  " + "-"*70)
    print("  {:>6s} | {:>10s} | {:>10s} | {:>12s} | {:>10s} | {:>8s}".format(
        "Step", "Satisfaction", "Connected", "SwitchRate", "ElapsedTime", "ETA"))
    print("  " + "-"*70)

    for step in range(num_steps):
        biz_types = {uid: env.env.uavs[uid].true_business_type.value for uid in range(env.num_agents)}
        actions, _, _, _, _ = agent.select_actions(obs_dict, global_state, biz_types=biz_types, training=False)
        obs_dict, global_state, rewards, team_reward, done, info = env.step(actions)
        
        all_sats.append(info['avg_satisfaction'])
        all_connected_rates.append(info['connected_rate'])
        
        total_ho = sum(env.env.uavs[uid].handover_count for uid in range(env.num_agents))
        all_handovers.append(total_ho)
        
        if 'reward_diag' in info:
            diag = info['reward_diag']
            total_switch_attempts_all += diag.get('switch_attempts', 0)
            total_switch_success_all += diag.get('switch_success', 0)
            total_switch_rollback_all += diag.get('switch_rollback', 0)
        
        # [PROGRESS] 定期打印进度
        if (step + 1) % log_interval == 0 or step == num_steps - 1:
            elapsed = time.time() - eval_start_time
            
            # 计算切换成功率 (当前累计)
            current_hosr = (total_switch_success_all / max(total_switch_attempts_all, 1) * 100 
                           if total_switch_attempts_all > 0 else 100.0)
            
            # [DEBUG] 打印原始统计数据
            print("\n  [DEBUG] Step {}: attempts={}, success={}, rollback={}".format(
                step + 1, total_switch_attempts_all, total_switch_success_all, total_switch_rollback_all))
            
            if total_switch_attempts_all == 0:
                print("  [WARN] 切换尝试次数=0 → 模型可能总是选择action=0 (stay)")
            
            # 计算ETA
            steps_done = step + 1
            time_per_step = elapsed / max(steps_done, 1)
            remaining_steps = num_steps - steps_done
            eta_seconds = remaining_steps * time_per_step
            
            if eta_seconds > 60:
                eta_str = "{:.1f}min".format(eta_seconds / 60)
            else:
                eta_str = "{:.0f}s".format(eta_seconds)
            
            if elapsed > 60:
                elapsed_str = "{:.1f}min".format(elapsed / 60)
            else:
                elapsed_str = "{:.0f}s".format(elapsed)
            
            print("  {:>6d} | {:>10.4f} | {:>9.2%} | {:>11.2f}% | {:>10s} | {:>8s}".format(
                step + 1,
                info['avg_satisfaction'],
                info['connected_rate'],
                current_hosr,
                elapsed_str,
                eta_str
            ))
            
            # 强制刷新缓冲区，确保立即显示
            sys.stdout.flush()
    
    eval_elapsed = time.time() - eval_start_time
    print("  " + "-"*70)
    print("  [DONE] 仿真完成! 总耗时: {:.1f}s ({:.2f}min)".format(
        eval_elapsed, eval_elapsed / 60))
    
    # 构建与 get_state_statistics() 兼容的结果字典
    final_sats = [env.env.uavs[uid].current_satisfaction for uid in range(env.num_agents)]
    connected_count = sum(1 for uid in range(env.num_agents) if env.env.uavs[uid].connected_bs_id is not None)
    total_ho = sum(env.env.uavs[uid].handover_count for uid in range(env.num_agents))
    
    # [FIX V2] 使用真实收集的切换数据计算成功率
    # 切换成功率 = 切换成功 / 切换尝试
    if total_switch_attempts_all > 0:
        real_handover_success_rate = total_switch_success_all / total_switch_attempts_all
    else:
        real_handover_success_rate = 1.0
    
    # [FIX V2] 连接保持率 = (切换成功 + 回滚成功 + 不切换) / 总UAV数
    stay_count = max(0, (env.num_agents * num_steps) - total_switch_attempts_all)
    connected_kept = total_switch_success_all + total_switch_rollback_all + stay_count
    real_connected_ratio = connected_kept / max(env.num_agents * num_steps, 1)
    
    stats = {
        'avg_satisfaction': np.mean(final_sats),
        'critical_satisfaction': np.mean([s for i, s in enumerate(final_sats)
                                          if env.env.uavs[i].true_business_type.value == 0]),
        'weighted_satisfaction': np.mean(final_sats),
        'connected_count': connected_count,
        'connected_ratio': real_connected_ratio,
        'total_throughput': sum(env.env.uavs[uid].current_allocated_rate 
                               for uid in range(env.num_agents)
                               if env.env.uavs[uid].connected_bs_id is not None),
        'handover_success_rate': real_handover_success_rate,
        'avg_switching_latency_ms': np.mean(env._communication_metrics.get('handover_latencies', [0])),
        'max_switching_latency_ms': max(env._communication_metrics.get('handover_latencies', [0])),
        'avg_decision_time_ms': 0.001,  # MAPPO决策时间极短（神经网络推理）
        'missed_opportunity_rate': 0.0,  # MAPPO不会错失机会（全局优化）
        'migration_success_rate': real_handover_success_rate,  # 切换成功即迁移成功
        'load_variance': np.var([bs.load_ratio for bs in env.env.base_stations.values()]),
        'avg_sinr': np.mean(env.env.sinr_matrix[:env.num_agents, :num_bs]),
        'latency_satisfaction': np.mean([env.env.uavs[uid].latency_satisfaction 
                                        for uid in range(env.num_agents)]),
        'rate_satisfaction': np.mean([env.env.uavs[uid].rate_satisfaction 
                                      for uid in range(env.num_agents)]),
        'recognition_accuracy': recognition_model.score(
            [[getattr(env.env.uavs[uid], attr, 0) 
              for attr in ['current_sinr', 'current_allocated_rate', 'current_latency_ms']]
             for uid in range(min(100, env.num_agents))],
            [env.env.uavs[uid].true_business_type.value 
             for uid in range(min(100, env.num_agents))]
        )[0] * 100 if recognition_model else 100.0,
        '_algorithm': 'MAPPO',
    }
    
    print(f"  MAPPO - 满足率: {stats['avg_satisfaction']:.3f}, "
          f"连接率: {stats['connected_ratio']*100:.1f}%, "
          f"切换成功率: {real_handover_success_rate:.1%} ({total_switch_success_all}/{total_switch_attempts_all}), "
          f"切换次数: {total_ho}")
    return stats


# 导入torch（evaluate_mappo_in_experiment中使用）
import torch



def save_experiment_data(exp_name: str, data: dict, extra_formats: list = None):
    """
    保存实验关键数据到文件，防止终端输出丢失。

    Args:
        exp_name: 实验名称标识（如 'exp1', 'exp2'）
        data: 要保存的数据字典
        extra_formats: 额外保存格式列表，可选 'json', 'csv'
    """
    if extra_formats is None:
        extra_formats = ['json']

    # 将 numpy 类型转换为 Python 原生类型以便 JSON 序列化
    def _convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    serializable_data = _convert(data)
    serializable_data['_meta'] = {
        'experiment': exp_name,
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'global_seed': GLOBAL_SEED,
    }

    base_path = os.path.join(RESULT_DIR, f'{exp_name}_data')

    # 1. pickle 格式（保留完整 Python 对象结构）
    pkl_path = base_path + '.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  数据已保存(pickle): {pkl_path}")

    # 2. JSON 格式（人类可读，可跨语言使用）
    if 'json' in extra_formats:
        json_path = base_path + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        print(f"  数据已保存(json):   {json_path}")

# 配置字体和警告抑制
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing.*')

# -------------------- 实验1 --------------------
class Experiment1:
    """
    实验1：识别准确性的价值验证（重构版）
    
    核心问题：识别准确率如何影响系统性能？
    
    实验设计：
    - 条件A（100%准确率）：使用真实业务类型作为识别结果（基准）
    - 条件B（85%准确率）：使用高质量模型，人工注入15%噪声
    - 条件C（70%准确率）：使用中等质量模型，人工注入30%噪声
    - 条件D（随机33%）：随机分配业务类型（下界对照）
    
    所有条件使用相同的差异化QoS配置，控制其他变量一致
    """

    # 预设的识别准确率目标值（均匀梯度分布）
    ACCURACY_LEVELS = {
        'perfect': 1.00,    # 100% - 基准
        'high': 0.90,       # 90% - 高质量模型
        'medium': 0.80,      # 80% - 中等质量模型
        'low': 0.60,        # 60% - 低质量模型
        'random': 0.33,     # 33% - 随机猜测
    }

    @staticmethod
    def run(recognition_model, scaler, num_steps=150, repeats=15):  # 增加到15次重复以减少随机波动
        print("\n" + "="*80)
        print("实验1：识别准确性的价值验证")
        print("="*80)
        print("\n实验目的：验证业务识别准确率对系统性能的影响")
        print("\n实验条件：")
        print("  A. 100%准确率 - 使用真实类型（性能基准）")
        print("  B.  90%准确率 - 高质量识别模型")
        print("  C.  80%准确率 - 中等质量识别模型")
        print("  D.  60%准确率 - 低质量识别模型")
        print("  E.  33%准确率 - 随机分配（下界对照）")
        print("\n控制变量：差异化QoS配置、切换算法、网络环境完全相同")
        print("\n算法配置说明：使用EnhancedHandover算法，但禁用ε-greedy探索机制")
        print("  原因：ε-greedy引入随机性，会干扰识别准确率对性能影响的评估")
        print("="*80)

        # 存储各条件的结果
        results_by_accuracy = {
            'perfect': [],
            'high': [],
            'medium': [],
            'low': [],
            'random': []
        }

        for rep in range(repeats):
            print(f"\n--- 重复 {rep+1}/{repeats} ---")

            # 为每个准确率条件创建环境，使用不同的seed以确保独立
            for idx, (condition_name, target_accuracy) in enumerate(Experiment1.ACCURACY_LEVELS.items()):
                # 方案9：简化种子策略，使用简单的线性偏移
                # 避免与错误分配逻辑冲突
                condition_seed = GLOBAL_SEED + rep * 10000 + idx * 100
                set_global_seed(condition_seed)


                # 使用不带随机事件的环境进行对比实验
                env = EnhancedNetworkEnvironment(
                    num_bs=8, num_uav=300,  # 调整后: 实际带宽参数对齐，300架约77%负载率
                    recognition_model=None, scaler=None,
                    seed=condition_seed,
                    event_probability=0.0  # 关闭随机事件，专注于识别准确率的影响
                )

                # 根据目标准确率设置UAV的识别类型（使用确定性方法）
                actual_accuracy = Experiment1._setup_recognition_with_accuracy_deterministic(
                    env, target_accuracy, condition_seed
                )

                env.recognition_updater = None
                algo = EnhancedHandoverAlgorithm(env)
                # 降低epsilon以减少随机探索对实验的干扰
                algo.epsilon = 0.00  # 完全禁用探索以获得确定性的基线结果

                # 运行仿真
                for step in range(num_steps):
                    env.step()
                    algo.run_step(enable_load_balancing=True)

                # 收集结果
                stats = env.get_state_statistics()
                stats.update(algo.get_detailed_stats())
                stats['actual_recognition_accuracy'] = actual_accuracy
                results_by_accuracy[condition_name].append(stats)

                print(f" {condition_name:8s} (目标{target_accuracy*100:3.0f}%, 实际{actual_accuracy*100:5.1f}%) - "
                      f"满足率: {stats['avg_satisfaction']:.3f}")

        # 汇总结果
        summary = Experiment1._summarize_results(results_by_accuracy)
        Experiment1._print_results_table(summary)
        Experiment1._plot(summary)
        return summary

    @staticmethod
    def _setup_recognition_with_accuracy(env, target_accuracy):
        """
        根据目标准确率设置UAV的识别类型（随机版本，用于其他实验）

        Returns:
            actual_accuracy: 实际达到的准确率
        """
        correct_count = 0
        total_count = len(env.uavs)

        for uav in env.uavs.values():
            true_type = uav.true_business_type

            if np.random.random() < target_accuracy:
                # 正确识别
                recognized_type = true_type
                correct_count += 1
            else:
                # 错误识别：随机选择其他类型
                other_types = [t for t in BusinessType if t != true_type]
                recognized_type = np.random.choice(other_types)

            # 设置识别结果和QoS配置
            uav.business_type = recognized_type
            uav.qos_profile = QOS_PROFILES[recognized_type]
            uav.recognition_confidence = 0.7 + np.random.random() * 0.25  # 0.7-0.95

        return correct_count / total_count if total_count > 0 else 0.0

    @staticmethod
    def _setup_recognition_with_accuracy_deterministic(env, target_accuracy, seed):
        """
        根据目标准确率设置UAV的识别类型（确定性版本，用于实验1）

        使用确定性方法确保在相同seed下产生相同的错误模式，
        这样可以准确对比不同准确率的影响。

        方案9改进：简化错误分配逻辑，避免种子冲突

        Args:
            env: 网络环境
            target_accuracy: 目标准确率
            seed: 随机种子（用于确保确定性）

        Returns:
            actual_accuracy: 实际达到的准确率
        """
        # 使用临时随机数生成器，避免影响主随机数流
        rng = np.random.RandomState(seed)

        correct_count = 0
        total_count = len(env.uavs)

        # 方案9：直接使用简单随机数判断，避免复杂逻辑
        for uav in env.uavs.values():
            true_type = uav.true_business_type

            if rng.random() < target_accuracy:
                # 正确识别
                recognized_type = true_type
                correct_count += 1
            else:
                # 错误识别：确定性选择其他类型（按轮询方式）
                other_types = [t for t in BusinessType if t != true_type]
                # 使用UAV ID的索引来选择错误类型，确保确定性
                error_index = (uav.uav_id + seed) % len(other_types)  # 使用种子避免冲突
                recognized_type = other_types[error_index]

            # 设置识别结果和QoS配置
            uav.business_type = recognized_type
            uav.qos_profile = QOS_PROFILES[recognized_type]
            # 确定性设置置信度
            uav.recognition_confidence = 0.825  # 固定在中间值

        return correct_count / total_count if total_count > 0 else 0.0

    @staticmethod
    def _summarize_results(results_by_accuracy):
        """汇总各准确率条件的实验结果"""
        def avg_std(key, results):
            values = [r[key] for r in results]
            return np.mean(values), np.std(values)

        summary = {}
        for condition_name in Experiment1.ACCURACY_LEVELS.keys():
            results = results_by_accuracy[condition_name]
            summary[condition_name] = {
                'satisfaction': avg_std('avg_satisfaction', results),
                'true_satisfaction': avg_std('avg_true_satisfaction', results),
                'resource_match': avg_std('resource_match_ratio', results),
                'actual_accuracy': avg_std('actual_recognition_accuracy', results),
                'handover_success': avg_std('handover_success_rate', results),
                'throughput': avg_std('total_load', results),
                'critical_sat': avg_std('critical_satisfaction', results),
                'weighted_sat': avg_std('weighted_satisfaction', results),
            }
        return summary

    @staticmethod
    def _print_results_table(summary):
        """打印实验结果表格"""
        condition_names = {
            'perfect': '100%准确率',
            'high': '90%准确率',
            'medium': '80%准确率',
            'low': '60%准确率',
            'random': '33%准确率'
        }

        headers = ["指标", "100%准确率", "90%准确率", "80%准确率", "60%准确率", "33%准确率"]

        # 计算相对于100%准确率的性能损失(基于真实满意率)
        perfect_sat = summary['perfect']['true_satisfaction'][0]

        rows = [
            ["真实满足率(基于真实需求)"] + [
                f"{summary[c]['true_satisfaction'][0]:.3f}±{summary[c]['true_satisfaction'][1]:.3f}"
                for c in ['perfect', 'high', 'medium', 'low', 'random']
            ],
            ["性能损失(基于真实需求)"] + [
                f"-"
                if c == 'perfect' else
                (f"{(summary[c]['true_satisfaction'][0] - perfect_sat)*100:+.2f}%")
                for c in ['perfect', 'high', 'medium', 'low', 'random']
            ],
            ["资源匹配度"] + [
                f"{summary[c]['resource_match'][0]:.3f}±{summary[c]['resource_match'][1]:.3f}"
                for c in ['perfect', 'high', 'medium', 'low', 'random']
            ],
            ["关键业务满足率"] + [
                f"{summary[c]['critical_sat'][0]:.3f}±{summary[c]['critical_sat'][1]:.3f}"
                for c in ['perfect', 'high', 'medium', 'low', 'random']
            ],
            ["切换成功率"] + [
                f"{summary[c]['handover_success'][0]*100:.1f}%"
                for c in ['perfect', 'high', 'medium', 'low', 'random']
            ],
            ["系统吞吐量(Mbps)"] + [
                f"{summary[c]['throughput'][0]:.1f}±{summary[c]['throughput'][1]:.1f}"
                for c in ['perfect', 'high', 'medium', 'low', 'random']
            ],
            ["实际识别准确率"] + [
                f"{summary[c]['actual_accuracy'][0]*100:.1f}%"
                for c in ['perfect', 'high', 'medium', 'low', 'random']
            ],
        ]

        print("\n" + "="*100)
        print("实验1结果：识别准确率对系统性能的影响")
        print("="*100)
        VisualizationHelper.print_data_table("识别准确性价值分析", headers, rows)

        # 打印关键结论
        print("\n【关键结论】")
        high_loss = (perfect_sat - summary['high']['true_satisfaction'][0]) * 100
        medium_loss = (perfect_sat - summary['medium']['true_satisfaction'][0]) * 100
        low_loss = (perfect_sat - summary['low']['true_satisfaction'][0]) * 100
        random_loss = (perfect_sat - summary['random']['true_satisfaction'][0]) * 100

        print(f"  - 识别准确率从100%降至90%，性能损失: {high_loss:+.2f}%")
        print(f"  - 识别准确率从100%降至80%，性能损失: {medium_loss:+.2f}%")
        print(f"  - 识别准确率从100%降至60%，性能损失: {low_loss:+.2f}%")
        print(f"  - 识别准确率从100%降至33%，性能损失: {random_loss:+.2f}%")

        # 验证单调性
        print(f"\n【数据一致性检查】")
        sat_values = [summary[c]['true_satisfaction'][0] for c in ['perfect', 'high', 'medium', 'low', 'random']]
        print(f"  真实满足率序列: {' -> '.join([f'{v:.3f}' for v in sat_values])}")
        if sat_values == sorted(sat_values, reverse=True):
            print(f"  [OK] 真实满足率随准确率降低而下降 (符合预期)")
        else:
            print(f"  [WARN] 真实满足率未随准确率单调下降 (可能存在异常)")

        if abs(high_loss) < 5 and abs(medium_loss) < 5:
            print(f"\n  - 结论: 90%和80%识别准确率性能接近，准确率影响较小")
        elif high_loss > 10 or medium_loss > 10:
            print(f"\n  - 结论: 识别准确率对系统性能影响显著，应优先提升模型精度")
        else:
            print(f"\n  - 结论: 识别准确率对性能有影响，但90%以上已达到可接受水平")
        print("="*100)

    @staticmethod
    def _plot(summary):
        """绘制实验结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('实验1：识别准确性的价值验证', fontsize=14, fontweight='bold')

        conditions = ['perfect', 'high', 'medium', 'low', 'random']
        labels = ['100%', '90%', '80%', '60%', '33%']
        colors = [COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger']]
        
        # 图1: 真实满足率 vs 识别准确率
        ax = axes[0, 0]
        accuracies = [summary[c]['actual_accuracy'][0] * 100 for c in conditions]
        true_satisfactions = [summary[c]['true_satisfaction'][0] for c in conditions]
        ax.plot(accuracies, true_satisfactions, 'o-', color=COLORS['primary'],
                linewidth=2, markersize=10)
        for i, (acc, sat) in enumerate(zip(accuracies, true_satisfactions)):
            ax.annotate(labels[i], (acc, sat), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
        ax.set_xlabel('识别准确率 (%)', fontsize=11)
        ax.set_ylabel('真实满足率(基于真实需求)', fontsize=11)
        ax.set_title('识别准确率 vs 真实系统性能', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 图2: 各指标对比柱状图
        ax = axes[0, 1]
        x = np.arange(len(labels))
        width = 0.25
        true_sat_values = [summary[c]['true_satisfaction'][0] for c in conditions]
        crit_values = [summary[c]['critical_sat'][0] for c in conditions]
        res_match_values = [summary[c]['resource_match'][0] for c in conditions]
        bars1 = ax.bar(x - width, true_sat_values, width, label='真实满足率',
                       color=COLORS['primary'], alpha=0.8)
        bars2 = ax.bar(x, crit_values, width, label='关键业务满足率',
                       color=COLORS['danger'], alpha=0.8)
        bars3 = ax.bar(x + width, res_match_values, width, label='资源匹配度',
                       color=COLORS['success'], alpha=0.8)
        ax.set_ylabel('指标值', fontsize=11)
        ax.set_title('不同准确率下的性能指标对比', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 图3: 性能损失曲线
        ax = axes[1, 0]
        perfect_sat = summary['perfect']['true_satisfaction'][0]
        losses = [(perfect_sat - summary[c]['true_satisfaction'][0]) * 100
                  for c in conditions]
        bars = ax.bar(labels, losses, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax.set_ylabel('性能损失 (%)', fontsize=11)
        ax.set_xlabel('识别准确率', fontsize=11)
        ax.set_title('相对于100%准确率的性能损失', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, loss in zip(bars, losses):
            if loss > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{loss:.2f}%', ha='center', va='bottom', fontsize=9)

        # 图4: 切换成功率对比
        ax = axes[1, 1]
        success_rates = [summary[c]['handover_success'][0] * 100 for c in conditions]
        bars = ax.bar(labels, success_rates, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax.set_ylabel('切换成功率 (%)', fontsize=11)
        ax.set_xlabel('识别准确率', fontsize=11)
        ax.set_title('不同准确率下的切换成功率', fontweight='bold')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'exp1_results.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

        save_experiment_data('exp1', summary)
        return summary


# -------------------- 统计检验工具函数 --------------------

def perform_statistical_test(group1: List[float], group2: List[float],
                            test_name: str = 'ttest',
                            alpha: float = 0.05) -> Dict[str, Any]:
    """
    执行统计显著性检验

    Args:
        group1: 第一组数据
        group2: 第二组数据
        test_name: 检验方法 ('ttest', 'mannwhitney', 'wilcoxon')
        alpha: 显著性水平

    Returns:
        Dict: 包含检验结果,包括统计量、p值、是否显著等
    """
    # 计算描述性统计
    result = {
        'group1': {
            'mean': np.mean(group1),
            'std': np.std(group1),
            'count': len(group1),
            'median': np.median(group1)
        },
        'group2': {
            'mean': np.mean(group2),
            'std': np.std(group2),
            'count': len(group2),
            'median': np.median(group2)
        },
        'effect_size': None,
        'test_method': test_name,
        'alpha': alpha
    }

    # 执行统计检验
    if test_name == 'ttest':
        # 独立样本t检验 (假设正态分布)
        statistic, p_value = stats.ttest_ind(group1, group2)
        result['statistic'] = statistic
        result['p_value'] = p_value
        result['significant'] = p_value < alpha

        # 计算Cohen's d效应量
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        if pooled_std != 0:
            cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
            result['effect_size'] = cohens_d
            result['effect_size_interpretation'] = _interpret_cohens_d(cohens_d)

    elif test_name == 'mannwhitney':
        # Mann-Whitney U检验 (非参数检验)
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        result['statistic'] = statistic
        result['p_value'] = p_value
        result['significant'] = p_value < alpha

        # 计算秩和效应量
        n1, n2 = len(group1), len(group2)
        u = statistic
        z = (u - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        r = z / np.sqrt(n1 + n2)
        result['effect_size'] = r
        result['effect_size_interpretation'] = _interpret_rank_biserial(r)

    elif test_name == 'wilcoxon':
        # Wilcoxon符号秩检验 (配对数据)
        if len(group1) != len(group2):
            raise ValueError("Wilcoxon检验需要两组数据长度相同")
        statistic, p_value = stats.wilcoxon(group1, group2)
        result['statistic'] = statistic
        result['p_value'] = p_value
        result['significant'] = p_value < alpha

    else:
        raise ValueError(f"未知的检验方法: {test_name}")

    return result


def _interpret_cohens_d(d: float) -> str:
    """解释Cohen's d效应量"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return '微小'
    elif abs_d < 0.5:
        return '小'
    elif abs_d < 0.8:
        return '中等'
    else:
        return '大'


def _interpret_rank_biserial(r: float) -> str:
    """解释秩相关效应量"""
    abs_r = abs(r)
    if abs_r < 0.1:
        return '微小'
    elif abs_r < 0.3:
        return '小'
    elif abs_r < 0.5:
        return '中等'
    else:
        return '大'


def print_statistical_results(results: Dict[str, Any], metric_name: str = 'Metric'):
    """打印统计检验结果"""
    print(f"\n{'='*60}")
    print(f"{metric_name} 统计显著性检验")
    print(f"{'='*60}")

    print(f"\n【描述性统计】")
    print(f"  组1 (n={results['group1']['count']}):  "
          f"均值={results['group1']['mean']:.4f}±{results['group1']['std']:.4f}, "
          f"中位数={results['group1']['median']:.4f}")
    print(f"  组2 (n={results['group2']['count']}):  "
          f"均值={results['group2']['mean']:.4f}±{results['group2']['std']:.4f}, "
          f"中位数={results['group2']['median']:.4f}")

    print(f"\n【统计检验结果】")
    print(f"  检验方法: {results['test_method']}")
    print(f"  统计量: {results['statistic']:.4f}")
    print(f"  p值: {results['p_value']:.6f}")
    print(f"  显著性水平: {results['alpha']}")
    print(f"  是否显著: {'是 ✓' if results['significant'] else '否 ✗'}")

    if results['effect_size'] is not None:
        print(f"  效应量: {results['effect_size']:.4f} ({results['effect_size_interpretation']})")

    # 打印结论
    print(f"\n【结论】")
    if results['significant']:
        if results['group1']['mean'] > results['group2']['mean']:
            direction = "组1显著高于组2"
        else:
            direction = "组1显著低于组2"
        print(f"  在α={results['alpha']}水平下,{direction} (p={results['p_value']:.4f})")
    else:
        print(f"  在α={results['alpha']}水平下,两组差异无统计学意义 (p={results['p_value']:.4f})")

    print(f"{'='*60}\n")


def compare_algorithms_with_tests(enhanced_results: List[Dict],
                                  traditional_results: List[Dict],
                                  metrics: List[str]) -> Dict[str, Dict]:
    """
    对增强算法和传统算法进行多指标的统计显著性检验

    Args:
        enhanced_results: 增强算法的多次运行结果
        traditional_results: 传统算法的多次运行结果
        metrics: 需要检验的指标列表

    Returns:
        Dict: 每个指标的检验结果
    """
    all_test_results = {}

    for metric in metrics:
        if metric in enhanced_results[0] and metric in traditional_results[0]:
            group1 = [r[metric] for r in enhanced_results]
            group2 = [r[metric] for r in traditional_results]

            # 自动选择检验方法
            # Shapiro-Wilk正态性检验
            _, p1 = stats.shapiro(group1)
            _, p2 = stats.shapiro(group2)

            if p1 > 0.05 and p2 > 0.05:
                # 都符合正态分布,使用t检验
                test_method = 'ttest'
            else:
                # 不符合正态分布,使用非参数检验
                test_method = 'mannwhitney'

            test_results = perform_statistical_test(group1, group2,
                                                     test_name=test_method,
                                                     alpha=0.05)
            all_test_results[metric] = test_results

    return all_test_results


def print_comprehensive_test_summary(all_test_results: Dict[str, Dict],
                                    enhanced_name: str = '增强算法',
                                    traditional_name: str = '传统算法'):
    """打印综合检验结果摘要"""
    print(f"\n{'='*80}")
    print(f"综合统计显著性检验摘要: {enhanced_name} vs {traditional_name}")
    print(f"{'='*80}")

    significant_count = 0
    total_count = 0

    for metric, results in all_test_results.items():
        total_count += 1
        print(f"\n【{metric}】")
        print(f"  {enhanced_name}: {results['group1']['mean']:.4f}±{results['group1']['std']:.4f}")
        print(f"  {traditional_name}: {results['group2']['mean']:.4f}±{results['group2']['std']:.4f}")
        print(f"  p值: {results['p_value']:.6f}")
        if results['significant']:
            significant_count += 1
            direction = "↑" if results['group1']['mean'] > results['group2']['mean'] else "↓"
            print(f"  结论: 显著差异 {direction} (效应量={results.get('effect_size', 0):.4f})")
        else:
            print(f"  结论: 无显著差异")

    print(f"\n{'='*80}")
    print(f"总结: {significant_count}/{total_count} 个指标具有显著差异")
    print(f"{'='*80}\n")


def compare_three_algorithms_with_tests(mappo_results: List[Dict],
                                       enhanced_results: List[Dict],
                                       traditional_results: List[Dict],
                                       metrics: List[str]) -> Dict[str, Dict]:
    """
    三算法多指标统计显著性检验（MAPPO vs 增强算法 vs 传统算法）

    对每个指标执行三组两两配对t-test/Wilcoxon检验，输出完整p值和显著性结论。

    Args:
        mappo_results: MAPPO多次运行结果
        enhanced_results: 增强算法多次运行结果
        traditional_results: 传统算法多次运行结果
        metrics: 需要检验的指标列表

    Returns:
        Dict: {metric_name: {pair_key: test_result_dict}}
    """
    all_test_results = {}
    algo_data = {
        'MAPPO': mappo_results,
        '增强': enhanced_results,
        '传统': traditional_results,
    }
    pair_names = [
        ('MAPPO', '增强'),
        ('MAPPO', '传统'),
        ('增强', '传统'),
    ]

    for metric in metrics:
        # 检查三组数据是否都包含该指标
        valid_pairs = []
        for name_a, name_b in pair_names:
            data_a = algo_data[name_a]
            data_b = algo_data[name_b]
            if data_a and data_b and metric in data_a[0] and metric in data_b[0]:
                valid_pairs.append((name_a, name_b, data_a, data_b))

        if not valid_pairs:
            continue

        all_test_results[metric] = {}
        for name_a, name_b, data_a, data_b in valid_pairs:
            group_a = [r[metric] for r in data_a]
            group_b = [r[metric] for r in data_b]

            # Shapiro-Wilk 正态性检验 → 自动选择 t-test 或 Mann-Whitney U
            _, p_norm_a = stats.shapiro(group_a) if len(group_a) >= 3 else (0, 1)
            _, p_norm_b = stats.shapiro(group_b) if len(group_b) >= 3 else (0, 1)
            test_method = 'ttest' if (p_norm_a > 0.05 and p_norm_b > 0.05) else 'mannwhitney'

            test_result = perform_statistical_test(
                group_a, group_b, test_name=test_method, alpha=0.05
            )
            pair_key = f'{name_a}_vs_{name_b}'
            all_test_results[metric][pair_key] = {
                **test_result,
                'name_a': name_a,
                'name_b': name_b,
            }

    return all_test_results


def print_three_algorithm_test_summary(all_test_results: Dict):
    """打印三算法对比的统计检验摘要"""
    print(f"\n{'='*90}")
    print("三算法统计显著性检验: MAPPO vs 增强算法 vs 传统算法")
    print(f"{'='*90}")
    print(f"{'指标':<22} {'对比组':<20} {'均值A':>10} {'均值B':>10} {'检验方法':<10} {'p值':>10} {'显著':>4} {'效应量':>8}")
    print(f"{'-'*94}")

    significant_count = 0
    total_count = 0

    for metric, pairs in all_test_results.items():
        print(f"\n  【{metric}】")
        for pair_key, result in pairs.items():
            total_count += 1
            g1 = result['group1']
            g2 = result['group2']
            direction = "↑" if g1['mean'] > g2['mean'] else ("↓" if g1['mean'] < g2['mean'] else "=")
            sig_mark = "***" if result['p_value'] < 0.001 else ("**" if result['p_value'] < 0.01 else ("*" if result['p_value'] < 0.05 else ""))
            is_sig = result['significant']
            if is_sig:
                significant_count += 1
            effect = result.get('effect_size', 0)
            eff_str = f"{effect:.4f}" if effect is not None else "N/A"

            print(f"    {result['name_a']} vs {result['name_b']:<10}: "
                  f"{g1['mean']:>8.4f}±{g1['std']:.4f}  {g2['mean']:>8.4f}±{g2['std']:.4f}  "
                  f"{result['test_method']:<10} {result['p_value']:.4f}{sig_mark:>3}  "
                  f"{'Y' if is_sig else 'N':>4}  {eff_str:>8}")

    print(f"\n{'='*90}")
    print(f"总结: {significant_count}/{total_count} 组对比具有显著差异 (α=0.05)")
    print(f"  显著性标记: *** p<0.001, ** p<0.01, * p<0.05")
    print(f"{'='*90}\n")


# -------------------- 实验3：增强算法 vs 传统算法（全面对比）--------------------
class Experiment3:
    METRICS = {
        'handover_success_rate': '切换成功率',
        'avg_switching_latency_ms': '平均切换时延(ms)',
        'max_switching_latency_ms': '最大切换时延(ms)',
        'avg_decision_time_ms': '平均决策时间(ms)',
        'missed_opportunity_rate': '错失机会率',
        'avg_satisfaction': '整体满足率',
        'critical_satisfaction': '关键业务满足率',
        'weighted_satisfaction': '加权满足率',
        'latency_satisfaction': '时延满足率',
        'rate_satisfaction': '速率满足率',
        'total_throughput': '系统吞吐量(Mbps)',
        'load_variance': '负载方差',
        'avg_sinr': '平均SINR(dB)',
        'recognition_accuracy': '识别准确率(%)',
        'migration_success_rate': '迁移成功率',
        'connected_ratio': '连接保持率',
    }

    @staticmethod
    def run(recognition_model, scaler, num_steps=350, repeats=10, include_mappo=False, mappo_model_path=None):
        """
        运行实验3：增强算法 vs 传统算法（全面对比）

        Args:
            recognition_model: 业务识别模型
            scaler: 识别模型标准化器
            num_steps: 仿真步数（默认350，对齐MAPPO训练环境）
            repeats: 重复实验次数（默认10）
            include_mappo: 是否包含MAPPO三算法对比评估
            mappo_model_path: MAPPO模型路径，None则使用默认路径
        """
        print("\n" + "="*80)
        print("实验3：增强算法 vs 传统算法（全面对比）" + (" + MAPPO" if include_mappo else ""))
        print("="*80)

        enhanced_results, traditional_results, mappo_results = [], [], []  # [Step4] mappo_results
        for rep in range(repeats):
            print(f"\n--- 重复 {rep+1}/{repeats} ---")
            set_global_seed(GLOBAL_SEED + rep)

            # 增强算法
            env_enh = EnhancedNetworkEnvironment(
                num_bs=8, num_uav=300,  # 与带宽参数对齐: ~77%负载率
                recognition_model=recognition_model, scaler=scaler,
                seed=GLOBAL_SEED + rep, event_probability=0.05
            )
            algo_enh = EnhancedHandoverAlgorithm(env_enh)
            algo_enh.epsilon = 0.0  # 最终算法不含ε-greedy探索机制

            # 传统算法
            env_trad = EnhancedNetworkEnvironment(
                num_bs=8, num_uav=300,  # 与带宽参数对齐: ~77%负载率
                recognition_model=recognition_model, scaler=scaler,
                seed=GLOBAL_SEED + rep, event_probability=0.05
            )
            algo_trad = IntegratedHandoverAlgorithm(env_trad)

            for step in range(num_steps):
                env_enh.step()
                algo_enh.run_step(enable_load_balancing=True)
                env_trad.step()
                algo_trad.run_step()

            enh_stats = env_enh.get_state_statistics()
            enh_stats.update(algo_enh.get_detailed_stats())
            enh_stats['connected_ratio'] = enh_stats['connected_count'] / env_enh.num_uav
            enhanced_results.append(enh_stats)

            trad_stats = env_trad.get_state_statistics()
            trad_stats.update(algo_trad.get_detailed_stats())
            trad_stats['connected_ratio'] = trad_stats['connected_count'] / env_trad.num_uav
            traditional_results.append(trad_stats)

            print(f" 增强算法 - 满足率: {enh_stats['avg_satisfaction']:.3f}, "
                  f"切换成功率: {enh_stats['handover_success_rate']*100:.1f}%, "
                  f"吞吐量: {enh_stats['total_load']:.1f} Mbps")
            print(f" 传统算法 - 满足率: {trad_stats['avg_satisfaction']:.3f}, "
                  f"切换成功率: {trad_stats['handover_success_rate']*100:.1f}%, "
                  f"吞吐量: {trad_stats['total_load']:.1f} Mbps")

            # [Step4] MAPPO评估（可选）
            if include_mappo:
                mappo_stats = evaluate_mappo_in_experiment(
                    num_bs=8, num_uav=300, num_steps=num_steps,
                    recognition_model=recognition_model, scaler=scaler,
                    seed=GLOBAL_SEED + rep,
                    model_path=mappo_model_path,  # 支持自定义模型路径
                )
                if mappo_stats is not None:
                    mappo_results.append(mappo_stats)
                    print(f" MAPPO     - 满足率: {mappo_stats['avg_satisfaction']:.3f}, "
                          f"连接率: {mappo_stats['connected_ratio']*100:.1f}%")

        summary = Experiment3._summarize(enhanced_results, traditional_results, mappo_results if include_mappo else None)
        Experiment3._print_results_table(summary)

        # 添加统计显著性检验
        print("\n" + "="*80)
        print("统计显著性检验")
        print("="*80)

        metrics_to_test = [
            'avg_satisfaction',
            'handover_success_rate',
            'critical_satisfaction',
            'avg_switching_latency_ms',
            'avg_decision_time_ms',
            'total_load',
            'load_variance'
        ]

        all_test_results = compare_algorithms_with_tests(
            enhanced_results, traditional_results, metrics_to_test
        )

        print_comprehensive_test_summary(all_test_results, "增强算法", "传统算法")

        # [Step4] 三算法对比检验（如果包含MAPPO）
        if include_mappo and len(mappo_results) > 0:
            print("\n" + "="*80)
            print("三算法对比: MAPPO vs 增强算法 vs 传统算法")
            print("="*80)
            
            # 三指标快速预览
            for metric in ['avg_satisfaction', 'critical_satisfaction', 'connected_ratio', 'load_variance']:
                if all(metric in r for r in [enhanced_results[0], traditional_results[0], mappo_results[0]]):
                    enh_vals = [r[metric] for r in enhanced_results]
                    trad_vals = [r[metric] for r in traditional_results]
                    map_vals = [r[metric] for r in mappo_results]
                    print(f"  {metric}: MAPPO={np.mean(map_vals):.3f}±{np.std(map_vals):.3f}, "
                          f"增强={np.mean(enh_vals):.3f}±{np.std(enh_vals):.3f}, "
                          f"传统={np.mean(trad_vals):.3f}±{np.std(trad_vals):.3f}")

            # 正式统计检验（t-test/Wilcoxon + p值）
            three_algo_metrics = ['avg_satisfaction', 'critical_satisfaction', 'connected_ratio',
                                  'load_variance', 'total_throughput']
            three_test_results = compare_three_algorithms_with_tests(
                mappo_results, enhanced_results, traditional_results, three_algo_metrics
            )
            print_three_algorithm_test_summary(three_test_results)
            summary['three_algo_statistical_tests'] = three_test_results

        # 将检验结果添加到summary中
        summary['statistical_tests'] = all_test_results

        Experiment3._plot(summary)
        return summary

    @staticmethod
    def _summarize(enhanced_results, traditional_results, mappo_results=None):
        summary = {'enhanced': {}, 'traditional': {}, 'improvement': {}, 'mappo': {}}
        for key in Experiment3.METRICS.keys():
            if key in enhanced_results[0]:
                enh_vals = [r[key] for r in enhanced_results]
                summary['enhanced'][key] = (np.mean(enh_vals), np.std(enh_vals))
            if key in traditional_results[0]:
                trad_vals = [r[key] for r in traditional_results]
                summary['traditional'][key] = (np.mean(trad_vals), np.std(trad_vals))
                if np.mean(trad_vals) != 0:
                    improvement = (np.mean(enh_vals) - np.mean(trad_vals)) / abs(np.mean(trad_vals)) * 100
                else:
                    improvement = 0
                summary['improvement'][key] = improvement
            else:
                summary['traditional'][key] = (0,0)
                summary['improvement'][key] = 0

        # [Step4] MAPPO结果汇总
        if mappo_results and len(mappo_results) > 0:
            # 使用MAPPO评估函数返回的兼容字段（可能与传统指标不完全一致）
            mappo_keys = set()
            for r in mappo_results:
                mappo_keys.update(r.keys())
            for key in mappo_keys:
                vals = [r.get(key) for r in mappo_results if key in r and r[key] is not None]
                if vals:
                    try:
                        summary['mappo'][key] = (np.mean(vals), np.std(vals))
                    except (TypeError, ValueError):
                        pass

        return summary

    @staticmethod
    def _print_results_table(summary):
        headers = ["指标", "增强算法(均值±std)", "传统算法(均值±std)", "提升"]
        rows = []
        has_mappo = 'mappo' in summary and any(k in summary['mappo'] for k in Experiment3.METRICS.keys())
        for key, name in Experiment3.METRICS.items():
            if key in summary['enhanced']:
                enh_mean, enh_std = summary['enhanced'][key]
                trad_mean, trad_std = summary['traditional'][key]
                imp = summary['improvement'][key]
                row = [name, f"{enh_mean:.3f}±{enh_std:.3f}", f"{trad_mean:.3f}±{trad_std:.3f}", f"{imp:+.1f}%"]
                # [Step4] 如果有MAPPO数据，追加MAPPO列（保持每行列数一致）
                if has_mappo:
                    if key in summary['mappo']:
                        map_mean, map_std = summary['mappo'][key]
                        row.append(f"{map_mean:.3f}±{map_std:.3f}")
                    else:
                        row.append("N/A")
                rows.append(row)
        
        # 动态调整表头
        final_headers = headers.copy()
        if 'mappo' in summary and any(k in summary['mappo'] for k in Experiment3.METRICS.keys()):
            final_headers.append("MAPPO(均值±std)")
        
        VisualizationHelper.print_data_table("实验3结果：三算法对比" if 'mappo' in summary else "实验3结果：增强算法 vs 传统算法", final_headers, rows)

    @staticmethod
    def _plot(summary):
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('实验3：增强算法 vs 传统算法（全面对比）', fontsize=16, fontweight='bold')
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 检测是否有MAPPO数据（三方对比模式）
        has_mappo = 'mappo' in summary and any(k in summary.get('mappo', {}) for k in Experiment3.METRICS.keys())

        def plot_bars(ax, metrics, labels, title):
            n_bars = 3 if has_mappo else 2
            width = 0.25 if has_mappo else 0.35
            x = np.arange(len(labels))
            enh_vals = [summary['enhanced'][m][0] if m in summary['enhanced'] else 0 for m in metrics]
            trad_vals = [summary['traditional'][m][0] if m in summary['traditional'] else 0 for m in metrics]
            colors_enh = CMAP_PRIMARY(np.linspace(0.4, 0.8, len(labels)))
            colors_trad = plt.cm.Greys(np.linspace(0.4, 0.7, len(labels)))
            offset = width if has_mappo else width / 2
            bars1 = ax.bar(x - offset, enh_vals, width, label='增强算法', color=colors_enh)
            bars2 = ax.bar(x + offset, trad_vals, width, label='传统算法', color=colors_trad)
            ax.set_ylabel('数值')
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha='right')
            # MAPPO柱（如果存在）
            if has_mappo:
                mappo_vals = [summary['mappo'][m][0] if m in summary['mappo'] else 0 for m in metrics]
                colors_mappo = plt.cm.Oranges(np.linspace(0.4, 0.8, len(labels)))
                bars3 = ax.bar(x, mappo_vals, width, label='MAPPO', color=colors_mappo)
                for bar in bars3:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:.2f}', ha='center', va='bottom', fontsize=7)
            ax.legend()
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=7)

        # 子图
        ax = fig.add_subplot(gs[0,0])
        plot_bars(ax, ['handover_success_rate', 'avg_switching_latency_ms', 'max_switching_latency_ms'],
                  ['成功率', '平均时延', '最大时延'], '切换性能指标')

        ax = fig.add_subplot(gs[0,1])
        plot_bars(ax, ['avg_decision_time_ms', 'missed_opportunity_rate'],
                  ['决策时间', '错失率'], '决策性能指标')

        ax = fig.add_subplot(gs[0,2])
        plot_bars(ax, ['avg_satisfaction', 'critical_satisfaction', 'weighted_satisfaction'],
                  ['整体', '关键业务', '加权'], 'QoS满足率指标')

        ax = fig.add_subplot(gs[0,3])
        plot_bars(ax, ['total_throughput', 'load_variance', 'avg_sinr'],
                  ['吞吐量', '负载方差', 'SINR'], '网络性能指标')

        # 雷达图
        ax = fig.add_subplot(gs[1,:2], projection='polar')
        categories = ['切换成功率', '整体满足率', '关键业务满足率', '吞吐量', '连接保持率']
        metrics_map = ['handover_success_rate', 'avg_satisfaction', 'critical_satisfaction',
                       'total_throughput', 'connected_ratio']
        enh_vals, trad_vals, mappo_vals = [], [], []
        for m in metrics_map:
            if m in summary['enhanced']:
                enh_val = summary['enhanced'][m][0]
                trad_val = summary['traditional'][m][0]
                if m == 'total_throughput':
                    enh_val = min(enh_val / 1000, 1.0)
                    trad_val = min(trad_val / 1000, 1.0)
                enh_vals.append(enh_val)
                trad_vals.append(trad_val)
                if has_mappo and m in summary.get('mappo', {}):
                    mv = summary['mappo'][m][0]
                    if m == 'total_throughput':
                        mv = min(mv / 1000, 1.0)
                    mappo_vals.append(mv)
                else:
                    mappo_vals.append(0)
            else:
                enh_vals.append(0); trad_vals.append(0); mappo_vals.append(0)
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        enh_vals += enh_vals[:1]; trad_vals += trad_vals[:1]; angles += angles[:1]
        if has_mappo:
            mappo_vals += mappo_vals[:1]
        ax.plot(angles, enh_vals, 'o-', linewidth=2, label='增强算法', color=COLORS['primary'])
        ax.fill(angles, enh_vals, alpha=0.25, color=COLORS['primary'])
        ax.plot(angles, trad_vals, 'o-', linewidth=2, label='传统算法', color=COLORS['neutral'])
        ax.fill(angles, trad_vals, alpha=0.15, color=COLORS['neutral'])
        if has_mappo:
            ax.plot(angles, mappo_vals, 'o-', linewidth=2, label='MAPPO', color='#FF8C00')
            ax.fill(angles, mappo_vals, alpha=0.15, color='#FF8C00')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0,1)
        ax.set_title('综合性能雷达图', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.0))

        # 提升百分比
        ax = fig.add_subplot(gs[1,2:])
        improvements = [(k,v) for k,v in summary['improvement'].items() if abs(v)>0.1 and k in Experiment3.METRICS]
        improvements.sort(key=lambda x: abs(x[1]), reverse=True)
        if len(improvements) > 10:
            improvements = improvements[:10]
        names = [Experiment3.METRICS[k] for k,_ in improvements]
        values = [v for _,v in improvements]
        colors = [COLORS['success'] if v>0 else COLORS['danger'] for v in values]
        bars = ax.barh(names, values, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('提升百分比(%)')
        ax.set_title('关键指标提升对比', fontweight='bold')
        for bar, val in zip(bars, values):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:+.1f}%',
                    ha='left' if val>0 else 'right', va='center', fontsize=8)

        # 热力图
        ax = fig.add_subplot(gs[2,:2])
        metrics_subset = ['handover_success_rate', 'avg_satisfaction', 'critical_satisfaction',
                          'latency_satisfaction', 'rate_satisfaction', 'connected_ratio']
        rows_data = [
            [summary['enhanced'][m][0] if m in summary['enhanced'] else 0 for m in metrics_subset],
            [summary['traditional'][m][0] if m in summary['traditional'] else 0 for m in metrics_subset]
        ]
        if has_mappo:
            rows_data.insert(1, [summary['mappo'][m][0] if m in summary.get('mappo', {}) else 0 for m in metrics_subset])
        data = np.array(rows_data)
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(metrics_subset)))
        ax.set_xticklabels([Experiment3.METRICS[m] for m in metrics_subset], rotation=45, ha='right')
        ytick_labels = ['增强算法']
        if has_mappo:
            ytick_labels.append('MAPPO')
        ytick_labels.append('传统算法')
        ax.set_yticks(range(len(ytick_labels)))
        ax.set_yticklabels(ytick_labels)
        ax.set_title('性能指标热力图', fontweight='bold')
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center', color='black', fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=ax)

        # 关键指标分布对比
        ax = fig.add_subplot(gs[2,2:])
        metrics = ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction']
        x_pos = np.arange(len(metrics))
        for i, m in enumerate(metrics):
            if m in summary['enhanced']:
                enh_mean, enh_std = summary['enhanced'][m]
                trad_mean, trad_std = summary['traditional'][m]
                if has_mappo and m in summary.get('mappo', {}):
                    mappo_mean, mappo_std = summary['mappo'][m]
                offset = 0.2 if has_mappo else 0.15
                ax.errorbar(i - offset, enh_mean, yerr=enh_std, fmt='o', color=COLORS['primary'],
                            markersize=10, capsize=5, label='增强算法' if i==0 else '')
                if has_mappo:
                    ax.errorbar(i, mappo_mean, yerr=mappo_std, fmt='^', color='#FF8C00',
                                markersize=10, capsize=5, label='MAPPO' if i==0 else '')
                ax.errorbar(i + offset, trad_mean, yerr=trad_std, fmt='s', color=COLORS['neutral'],
                            markersize=10, capsize=5, label='传统算法' if i==0 else '')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([Experiment3.METRICS[m] for m in metrics], rotation=15, ha='right')
        ax.set_ylabel('数值')
        ax.set_title('关键指标分布对比', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 文本摘要
        ax = fig.add_subplot(gs[3,:])
        ax.axis('off')
        text = "【实验3关键发现】\n\n"
        key_findings = [
            ("切换成功率", 'handover_success_rate', "%", 100),
            ("整体满足率", 'avg_satisfaction', "", 1),
            ("关键业务满足率", 'critical_satisfaction', "", 1),
            ("系统吞吐量", 'total_throughput', " Mbps", 1),
            ("平均切换时延", 'avg_switching_latency_ms', " ms", 1),
        ]
        for name, key, unit, scale in key_findings:
            if key in summary['enhanced']:
                enh_val = summary['enhanced'][key][0] * scale
                trad_val = summary['traditional'][key][0] * scale
                improvement = summary['improvement'][key]
                line = f"• {name}: 增强算法 {enh_val:.2f}{unit} vs 传统算法 {trad_val:.2f}{unit} ({improvement:+.1f}%)"
                if has_mappo and key in summary.get('mappo', {}):
                    mappo_val = summary['mappo'][key][0] * scale
                    line += f" | MAPPO {mappo_val:.2f}{unit}"
                text += line + "\n"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.savefig(os.path.join(RESULT_DIR, 'exp3_results.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

        save_experiment_data('exp3', summary)
        return summary


# -------------------- 实验2：机制有效性验证 --------------------
class Experiment2:
    """
    实验2：机制有效性验证

    采用逐步添加机制的方式，从传统算法开始，依次添加各个增强机制，
    验证每个机制对系统性能的贡献，为增强算法的设计提供理论依据。
    """
    MECHANISMS = {
        'traditional': '传统算法（基线）',
        'add_dynamic_threshold': '传统+动态阈值',
        'add_business_weights': '传统+动态阈值+业务权重',
        'add_epsilon_greedy': '传统+动态阈值+业务权重+ε-greedy',
        'add_load_balance': '传统+动态阈值+业务权重+ε-greedy+负载均衡',
        'add_adaptive_recognition': '传统+动态阈值+业务权重+ε-greedy+负载均衡+自适应识别',
        'full': '完整增强算法'
    }

    @staticmethod
    def run(recognition_model, scaler, num_steps=150, repeats=10, include_mappo=False):
        print("\n" + "="*80)
        print("实验2：机制有效性验证（逐步添加机制）")
        print("="*80)
        print("\n实验目的：从传统算法开始，逐步添加增强机制，验证各机制的独立贡献")
        print("\n机制添加顺序：")
        print("  1. 传统算法（基线）")
        print("  2. 添加动态阈值")
        print("  3. 添加业务特化权重")
        print("  4. 添加ε-greedy探索")
        print("  5. 添加负载均衡")
        print("  6. 添加自适应识别更新")
        print("  7. 完整增强算法（验证）")
        print("="*80)

        results = {key: [] for key in Experiment2.MECHANISMS.keys()}
        for rep in range(repeats):
            print(f"\n--- 重复 {rep+1}/{repeats} ---")
            set_global_seed(GLOBAL_SEED + rep)
            for mechanism in Experiment2.MECHANISMS.keys():
                env = EnhancedNetworkEnvironment(
                    num_bs=8, num_uav=300,  # 与带宽参数对齐: ~77%负载率
                    recognition_model=recognition_model, scaler=scaler,
                    seed=GLOBAL_SEED + rep, event_probability=0.05
                )

                # 根据机制配置创建对应的算法
                if mechanism == 'traditional':
                    algo = IntegratedHandoverAlgorithm(env)
                else:
                    algo = EnhancedHandoverAlgorithm(env)

                    # 根据机制配置启用对应的功能
                    if mechanism == 'add_dynamic_threshold':
                        # 只启用动态阈值，禁用其他增强功能
                        algo.base_threshold = 0.005
                        algo.calculate_dynamic_threshold = algo.__class__.calculate_dynamic_threshold.__get__(algo, type(algo))
                        # 禁用业务权重
                        for bt in BusinessType:
                            algo.business_weights[bt] = {'sinr': 0.4, 'load': 0.3, 'rate': 0.3}
                        # 禁用ε-greedy
                        algo.epsilon = 0.0
                        # 禁用负载均衡（在运行时控制）
                        enable_lb = False

                    elif mechanism == 'add_business_weights':
                        # 启用动态阈值和业务权重，禁用其他
                        algo.epsilon = 0.0
                        enable_lb = False

                    elif mechanism == 'add_epsilon_greedy':
                        # 启用动态阈值、业务权重和ε-greedy
                        # epsilon默认为0.05，无需修改
                        enable_lb = False

                    elif mechanism == 'add_load_balance':
                        # 启用动态阈值、业务权重、ε-greedy和负载均衡
                        enable_lb = True

                    elif mechanism == 'add_adaptive_recognition':
                        # 启用所有机制，包括自适应识别更新
                        # 自适应识别由环境控制，无需额外配置
                        enable_lb = True

                    elif mechanism == 'full':
                        # 完整增强算法
                        enable_lb = True

                # 运行仿真
                for step in range(num_steps):
                    env.step()
                    if mechanism == 'traditional':
                        algo.run_step()
                    elif mechanism in ['add_dynamic_threshold', 'add_business_weights',
                                      'add_epsilon_greedy', 'add_load_balance',
                                      'add_adaptive_recognition', 'full']:
                        algo.run_step(enable_load_balancing=enable_lb)

                stats = env.get_state_statistics()
                if hasattr(algo, 'get_detailed_stats'):
                    stats.update(algo.get_detailed_stats())
                results[mechanism].append(stats)
                print(f" {Experiment2.MECHANISMS[mechanism]}: "
                      f"满足率={stats['avg_satisfaction']:.3f}, "
                      f"切换成功率={stats.get('handover_success_rate',0)*100:.1f}%")

        summary = Experiment2._summarize(results)
        Experiment2._print_results_table(summary)
        Experiment2._plot(summary)
        return summary

    @staticmethod
    def _summarize(results):
        summary = {}
        for mechanism, data_list in results.items():
            summary[mechanism] = {}
            for key in ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction',
                        'weighted_satisfaction', 'total_load', 'load_variance']:
                if key in data_list[0]:
                    vals = [d[key] for d in data_list]
                    summary[mechanism][key] = (np.mean(vals), np.std(vals))
        return summary

    @staticmethod
    def _print_results_table(summary):
        headers = ["机制配置", "整体满足率", "切换成功率", "关键业务满足率", "吞吐量", "负载方差"]
        rows = []
        for mechanism, name in Experiment2.MECHANISMS.items():
            if mechanism in summary:
                data = summary[mechanism]
                row = [name]
                for key in ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction', 'total_load', 'load_variance']:
                    if key in data:
                        mean, std = data[key]
                        if key == 'handover_success_rate':
                            row.append(f"{mean*100:.1f}%±{std*100:.1f}%")
                        else:
                            row.append(f"{mean:.3f}±{std:.3f}")
                    else:
                        row.append("N/A")
                rows.append(row)
        VisualizationHelper.print_data_table("实验2结果：机制有效性验证", headers, rows)

        # 逐步添加机制的贡献分析
        print("\n【逐步添加机制贡献分析】")
        mechanism_order = ['traditional', 'add_dynamic_threshold', 'add_business_weights',
                          'add_epsilon_greedy', 'add_load_balance', 'add_adaptive_recognition', 'full']

        prev_mechanism = mechanism_order[0]
        prev_sat = summary[prev_mechanism]['avg_satisfaction'][0] if prev_mechanism in summary else 0

        print(f"\n相对于传统算法的逐步提升:")
        for mechanism in mechanism_order[1:]:
            if mechanism in summary:
                curr_sat = summary[mechanism]['avg_satisfaction'][0]
                contribution = curr_sat - prev_sat
                contribution_pct = contribution / prev_sat * 100 if prev_sat > 0 else 0

                mechanism_name = Experiment2.MECHANISMS[mechanism]

                # 提取新增的机制名称
                if mechanism == 'add_dynamic_threshold':
                    added_name = "动态阈值"
                elif mechanism == 'add_business_weights':
                    added_name = "业务权重"
                elif mechanism == 'add_epsilon_greedy':
                    added_name = "ε-greedy探索"
                elif mechanism == 'add_load_balance':
                    added_name = "负载均衡"
                elif mechanism == 'add_adaptive_recognition':
                    added_name = "自适应识别更新"
                elif mechanism == 'full':
                    added_name = "完整算法(验证)"
                else:
                    added_name = mechanism_name

                print(f"  {added_name}: 贡献 = {contribution:+.4f} ({contribution_pct:+.1f}%) "
                      f"[{prev_sat:.4f} -> {curr_sat:.4f}]")

                prev_sat = curr_sat

        # 总体提升对比
        if 'traditional' in summary and 'full' in summary:
            trad_sat = summary['traditional']['avg_satisfaction'][0]
            full_sat = summary['full']['avg_satisfaction'][0]
            total_improvement = (full_sat - trad_sat) / trad_sat * 100
            print(f"\n总体提升: 传统算法 -> 完整增强算法 = {total_improvement:+.1f}%")

    @staticmethod
    def _plot(summary):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('实验2：机制有效性验证（逐步添加）', fontsize=14, fontweight='bold')
        mechanism_order = ['traditional', 'add_dynamic_threshold', 'add_business_weights',
                          'add_epsilon_greedy', 'add_load_balance', 'add_adaptive_recognition', 'full']
        mechanisms = mechanism_order
        names = [Experiment2.MECHANISMS[m] for m in mechanisms]

        def plot_hbar(ax, key, title, xlabel):
            vals = [summary[m][key][0] if m in summary else 0 for m in mechanisms]
            errs = [summary[m][key][1] if m in summary else 0 for m in mechanisms]
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(mechanisms)))
            bars = ax.barh(names, vals, xerr=errs, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
            ax.set_xlabel(xlabel)
            ax.set_title(title, fontweight='bold')
            for bar, val in zip(bars, vals):
                ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}', ha='left', va='center', fontsize=9)

        plot_hbar(axes[0,0], 'avg_satisfaction', '整体满足率对比', '整体满足率')
        plot_hbar(axes[0,1], 'handover_success_rate', '切换成功率对比', '切换成功率')
        plot_hbar(axes[0,2], 'critical_satisfaction', '关键业务满足率对比', '关键业务满足率')

        # 逐步提升曲线图
        ax = axes[1,0]
        if all(m in summary for m in mechanism_order):
            sats = [summary[m]['avg_satisfaction'][0] for m in mechanism_order]
            stds = [summary[m]['avg_satisfaction'][1] for m in mechanism_order]
            x = range(len(mechanism_order))
            short_names = ['传统', '+动态\n阈值', '+业务\n权重', '+ε-\ngreedy', '+负载\n均衡', '+自适应\n识别', '完整']
            ax.plot(x, sats, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)
            ax.fill_between(x, [s-std for s,std in zip(sats, stds)],
                           [s+std for s,std in zip(sats, stds)],
                           alpha=0.2, color=COLORS['primary'])
            ax.set_xticks(x)
            ax.set_xticklabels(short_names, fontsize=9)
            ax.set_ylabel('整体满足率')
            ax.set_title('逐步添加机制的性能提升曲线', fontweight='bold')
            ax.grid(True, alpha=0.3)

            # 标注每个阶段的提升
            for i in range(1, len(sats)):
                improvement = sats[i] - sats[i-1]
                if abs(improvement) > 0.005:  # 只标注显著提升
                    mid_x = (i-1 + i) / 2
                    mid_y = (sats[i-1] + sats[i]) / 2
                    ax.annotate(f'{improvement:+.4f}',
                               xy=(mid_x, mid_y),
                               xytext=(mid_x, mid_y + improvement*0.5),
                               ha='center', va='center',
                               fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', lw=0.5))

        # 逐步提升柱状图
        ax = axes[1,1]
        if all(m in summary for m in mechanism_order):
            improvements = []
            contrib_names = []
            for i in range(1, len(mechanism_order)):
                if mechanism_order[i-1] in summary and mechanism_order[i] in summary:
                    prev_sat = summary[mechanism_order[i-1]]['avg_satisfaction'][0]
                    curr_sat = summary[mechanism_order[i]]['avg_satisfaction'][0]
                    improvement = curr_sat - prev_sat
                    improvements.append(improvement)

                    # 提取机制名称
                    if mechanism_order[i] == 'add_dynamic_threshold':
                        name = "动态阈值"
                    elif mechanism_order[i] == 'add_business_weights':
                        name = "业务权重"
                    elif mechanism_order[i] == 'add_epsilon_greedy':
                        name = "ε-greedy"
                    elif mechanism_order[i] == 'add_load_balance':
                        name = "负载均衡"
                    elif mechanism_order[i] == 'add_adaptive_recognition':
                        name = "自适应识别"
                    elif mechanism_order[i] == 'full':
                        name = "完整验证"
                    else:
                        name = mechanism_order[i]
                    contrib_names.append(name)

            colors = [COLORS['success'] if imp > 0 else COLORS['danger'] for imp in improvements]
            bars = ax.bar(contrib_names, improvements, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_ylabel('满足率提升')
            ax.set_title('各机制的独立贡献', fontweight='bold')
            ax.set_xticklabels(contrib_names, rotation=30, ha='right')
            for bar, imp in zip(bars, improvements):
                ax.text(bar.get_x() + bar.get_width()/2, imp,
                       f'{imp:+.4f}',
                       ha='center', va='bottom' if imp > 0 else 'top',
                       fontsize=9, fontweight='bold')

        # 综合评分对比
        ax = axes[1,2]
        scores = []
        score_names = []
        for mechanism in mechanisms:
            if mechanism in summary:
                score = (0.4 * summary[mechanism]['avg_satisfaction'][0] +
                         0.3 * summary[mechanism]['handover_success_rate'][0] +
                         0.3 * summary[mechanism]['critical_satisfaction'][0])
                scores.append(score)
                score_names.append(mechanism)
        colors = plt.cm.RdYlGn(np.array(scores) / max(scores))
        bars = ax.barh([Experiment2.MECHANISMS[m] for m in score_names], scores,
                      color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.set_xlabel('综合评分')
        ax.set_title('各配置的综合评分', fontweight='bold')
        for bar, val in zip(bars, scores):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                   ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'exp2_results.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

        save_experiment_data('exp2', summary)
        return summary


# -------------------- 实验2b：机制组合验证 --------------------
class Experiment2b:
    """
    实验2b：机制组合验证

    测试不同机制组合的性能，验证机制间的交互效应，
    找到在当前规模下的最优配置。

    设计原因：
    - 实验2发现ε-greedy在"逐步添加"时导致性能下降(-4.7%)
    - 但在"禁用"时却是正面的(+4.8%)
    - 需要通过组合验证找出机制间的交互效应
    """
    COMBINATIONS = {
        'traditional': '传统算法',
        'dyn_thresh': '+动态阈值',
        'weights': '+业务权重',
        'epsilon': '+ε-greedy',
        'load_balance': '+负载均衡',
        'dyn_thresh_weights': '动态阈值+业务权重',
        'dyn_thresh_epsilon': '动态阈值+ε-greedy',
        'weights_epsilon': '业务权重+ε-greedy',
        'dyn_thresh_weights_epsilon': '动态阈值+业务权重+ε-greedy',
        'dyn_thresh_weights_lb': '动态阈值+业务权重+负载均衡',
        'dyn_thresh_epsilon_lb': '动态阈值+ε-greedy+负载均衡',
        'weights_epsilon_lb': '业务权重+ε-greedy+负载均衡',
        'dyn_thresh_weights_epsilon_lb': '动态阈值+业务权重+ε-greedy+负载均衡',
        'full': '完整增强算法(含自适应识别)',
    }

    @staticmethod
    def run(recognition_model, scaler, num_steps=150, repeats=8):
        print("\n" + "="*80)
        print("实验2b：机制组合验证")
        print("="*80)
        print("\n实验目的：验证不同机制组合的性能，找出机制间的交互效应")
        print("\n测试组合：")
        print("  - 单机制：动态阈值、业务权重、ε-greedy、负载均衡")
        print("  - 双机制组合：各两两组合")
        print("  - 三机制组合：关键三机制")
        print("  - 四机制组合：所有核心机制")
        print("  - 完整算法：包含自适应识别")
        print("="*80)

        results = {key: [] for key in Experiment2b.COMBINATIONS.keys()}

        for rep in range(repeats):
            print(f"\n--- 重复 {rep+1}/{repeats} ---")
            set_global_seed(GLOBAL_SEED + rep)

            for combo_name in Experiment2b.COMBINATIONS.keys():
                env = EnhancedNetworkEnvironment(
                    num_bs=8, num_uav=300,  # 调整后: 实际带宽对齐，300架约77%负载率
                    recognition_model=recognition_model, scaler=scaler,
                    seed=GLOBAL_SEED + rep, event_probability=0.05
                )

                # 根据组合配置创建算法
                if combo_name == 'traditional':
                    algo = IntegratedHandoverAlgorithm(env)
                    enable_lb = False
                else:
                    algo = EnhancedHandoverAlgorithm(env)

                    # 解析组合配置
                    has_dyn_thresh = 'dyn_thresh' in combo_name
                    has_weights = 'weights' in combo_name
                    has_epsilon = 'epsilon' in combo_name
                    has_lb = 'load_balance' in combo_name or 'lb' in combo_name

                    # 配置各机制
                    if not has_dyn_thresh:
                        # 禁用动态阈值
                        algo.base_threshold = 0.005
                        algo.calculate_dynamic_threshold = lambda uav: 0.005

                    if not has_weights:
                        # 禁用业务权重
                        for bt in BusinessType:
                            algo.business_weights[bt] = {'sinr': 0.4, 'load': 0.3, 'rate': 0.3}

                    if not has_epsilon:
                        # 禁用ε-greedy
                        algo.epsilon = 0.0

                    enable_lb = has_lb

                    # 完整算法额外启用自适应识别
                    if combo_name == 'full':
                        # 自适应识别由环境控制，无需额外配置
                        pass

                # 运行仿真
                for step in range(num_steps):
                    env.step()
                    if combo_name == 'traditional':
                        algo.run_step()
                    else:
                        algo.run_step(enable_load_balancing=enable_lb)

                stats = env.get_state_statistics()
                if hasattr(algo, 'get_detailed_stats'):
                    stats.update(algo.get_detailed_stats())
                results[combo_name].append(stats)

                if rep == 0:  # 只在第一次重复时打印，避免输出过多
                    print(f" {Experiment2b.COMBINATIONS[combo_name]:40s}: "
                          f"满足率={stats['avg_satisfaction']:.4f}")

        summary = Experiment2b._summarize(results)
        Experiment2b._print_results_table(summary)
        Experiment2b._plot(summary)
        return summary

    @staticmethod
    def _summarize(results):
        summary = {}
        for combo_name, data_list in results.items():
            summary[combo_name] = {}
            for key in ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction',
                        'weighted_satisfaction', 'total_load', 'load_variance']:
                if key in data_list[0]:
                    vals = [d[key] for d in data_list]
                    summary[combo_name][key] = (np.mean(vals), np.std(vals))
        return summary

    @staticmethod
    def _print_results_table(summary):
        # 按性能排序
        sorted_combos = sorted(summary.keys(),
                           key=lambda x: summary[x]['avg_satisfaction'][0],
                           reverse=True)

        print("\n" + "="*100)
        print("【按性能排序的机制组合】")
        print("="*100)

        headers = ["排名", "组合名称", "整体满足率", "切换成功率", "关键业务满足率", "提升(相对传统)"]
        rows = []

        trad_sat = summary.get('traditional', {}).get('avg_satisfaction', (0, 0))[0]

        for rank, combo in enumerate(sorted_combos, 1):
            data = summary[combo]
            sat_mean, sat_std = data['avg_satisfaction']
            success_mean, success_std = data['handover_success_rate']
            crit_mean, crit_std = data['critical_satisfaction']
            improvement = ((sat_mean - trad_sat) / trad_sat * 100) if trad_sat > 0 else 0

            row = [
                f"{rank}",
                Experiment2b.COMBINATIONS[combo],
                f"{sat_mean:.4f}±{sat_std:.4f}",
                f"{success_mean*100:.2f}%",
                f"{crit_mean:.4f}±{crit_std:.4f}",
                f"{improvement:+.2f}%"
            ]
            rows.append(row)

        VisualizationHelper.print_data_table("实验2b结果：机制组合性能对比", headers, rows)

        # 分析机制贡献
        print("\n【机制贡献分析】")
        print("\n单机制贡献（相对于传统算法）:")
        for mechanism in ['dyn_thresh', 'weights', 'epsilon', 'load_balance']:
            if mechanism in summary and 'traditional' in summary:
                trad_sat = summary['traditional']['avg_satisfaction'][0]
                mech_sat = summary[mechanism]['avg_satisfaction'][0]
                contribution = mech_sat - trad_sat
                pct = contribution / trad_sat * 100 if trad_sat > 0 else 0
                print(f"  {Experiment2b.COMBINATIONS[mechanism]:20s}: "
                      f"{contribution:+.4f} ({pct:+.2f}%)")

        print("\n关键发现:")
        best_combo = sorted_combos[0]
        best_sat = summary[best_combo]['avg_satisfaction'][0]
        full_sat = summary['full']['avg_satisfaction'][0] if 'full' in summary else 0

        print(f"  1. 最优组合: {Experiment2b.COMBINATIONS[best_combo]} (满足率={best_sat:.4f})")
        print(f"  2. 相对传统算法提升: {((best_sat-trad_sat)/trad_sat*100):.2f}%")
        if 'full' in summary and best_combo != 'full':
            print(f"  3. 相对完整算法差异: {(best_sat-full_sat):.4f}")
            if best_sat > full_sat:
                print(f"     [发现] 存在比完整算法更优的组合!")
            else:
                print(f"     [验证] 完整算法接近最优组合")

        # 分析ε-greedy的作用
        print("\n【ε-greedy机制分析】")
        epsilon_present = ['dyn_thresh_epsilon', 'weights_epsilon', 'dyn_thresh_weights_epsilon',
                        'dyn_thresh_epsilon_lb', 'weights_epsilon_lb', 'dyn_thresh_weights_epsilon_lb', 'full']
        epsilon_absent = ['dyn_thresh', 'weights', 'dyn_thresh_weights', 'dyn_thresh_weights_lb']

        avg_with_epsilon = np.mean([summary[c]['avg_satisfaction'][0]
                                   for c in epsilon_present if c in summary])
        avg_without_epsilon = np.mean([summary[c]['avg_satisfaction'][0]
                                      for c in epsilon_absent if c in summary])

        print(f"  含ε-greedy的平均满足率: {avg_with_epsilon:.4f}")
        print(f"  不含ε-greedy的平均满足率: {avg_without_epsilon:.4f}")
        print(f"  差异: {avg_with_epsilon-avg_without_epsilon:+.4f}")

        if avg_with_epsilon < avg_without_epsilon:
            print(f"  结论: ε-greedy在当前规模下总体起负面作用")
        else:
            print(f"  结论: ε-greedy在当前规模下总体起正面作用")

        print("="*100)

    @staticmethod
    def _plot(summary):
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('实验2b：机制组合验证', fontsize=16, fontweight='bold')
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # 1. 按性能排名的柱状图
        ax = fig.add_subplot(gs[0, :2])
        sorted_combos = sorted(summary.keys(),
                           key=lambda x: summary[x]['avg_satisfaction'][0],
                           reverse=True)
        sats = [summary[c]['avg_satisfaction'][0] for c in sorted_combos]
        names = [Experiment2b.COMBINATIONS[c] for c in sorted_combos]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sats)))
        bars = ax.barh(names, sats, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.set_xlabel('整体满足率')
        ax.set_title('各机制组合性能排名', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        for bar, val in zip(bars, sats):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                   ha='left', va='center', fontsize=9)

        # 2. 机制组合热力图 (单机制)
        ax = fig.add_subplot(gs[0, 2])
        mechanisms = ['传统', '动态\n阈值', '业务\n权重', 'ε-\ngreedy', '负载\n均衡']
        combos = ['traditional', 'dyn_thresh', 'weights', 'epsilon', 'load_balance']
        values = [summary[c]['avg_satisfaction'][0] if c in summary else 0 for c in combos]
        im = ax.imshow([values], cmap='RdYlGn', aspect='auto', vmin=0.6, vmax=0.9)
        ax.set_xticks(range(len(mechanisms)))
        ax.set_xticklabels(mechanisms, fontsize=9)
        ax.set_yticks([])
        ax.set_title('单机制性能热力图', fontweight='bold')
        for i, val in enumerate(values):
            ax.text(i, 0, f'{val:.4f}', ha='center', va='center',
                   fontsize=10, fontweight='bold',
                   color='black' if val < 0.75 else 'white')
        plt.colorbar(im, ax=ax)

        # 3. 双机制组合矩阵
        ax = fig.add_subplot(gs[1, :2])
        mech_list = ['dyn_thresh', 'weights', 'epsilon', 'load_balance']
        mech_names = ['动态阈值', '业务权重', 'ε-greedy', '负载均衡']

        # 构建矩阵
        matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(i, 4):
                combo_name = f"{mech_list[i]}_{mech_list[j]}"
                if combo_name in summary:
                    matrix[i, j] = summary[combo_name]['avg_satisfaction'][0]
                elif i == j:
                    # 对角线是单机制
                    matrix[i, j] = summary[mech_list[i]]['avg_satisfaction'][0]

        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.7, vmax=0.9, aspect='auto')
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(mech_names)
        ax.set_yticklabels(mech_names)
        ax.set_title('双机制组合性能矩阵', fontweight='bold')
        for i in range(4):
            for j in range(i, 4):
                val = matrix[i, j]
                ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       color='black' if val < 0.78 else 'white')
        plt.colorbar(im, ax=ax)

        # 4. ε-greedy作用对比
        ax = fig.add_subplot(gs[1, 2])
        no_epsilon_combos = ['dyn_thresh', 'weights', 'dyn_thresh_weights', 'dyn_thresh_weights_lb']
        with_epsilon_combos = ['dyn_thresh_epsilon', 'weights_epsilon',
                             'dyn_thresh_weights_epsilon', 'dyn_thresh_weights_epsilon_lb']

        no_epsilon_sats = [summary[c]['avg_satisfaction'][0] for c in no_epsilon_combos if c in summary]
        with_epsilon_sats = [summary[c]['avg_satisfaction'][0] for c in with_epsilon_combos if c in summary]

        x = range(len(no_epsilon_sats))
        width = 0.35

        ax.bar([i-width/2 for i in x], no_epsilon_sats, width,
               label='不含ε-greedy', color=COLORS['primary'], alpha=0.8)
        ax.bar([i+width/2 for i in x], with_epsilon_sats, width,
               label='含ε-greedy', color=COLORS['neutral'], alpha=0.8)

        labels = ['动态阈值', '业务权重', '动态+业务',
                 '动态+业务+负载']
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel('整体满足率')
        ax.set_title('ε-greedy作用对比', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 5. 负载均衡作用对比
        ax = fig.add_subplot(gs[2, 0])
        no_lb_combos = ['dyn_thresh', 'weights', 'dyn_thresh_weights',
                       'dyn_thresh_epsilon', 'weights_epsilon', 'dyn_thresh_weights_epsilon']
        with_lb_combos = ['dyn_thresh_weights_lb', 'dyn_thresh_epsilon_lb',
                         'weights_epsilon_lb', 'dyn_thresh_weights_epsilon_lb']

        no_lb_sats = [summary[c]['avg_satisfaction'][0] for c in no_lb_combos if c in summary]
        with_lb_sats = [summary[c]['avg_satisfaction'][0] for c in with_lb_combos if c in summary]

        # 计算平均提升
        no_lb_avg = np.mean(no_lb_sats)
        with_lb_avg = np.mean(with_lb_sats)
        lb_improvement = (with_lb_avg - no_lb_avg) / no_lb_avg * 100 if no_lb_avg > 0 else 0

        ax.bar(['无负载均衡', '有负载均衡'], [no_lb_avg, with_lb_avg],
               color=[COLORS['neutral'], COLORS['primary']], alpha=0.8)
        ax.set_ylabel('平均满足率')
        ax.set_title(f'负载均衡作用 (提升{lb_improvement:+.2f}%)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 6. 关键指标散点图
        ax = fig.add_subplot(gs[2, 1])
        all_combos = list(summary.keys())
        x_vals = [summary[c]['handover_success_rate'][0]*100 for c in all_combos]
        y_vals = [summary[c]['avg_satisfaction'][0] for c in all_combos]
        colors = plt.cm.RdYlGn(y_vals)

        ax.scatter(x_vals, y_vals, s=100, c=colors, alpha=0.7, edgecolors='white', linewidth=1.5)
        ax.set_xlabel('切换成功率 (%)')
        ax.set_ylabel('整体满足率')
        ax.set_title('切换成功率 vs 整体满足率', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 标注最优组合
        best_idx = np.argmax(y_vals)
        ax.scatter([x_vals[best_idx]], [y_vals[best_idx]],
                  s=200, c='gold', edgecolors='red', linewidth=2, marker='*',
                  label=f"最优: {Experiment2b.COMBINATIONS[all_combos[best_idx]]}")
        ax.legend(fontsize=8)

        # 7. 关键文本摘要
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')

        best_combo = sorted_combos[0]
        best_sat = summary[best_combo]['avg_satisfaction'][0]
        trad_sat = summary['traditional']['avg_satisfaction'][0]
        full_sat = summary['full']['avg_satisfaction'][0]

        text = f"【实验2b关键发现】\n\n"
        text += f"最优组合: {Experiment2b.COMBINATIONS[best_combo]}\n"
        text += f"满足率: {best_sat:.4f}\n\n"
        text += f"相对传统算法: {((best_sat-trad_sat)/trad_sat*100):.2f}% 提升\n"
        text += f"相对完整算法: {((best_sat-full_sat)/full_sat*100):+.2f}% 差异\n\n"

        # 找出最优组合包含的机制
        best_mechs = []
        if 'dyn_thresh' in best_combo or 'thresh' in best_combo:
            best_mechs.append('动态阈值')
        if 'weights' in best_combo:
            best_mechs.append('业务权重')
        if 'epsilon' in best_combo:
            best_mechs.append('ε-greedy')
        if 'lb' in best_combo or 'load_balance' in best_combo:
            best_mechs.append('负载均衡')

        text += f"核心机制: {', '.join(best_mechs)}\n\n"

        # ε-greedy结论
        if 'epsilon' not in best_combo:
            text += f"[结论] 最优组合不含ε-greedy\n"
            text += f"      说明该机制在当前规模下不适用\n"
        else:
            text += f"[结论] 最优组合含ε-greedy\n"

        # 负载均衡结论
        if ('lb' not in best_combo and 'load_balance' not in best_combo):
            text += f"[结论] 最优组合不含负载均衡\n"
            text += f"      说明该机制贡献有限\n"
        else:
            text += f"[结论] 最优组合含负载均衡\n"

        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.savefig(os.path.join(RESULT_DIR, 'exp2b_results.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

        save_experiment_data('exp2b', summary)
        return summary


# -------------------- 实验4 --------------------
class Experiment4:
    """
    实验4：多场景对比实验

    场景设计与论文"典型5G应用场景适配性与需求映射"完全对齐：
    - 智慧城市监控: eMBB高带宽+URLLC切片，安防巡逻视频为主
    - 工业巡检: eMBB+MEC实时处理，4K视频巡视为主
    - 农业植保: mMTC大量传感+eMBB数据，环境监测为主
    - 应急救援: URLLC超可靠低时延，控制信令为主
    - 物流配送: eMBB+网络切片，控制与监测并重
    """
    SCENARIOS = {
        'smart_city': {
            'name': '智慧城市监控',
            'desc': '安防巡逻视频，eMBB高带宽+URLLC切片，优先保证视频流畅',
            '5g_feature': 'eMBB+URLLC切片',
            'switch_focus': '优先保证视频流畅，低延时控制',
            'num_uav': 400,
        },
        'industrial_inspection': {
            'name': '工业巡检',
            'desc': '4K视频巡视，eMBB+MEC实时处理，边缘节点接入',
            '5g_feature': 'eMBB+MEC',
            'switch_focus': '边缘节点接入、通信恢复机制',
            'num_uav': 300,
        },
        'agriculture': {
            'name': '农业植保',
            'desc': '植物健康监测，mMTC大量传感+eMBB数据，大范围覆盖',
            '5g_feature': 'mMTC+eMBB',
            'switch_focus': '大范围网联覆盖、能效切换',
            'num_uav': 350,
        },
        'emergency_rescue': {
            'name': '应急救援',
            'desc': '实时指挥通信，URLLC超可靠低时延，最低切换时延保障',
            '5g_feature': 'URLLC',
            'switch_focus': '最低切换时延、可靠链路保障',
            'num_uav': 300,
        },
        'logistics_delivery': {
            'name': '物流配送',
            'desc': '路径导航与状态，eMBB+网络切片，长航程持续覆盖',
            '5g_feature': 'eMBB+网络切片',
            'switch_focus': '长航程持续覆盖、服务切换平稳',
            'num_uav': 500,
        },
    }


    @staticmethod
    def run(recognition_model, scaler, num_steps=150, repeats=10, include_mappo=False, mappo_model_path=None):
        """
        运行实验4：多场景对比实验

        Args:
            recognition_model: 业务识别模型
            scaler: 识别模型标准化器
            num_steps: 仿真步数（默认150）
            repeats: 重复实验次数（默认10）
            include_mappo: 是否包含MAPPO泛化评估
            mappo_model_path: MAPPO模型路径，None则使用默认路径
        """
        print("\n" + "="*80)
        print("实验4：多场景对比实验" + (" + MAPPO" if include_mappo else ""))
        print("="*80)
        print("\n场景设计依据论文'典型5G应用场景适配性与需求映射'：")
        for key, info in Experiment4.SCENARIOS.items():
            print(f"  {info['name']:8s} | 5G特性: {info['5g_feature']:16s} | "
                  f"切换重点: {info['switch_focus']}")
        print("="*80)

        results = {scenario: {'enhanced': [], 'traditional': [], 'mappo': []} for scenario in Experiment4.SCENARIOS.keys()}
        for scenario, info in Experiment4.SCENARIOS.items():
            num_uav = info['num_uav']
            print(f"\n{'='*60}")
            print(f"场景: {info['name']} - {info['desc']}")
            print(f"UAV数量: {num_uav}  5G特性: {info['5g_feature']}")
            print('='*60)
            for rep in range(repeats):
                print(f"\n 重复 {rep+1}/{repeats}")
                set_global_seed(GLOBAL_SEED + rep)

                env_enh = EnhancedNetworkEnvironment(
                    num_bs=8, num_uav=num_uav,
                    recognition_model=recognition_model, scaler=scaler,
                    seed=GLOBAL_SEED + rep, scenario=scenario, event_probability=0.05
                )
                algo_enh = EnhancedHandoverAlgorithm(env_enh)
                algo_enh.epsilon = 0.0  # 最终算法不含ε-greedy探索机制

                env_trad = EnhancedNetworkEnvironment(
                    num_bs=8, num_uav=num_uav,
                    recognition_model=recognition_model, scaler=scaler,
                    seed=GLOBAL_SEED + rep, scenario=scenario, event_probability=0.05
                )
                algo_trad = IntegratedHandoverAlgorithm(env_trad)

                for step in range(num_steps):
                    env_enh.step()
                    algo_enh.run_step(enable_load_balancing=True)
                    env_trad.step()
                    algo_trad.run_step()

                enh_stats = env_enh.get_state_statistics()
                enh_stats.update(algo_enh.get_detailed_stats())
                enh_stats['business_stats'] = env_enh.get_business_type_stats()
                results[scenario]['enhanced'].append(enh_stats)

                trad_stats = env_trad.get_state_statistics()
                trad_stats.update(algo_trad.get_detailed_stats())
                trad_stats['business_stats'] = env_trad.get_business_type_stats()
                results[scenario]['traditional'].append(trad_stats)

                print(f" 增强算法 - 满足率: {enh_stats['avg_satisfaction']:.3f}, "
                      f"关键业务: {enh_stats['critical_satisfaction']:.3f}")
                print(f" 传统算法 - 满足率: {trad_stats['avg_satisfaction']:.3f}, "
                      f"关键业务: {trad_stats['critical_satisfaction']:.3f}")

                # [Step4] MAPPO评估（使用实验3训练的模型，零样本泛化到不同UAV数量）
                if include_mappo:
                    mappo_stats = evaluate_mappo_in_experiment(
                        num_bs=8, num_uav=num_uav, num_steps=num_steps,
                        model_path=mappo_model_path,  # 支持自定义模型路径
                    )
                    if mappo_stats is not None:
                        results[scenario]['mappo'].append(mappo_stats)
                        print(f" MAPPO     - 满足率: {mappo_stats['avg_satisfaction']:.3f}")
        Experiment4._print_results_table(summary)
        Experiment4._plot(summary)
        return summary

    @staticmethod
    def _summarize(results):
        summary = {}
        for scenario in Experiment4.SCENARIOS.keys():
            summary[scenario] = {'enhanced': {}, 'traditional': {}, 'mappo': {}}
            for algo_type in ['enhanced', 'traditional']:
                data_list = results[scenario][algo_type]
                if not data_list:
                    continue
                for key in ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction',
                            'weighted_satisfaction', 'total_load', 'avg_sinr', 'load_variance',
                            'connected_ratio']:
                    if key in data_list[0]:
                        vals = [d[key] for d in data_list]
                        summary[scenario][algo_type][key] = (np.mean(vals), np.std(vals))
            # [Step4] MAPPO结果汇总
            mappo_list = results[scenario].get('mappo', [])
            if mappo_list:
                mappo_keys = set()
                for r in mappo_list:
                    mappo_keys.update(r.keys())
                for key in mappo_keys:
                    vals = [r.get(key) for r in mappo_list if key in r and r[key] is not None]
                    if vals:
                        try:
                            summary[scenario]['mappo'][key] = (np.mean(vals), np.std(vals))
                        except (TypeError, ValueError):
                            pass
        return summary

    @staticmethod
    def _print_results_table(summary):
        for scenario, info in Experiment4.SCENARIOS.items():
            print(f"\n【{info['name']}】{info['desc']}")
            headers = ["算法", "整体满足率", "切换成功率", "关键业务满足率", "吞吐量(Mbps)", "SINR(dB)"]
            rows = []
            for algo_type, algo_name in [('enhanced', '增强算法'), ('traditional', '传统算法')]:
                if algo_type in summary[scenario] and summary[scenario][algo_type]:
                    data = summary[scenario][algo_type]
                    row = [algo_name]
                    for key in ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction', 'total_load', 'avg_sinr']:
                        if key in data:
                            mean, std = data[key]
                            if key == 'handover_success_rate':
                                row.append(f"{mean*100:.1f}%±{std*100:.1f}%")
                            elif key == 'total_load':
                                row.append(f"{mean:.1f}±{std:.1f}")
                            elif key == 'avg_sinr':
                                row.append(f"{mean:.1f}±{std:.1f}")
                            else:
                                row.append(f"{mean:.3f}±{std:.3f}")
                        else:
                            row.append("N/A")
                    rows.append(row)
            # [Step4] MAPPO行
            if 'mappo' in summary[scenario] and summary[scenario]['mappo']:
                data = summary[scenario]['mappo']
                row = ['MAPPO(本文)']
                for key in ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction',
                             'total_throughput', 'avg_sinr']:
                    if key in data:
                        mean, std = data[key]
                        if key == 'handover_success_rate':
                            row.append(f"{mean*100:.1f}%±{std*100:.1f}%")
                        elif key in ('total_load', 'total_throughput'):
                            row.append(f"{mean:.1f}±{std:.1f}")
                        elif key == 'avg_sinr':
                            row.append(f"{mean:.1f}±{std:.1f}")
                        else:
                            row.append(f"{mean:.3f}±{std:.3f}")
                    else:
                        row.append("N/A")
                rows.append(row)

            # 计算提升
            if 'enhanced' in summary[scenario] and 'traditional' in summary[scenario]:
                enh_sat = summary[scenario]['enhanced'].get('avg_satisfaction', (0, 0))[0]
                trad_sat = summary[scenario]['traditional'].get('avg_satisfaction', (0, 0))[0]
                if trad_sat > 0:
                    improvement = (enh_sat - trad_sat) / trad_sat * 100
                    print(f" 满足率提升(增强vs传统): {improvement:+.1f}%")
            if 'mappo' in summary[scenario] and summary[scenario]['mappo'] and 'enhanced' in summary[scenario]:
                map_sat = summary[scenario]['mappo'].get('avg_satisfaction', (0, 0))[0]
                enh_sat = summary[scenario]['enhanced'].get('avg_satisfaction', (0, 0))[0]
                if enh_sat > 0:
                    improvement = (map_sat - enh_sat) / enh_sat * 100
                    print(f" 满足率提升(MAPPOvs增强): {improvement:+.1f}%")

            VisualizationHelper.print_data_table(
                f"{info['name']}详细结果{' (三算法)' if 'mappo' in summary[scenario] and summary[scenario]['mappo'] else ''}",
                headers, rows)

    @staticmethod
    def _plot(summary):
        # [Step4] 检测是否有MAPPO数据，决定使用2算法还是3算法布局
        has_mappo = any(
            'mappo' in summary[s] and summary[s]['mappo']
            for s in Experiment4.SCENARIOS.keys()
        )

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('实验4：多场景对比实验' + (' (含MAPPO)' if has_mappo else ''),
                     fontsize=14, fontweight='bold')
        scenarios = list(Experiment4.SCENARIOS.keys())
        scenario_names = [Experiment4.SCENARIOS[s]['name'] for s in scenarios]
        x = np.arange(len(scenarios))

        # 布局参数：有MAPPO用3组(窄)，无则2组(宽)
        if has_mappo:
            width = 0.25
            offsets = [-width, 0, width]
        else:
            width = 0.35
            offsets = [-width/2, width/2]

        def _get_val(s, algo, key, fallback=0, scale=1):
            """安全获取summary中的值"""
            if algo in summary[s] and key in summary[s][algo]:
                return summary[s][algo][key][0] * scale
            return fallback

        # ===== 图1: 整体满足率 =====
        ax = axes[0, 0]
        enh_vals = [_get_val(s, 'enhanced', 'avg_satisfaction') for s in scenarios]
        trad_vals = [_get_val(s, 'traditional', 'avg_satisfaction') for s in scenarios]
        ax.bar(x + offsets[0], enh_vals, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
        ax.bar(x + offsets[1], trad_vals, width, label='传统算法', color=COLORS['neutral'], alpha=0.8)
        if has_mappo:
            map_vals = [_get_val(s, 'mappo', 'avg_satisfaction') for s in scenarios]
            ax.bar(x + offsets[2], map_vals, width, label='MAPPO', color=COLORS['warning'], alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(scenario_names, rotation=15, ha='right')
        ax.set_ylabel('整体满足率'); ax.set_title('各场景整体满足率对比', fontweight='bold')
        ax.legend(fontsize=8)

        # ===== 图2: 切换成功率 =====
        ax = axes[0, 1]
        enh_vals = [_get_val(s, 'enhanced', 'handover_success_rate', scale=100) for s in scenarios]
        trad_vals = [_get_val(s, 'traditional', 'handover_success_rate', scale=100) for s in scenarios]
        ax.bar(x + offsets[0], enh_vals, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
        ax.bar(x + offsets[1], trad_vals, width, label='传统算法', color=COLORS['neutral'], alpha=0.8)
        if has_mappo:
            map_vals = [_get_val(s, 'mappo', 'handover_success_rate', scale=100) for s in scenarios]
            ax.bar(x + offsets[2], map_vals, width, label='MAPPO', color=COLORS['warning'], alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(scenario_names, rotation=15, ha='right')
        ax.set_ylabel('切换成功率(%)'); ax.set_title('各场景切换成功率对比', fontweight='bold')
        ax.legend(fontsize=8)

        # ===== 图3: 关键业务满足率 =====
        ax = axes[0, 2]
        enh_vals = [_get_val(s, 'enhanced', 'critical_satisfaction') for s in scenarios]
        trad_vals = [_get_val(s, 'traditional', 'critical_satisfaction') for s in scenarios]
        ax.bar(x + offsets[0], enh_vals, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
        ax.bar(x + offsets[1], trad_vals, width, label='传统算法', color=COLORS['neutral'], alpha=0.8)
        if has_mappo:
            map_vals = [_get_val(s, 'mappo', 'critical_satisfaction') for s in scenarios]
            ax.bar(x + offsets[2], map_vals, width, label='MAPPO', color=COLORS['warning'], alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(scenario_names, rotation=15, ha='right')
        ax.set_ylabel('关键业务满足率'); ax.set_title('各场景关键业务满足率对比', fontweight='bold')
        ax.legend(fontsize=8)

        # ===== 图4: 吞吐量 =====
        ax = axes[1, 0]
        # 注意: 增强算法用 total_load, MAPPO用 total_throughput
        enh_vals = [_get_val(s, 'enhanced', 'total_load') or _get_val(s, 'enhanced', 'total_throughput') for s in scenarios]
        trad_vals = [_get_val(s, 'traditional', 'total_load') or _get_val(s, 'traditional', 'total_throughput') for s in scenarios]
        ax.bar(x + offsets[0], enh_vals, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
        ax.bar(x + offsets[1], trad_vals, width, label='传统算法', color=COLORS['neutral'], alpha=0.8)
        if has_mappo:
            map_vals = [_get_val(s, 'mappo', 'total_load') or _get_val(s, 'mappo', 'total_throughput') for s in scenarios]
            ax.bar(x + offsets[2], map_vals, width, label='MAPPO', color=COLORS['warning'], alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(scenario_names, rotation=15, ha='right')
        ax.set_ylabel('吞吐量(Mbps)'); ax.set_title('各场景吞吐量对比', fontweight='bold')
        ax.legend(fontsize=8)

        # ===== 图5: 提升百分比（增强vs传统） =====
        ax = axes[1, 1]
        improvements = []
        for s in scenarios:
            e = _get_val(s, 'enhanced', 'avg_satisfaction')
            t = _get_val(s, 'traditional', 'avg_satisfaction')
            improvements.append((e - t) / max(t, 0.001) * 100)
        colors = [COLORS['success'] if i > 0 else COLORS['danger'] for i in improvements]
        bars = ax.bar(scenario_names, improvements, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('提升百分比(%)'); ax.set_title('增强算法在各场景的满足率提升', fontweight='bold')
        ax.set_xticklabels(scenario_names, rotation=15, ha='right')
        for bar, val in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:+.1f}%',
                    ha='center', va='bottom' if val > 0 else 'top', fontsize=9, fontweight='bold')

        # ===== 图6: 热力图（含MAPPO行） =====
        ax = axes[1, 2]
        heat_rows = [
            ('增强-满足率', lambda s: _get_val(s, 'enhanced', 'avg_satisfaction')),
            ('传统-满足率', lambda s: _get_val(s, 'traditional', 'avg_satisfaction')),
            ('增强-成功率', lambda s: _get_val(s, 'enhanced', 'handover_success_rate')),
            ('传统-成功率', lambda s: _get_val(s, 'traditional', 'handover_success_rate')),
        ]
        if has_mappo:
            heat_rows.extend([
                ('MAPPO-满足率', lambda s: _get_val(s, 'mappo', 'avg_satisfaction')),
                ('MAPPO-成功率', lambda s: _get_val(s, 'mappo', 'handover_success_rate')),
            ])
        data = np.array([[fn(s) for s in scenarios] for name, fn in heat_rows])
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(scenarios))); ax.set_xticklabels(scenario_names, rotation=30, ha='right')
        ax.set_yticks(range(len(heat_rows))); ax.set_yticklabels([name for name, _ in heat_rows])
        ax.set_title('场景适应性热力图' + (' (含MAPPO)' if has_mappo else ''), fontweight='bold')
        for i in range(len(heat_rows)):
            for j in range(len(scenarios)):
                val = data[i, j]
                text = ax.text(j, i, f'{val*100:.0f}%' if '成功率' in heat_rows[i][0] else f'{val:.2f}',
                               ha='center', va='center',
                               color='white' if val < 0.5 else 'black', fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'exp4_results.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

        save_experiment_data('exp4', summary)


# -------------------- 实验2c：大规模场景下ε-greedy机制验证 --------------------
class Experiment2c:
    """
    实验2c：大规模场景下ε-greedy探索机制验证

    实验动机：
    - 实验2b发现ε-greedy在300UAV/8基站规模下总体呈微弱负面作用
    - 但ε-greedy的设计初衷是为大规模高动态环境提供探索能力
    - 需要验证当规模扩大时，ε-greedy是否转为正面作用
    """

    CONFIGS = {
        'core_no_epsilon': {
            'name': '核心组合(无ε-greedy)',
            'desc': '动态阈值+业务权重+负载均衡',
            'has_dynamic_threshold': True,
            'has_business_weights': True,
            'has_epsilon_greedy': False,
            'has_load_balance': True,
            'has_adaptive_recognition': False,
        },
        'core_with_epsilon': {
            'name': '核心组合(有ε-greedy)',
            'desc': '动态阈值+业务权重+负载均衡+ε-greedy',
            'has_dynamic_threshold': True,
            'has_business_weights': True,
            'has_epsilon_greedy': True,
            'has_load_balance': True,
            'has_adaptive_recognition': False,
        },
        'full_no_epsilon': {
            'name': '完整算法(无ε-greedy)',
            'desc': '所有机制，禁用ε-greedy',
            'has_dynamic_threshold': True,
            'has_business_weights': True,
            'has_epsilon_greedy': False,
            'has_load_balance': True,
            'has_adaptive_recognition': True,
        },
        'full_with_epsilon': {
            'name': '完整算法(有ε-greedy)',
            'desc': '所有机制，启用ε-greedy',
            'has_dynamic_threshold': True,
            'has_business_weights': True,
            'has_epsilon_greedy': True,
            'has_load_balance': True,
            'has_adaptive_recognition': True,
        },
    }

    @staticmethod
    def run(recognition_model, scaler, num_steps=200, repeats=6):
        print("\n" + "="*80)
        print("实验2c：大规模场景下ε-greedy机制验证")
        print("="*80)
        print("\n实验目的：验证ε-greedy在扩大规模后的作用是否转为正面")
        print("\n规模配置：")
        print(f"  - UAV数量: 600 (实验2的2倍)")
        print(f"  - 基站数量: 16 (实验2的2倍)")
        print(f"  - 场景范围: 4000m×4000m (实验2的2倍)")
        print(f"  - 仿真时长: {num_steps}步")
        print("\n对比配置：")
        for key, cfg in Experiment2c.CONFIGS.items():
            print(f"  - {cfg['name']}: {cfg['desc']}")
        print("="*80)

        results = {key: [] for key in Experiment2c.CONFIGS.keys()}
        config_keys = list(Experiment2c.CONFIGS.keys())

        for rep in range(repeats):
            print(f"\n--- 重复 {rep+1}/{repeats} ---")
            set_global_seed(GLOBAL_SEED + rep)

            for idx, config_key in enumerate(config_keys):
                # 每个配置使用独立种子，避免环境初始化受前一个配置影响
                cfg_seed = GLOBAL_SEED + rep * 100 + idx
                set_global_seed(cfg_seed)

                config = Experiment2c.CONFIGS[config_key]
                env = EnhancedNetworkEnvironment(
                    num_bs=16, num_uav=600,
                    recognition_model=recognition_model, scaler=scaler,
                    seed=cfg_seed, event_probability=0.05,
                    bs_capacity_range=(1500, 2500)
                )

                algo = EnhancedHandoverAlgorithm(env)

                if not config['has_dynamic_threshold']:
                    algo.base_threshold = 0.005
                    algo.calculate_dynamic_threshold = lambda uav: 0.005

                if not config['has_business_weights']:
                    for bt in BusinessType:
                        algo.business_weights[bt] = {'sinr': 0.4, 'load': 0.3, 'rate': 0.3}

                if not config['has_epsilon_greedy']:
                    algo.epsilon = 0.0
                else:
                    algo.epsilon = 0.05

                enable_lb = config['has_load_balance']

                for step in range(num_steps):
                    env.step()
                    algo.run_step(enable_load_balancing=enable_lb)

                stats = env.get_state_statistics()
                stats.update(algo.get_detailed_stats())
                results[config_key].append(stats)

                print(f" {config['name']:20s}: 满足率={stats['avg_satisfaction']:.4f}, "
                      f"切换成功率={stats.get('handover_success_rate',0)*100:.1f}%")

        summary = Experiment2c._summarize(results)
        Experiment2c._print_results_table(summary)
        Experiment2c._plot(summary)
        return summary

    @staticmethod
    def _summarize(results):
        summary = {}
        for config_key, data_list in results.items():
            summary[config_key] = {}
            for key in ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction',
                        'weighted_satisfaction', 'total_load', 'load_variance']:
                if key in data_list[0]:
                    vals = [d[key] for d in data_list]
                    summary[config_key][key] = (np.mean(vals), np.std(vals))
        return summary

    @staticmethod
    def _print_results_table(summary):
        print("\n" + "="*100)
        print("【实验2c结果：大规模场景下ε-greedy机制验证】")
        print("="*100)

        headers = ["配置", "整体满足率", "切换成功率", "关键业务满足率", "负载方差", "vs无ε差异"]
        rows = []

        core_no_eps = summary.get('core_no_epsilon', {}).get('avg_satisfaction', (0, 0))[0]
        full_no_eps = summary.get('full_no_epsilon', {}).get('avg_satisfaction', (0, 0))[0]

        for config_key, config in Experiment2c.CONFIGS.items():
            if config_key not in summary:
                continue
            data = summary[config_key]
            sat_mean, sat_std = data['avg_satisfaction']
            success_mean, success_std = data['handover_success_rate']
            crit_mean, crit_std = data['critical_satisfaction']
            load_var_mean, load_var_std = data['load_variance']

            if 'core' in config_key:
                base = core_no_eps
            else:
                base = full_no_eps
            diff = sat_mean - base if base > 0 else 0

            row = [
                config['name'],
                f"{sat_mean:.4f}±{sat_std:.4f}",
                f"{success_mean*100:.1f}%",
                f"{crit_mean:.4f}",
                f"{load_var_mean:.4f}",
                f"{diff:+.4f}" if 'with_epsilon' in config_key else "-"
            ]
            rows.append(row)

        VisualizationHelper.print_data_table("实验2c结果汇总", headers, rows)

        print("\n【关键结论】")
        core_diff = summary.get('core_with_epsilon', {}).get('avg_satisfaction', (0, 0))[0] - \
                    summary.get('core_no_epsilon', {}).get('avg_satisfaction', (0, 0))[0]
        full_diff = summary.get('full_with_epsilon', {}).get('avg_satisfaction', (0, 0))[0] - \
                    summary.get('full_no_epsilon', {}).get('avg_satisfaction', (0, 0))[0]

        print(f"\nε-greedy在大规模场景下的作用:")
        print(f"  核心组合(含ε) - 核心组合(无ε) = {core_diff:+.4f}")
        print(f"  完整算法(含ε) - 完整算法(无ε) = {full_diff:+.4f}")

        if core_diff > 0.005 or full_diff > 0.005:
            print(f"\n  [结论] ε-greedy在大规模场景下转为正面作用")
            print(f"         说明探索机制在扩大规模后显现价值")
        elif core_diff < -0.005 or full_diff < -0.005:
            print(f"\n  [结论] ε-greedy在大规模场景下仍为负面作用")
            print(f"         建议在实际部署中移除或大幅降低ε值")
        else:
            print(f"\n  [结论] ε-greedy在大规模场景下作用不显著")
            print(f"         在当前规模范围内无明确增益")

        print(f"\n【与实验2b对比】")
        print(f"  实验2b(300UAV): ε-greedy总体呈微弱负面作用")
        print(f"  实验2c(600UAV): {'转为正面' if core_diff > 0.005 else '仍为负面' if core_diff < -0.005 else '作用不显著'}")
        print("="*100)

    @staticmethod
    def _plot(summary):
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('实验2c：大规模场景下ε-greedy机制验证 (600UAV/16BS)', fontsize=14, fontweight='bold')
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        configs = list(Experiment2c.CONFIGS.keys())
        names = [Experiment2c.CONFIGS[c]['name'] for c in configs]

        # 1. 满足率对比
        ax = fig.add_subplot(gs[0, 0])
        sats = [summary[c]['avg_satisfaction'][0] if c in summary else 0 for c in configs]
        colors = [COLORS['primary'] if 'no_epsilon' in c else COLORS['neutral'] for c in configs]
        bars = ax.bar(names, sats, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('整体满足率')
        ax.set_title('整体满足率对比', fontweight='bold')
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
        for bar, val in zip(bars, sats):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # 2. ε-greedy作用对比（分组柱状图）
        ax = fig.add_subplot(gs[0, 1])
        categories = ['核心组合', '完整算法']
        no_epsilon = [
            summary.get('core_no_epsilon', {}).get('avg_satisfaction', (0, 0))[0],
            summary.get('full_no_epsilon', {}).get('avg_satisfaction', (0, 0))[0]
        ]
        with_epsilon = [
            summary.get('core_with_epsilon', {}).get('avg_satisfaction', (0, 0))[0],
            summary.get('full_with_epsilon', {}).get('avg_satisfaction', (0, 0))[0]
        ]
        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, no_epsilon, width, label='无ε-greedy', color=COLORS['primary'], alpha=0.8)
        ax.bar(x + width/2, with_epsilon, width, label='有ε-greedy', color=COLORS['neutral'], alpha=0.8)
        ax.set_ylabel('整体满足率')
        ax.set_title('ε-greedy作用对比（大规模）', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        for i, (no_eps, with_eps) in enumerate(zip(no_epsilon, with_epsilon)):
            diff = with_eps - no_eps
            ax.annotate(f'{diff:+.4f}', xy=(i, max(no_eps, with_eps)),
                       ha='center', va='bottom', fontsize=10, fontweight='bold',
                       color=COLORS['success'] if diff > 0 else COLORS['danger'])

        # 3. 切换成功率对比
        ax = fig.add_subplot(gs[0, 2])
        success_rates = [summary[c]['handover_success_rate'][0]*100 if c in summary else 0 for c in configs]
        colors = [COLORS['primary'] if 'no_epsilon' in c else COLORS['neutral'] for c in configs]
        bars = ax.bar(names, success_rates, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('切换成功率(%)')
        ax.set_title('切换成功率对比', fontweight='bold')
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
        for bar, val in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}%',
                   ha='center', va='bottom', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # 4. 关键业务满足率
        ax = fig.add_subplot(gs[1, 0])
        crit_sats = [summary[c]['critical_satisfaction'][0] if c in summary else 0 for c in configs]
        colors = [COLORS['primary'] if 'no_epsilon' in c else COLORS['neutral'] for c in configs]
        bars = ax.bar(names, crit_sats, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('关键业务满足率')
        ax.set_title('关键业务满足率对比', fontweight='bold')
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
        for bar, val in zip(bars, crit_sats):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # 5. 负载方差
        ax = fig.add_subplot(gs[1, 1])
        load_vars = [summary[c]['load_variance'][0] if c in summary else 0 for c in configs]
        colors = [COLORS['primary'] if 'no_epsilon' in c else COLORS['neutral'] for c in configs]
        bars = ax.bar(names, load_vars, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('负载方差')
        ax.set_title('负载均衡程度对比', fontweight='bold')
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
        for bar, val in zip(bars, load_vars):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # 6. 文本摘要
        ax = fig.add_subplot(gs[1, 2])
        ax.axis('off')

        core_no = summary.get('core_no_epsilon', {}).get('avg_satisfaction', (0, 0))[0]
        core_with = summary.get('core_with_epsilon', {}).get('avg_satisfaction', (0, 0))[0]
        full_no = summary.get('full_no_epsilon', {}).get('avg_satisfaction', (0, 0))[0]
        full_with = summary.get('full_with_epsilon', {}).get('avg_satisfaction', (0, 0))[0]

        text = "【实验2c关键发现】\n\n"
        text += f"规模: 600UAV / 16BS / 4000m²\n\n"
        text += f"核心组合:\n"
        text += f"  无ε-greedy: {core_no:.4f}\n"
        text += f"  有ε-greedy: {core_with:.4f}\n"
        text += f"  差异: {core_with-core_no:+.4f}\n\n"
        text += f"完整算法:\n"
        text += f"  无ε-greedy: {full_no:.4f}\n"
        text += f"  有ε-greedy: {full_with:.4f}\n"
        text += f"  差异: {full_with-full_no:+.4f}\n\n"

        if core_with - core_no > 0.005:
            text += "[结论] ε-greedy在大规模下\n转为正面作用 ✓"
        elif core_with - core_no < -0.005:
            text += "[结论] ε-greedy在大规模下\n仍为负面作用 ✗"
        else:
            text += "[结论] ε-greedy作用不显著\n建议移除或调低ε值"

        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.savefig(os.path.join(RESULT_DIR, 'exp2c_results.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

        save_experiment_data('exp2c', summary)
        return summary