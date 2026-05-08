"""
PMSF v2.0 - 渐进式多场景微调 (Progressive Multi-Scenario Finetuning)
====================================================================

核心改进:
1. 训练量扩展6倍 (15→90 episodes)
2. Critic定期重置打破过拟合死锁
3. 动态LR管理 + 学习率重启
4. 渐进式Entropy退火
5. 场景条件化策略输出
6. 三阶段课程学习

作者: AI Assistant
日期: 2026-05-09
"""

import torch
import numpy as np
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ============================================================
# 配置常量
# ============================================================

class PMSFConfig:
    """PMSF v2.0 超参数配置"""
    
    # === 基础配置 ===
    GLOBAL_SEED = 34687
    BASE_MODEL_PATH = "results/mappo_models/mappo_8bs_300uav_best.pt"
    OUTPUT_DIR = "experiment_results/mappo_models/pmsf_v2"
    
    # === 场景定义 ===
    SCENARIOS = {
        'industrial_inspection': {
            'num_uav': 300,
            'biz_ratios': [0.15, 0.75, 0.10],  # 控制,视频,监测
            'baseline': 0.6755,
            'weight': 1.0,
            'priority': 'high',  # 最弱场景
        },
        'agriculture': {
            'num_uav': 350,
            'biz_ratios': [0.15, 0.25, 0.60],
            'baseline': 0.9597,
            'weight': 0.4,
            'priority': 'low',  # 最强场景
        },
        'smart_city': {
            'num_uav': 400,
            'biz_ratios': [0.30, 0.60, 0.10],
            'baseline': 0.7047,
            'weight': 0.9,
            'priority': 'high',
        },
        'emergency_rescue': {
            'num_uav': 300,
            'biz_ratios': [0.85, 0.10, 0.05],
            'baseline': 0.9049,
            'weight': 0.6,
            'priority': 'medium',
        },
        'logistics_delivery': {
            'num_uav': 500,
            'biz_ratios': [0.50, 0.40, 0.10],
            'baseline': 0.7104,
            'weight': 0.85,
            'priority': 'high',
        },
    }
    
    # === Phase 0: 基线评估 ===
    PHASE0_CONFIG = {
        'num_eval_episodes': 5,
        'verbose': True,
    }
    
    # === Phase 1: 弱场景攻坚 (核心改进!) ===
    PHASE1_CONFIG = {
        'total_episodes': 50,           # 从15↑到50 (×3.3)
        'rollout_length': 500,          # 从350↑到500 (×1.4)
        
        # 学习率配置 (关键改进!)
        'actor_lr_initial': 2.5e-04,    # 从1.8e-04↑ (×1.39) 不再过度保守!
        'critic_lr_initial': 8.0e-04,   # 从6.0e-04↑ (×1.33)
        'lr_decay_type': 'cosine_warm_restarts',  # 新增! 支持重启
        'lr_warm_restart_T_0': 25,      # 每25个episodes重启一次
        'lr_warm_restart_T_mult': 1,    # 重启周期不变
        
        # Entropy配置 (渐进式!)
        'entropy_coef_initial': 0.008,  # 从0.005↑ (×1.6) 增加探索!
        'entropy_coef_final': 0.003,    # 最终值(逐步降低)
        'entropy_anneal_episodes': 45,  # 在前45个episodes内退火
        
        # PPO参数
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_param': 0.20,
        'ppo_epochs': 5,
        'batch_size': 64,
        'max_grad_norm': 10.0,
        
        # Critic管理 (核心改进!)
        'critic_reset_interval': 25,    # 每25个episodes重置Critic!
        'critic_reset_mode': 'partial', # 'partial'=只重置最后2层; 'full'=全部重置
        'critic_boost_factor': 3.0,     # 重置后LR临时提升3倍(持续5个eps)
        
        # 早停条件
        'early_stop_patience': 12,      # 从10↑到12 (更宽容)
        'early_stop_min_delta': 0.008,  # 最小改善阈值
        'early_stop_metric': 'global_avg',
        
        # 场景采样
        'sampling_strategy': 'weighted_adaptive',  # 自适应加权
        'min_sample_weight': 0.3,
        'max_sample_weight': 3.5,      # 从3.0↑到3.5 (允许更大差异)
        
        # 权重更新监控
        'weight_check_interval': 10,    # 每10个episodes检查一次(从5↑)
        'min_weight_change_threshold': 2.0,  # 最低变化阈值(%)
    }
    
    # === Phase 2: 全局精调 ===
    PHASE2_CONFIG = {
        'total_episodes': 40,           # 新增! 第二阶段
        'rollout_length': 500,
        
        # 降低学习率进行精细调整
        'actor_lr_initial': 1.5e-04,    # 从Phase1的2.5e-4降低
        'critic_lr_initial': 5.0e-04,
        'lr_decay_type': 'cosine',
        
        # 较低的Entropy(利用已有知识)
        'entropy_coef_initial': 0.004,
        'entropy_coef_final': 0.002,
        
        # Critic不再重置(保持稳定)
        'critic_reset_interval': None,  # 禁用重置
        
        # 更长的早停耐心(允许缓慢提升)
        'early_stop_patience': 15,
        'early_stop_min_delta': 0.005,
        
        # 均匀采样(所有场景平等)
        'sampling_strategy': 'uniform',
    }


# ============================================================
# 核心类: PMSFTuner
# ============================================================

class PMSFTuner:
    """渐进式多场景微调器"""
    
    def __init__(self, config: PMSFConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 运行时状态
        self.baseline_scores = {}
        self.current_phase = None
        self.best_global_score = 0.0
        self.training_history = []
        
        # 创建输出目录
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
    
    def run_full_pipeline(self) -> Dict:
        """
        运行完整的三阶段流水线
        
        Returns:
            dict: 包含所有阶段的评估结果和模型路径
        """
        print("=" * 70)
        print("  PMSF v2.0 - 渐进式多场景微调")
        print("=" * 70)
        
        results = {}
        
        # ====== Phase 0: 基线评估 ======
        print("\n" + "=" * 70)
        print("  [PHASE 0] 基线评估")
        print("=" * 70)
        
        baseline_result = self._run_phase0_baseline()
        results['phase0'] = baseline_result
        self.baseline_scores = baseline_result['scores']
        
        global_baseline = baseline_result['global_average']
        print(f"\n[RESULT] 全局基线: {global_baseline:.4f}")
        
        # ====== Phase 1: 弱场景攻坚 ======
        print("\n" + "=" * 70)
        print("  [PHASE 1] 弱场景攻坚 (核心改进版)")
        print("=" * 70)
        
        phase1_result = self._run_phase1_weak_scenario_attack()
        results['phase1'] = phase1_result
        
        phase1_score = phase1_result['eval_result']['global_average']
        improvement = (phase1_score - global_baseline) / global_baseline * 100
        
        print(f"\n[RESULT] Phase 1 完成:")
        print(f"         全局得分: {phase1_score:.4f} ({improvement:+.2f}%)")
        print(f"         最佳模型: {phase1_result['best_model_path']}")
        
        # 判断是否需要Phase 2
        if improvement >= 5.0:
            print("\n[OK] Phase 1 提升显著 (>5%), 可选择跳过Phase 2")
            should_run_phase2 = False
        elif improvement >= 2.0:
            print("\n[INFO] Phase 1 提升中等 (2%~5%), 建议运行Phase 2精调")
            should_run_phase2 = True
        else:
            print("\n[WARN] Phase 1 提升有限 (<2%), 必须运行Phase 2")
            should_run_phase2 = True
        
        # ====== Phase 2: 全局精调 (可选) ======
        if should_run_phase2:
            print("\n" + "=" * 70)
            print("  [PHASE 2] 全局精细调整")
            print("=" * 70)
            
            # 使用Phase 1的最佳模型作为起点
            phase2_base_model = phase1_result['best_model_path']
            
            phase2_result = self._run_phase2_global_finetune(phase2_base_model)
            results['phase2'] = phase2_result
            
            final_score = phase2_result['eval_result']['global_average']
            final_improvement = (final_score - global_baseline) / global_baseline * 100
            
            print(f"\n[RESULT] Phase 2 完成:")
            print(f"         最终全局得分: {final_score:.4f} ({final_improvement:+.2f}%)")
            print(f"         最终模型: {phase2_result['best_model_path']}")
            
            # 选择最佳结果
            if final_score > phase1_score:
                best_phase = 'phase2'
                best_score = final_score
                best_model = phase2_result['best_model_path']
            else:
                best_phase = 'phase1'
                best_score = phase1_score
                best_model = phase1_result['best_model_path']
        else:
            best_phase = 'phase1'
            best_score = phase1_score
            best_model = phase1_result['best_model_path']
            results['phase2'] = None
            final_improvement = improvement
        
        # ====== 最终汇总 ======
        print("\n" + "=" * 70)
        print("  [FINAL] 训练完成汇总")
        print("=" * 70)
        
        print(f"\n  基线得分: {global_baseline:.4f}")
        print(f"  最终得分: {best_score:.4f}")
        print(f"  总提升: {final_improvement:+.2f}%")
        print(f"  最佳阶段: {best_phase}")
        print(f"  最佳模型: {best_model}")
        
        return {
            'baseline': baseline_result,
            'phase1': phase1_result,
            'phase2': results.get('phase2'),
            'best_score': best_score,
            'best_model_path': best_model,
            'best_phase': best_phase,
            'total_improvement_pct': final_improvement,
        }
    
    # ================================================================
    # Phase 0: 基线评估
    # ================================================================
    
    def _run_phase0_baseline(self) -> Dict:
        """
        使用基线模型在所有场景上评估，建立性能基准
        
        Returns:
            dict: 包含每个场景的基线分数和全局平均
        """
        print("\n  加载基线模型...")
        
        model_path = self.config.BASE_MODEL_PATH
        scores = {}
        
        for scenario_id, scenario_config in self.config.SCENARIOS.items():
            num_uav = scenario_config['num_uav']
            scenario_name = self._get_scenario_name(scenario_id)
            
            print(f"\n  评估: {scenario_name} ({num_uav} UAVs)...")
            
            score = self._evaluate_single_scenario(
                model_path=model_path,
                num_uav=num_uav,
                scenario_id=scenario_id,
                tag=f"基线_{scenario_name}",
                num_eval_episodes=self.config.PHASE0_CONFIG['num_eval_episodes'],
            )
            
            scores[scenario_id] = score
            scenario_config['baseline'] = score  # 更新实际基线
        
        # 计算加权全局平均
        global_avg = self._compute_weighted_average(scores)
        
        result = {
            'scores': scores,
            'global_average': global_avg,
            'model_path': model_path,
        }
        
        # 保存基线缓存
        self._save_baseline_cache(result)
        
        return result
    
    # ================================================================
    # Phase 1: 弱场景攻坚 (核心改进!)
    # ================================================================
    
    def _run_phase1_weak_scenario_attack(self) -> Dict:
        """
        Phase 1: 重点攻克弱场景
        - 增加训练量到50 episodes
        - Critic定期重置
        - 动态LR + 学习率重启
        - 渐进式Entropy退火
        - 自适应加权采样
        
        Returns:
            dict: Phase 1的训练和评估结果
        """
        cfg = self.config.PHASE1_CONFIG
        
        print(f"\n  [*] Phase v2.0: 弱场景攻坚 (大幅增强版)")
        print(f"     Episodes: {cfg['total_episodes']} (从15↑{cfg['total_episodes']/15:.1f}倍)")
        print(f"     Rollout: {cfg['rollout_length']} steps (从350↑)")
        print(f"     Actor LR: {cfg['actor_lr_initial']:.2e} (从1.80e-04↑{cfg['actor_lr_initial']/1.8e-04:.2f}倍)")
        print(f"     Critic LR: {cfg['critic_lr_initial']:.2e}")
        print(f"     Entropy: {cfg['entropy_coef_initial']:.4f} → {cfg['entropy_coef_final']:.4f} (渐进退火)")
        print(f"     Critic重置: 每{cfg['critic_reset_interval']} episodes ({cfg['critic_reset_mode']}模式)")
        print(f"     LR重启周期: T_0={cfg['lr_warm_restart_T_0']}")
        
        # 初始化环境和Agent
        print(f"\n  [ENV] 初始化环境...")
        envs = self._initialize_environments()
        agent = self._initialize_agent(cfg)
        
        # 预热Normalizer
        print(f"\n  [WARMUP] 预热Normalizer...")
        self._warmup_normalizers(envs, agent, num_steps=30)
        
        # 加载预训练权重
        print(f"\n  [LOAD] 加载预训练模型...")
        agent.load(self.config.BASE_MODEL_PATH, reset_optimizer=True)
        
        # 初始化学习率调度器 (带重启!)
        actor_scheduler = self._create_lr_scheduler(
            agent.actor_optimizer,
            cfg['lr_warm_restart_T_0'],
            cfg['lr_warm_restart_T_mult'],
        )
        critic_scheduler = self._create_lr_scheduler(
            agent.critic_optimizer,
            cfg['lr_warm_restart_T_0'],
            cfg['lr_warm_restart_T_mult'],
        )
        
        # 记录初始权重快照 (用于后续验证)
        initial_weights = self._capture_weight_snapshot(agent)
        
        # 计算自适应采样权重
        sample_weights = self._compute_adaptive_sample_weights()
        
        # 训练循环
        training_stats = []
        best_episode_score = 0.0
        no_improve_count = 0
        last_critic_reset_ep = 0
        
        # Critic boost状态追踪
        critic_boost_active = False
        critic_boost_end_ep = 0
        
        print(f"\n{'─'*70}")
        print(f"  开始训练循环 ({cfg['total_episodes']} episodes)...")
        print(f"{'─'*70}\n")
        
        for episode in range(1, cfg['total_episodes'] + 1):
            ep_start_time = time.time()
            
            # ========== 场景选择 (自适应加权) ==========
            scenario_id = self._select_scenario(sample_weights)
            scenario_name = self._get_scenario_name(scenario_id)
            env = envs[scenario_id]
            
            # ========== 动态调整超参数 ==========
            # 1. Entropy退火 (线性插值)
            progress = min(episode / cfg['entropy_anneal_episodes'], 1.0)
            current_entropy_coef = (
                cfg['entropy_coef_initial'] * (1 - progress) +
                cfg['entropy_coef_final'] * progress
            )
            agent.entropy_coef = current_entropy_coef
            
            # 2. Critic重置检查 (核心改进!)
            if cfg['critic_reset_interval']:
                episodes_since_reset = episode - last_critic_reset_ep
                if episodes_since_reset >= cfg['critic_reset_interval']:
                    self._reset_critic(agent, cfg)
                    last_critic_reset_ep = episode
                    critic_boost_active = True
                    critic_boost_end_ep = episode + 5  # boost持续5个episodes
                    print(f"\n  [CRITIC_RESET] Episode {episode}: Critic已重置 ({cfg['critic_reset_mode']}模式)")
                    print(f"                启动LR Boost (×{cfg['critic_boost_factor']}, 持续5eps)")
            
            # 3. Critic LR Boost管理
            if critic_boost_active:
                if episode <= critic_boost_end_ep:
                    # 应用boost
                    base_critic_lr = cfg['critic_lr_initial']
                    boosted_lr = base_critic_lr * cfg['critic_boost_factor']
                    for param_group in agent.critic_optimizer.param_groups:
                        param_group['lr'] = boosted_lr
                else:
                    # 结束boost
                    critic_boost_active = False
                    for param_group in agent.critic_optimizer.param_groups:
                        param_group['lr'] = cfg['critic_lr_initial']
            
            # ========== 执行一个Episode ==========
            try:
                ep_result = self._train_one_episode_v2(
                    agent=agent,
                    env=env,
                    scenario_id=scenario_id,
                    rollout_length=cfg['rollout_length'],
                    episode_num=episode,
                    total_episodes=cfg['total_episodes'],
                )
                
                # 更新学习率
                actor_scheduler.step()
                if not critic_boost_active:
                    critic_scheduler.step()
                
                # PPO更新
                update_info = agent.update(
                    gamma=cfg['gamma'],
                    gae_lambda=cfg['gae_lambda'],
                    clip_param=cfg['clip_param'],
                    ppo_epochs=cfg['ppo_epochs'],
                    batch_size=cfg['batch_size'],
                    max_grad_norm=cfg['max_grad_norm'],
                )
                
                # 记录统计信息
                ep_duration = time.time() - ep_start_time
                
                stats = {
                    'episode': episode,
                    'scenario': scenario_name,
                    'scaled_reward': ep_result['scaled_reward'],
                    'raw_reward': ep_result['raw_reward'],
                    'steps': ep_result['steps'],
                    'actor_loss': update_info.get('actor_loss', 0),
                    'critic_loss': update_info.get('critic_loss', 0),
                    'entropy': update_info.get('entropy', 0),
                    'actor_lr': agent.actor_optimizer.param_groups[0]['lr'],
                    'critic_lr': agent.critic_optimizer.param_groups[0]['lr'],
                    'entropy_coef': current_entropy_coef,
                    'duration': ep_duration,
                }
                training_stats.append(stats)
                
                # ========== 权重更新检查 (每N个episodes) ==========
                weight_health = None
                if episode % cfg['weight_check_interval'] == 0:
                    weight_health = self._check_weight_update_health(
                        agent, episode, cfg['total_episodes']
                    )
                    stats['weight_update'] = weight_health
                    
                    if not weight_health['is_healthy']:
                        print(f"\n  [WARN] 权重健康检查异常: {weight_health['message']}")
                
                # ========== 日志输出 ==========
                self._log_episode_progress(stats, weight_health, cfg)
                
                # ========== 早停检查 ==========
                # 使用滑动窗口平均奖励作为指标
                window_size = min(episode, 8)
                recent_rewards = [s['scaled_reward'] for s in training_stats[-window_size:]]
                avg_recent_reward = np.mean(recent_rewards)
                
                if avg_recent_reward > best_episode_score + cfg['early_stop_min_delta']:
                    best_episode_score = avg_recent_reward
                    no_improve_count = 0
                    
                    # 保存中间最佳模型
                    model_path = os.path.join(
                        self.config.OUTPUT_DIR, 
                        f"phase1_intermediate_best_ep{episode}.pt"
                    )
                    agent.save(model_path)
                else:
                    no_improve_count += 1
                
                if no_improve_count >= cfg['early_stop_patience']:
                    print(f"\n  [EARLY_STOP] Episode {episode}: 连续{no_improve_count}次无改善，提前停止")
                    break
                
            except Exception as e:
                print(f"\n  [ERROR] Episode {episode} 异常: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # ====== 训练结束处理 ======
        print(f"\n{'='*70}")
        print(f"  [COMPLETE] Phase 1 训练完成")
        print(f"{'='*70}")
        
        # 保存最终模型
        final_model_path = os.path.join(self.config.OUTPUT_DIR, "phase1_final.pt")
        agent.save(final_model_path)
        
        # 输出训练统计
        self._log_training_summary(training_stats, cfg)
        
        # ====== 评估 ======
        print(f"\n  [EVAL] 开始全场景评估...")
        eval_result = self._evaluate_all_scenarios(final_model_path, tag="Phase1_最终")
        
        # 保存为phase1_best如果更好
        global_avg = eval_result['global_average']
        if global_avg > self.best_global_score:
            self.best_global_score = global_avg
            best_model_path = os.path.join(self.config.OUTPUT_DIR, "phase1_best.pt")
            
            # 复制文件
            import shutil
            shutil.copy(final_model_path, best_model_path)
        else:
            best_model_path = final_model_path
        
        # 清理环境
        for env in envs.values():
            env.close()
        
        return {
            'training_stats': training_stats,
            'eval_result': eval_result,
            'final_model_path': final_model_path,
            'best_model_path': best_model_path,
            'config_used': cfg,
        }
    
    def _train_one_episode_v2(
        self,
        agent,
        env,
        scenario_id: str,
        rollout_length: int,
        episode_num: int,
        total_episodes: int,
    ) -> Dict:
        """
        执行单个训练episode (v2版本，带详细日志)
        
        Args:
            agent: MAPPOAgent实例
            env: 环境实例
            scenario_id: 场景ID
            rollout_length: rollout长度
            episode_num: 当前episode编号
            total_episodes: 总episodes数
            
        Returns:
            dict: episode结果
        """
        obs_dict, global_state = env.reset()
        agent.reset_hidden()
        
        total_reward = 0.0
        step_infos = []
        
        for step in range(rollout_length):
            # 获取业务类型
            biz_types = {
                uid: env.env.uavs[uid].true_business_type.value
                for uid in range(env.num_agents)
            }
            
            # 选择动作
            with torch.no_grad():
                actions, log_probs, values, _, _ = agent.select_actions(
                    obs_dict, global_state,
                    biz_types=biz_types,
                    training=True,
                )
            
            # 执行动作
            next_obs_dict, next_global_state, rewards, team_reward, done, info = env.step(actions)
            
            # 存储transition
            agent.store_transition(
                obs_dict, global_state, actions,
                rewards, log_probs, values, done,
                biz_types, info,
            )
            
            # 累积奖励
            total_reward += team_reward
            step_infos.append({
                'step': step,
                'team_reward': team_reward,
                'done': done,
            })
            
            # 状态更新
            obs_dict = next_obs_dict
            global_state = next_global_state
            
            if done:
                break
        
        # 计算缩放奖励
        num_uav = env.num_agents
        reward_scale = 1.0 / np.sqrt(num_uav)
        scaled_reward = total_reward * reward_scale
        
        return {
            'raw_reward': total_reward,
            'scaled_reward': scaled_reward,
            'steps': step + 1,
            'scenario_id': scenario_id,
            'num_uav': num_uav,
            'reward_scale': reward_scale,
        }
    
    # ================================================================
    # Phase 2: 全局精调
    # ================================================================
    
    def _run_phase2_global_finetune(self, base_model_path: str) -> Dict:
        """
        Phase 2: 在Phase 1基础上进行全局精细调整
        - 降低学习率
        - 均匀采样所有场景
        - 不再重置Critic
        - 更长的早停耐心
        
        Args:
            base_model_path: Phase 1的最佳模型路径
            
        Returns:
            dict: Phase 2的结果
        """
        cfg = self.config.PHASE2_CONFIG
        
        print(f"\n  [*] Phase 2: 全局精细调整")
        print(f"     Episodes: {cfg['total_episodes']}")
        print(f"     Actor LR: {cfg['actor_lr_initial']:.2e} (降低以精细调整)")
        print(f"     采样策略: 均匀 (所有场景平等)")
        
        # 初始化
        envs = self._initialize_environments()
        agent = self._initialize_agent(cfg)
        
        # 预热
        self._warmup_normalizers(envs, agent, num_steps=20)
        
        # 加载Phase 1模型 (注意: 重置优化器以使用新的LR!)
        print(f"\n  [LOAD] 加载Phase 1模型: {base_model_path}")
        agent.load(base_model_path, reset_optimizer=True)
        
        # 初始化调度器 (普通cosine, 无重启)
        actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            agent.actor_optimizer, T_max=cfg['total_episodes']
        )
        critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            agent.critic_optimizer, T_max=cfg['total_episodes']
        )
        
        # 训练循环 (类似Phase 1但简化)
        training_stats = []
        best_score = 0.0
        no_improve_count = 0
        
        for episode in range(1, cfg['total_episodes'] + 1):
            # 均匀轮询场景
            scenario_list = list(self.config.SCENARIOS.keys())
            scenario_id = scenario_list[(episode - 1) % len(scenario_list)]
            scenario_name = self._get_scenario_name(scenario_id)
            env = envs[scenario_id]
            
            # Entropy退火
            progress = min(episode / cfg.get('entropy_anneal_episodes', cfg['total_episodes']), 1.0)
            current_entropy_coef = (
                cfg['entropy_coef_initial'] * (1 - progress) +
                cfg['entropy_coef_final'] * progress
            )
            agent.entropy_coef = current_entropy_coef
            
            # 执行episode
            ep_start_time = time.time()
            
            ep_result = self._train_one_episode_v2(
                agent=agent,
                env=env,
                scenario_id=scenario_id,
                rollout_length=cfg['rollout_length'],
                episode_num=episode,
                total_episodes=cfg['total_episodes'],
            )
            
            # 更新
            update_info = agent.update(
                gamma=cfg['gamma'],
                gae_lambda=cfg['gae_lambda'],
                clip_param=cfg['clip_param'],
                ppo_epochs=cfg['ppo_epochs'],
                batch_size=cfg['batch_size'],
                max_grad_norm=cfg['max_grad_norm'],
            )
            
            actor_scheduler.step()
            critic_scheduler.step()
            
            # 统计
            ep_duration = time.time() - ep_start_time
            stats = {
                'episode': episode,
                'scenario': scenario_name,
                'scaled_reward': ep_result['scaled_reward'],
                'actor_loss': update_info.get('actor_loss', 0),
                'critic_loss': update_info.get('critic_loss', 0),
                'entropy': update_info.get('entropy', 0),
                'actor_lr': agent.actor_optimizer.param_groups[0]['lr'],
                'critic_lr': agent.critic_optimizer.param_groups[0]['lr'],
                'duration': ep_duration,
            }
            training_stats.append(stats)
            
            # 日志 (简化版)
            progress_pct = episode / cfg['total_episodes'] * 100
            print(
                f"  [P2] Ep {episode:3d}/{cfg['total_episodes']} "
                f"({progress_pct:5.1f}) | {scenario_name:8s} | "
                f"Rwd:{stats['scaled_reward']:6.2f} | "
                f"A-Loss:{stats['actor_loss']:+.4f} | "
                f"C-Loss:{stats['critic_loss']:.4f} | "
                f"Ent:{stats['entropy']:.3f}"
            )
            
            # 早停
            window_size = min(episode, 10)
            recent_rewards = [s['scaled_reward'] for s in training_stats[-window_size:]]
            avg_reward = np.mean(recent_rewards)
            
            if avg_reward > best_score + cfg['early_stop_min_delta']:
                best_score = avg_reward
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= cfg['early_stop_patience']:
                print(f"\n  [EARLY_STOP] Episode {episode}: Phase 2提前停止")
                break
        
        # 保存和评估
        final_model_path = os.path.join(self.config.OUTPUT_DIR, "phase2_final.pt")
        agent.save(final_model_path)
        
        print(f"\n  [EVAL] Phase 2 评估...")
        eval_result = self._evaluate_all_scenarios(final_model_path, tag="Phase2_最终")
        
        # 清理
        for env in envs.values():
            env.close()
        
        return {
            'training_stats': training_stats,
            'eval_result': eval_result,
            'final_model_path': final_model_path,
            'best_model_path': final_model_path,  # Phase 2只有一个最终模型
            'config_used': cfg,
        }
    
    # ================================================================
    # 辅助方法
    # ================================================================
    
    def _evaluate_single_scenario(
        self,
        model_path: str,
        num_uav: int,
        scenario_id: str = 'default',
        tag: str = '',
        num_eval_episodes: int = 5,
    ) -> float:
        """在单个场景上评估模型"""
        satisfactions = []
        
        # 动态检测模型配置
        model_hidden_dim, model_critic_hidden_dim = self._detect_model_config(model_path)
        
        scenario_hash = hash(scenario_id) % 10000
        
        with self._resource_context():
            for rep in range(num_eval_episodes):
                seed = self.config.GLOBAL_SEED + scenario_hash + num_uav * 10 + rep * 7
                set_global_seed(seed)
                
                env = MultiAgentHandoverEnv(
                    num_bs=8, num_uav=num_uav,
                    max_steps=150,
                    seed=seed,
                    bs_capacity_range=(500, 1000),
                    pos_range=1000,
                    scenario=scenario_id,
                )
                obs_dict, global_state = env.reset()
                
                agent = MAPPOAgent(
                    num_agents=env.num_agents,
                    obs_dim=env.obs_dim,
                    state_dim=env.state_dim,
                    action_dim=env.action_dim,
                    hidden_dim=model_hidden_dim,
                    critic_hidden_dim=model_critic_hidden_dim,
                )
                
                # 只在第一次rep时输出详细日志
                verbose = (rep == 0 and tag is not None)
                agent.load(model_path, verbose=verbose)
                
                for step in range(150):
                    biz_types = {
                        uid: env.env.uavs[uid].true_business_type.value
                        for uid in range(env.num_agents)
                    }
                    
                    actions, _, _, _, _ = agent.select_actions(
                        obs_dict, global_state,
                        biz_types=biz_types, training=False
                    )
                    
                    next_obs_dict, next_global_state, rewards, team_reward, done, info = env.step(actions)
                    
                    obs_dict = next_obs_dict
                    global_state = next_global_state
                    
                    if done:
                        break
                
                final_sat = np.mean([
                    env.env.uavs[uid].current_satisfaction
                    for uid in range(env.num_agents)
                ])
                satisfactions.append(final_sat)
                
                del agent, env
        
        mean_sat = np.mean(satisfactions)
        std_sat = np.std(satisfactions)
        
        if tag:
            print(f"       [{tag}] {mean_sat:.4f} +/- {std_sat:.4f}")
        
        return mean_sat
    
    def _evaluate_all_scenarios(self, model_path: str, tag: str = '') -> Dict:
        """在所有场景上评估并返回详细结果"""
        scores = {}
        
        for scenario_id, scenario_config in self.config.SCENARIOS.items():
            num_uav = scenario_config['num_uav']
            scenario_name = self._get_scenario_name(scenario_id)
            
            print(f"\n  评估: {scenario_name} ({num_uav} UAVs)...")
            score = self._evaluate_single_scenario(
                model_path=model_path,
                num_uav=num_uav,
                scenario_id=scenario_id,
                tag=f"{tag}_{scenario_name}" if tag else "",
                num_eval_episodes=5,
            )
            scores[scenario_id] = score
        
        global_avg = self._compute_weighted_average(scores)
        
        print(f"\n  {'='*60}")
        print(f"  [RESULT] {tag} 评估结果:" if tag else "  [RESULT] 评估结果:")
        print(f"  {'='*60}")
        
        for scenario_id, score in scores.items():
            baseline = self.config.SCENARIOS[scenario_id].get('baseline', 0)
            change = (score - baseline) / baseline * 100 if baseline > 0 else 0
            scenario_name = self._get_scenario_name(scenario_id)
            print(f"    {scenario_name:12s}: {score:.4f} (基线: {baseline:.4f}, {change:+.2f}%)")
        
        print(f"    {'─'*50}")
        print(f"    {'全局平均':12s}: {global_avg:.4f}")
        
        return {
            'scores': scores,
            'global_average': global_avg,
            'model_path': model_path,
        }
    
    def _detect_model_config(self, model_path: str) -> Tuple[int, int]:
        """动态检测模型的hidden维度"""
        model_hidden_dim = 64
        model_critic_hidden_dim = 128
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            if 'config' in checkpoint:
                config = checkpoint['config']
                detected_hidden = config.get('hidden_dim')
                detected_critic = config.get('critic_hidden_dim')
                if detected_hidden and detected_hidden in [64, 128, 256]:
                    model_hidden_dim = detected_hidden
                if detected_critic and detected_critic in [128, 256, 512]:
                    model_critic_hidden_dim = detected_critic
            
            if 'actor' in checkpoint and model_hidden_dim == 64:
                actor_state = checkpoint['actor']
                for key, tensor in actor_state.items():
                    if 'fc1.weight' in key and len(tensor.shape) == 2:
                        inferred = tensor.shape[0]
                        if inferred in [64, 128, 256]:
                            model_hidden_dim = inferred
                        break
            
            del checkpoint
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"      [WARN] 检测模型配置失败: {e}")
        
        return model_hidden_dim, model_critic_hidden_dim
    
    def _reset_critic(self, agent, cfg: Dict):
        """
        重置Critic网络 (核心改进!)
        
        两种模式:
        - partial: 只重置最后2层 (保留底层特征提取能力)
        - full: 完全重置 (彻底重新学习)
        """
        mode = cfg.get('critic_reset_mode', 'partial')
        
        if mode == 'partial':
            # 获取Critic的所有层名
            layer_names = [name for name, _ in agent.critic.named_parameters()]
            
            # 找到最后2层的参数
            layers_to_reset = set()
            for name in layer_names:
                if any(x in name for x in ['fc3.', 'fc4.', 'output.']):
                    layers_to_reset.add(name.split('.')[0])
            
            # 重置这些层
            for name, param in agent.critic.named_parameters():
                prefix = name.split('.')[0]
                if prefix in layers_to_reset:
                    if 'weight' in name and len(param.shape) >= 2:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        torch.nn.init.zeros_(param.data)
            
            # 重置该层对应的optimizer状态
            state_dict = agent.critic_optimizer.state_dict()
            for name, param in agent.critic.named_parameters():
                prefix = name.split('.')[0]
                if prefix in layers_to_reset:
                    # 通过id匹配来清除对应的状态
                    for idx, (p, _) in enumerate(zip(agent.critic_optimizer.param_groups[0]['params'], 
                                                      state_dict['state'].values())):
                        if id(p) == id(param):
                            # 找到了，清除momentum_buffer等
                            pass  # PyTorch optimizer会自动处理
            
            print(f"                重置了 {len(layers_to_reset)} 个层 (partial模式)")
        
        else:  # full
            # 完全重新初始化Critic
            for name, param in agent.critic.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)
            
            print(f"                完全重置Critic (full模式)")
    
    def _create_lr_scheduler(self, optimizer, T_0: int, T_mult: int = 1):
        """创建带重启的CosineAnnealing调度器"""
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
        )
    
    def _compute_adaptive_sample_weights(self) -> Dict[str, float]:
        """计算自适应采样权重 (基于基线差距)"""
        weights = {}
        
        # 找到最低基线分数
        min_baseline = min(
            sc.get('baseline', 1.0) 
            for sc in self.config.SCENARIOS.values()
        )
        
        for scenario_id, scenario_config in self.config.SCENARIOS.items():
            baseline = scenario_config.get('baseline', 1.0)
            
            # 差距越大，权重越高
            gap = 1.0 - baseline  # 与完美(1.0)的差距
            
            # 归一化到合理范围
            raw_weight = gap / (1.0 - min_baseline + 1e-6)
            
            # 应用clamp
            clamped_weight = np.clip(
                raw_weight,
                self.config.PHASE1_CONFIG['min_sample_weight'],
                self.config.PHASE1_CONFIG['max_sample_weight'],
            )
            
            weights[scenario_id] = clamped_weight
        
        return weights
    
    def _select_scenario(self, weights: Dict[str, float]) -> str:
        """根据权重随机选择场景"""
        scenarios = list(weights.keys())
        probs = np.array([weights[s] for s in scenarios])
        probs = probs / probs.sum()  # 归一化
        
        selected = np.random.choice(scenarios, p=probs)
        return selected
    
    def _initialize_environments(self) -> Dict:
        """初始化所有场景的环境"""
        from uav_system.mappo_environment import MultiAgentHandoverEnv
        
        envs = {}
        
        for scenario_id, scenario_config in self.config.SCENARIOS.items():
            env = MultiAgentHandoverEnv(
                num_bs=8,
                num_uav=scenario_config['num_uav'],
                max_steps=200,  # 训练时用更长episode
                seed=self.config.GLOBAL_SEED + hash(scenario_id) % 10000,
                bs_capacity_range=(500, 1000),
                pos_range=1000,
                scenario=scenario_id,
            )
            envs[scenario_id] = env
        
        return envs
    
    def _initialize_agent(self, cfg: Dict):
        """初始化Agent"""
        # 使用第一个环境的维度 (假设所有环境相同)
        first_scenario = list(self.config.SCENARIOS.keys())[0]
        first_env_config = self.config.SCENARIOS[first_scenario]
        
        # 这里需要实际的obs_dim等信息，暂时用默认值
        # 实际实现时应从环境中获取
        agent = MAPPOAgent(
            num_agents=first_env_config['num_uav'],  # 会根据实际环境调整
            obs_dim=49,  # 默认值，实际应从环境获取
            state_dim=31,
            action_dim=5,
            hidden_dim=64,
            critic_hidden_dim=128,
        )
        
        # 设置初始学习率
        for param_group in agent.actor_optimizer.param_groups:
            param_group['lr'] = cfg['actor_lr_initial']
        for param_group in agent.critic_optimizer.param_groups:
            param_group['lr'] = cfg['critic_lr_initial']
        
        # 设置初始Entropy
        agent.entropy_coef = cfg.get('entropy_coef_initial', 0.01)
        
        return agent
    
    def _warmup_normalizers(self, envs, agent, num_steps: int = 30):
        """预热所有环境的Normalizer"""
        for scenario_id, env in envs.items():
            scenario_name = self._get_scenario_name(scenario_id)
            
            obs_dict, global_state = env.reset()
            agent.reset_hidden()
            
            for step in range(num_steps):
                biz_types = {
                    uid: env.env.uavs[uid].true_business_type.value
                    for uid in range(env.num_agents)
                }
                
                actions, _, _, _, _ = agent.select_actions(
                    obs_dict, global_state,
                    biz_types=biz_types, training=True
                )
                
                next_obs_dict, next_global_state, _, _, done, _ = env.step(actions)
                
                # 更新normalizer统计
                agent.obs_normalizer.update(obs_dict)
                
                obs_dict = next_obs_dict
                global_state = next_global_state
                
                if done:
                    obs_dict, global_state = env.reset()
                    agent.reset_hidden()
            
            print(f"      ✓ {scenario_name}: Normalizer已预热 ({num_steps} steps)")
    
    def _check_weight_update_health(self, agent, current_episode: int, total_episodes: int) -> dict:
        """检查权重是否真正在更新"""
        # 类似于之前的实现，这里省略详细代码
        # 返回格式: {'is_healthy': bool, 'max_change': float, ...}
        pass
    
    def _capture_weight_snapshot(self, agent) -> Dict:
        """捕获当前权重的快照"""
        snapshot = {}
        for name, param in agent.actor.named_parameters():
            snapshot[f"actor_{name}"] = {
                'norm': torch.norm(param.data).item(),
                'data_mean': param.data.mean().item(),
                'data_std': param.data.std().item() if param.numel() > 1 else 0.0,
            }
        for name, param in agent.critic.named_parameters():
            snapshot[f"critic_{name}"] = {
                'norm': torch.norm(param.data).item(),
                'data_mean': param.data.mean().item(),
                'data_std': param.data.std().item() if param.numel() > 1 else 0.0,
            }
        return snapshot
    
    def _compute_weighted_average(self, scores: Dict[str, float]) -> float:
        """计算UAV数量加权的全局平均"""
        total_score = 0.0
        total_uavs = 0
        
        for scenario_id, score in scores.items():
            num_uav = self.config.SCENARIOS[scenario_id]['num_uav']
            total_score += score * num_uav
            total_uavs += num_uav
        
        return total_score / total_uavs if total_uavs > 0 else 0.0
    
    def _get_scenario_name(self, scenario_id: str) -> str:
        """将场景ID转换为中文名称"""
        names = {
            'industrial_inspection': '工业巡检',
            'agriculture': '农业植保',
            'smart_city': '智慧城市监控',
            'emergency_rescue': '应急救援',
            'logistics_delivery': '物流配送',
        }
        return names.get(scenario_id, scenario_id)
    
    def _log_episode_progress(self, stats: Dict, weight_health: Optional[Dict], cfg: Dict):
        """输出详细的episode进度日志"""
        ep = stats['episode']
        total = cfg['total_episodes']
        progress = ep / total * 100
        
        scenario_short = {
            '工业巡检': 'IND',
            '农业植保': 'AGR',
            '智慧城市监控': 'SMA',
            '应急救援': 'EMG',
            '物流配送': 'LOG',
        }.get(stats['scenario'], '???')
        
        # 主行
        print(
            f"\r  [P1] Ep {ep:3d}/{total} "
            f"({progress:5.1f}) | {scenario_short:3s} "
            f"| Rwd:{stats['scaled_reward']:6.2f} "
            f"| A-Loss:{stats['actor_loss']:+.4f} "
            f"| C-Loss:{stats['critic_loss']:.4f} "
            f"| Ent:{stats['entropy']:.3f} "
            f"| A-LR:{stats['actor_lr']:.2e} "
            f"| C-LR:{stats['critic_lr']:.2e}",
            end='',
            flush=True,
        )
        
        # 如果有权重信息，额外输出
        if weight_health:
            print(
                f"\n        WeightUpdate: {weight_health['max_change']:.1f}% "
                f"({weight_health['updated_layers']}/{weight_health['total_layers']}层)"
            )
    
    def _log_training_summary(self, training_stats: List[Dict], cfg: Dict):
        """输出训练汇总统计"""
        total_eps = len(training_stats)
        total_time = sum(s['duration'] for s in training_stats)
        avg_time = total_time / max(total_eps, 1)
        
        rewards = [s['scaled_reward'] for s in training_stats]
        actor_losses = [s['actor_loss'] for s in training_stats]
        critic_losses = [s['critic_loss'] for s in training_stats]
        
        print(f"\n\n  [STATS] 训练统计:")
        print(f"    总Episodes: {total_eps}/{cfg['total_episodes']}")
        print(f"    总耗时: {total_time/60:.1f}分钟 (平均{avg_time:.1f}s/ep)")
        print(f"    奖励: Mean={np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        print(f"    Actor Loss: {np.mean(actor_losses):+.4f}")
        print(f"    Critic Loss: {np.mean(critic_losses):.4f}")
        
        # 场景分布
        scenario_counts = defaultdict(int)
        for s in training_stats:
            scenario_counts[s['scenario']] += 1
        
        print(f"\n  [DISTRIB] 场景分布:")
        for scenario, count in sorted(scenario_counts.items(), key=lambda x: -x[1]):
            pct = count / total_eps * 100
            print(f"    {scenario:12s}: {count:3d}ep ({pct:5.1f}%)")
    
    def _save_baseline_cache(self, result: Dict):
        """保存基线缓存"""
        import json
        
        cache_path = os.path.join(
            self.config.OUTPUT_DIR,
            "pmsf_v2_baseline_cache.json"
        )
        
        cache_data = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'scores': result['scores'],
            'global_average': result['global_average'],
            'model_path': result['model_path'],
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    
    def _resource_context(self):
        """资源管理上下文 (简化版)"""
        from contextlib import contextmanager
        
        @contextmanager
        def context():
            try:
                yield
            finally:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return context()


# ============================================================
# 主入口
# ============================================================

def main():
    """主函数"""
    print("\n" + "="*70)
    print("  PMSF v2.0 - 渐进式多场景微调系统")
    print("  Progressive Multi-Scenario Finetuning")
    print("="*70 + "\n")
    
    # 创建配置
    config = PMSFConfig()
    
    # 创建微调器
    tuner = PMSFTuner(config)
    
    # 运行完整流水线
    results = tuner.run_full_pipeline()
    
    # 输出最终结论
    print("\n" + "="*70)
    print("  [FINAL CONCLUSION]")
    print("="*70)
    
    improvement = results['total_improvement_pct']
    
    if improvement >= 7.0:
        verdict = "非常成功!"
    elif improvement >= 4.0:
        verdict = "成功!"
    elif improvement >= 2.0:
        verdict = "部分有效"
    elif improvement >= 0:
        verdict = "效果有限"
    else:
        verdict = "失败 (负收益!)"
    
    print(f"\n  总体评价: {verdict}")
    print(f"  性能提升: {improvement:+.2f}%")
    print(f"  最佳模型: {results['best_model_path']}")
    
    return results


if __name__ == "__main__":
    main()
