#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
curriculum_learning.py 全面代码审查报告
==========================================

审查日期: 2026-05-09
审查范围: 日志系统、潜在警告、硬编码问题
审查结果: 发现X个问题，其中P0级Y个需立即修复

==================================================
"""

# ============================================================
# 问题清单 (按严重程度排序)
# ============================================================

ISSUES_FOUND = {
    'P0_CRITICAL': [
        {
            'id': 'LOG-001',
            'location': 'Line 680 (已修复)',
            'type': 'Format Error',
            'description': 'ValueError: Invalid format specifier \'+\' for float',
            'impact': '程序无法启动',
            'status': 'FIXED',
            'fix': '使用条件符号: sign = \'+\' if improvement >= 0 else \'\'',
        },
    ],
    
    'P1_WARNING': [
        {
            'id': 'HARD-001',
            'location': 'Lines 1210-1211, 422, 146, 1260',
            'type': 'Hardcoded Values',
            'description': '多处魔法数字: hidden_dim=64, critic_hidden_dim=128, '
                         'warmup_steps=30, eval_steps=150, embedding_dim=64',
            'impact': '可维护性差，不同环境可能需要不同值',
            'status': 'NEEDS_FIX',
            'recommendation': '提取到CurriculumConfig配置类中',
        },
        {
            'id': 'WARN-001',
            'location': 'Line 1208',
            'type': 'Magic Number',
            'description': 'weight = 1.0 / (scfg.baseline_score + 0.1) 中的0.1',
            'impact': '语义不明，难以调整',
            'status': 'NEEDS_FIX',
            'recommendation': '定义为常量 BASELINE_EPSILON = 0.1 并添加注释',
        },
        {
            'id': 'WARN-002',
            'location': 'Line 1412',
            'type': 'Magic Number',
            'description': 'improvement < 0.001 中的阈值0.001',
            'impact': '早停敏感度不可调',
            'status': 'NEEDS_FIX',
            'recommendation': '添加到phase_config中: min_improvement_threshold',
        },
    ],
    
    'P2_IMPROVEMENT': [
        {
            'id': 'LOG-002',
            'location': 'Throughout file',
            'type': 'Logging System',
            'description': '缺少结构化日志系统，只有print语句',
            'impact': '难以过滤和分析日志，不支持输出到文件',
            'status': 'OPTIONAL',
            'recommendation': '添加logging模块支持或自定义Logger类',
        },
        {
            'id': 'SAFE-001',
            'location': 'Multiple locations',
            'type': 'Defensive Programming',
            'description': '部分字典访问缺少默认值保护',
            'impact': '可能的KeyError',
            'status': 'OPTIONAL',
            'recommendation': '统一使用 .get(key, default) 方法',
        },
        {
            'id': 'PERF-001',
            'location': 'Line 1347-1375 (_train_one_episode)',
            'type': 'Performance',
            'description': '每个episode都进行完整的PPO更新循环',
            'impact': '训练速度可能较慢',
            'status': 'OPTIONAL',
            'recommendation': '考虑累积多个episode后再更新（如果显存允许）',
        },
    ],
}

# ============================================================
# 修复方案实施
# ============================================================

def generate_fixes():
    """生成所有需要的代码修复"""
    
    fixes = []
    
    # Fix 1: 将硬编码值提取到配置类
    fix_config = '''
@dataclass
class CurriculumConfig:
    """课程学习配置 - 增强版"""
    
    # ====== 新增: 模型架构参数 ======
    default_hidden_dim: int = 64           # 默认Actor隐藏层维度
    default_critic_hidden_dim: int = 128   # 默认Critic隐藏层维度
    default_obs_dim: int = 49              # 默认观测维度
    default_state_dim: int = 31            # 默认状态维度
    default_action_dim: int = 3            # 默认动作维度
    
    # ====== 新增: 训练过程参数 ======
    warmup_steps: int = 30                 # Normalizer预热步数
    quick_eval_steps: int = 150            # 快速评估步数
    full_eval_steps: int = 350             # 完整评估步数
    rollout_length: int = 500              # Rollout长度
    
    # ====== 新增: 算法超参数 ======
    baseline_epsilon: float = 0.1          # 基线分数的平滑因子
    min_improvement_threshold: float = 0.001  # 早停最小改进阈值
    weight_clamp_min: float = 0.3          # 采样权重下界
    weight_clamp_max: float = 3.5          # 采样权重上界
    
    # ... 其他现有配置 ...
'''
    fixes.append(('CONFIG_ENHANCEMENT', fix_config))
    
    # Fix 2: 改进_detect_model_config方法
    fix_detect = '''
def _detect_model_config(self) -> Tuple[int, int]:
    """检测模型的hidden_dim配置 (增强版)"""
    # 从配置获取默认值 (而非硬编码)
    model_hidden_dim = self.config.default_hidden_dim          # 64
    model_critic_hidden_dim = self.config.default_critic_hidden_dim  # 128
    
    try:
        checkpoint = torch.load(self.base_model_path, map_location='cpu', 
                               weights_only=False)
        
        if 'config' in checkpoint:
            cfg = checkpoint['config']
            model_hidden_dim = cfg.get('hidden_dim', model_hidden_dim)
            model_critic_hidden_dim = cfg.get('critic_hidden_dim', 
                                              model_critic_hidden_dim)
            print(f"       [DETECT] 从模型config读取: "
                  f"hidden={model_hidden_dim}, critic_hidden={model_critic_hidden_dim}")
        else:
            # 通过权重大小推断
            if 'actor' in checkpoint:
                for key, tensor in checkpoint['actor'].items():
                    if 'fc1.weight' in key:
                        inferred = tensor.shape[0]
                        if inferred in [32, 64, 128, 256]:
                            model_hidden_dim = inferred
                            print(f"       [DETECT] 推断actor hidden_dim={inferred}")
                        break
            
            if 'critic' in checkpoint:
                for key, tensor in checkpoint['critic'].items():
                    if 'fc1.weight' in key:
                        inferred = tensor.shape[0]
                        if inferred in [64, 128, 256, 512]:
                            model_critic_hidden_dim = inferred
                            print(f"       [DETECT] 推断critic hidden_dim={inferred}")
                        break
        
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"       [WARN] 无法检测模型配置: {e}")
        print(f"             使用默认值: actor_hidden={model_hidden_dim}, "
              f"critic_hidden={model_critic_hidden_dim}")
    
    return model_hidden_dim, model_critic_hidden_dim
'''
    fixes.append(('DETECT_ENHANCEMENT', fix_detect))
    
    # Fix 3: 改进场景选择算法
    fix_selection = '''
def _select_scenario(self, available_scenarios: List[str], 
                    phase_config: Dict) -> str:
    """
    智能场景选择 (增强版 - 使用配置参数)
    
    改进点:
    - 使用配置中的clamp参数而非硬编码
    - 添加空列表保护
    - 更清晰的权重计算逻辑
    """
    if not available_scenarios:
        raise ValueError("available_scenarios不能为空")
    
    priority = phase_config.get('priority', 'balance')
    
    # 获取配置参数
    eps = self.config.baseline_epsilon  # 0.1 (替代硬编码的0.1)
    w_min = self.config.weight_clamp_min  # 0.3
    w_max = self.config.weight_clamp_max  # 3.5
    
    if priority == 'breakthrough':
        weights = []
        for sid in available_scenarios:
            scfg = self.scenarios[sid]
            gap = scfg.target_score - scfg.baseline_score
            # 差距越大权重越高，使用eps避免除零
            base_weight = 1.0 + gap * 3
            weight = max(w_min, min(w_max, base_weight))
            weights.append(weight)
            
    elif priority == 'maintain':
        weights = [1.0] * len(available_scenarios)
        
    else:  # balance
        weights = []
        for sid in available_scenarios:
            scfg = self.scenarios[sid]
            # 使用eps平滑基线分数，避免极端权重
            weight = 1.0 / (scfg.baseline_score + eps)
            weight = max(w_min, min(w_max, weight))
            weights.append(weight)
    
    # 归一化 (防止浮点误差导致总和不为1)
    total_weight = sum(weights)
    if total_weight <= 0:
        # 极端情况: 均匀分布
        probs = [1.0 / len(weights)] * len(weights)
    else:
        probs = [w / total_weight for w in weights]
    
    # 采样
    chosen_idx = np.random.choice(len(available_scenarios), p=probs)
    return available_scenarios[chosen_idx]
'''
    fixes.append(('SELECTION_ENHANCEMENT', fix_selection))
    
    # Fix 4: 改进早停检查
    fix_earlystop = '''
def _check_early_stop(self, scores: List[float], phase_config: Dict) -> bool:
    """
    检查是否触发早停 (增强版)
    
    改进点:
    - 使用配置中的阈值
    - 添加边界检查
    - 更详细的日志
    """
    patience = phase_config.get('early_stop_patience', 10)
    threshold = phase_config.get('min_improvement_threshold', 
                                 self.config.min_improvement_threshold)  # 0.001
    
    if not scores or len(scores) < patience:
        return False
    
    recent = scores[-patience:]
    improvement = recent[-1] - recent[0]
    
    should_stop = improvement < threshold
    
    if should_stop and len(scores) % 5 == 0:  # 每5次检查打印一次
        print(f"       [EARLY_STOP_CHECK] 最近{patience}次评估: "
              f"{recent[0]:.4f} → {recent[-1]:.4f} "
              f"(改进={improvement:+.4f}, 阈值={threshold})")
    
    return should_stop
'''
    fixes.append(('EARLYSTOP_ENHANCEMENT', fix_earlystop))
    
    # Fix 5: 添加简单的日志系统
    fix_logging = '''
class SimpleLogger:
    """
    简单日志系统 (替代纯print)
    
    功能:
    - 支持多级别日志 (DEBUG/INFO/WARN/ERROR)
    - 同时输出到控制台和文件
    - 支持时间戳和调用位置
    """
    
    def __init__(self, log_file: str, console_level: str = 'INFO',
                 file_level: str = 'DEBUG'):
        self.log_file = log_file
        self.console_level = getattr(logging, console_level.upper())
        self.file_level = getattr(logging, file_level.upper())
        
        # 配置logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)-5s] %(message)s',
            datefmt='%H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, encoding='utf-8'),
            ]
        )
        self.logger = logging.getLogger('PMSFv3')
        
        # 避免重复日志
        self.logger.propagate = False
    
    def debug(self, msg): self.logger.debug(msg)
    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
'''
    fixes.append(('LOGGING_SYSTEM', fix_logging))
    
    return fixes


# ============================================================
# 应用修复到实际代码
# ============================================================

def apply_fixes_to_curriculum_learning():
    """
    将所有修复应用到 curriculum_learning.py
    
    注意: 这是一个指导性函数，实际修改需要手动编辑文件
    或使用SearchReplace工具逐个应用
    """
    
    print("="*70)
    print("  curriculum_learning.py 代码修复指南")
    print("="*70)
    
    fixes = generate_fixes()
    
    for i, (name, code) in enumerate(fixes, 1):
        print(f"\n{'─'*70}")
        print(f"  Fix #{i}: {name}")
        print(f"{'─'*70}")
        print(code)
        print()
    
    print("\n" + "="*70)
    print("  修复优先级建议")
    print("="*70)
    print("""
  必须立即修复 (P0):
    ✓ LOG-001: 格式化错误 (已完成)

  强烈推荐修复 (P1):
    □ HARD-001: 提取硬编码值到配置类
    □ WARN-001: 定义baseline_epsilon常量  
    □ WARN-002: 提取min_improvement_threshold

  可选改进 (P2):
    □ LOG-002: 添加结构化日志系统
    □ SAFE-001: 加强字典访问保护
    □ PERF-001: 优化训练性能

  实施顺序建议:
    1. 先应用 HARD-001 (影响最大)
    2. 再应用 WARN-001 和 WARN-002 (快速修复)
    3. 最后考虑 P2 改进 (锦上添花)
""")


if __name__ == '__main__':
    apply_fixes_to_curriculum_learning()
