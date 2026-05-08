"""
快速诊断脚本: 对比基线模型 vs Phase 1 模型的权重差异

用途:
1. 检查两个模型的权重norm
2. 对比参数差异
3. 判断Phase 1是否真的更新了权重

运行方式:
    python compare_model_weights.py
    
预计耗时: <10秒
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def analyze_model(model_path: str, label: str) -> dict:
    """分析单个模型的权重"""
    print(f"\n{'='*70}")
    print(f"[ANALYZE] {label}")
    print(f"{'='*70}")
    print(f"  文件: {os.path.basename(model_path)}")
    print(f"  大小: {os.path.getsize(model_path)/1024:.1f} KB")
    
    try:
        # [FIX] PyTorch 2.6+ 兼容性
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location='cpu')
        
        info = {
            'path': model_path,
            'label': label,
            'keys': list(checkpoint.keys()),
        }
        
        if 'actor' in checkpoint:
            actor_state = checkpoint['actor']
            
            # 统计信息
            total_params = sum(p.numel() for p in actor_state.values())
            total_norm = 0.0
            layer_norms = {}
            
            for name, param in actor_state.items():
                param_norm = torch.norm(param).item()
                total_norm += param_norm ** 2
                
                # 提取层名
                layer_name = name.split('.')[0] if '.' in name else name
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = []
                layer_norms[layer_name].append((name, param_norm, param.shape))
            
            total_norm = total_norm ** 0.5
            
            print(f"\n  [ACTOR] 权重统计:")
            print(f"     层数: {len(actor_state)}")
            print(f"     总参数量: {total_params:,}")
            print(f"     整体Frobenius范数: {total_norm:.4f}")
            
            # 显示每层的norm
            print(f"\n     各层权重详情:")
            for layer_name, layers in sorted(layer_norms.items()):
                for name, norm, shape in layers[:3]:  # 每层最多显示3个
                    print(f"       [{name}] norm={norm:>10.4f} | shape={str(shape):>20s}")
                
            info['actor'] = {
                'num_layers': len(actor_state),
                'total_params': total_params,
                'total_norm': total_norm,
                'layer_norms': {k: [(n, s) for n, _, s in v] for k, v in layer_norms.items()},
            }
            
            # [KEY] 特别关注第一层的norm
            first_fc_norm = None
            for name, param in actor_state.items():
                if ('fc1.weight' in name or 'layers.0.weight' in name) and len(param.shape) == 2:
                    first_fc_norm = torch.norm(param).item()
                    print(f"\n     [KEY] 第一层全连接权重norm: {first_fc_norm:.4f}")
                    info['actor']['first_layer_norm'] = first_fc_norm
                    break
        
        if 'critic' in checkpoint:
            critic_state = checkpoint['critic']
            total_params = sum(p.numel() for p in critic_state.values())
            total_norm = sum(torch.norm(p).item()**2 for p in critic_state.values()) ** 0.5
            
            print(f"\n  [CRITIC] 权重统计:")
            print(f"     层数: {len(critic_state)}")
            print(f"     总参数量: {total_params:,}")
            print(f"     整体Frobenius范数: {total_norm:.4f}")
            
            info['critic'] = {
                'num_layers': len(critic_state),
                'total_params': total_params,
                'total_norm': total_norm,
            }
        
        del checkpoint
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return info
        
    except Exception as e:
        print(f"\n  [ERROR] 分析失败: {e}")
        return {'error': str(e)}


def compare_models(base_info: dict, phase1_info: dict):
    """对比两个模型"""
    print(f"\n\n{'='*70}")
    print(f"[COMPARE] 基线模型 vs Phase 1 模型 对比")
    print(f"{'='*70}")
    
    # 对比Actor
    if 'actor' in base_info and 'actor' in phase1_info:
        base_actor = base_info['actor']
        phase1_actor = phase1_info['actor']
        
        print(f"\n  [ACTOR 对比]")
        print(f"     {'指标':<25s} │ {'基线模型':>15s} │ {'Phase 1模型':>15s} │ {'变化':>10s}")
        print(f"     {'-'*25}-┼-{'-'*15}-┼-{'-'*15}-┼-{'-'*10}-")
        
        # 参数数量
        print(f"     {'总参数量':<25s} │ {base_actor['total_params']:>15,d} │ {phase1_actor['total_params']:>15,d} │", end='')
        param_diff = phase1_actor['total_params'] - base_actor['total_params']
        if param_diff != 0:
            print(f" {param_diff:+d} ⚠️")
        else:
            print(f" {'相同':>10s}")
        
        # 整体norm
        print(f"     {'整体Frobenius范数':<25s} │ {base_actor['total_norm']:>15.4f} │ {phase1_actor['total_norm']:>15.4f}", end='')
        norm_change = (phase1_actor['total_norm'] - base_actor['total_norm']) / max(base_actor['total_norm'], 1e-10) * 100
        print(f" │ {norm_change:>+9.2f}%")
        
        # 第一层norm
        if 'first_layer_norm' in base_actor and 'first_layer_norm' in phase1_actor:
            base_first = base_actor['first_layer_norm']
            phase1_first = phase1_actor['first_layer_norm']
            print(f"     {'第一层FC权重norm':<25s} │ {base_first:>15.4f} │ {phase1_first:>15.4f}", end='')
            first_change = (phase1_first - base_first) / max(base_first, 1e-10) * 100
            print(f" │ {first_change:>+9.2f}%")
            
            # [KEY] 判断是否有实质性更新
            if abs(first_change) < 1.0:
                print(f"\n     [🚨 致命发现] 第一层权重几乎没变 ({first_change:+.2f}%)!")
                print(f"              这意味着 Phase 1 训练可能没有真正更新Actor权重!")
            elif abs(first_change) < 10.0:
                print(f"\n     [⚠️ 警告] 第一层权重变化较小 ({first_change:+.2f}%)")
                print(f"              可能训练不足或学习率过小")
            else:
                print(f"\n     [✅ 正常] 第一层权重有明显变化 ({first_change:+.2f}%)")
    
    # 对比Critic
    if 'critic' in base_info and 'critic' in phase1_info:
        base_critic = base_info['critic']
        phase1_critic = phase1_info['critic']
        
        print(f"\n  [CRITIC 对比]")
        print(f"     {'指标':<25s} │ {'基线模型':>15s} │ {'Phase 1模型':>15s} │ {'变化':>10s}")
        print(f"     {'-'*25}-┼-{'-'*15}-┼-{'-'*15}-┼-{'-'*10}-")
        
        print(f"     {'总参数量':<25s} │ {base_critic['total_params']:>15,d} │ {phase1_critic['total_params']:>15,d}", end='')
        param_diff = phase1_critic['total_params'] - base_critic['total_params']
        if param_diff != 0:
            print(f" {param_diff:+d} ⚠️")
        else:
            print(f" {'相同':>10s}")
        
        print(f"     {'整体Frobenius范数':<25s} │ {base_critic['total_norm']:>15.4f} │ {phase1_critic['total_norm']:>15.4f}", end='')
        norm_change = (phase1_critic['total_norm'] - base_critic['total_norm']) / max(base_critic['total_norm'], 1e-10) * 100
        print(f" │ {norm_change:>+9.2f}%")
    
    print(f"\n{'='*70}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" " * 25 + "模型权重对比工具")
    print(" " * 15 + "(基线模型 vs Phase 1 模型)")
    print("="*70)
    
    # 模型路径
    model_dir = r"experiment_results\mappo_models\finetune_multi_v2"
    baseline_path = os.path.join(
        r"experiment_results\mappo_models",
        "mappo_8bs_300uav_best.pt"
    )
    phase1_path = os.path.join(model_dir, "phase1_r1.pt")
    
    # 检查文件
    for path, label in [(baseline_path, "基线模型"), (phase1_path, "Phase 1模型")]:
        if not os.path.exists(path):
            print(f"\n[ERROR] {label}不存在: {path}")
            return
        print(f"\n[OK] 找到{label}: {os.path.basename(path)} ({os.path.getsize(path)/1024:.1f} KB)")
    
    # 分析两个模型
    base_info = analyze_model(baseline_path, "基线模型 (mappo_8bs_300uav_best.pt)")
    phase1_info = analyze_model(phase1_path, "Phase 1 模型 (phase1_r1.pt)")
    
    # 对比
    if 'error' not in base_info and 'error' not in phase1_info:
        compare_models(base_info, phase1_info)
        
        # 最终判断
        print(f"\n[CONCLUSION]")
        if 'actor' in phase1_info and 'first_layer_norm' in phase1_info.get('actor', {}):
            # 这里会在compare_models中输出详细判断
            pass
        else:
            print(f"  无法自动判断，请查看上述详细数据")
    
    print(f"\n")


if __name__ == "__main__":
    main()
