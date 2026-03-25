"""
可视化工具模块

提供数据表格打印、图表绘制、模型可视化等功能。
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Any, Dict
from .config import COLORS, CMAP_PRIMARY, CMAP_SUCCESS, CMAP_WARNING, RESULT_DIR


class VisualizationHelper:
    """可视化辅助工具类"""

    @staticmethod
    def create_gradient_bar(ax, x, heights, width=0.6, cmap=CMAP_PRIMARY, alpha=0.8):
        """创建渐变色柱状图"""
        colors = cmap(np.linspace(0.3, 0.9, len(x)))
        bars = ax.bar(x, heights, width, color=colors, alpha=alpha, edgecolor='white', linewidth=1.5)
        for bar, height in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        return bars

    @staticmethod
    def create_comparison_bars(ax, labels, values1, values2, label1, label2,
                               cmap1=CMAP_PRIMARY, cmap2=CMAP_SUCCESS):
        """创建对比柱状图"""
        x = np.arange(len(labels))
        width = 0.35
        colors1 = cmap1(np.linspace(0.4, 0.8, len(labels)))
        colors2 = cmap2(np.linspace(0.4, 0.8, len(labels)))
        bars1 = ax.bar(x - width / 2, values1, width, label=label1, color=colors1, edgecolor='white', linewidth=1.5)
        bars2 = ax.bar(x + width / 2, values2, width, label=label2, color=colors2, edgecolor='white', linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        return bars1, bars2

    @staticmethod
    def add_value_labels(ax, bars, fmt='{:.2f}'):
        """在柱状图上添加数值标签"""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    fmt.format(height), ha='center', va='bottom', fontsize=8)

    @staticmethod
    def create_filled_line_plot(ax, x, y, label, color, alpha=0.3):
        """创建带填充区域的折线图"""
        ax.plot(x, y, label=label, color=color, linewidth=2)
        ax.fill_between(x, y, alpha=alpha, color=color)

    @staticmethod
    def print_data_table(title: str, headers: List[str], rows: List[List[Any]],
                         col_widths: List[int] = None):
        """打印格式化的数据表格"""
        print("\n" + "=" * 80)
        print(f"[数据表] {title}")
        print("=" * 80)
        if col_widths is None:
            col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
                          for i, h in enumerate(headers)]
        header_line = "|".join(f"{h:^{w}}" for h, w in zip(headers, col_widths))
        print("|" + header_line + "|")
        print("|" + "|".join("-" * w for w in col_widths) + "|")
        for row in rows:
            row_line = "|".join(f"{str(v):^{w}}" for v, w in zip(row, col_widths))
            print("|" + row_line + "|")
        print("=" * 80)

    @staticmethod
    def save_results_to_csv(filename: str, data: Dict[str, Any]):
        """将结果保存为CSV文件"""
        filepath = os.path.join(RESULT_DIR, filename)
        pd.DataFrame([data]).to_csv(filepath, index=False)


class RecognitionModelVisualizer:
    """业务识别模型可视化"""

    @staticmethod
    def visualize_model(model, all_model_results=None, save_dir=RESULT_DIR, show=False):
        """
        生成业务识别模型可视化图表

        Args:
            model: 训练好的模型
            all_model_results: 所有模型的对比结果列表
            save_dir: 保存目录
            show: 是否显示图形窗口
        """
        print("\n生成业务识别模型可视化...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('业务识别模型分析', fontsize=14, fontweight='bold')

        # 图1: 模型性能对比
        ax = axes[0, 0]
        if all_model_results:
            model_types = [r['type'].upper() for r in all_model_results]
            combined_scores = [r['combined_score'] for r in all_model_results]
            colors = [COLORS['warning'] if r['type'] == model.model_type else COLORS['primary']
                      for r in all_model_results]
            bars = ax.bar(model_types, combined_scores, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
            ax.set_ylim(0, max(combined_scores) * 1.1)
            ax.set_ylabel('综合得分')
            ax.set_title('各模型性能对比（多目标优化）', fontweight='bold')
            ax.axhline(y=max(combined_scores), color=COLORS['warning'], linestyle='--', linewidth=1, alpha=0.5)
            for bar, score in zip(bars, combined_scores):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(facecolor=COLORS['warning'], edgecolor='white', label=f'选中模型 ({model.model_type.upper()})'),
                Patch(facecolor=COLORS['primary'], edgecolor='white', label='其他模型')
            ], loc='upper right', fontsize=8)

        # 图2: 混淆矩阵
        ax = axes[0, 1]
        cm = model.model_info.get('confusion_matrix')
        if cm is not None:
            cm = np.array(cm)
            im = ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(len(cm)))
            ax.set_yticks(range(len(cm)))
            type_labels = ['控制信令', '视频回传', '环境监测']
            ax.set_xticklabels(type_labels, rotation=30, ha='right')
            ax.set_yticklabels(type_labels)
            ax.set_xlabel('预测标签')
            ax.set_ylabel('真实标签')
            ax.set_title('混淆矩阵', fontweight='bold')
            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='white' if cm[i, j] > cm.max() / 2 else 'black',
                            fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax)

        # 图3: 性能指标
        ax = axes[1, 0]
        metrics_names = ['准确率', 'F1分数', '交叉验证均值']
        metrics_keys = ['accuracy', 'f1_score', 'cross_val_mean']
        values = [model.model_info.get(m, 0) for m in metrics_keys]
        colors = [COLORS['primary'], COLORS['success'], COLORS['info']]
        bars = ax.bar(metrics_names, values, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('数值')
        ax.set_title('模型性能指标', fontweight='bold')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 图4: 模型信息
        ax = axes[1, 1]
        ax.axis('off')
        info_text = (
            "【模型信息】\n\n"
            f"模型类型: {model.model_info.get('model_type', 'N/A')}\n"
            f"模型版本: {model.model_info.get('version', 'N/A')}\n"
            f"训练时间: {model.model_info.get('training_timestamp', 'N/A')[:19]}\n\n"
            f"【性能指标】\n"
            f"准确率: {model.model_info.get('accuracy', 0) * 100:.2f}%\n"
            f"F1分数: {model.model_info.get('f1_score', 0):.3f}\n"
            f"交叉验证: {model.model_info.get('cross_val_mean', 0) * 100:.2f}%\n"
            f"推理延迟: {model.model_info.get('inference_latency_ms', 0):.3f} ms\n"
            f"训练时间: {model.model_info.get('training_time_s', 0):.2f} s\n"
        )
        ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_visualization.png'), dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f"模型可视化已保存到 {save_dir}/model_visualization.png")
