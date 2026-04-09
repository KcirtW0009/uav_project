# -*- coding: utf-8 -*-
"""
指标数据可视化与报告导出模块

提供以下功能：
1. 指标数据的可视化展示
2. 自定义时间区间的性能趋势分析
3. 报告导出（PDF、HTML、CSV）
4. 多算法对比分析
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class PerformanceVisualizer:
    """
    性能指标可视化器
    """
    
    def __init__(self, output_dir: str = 'visualizations'):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics_data = {}
    
    def add_metrics(self, algorithm_name: str, metrics: List[Dict]):
        """
        添加算法的指标数据
        
        Args:
            algorithm_name: 算法名称
            metrics: 指标数据列表，每个元素是一个步骤或 episode 的指标
        """
        self.metrics_data[algorithm_name] = metrics
    
    def plot_time_series(self, metric_name: str, title: str, ylabel: str,
                       start_time: Optional[int] = None, end_time: Optional[int] = None,
                       save_path: Optional[str] = None):
        """
        绘制时间序列图
        
        Args:
            metric_name: 指标名称
            title: 图表标题
            ylabel: Y轴标签
            start_time: 开始时间步
            end_time: 结束时间步
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 6))
        
        for algorithm, metrics in self.metrics_data.items():
            # 提取时间序列数据
            steps = []
            values = []
            
            for i, metric in enumerate(metrics):
                if start_time is not None and i < start_time:
                    continue
                if end_time is not None and i >= end_time:
                    break
                
                if metric_name in metric:
                    steps.append(i)
                    values.append(metric[metric_name])
            
            if values:
                plt.plot(steps, values, label=algorithm, linewidth=2)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('时间步', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {full_path}")
        
        plt.show()
    
    def plot_comparison_bar(self, metric_name: str, title: str, ylabel: str,
                          save_path: Optional[str] = None):
        """
        绘制对比柱状图
        
        Args:
            metric_name: 指标名称
            title: 图表标题
            ylabel: Y轴标签
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 6))
        
        algorithms = []
        values = []
        std_values = []
        
        for algorithm, metrics in self.metrics_data.items():
            # 提取指标值
            metric_values = [m[metric_name] for m in metrics if metric_name in m]
            if metric_values:
                algorithms.append(algorithm)
                values.append(np.mean(metric_values))
                std_values.append(np.std(metric_values))
        
        if values:
            bars = plt.bar(algorithms, values, yerr=std_values, capsize=5)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {full_path}")
        
        plt.show()
    
    def plot_heatmap(self, metrics: List[str], title: str, save_path: Optional[str] = None):
        """
        绘制热力图
        
        Args:
            metrics: 指标列表
            title: 图表标题
            save_path: 保存路径
        """
        # 准备数据
        data = {}
        for algorithm, algorithm_metrics in self.metrics_data.items():
            data[algorithm] = {}
            for metric in metrics:
                metric_values = [m[metric] for m in algorithm_metrics if metric in m]
                if metric_values:
                    data[algorithm][metric] = np.mean(metric_values)
                else:
                    data[algorithm][metric] = 0
        
        # 转换为DataFrame
        df = pd.DataFrame(data).T
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='.4f', linewidths=0.5)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {full_path}")
        
        plt.show()
    
    def export_report(self, report_name: str, format: str = 'html'):
        """
        导出报告
        
        Args:
            report_name: 报告名称
            format: 导出格式 ('html', 'csv', 'json')
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'html':
            self._export_html_report(report_name, timestamp)
        elif format == 'csv':
            self._export_csv_report(report_name, timestamp)
        elif format == 'json':
            self._export_json_report(report_name, timestamp)
        else:
            print(f"不支持的格式: {format}")
    
    def _export_html_report(self, report_name: str, timestamp: str):
        """
        导出HTML报告
        """
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metric-summary {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f0f8ff;
            border-radius: 5px;
        }}
        .algorithm-section {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .timestamp {{
            color: #666;
            font-size: 12px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report_name}</h1>
        <div class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <h2>1. 算法概览</h2>
        <table>
            <tr>
                <th>算法名称</th>
                <th>数据点数量</th>
                <th>平均满意度</th>
                <th>平均切换次数</th>
                <th>平均延迟(ms)</th>
                <th>平均丢包率</th>
                <th>平均吞吐量(Mbps)</th>
                <th>平均带宽利用率</th>
            </tr>
        """
        
        # 添加算法概览数据
        for algorithm, metrics in self.metrics_data.items():
            if metrics:
                avg_satisfaction = np.mean([m.get('satisfaction', 0) for m in metrics])
                avg_handovers = np.mean([m.get('handover_count', 0) for m in metrics])
                avg_latency = np.mean([m.get('avg_latency', 0) for m in metrics])
                avg_packet_loss = np.mean([m.get('avg_packet_loss', 0) for m in metrics])
                avg_throughput = np.mean([m.get('avg_throughput', 0) for m in metrics])
                avg_bandwidth = np.mean([m.get('bandwidth_utilization', 0) for m in metrics])
                
                html_content += f"""
            <tr>
                <td>{algorithm}</td>
                <td>{len(metrics)}</td>
                <td>{avg_satisfaction:.4f}</td>
                <td>{avg_handovers:.1f}</td>
                <td>{avg_latency:.4f}</td>
                <td>{avg_packet_loss:.4f}</td>
                <td>{avg_throughput:.2f}</td>
                <td>{avg_bandwidth:.4f}</td>
            </tr>
                """
        
        html_content += f"""
        </table>
        
        <h2>2. 详细指标分析</h2>
        """
        
        # 添加详细指标分析
        for algorithm, metrics in self.metrics_data.items():
            if metrics:
                html_content += f"""
        <div class="algorithm-section">
            <h3>{algorithm}</h3>
            <div class="metric-summary">
                <h4>关键指标摘要</h4>
                <ul>
                    <li>满意度: {np.mean([m.get('satisfaction', 0) for m in metrics]):.4f} ± {np.std([m.get('satisfaction', 0) for m in metrics]):.4f}</li>
                    <li>切换次数: {np.mean([m.get('handover_count', 0) for m in metrics]):.1f} ± {np.std([m.get('handover_count', 0) for m in metrics]):.1f}</li>
                    <li>网络延迟: {np.mean([m.get('avg_latency', 0) for m in metrics]):.4f} ± {np.std([m.get('avg_latency', 0) for m in metrics]):.4f} ms</li>
                    <li>丢包率: {np.mean([m.get('avg_packet_loss', 0) for m in metrics]):.4f} ± {np.std([m.get('avg_packet_loss', 0) for m in metrics]):.4f}</li>
                    <li>吞吐量: {np.mean([m.get('avg_throughput', 0) for m in metrics]):.2f} ± {np.std([m.get('avg_throughput', 0) for m in metrics]):.2f} Mbps</li>
                    <li>带宽利用率: {np.mean([m.get('bandwidth_utilization', 0) for m in metrics]):.4f} ± {np.std([m.get('bandwidth_utilization', 0) for m in metrics]):.4f}</li>
                </ul>
            </div>
        </div>
                """
        
        html_content += f"""
    </div>
</body>
</html>
        """
        
        # 保存HTML文件
        file_path = os.path.join(self.output_dir, f"{report_name}_{timestamp}.html")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已保存到: {file_path}")
    
    def _export_csv_report(self, report_name: str, timestamp: str):
        """
        导出CSV报告
        """
        # 准备数据
        all_data = []
        for algorithm, metrics in self.metrics_data.items():
            for i, metric in enumerate(metrics):
                row = {'algorithm': algorithm, 'step': i}
                row.update(metric)
                all_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(all_data)
        
        # 保存CSV文件
        file_path = os.path.join(self.output_dir, f"{report_name}_{timestamp}.csv")
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        print(f"CSV报告已保存到: {file_path}")
    
    def _export_json_report(self, report_name: str, timestamp: str):
        """
        导出JSON报告
        """
        # 准备数据
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'algorithms': {}
        }
        
        for algorithm, metrics in self.metrics_data.items():
            report_data['algorithms'][algorithm] = {
                'metrics': metrics,
                'summary': {
                    'avg_satisfaction': np.mean([m.get('satisfaction', 0) for m in metrics]),
                    'avg_handovers': np.mean([m.get('handover_count', 0) for m in metrics]),
                    'avg_latency': np.mean([m.get('avg_latency', 0) for m in metrics]),
                    'avg_packet_loss': np.mean([m.get('avg_packet_loss', 0) for m in metrics]),
                    'avg_throughput': np.mean([m.get('avg_throughput', 0) for m in metrics]),
                    'avg_bandwidth_utilization': np.mean([m.get('bandwidth_utilization', 0) for m in metrics]),
                }
            }
        
        # 保存JSON文件
        file_path = os.path.join(self.output_dir, f"{report_name}_{timestamp}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"JSON报告已保存到: {file_path}")


def create_performance_report(experiment_results: Dict[str, List[Dict]], 
                           report_name: str = 'performance_report',
                           output_dir: str = 'reports'):
    """
    创建性能报告
    
    Args:
        experiment_results: 实验结果，键为算法名称，值为指标数据列表
        report_name: 报告名称
        output_dir: 输出目录
    """
    # 创建可视化器
    visualizer = PerformanceVisualizer(output_dir)
    
    # 添加数据
    for algorithm, metrics in experiment_results.items():
        visualizer.add_metrics(algorithm, metrics)
    
    # 生成可视化图表
    print("生成可视化图表...")
    
    # 满意度时间序列
    visualizer.plot_time_series(
        'satisfaction',
        '满意度趋势',
        '满意度',
        save_path='satisfaction_trend.png'
    )
    
    # 切换次数时间序列
    visualizer.plot_time_series(
        'handover_count',
        '切换次数趋势',
        '切换次数',
        save_path='handover_trend.png'
    )
    
    # 延迟时间序列
    visualizer.plot_time_series(
        'avg_latency',
        '网络延迟趋势',
        '延迟 (ms)',
        save_path='latency_trend.png'
    )
    
    # 吞吐量时间序列
    visualizer.plot_time_series(
        'avg_throughput',
        '吞吐量趋势',
        '吞吐量 (Mbps)',
        save_path='throughput_trend.png'
    )
    
    # 丢包率时间序列
    visualizer.plot_time_series(
        'avg_packet_loss',
        '丢包率趋势',
        '丢包率',
        save_path='packet_loss_trend.png'
    )
    
    # 带宽利用率时间序列
    visualizer.plot_time_series(
        'bandwidth_utilization',
        '带宽利用率趋势',
        '带宽利用率',
        save_path='bandwidth_trend.png'
    )
    
    # 算法对比柱状图
    visualizer.plot_comparison_bar(
        'satisfaction',
        '算法满意度对比',
        '平均满意度',
        save_path='satisfaction_comparison.png'
    )
    
    visualizer.plot_comparison_bar(
        'handover_count',
        '算法切换次数对比',
        '平均切换次数',
        save_path='handover_comparison.png'
    )
    
    visualizer.plot_comparison_bar(
        'avg_latency',
        '算法网络延迟对比',
        '平均延迟 (ms)',
        save_path='latency_comparison.png'
    )
    
    # 热力图
    visualizer.plot_heatmap(
        ['satisfaction', 'handover_count', 'avg_latency', 'avg_packet_loss', 'avg_throughput', 'bandwidth_utilization'],
        '算法性能热力图',
        save_path='performance_heatmap.png'
    )
    
    # 导出报告
    print("导出报告...")
    visualizer.export_report(report_name, format='html')
    visualizer.export_report(report_name, format='csv')
    visualizer.export_report(report_name, format='json')
    
    print("报告生成完成！")


if __name__ == "__main__":
    # 示例用法
    sample_data = {
        '传统算法': [
            {'step': 0, 'satisfaction': 0.7, 'handover_count': 10, 'avg_latency': 50, 'avg_packet_loss': 0.01, 'avg_throughput': 10, 'bandwidth_utilization': 0.6},
            {'step': 1, 'satisfaction': 0.72, 'handover_count': 12, 'avg_latency': 48, 'avg_packet_loss': 0.009, 'avg_throughput': 10.5, 'bandwidth_utilization': 0.62},
            {'step': 2, 'satisfaction': 0.75, 'handover_count': 8, 'avg_latency': 45, 'avg_packet_loss': 0.008, 'avg_throughput': 11, 'bandwidth_utilization': 0.65},
        ],
        '增强算法': [
            {'step': 0, 'satisfaction': 0.75, 'handover_count': 8, 'avg_latency': 45, 'avg_packet_loss': 0.008, 'avg_throughput': 11, 'bandwidth_utilization': 0.65},
            {'step': 1, 'satisfaction': 0.78, 'handover_count': 6, 'avg_latency': 42, 'avg_packet_loss': 0.007, 'avg_throughput': 11.5, 'bandwidth_utilization': 0.68},
            {'step': 2, 'satisfaction': 0.82, 'handover_count': 5, 'avg_latency': 38, 'avg_packet_loss': 0.006, 'avg_throughput': 12, 'bandwidth_utilization': 0.7},
        ],
        'MAPPO算法': [
            {'step': 0, 'satisfaction': 0.72, 'handover_count': 9, 'avg_latency': 47, 'avg_packet_loss': 0.009, 'avg_throughput': 10.2, 'bandwidth_utilization': 0.63},
            {'step': 1, 'satisfaction': 0.79, 'handover_count': 7, 'avg_latency': 40, 'avg_packet_loss': 0.006, 'avg_throughput': 11.8, 'bandwidth_utilization': 0.69},
            {'step': 2, 'satisfaction': 0.85, 'handover_count': 4, 'avg_latency': 35, 'avg_packet_loss': 0.005, 'avg_throughput': 12.5, 'bandwidth_utilization': 0.72},
        ],
    }
    
    create_performance_report(sample_data, report_name='示例性能报告')