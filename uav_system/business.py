"""
=============================================================================
  UAV业务识别与切换决策系统 - 业务类型定义模块 (business.py)
=============================================================================

【模块概述】
本模块是整个系统的"业务建模层"，定义了UAV网络中的三种典型业务类型、
它们的QoS（服务质量）需求配置，以及用于生成仿真数据的特征参数。

【设计哲学】

1. **3GPP 5G切片对齐**:
   业务类型直接对应5G的三大应用场景:
   - 控制信令 → URLLC (超可靠低延迟通信)
   - 视频回传 → eMBB (增强型移动宽带)
   - 环境监测 → mMTC (海量机器类通信)
   
   确保研究成果具有实际工程参考价值。

2. **数据驱动的QoS参数**:
   所有QoS阈值均来自权威来源:
   - 华为5G白皮书 (2024)
   - 3GPP TS 22.125 (5G服务要求)
   - 学术论文KPI基准表
   
   避免主观臆断，提升可信度。

3. **差异化降级策略**:
   不同业务对资源不足的容忍度不同:
   - 控制信令: 仅允许5%降级(安全关键)
   - 视频回传: 允许40%降级(质量可调)
   - 环境监测: 允许70%降级(尽力而为)
   
   为增强算法的抢占/降级机制提供决策依据。

4. **可扩展的枚举设计**:
   使用Python Enum确保类型安全
   支持未来扩展新业务类型(如AR/VR、边缘计算等)
   集中管理避免魔法数字散落各处

【核心组件】
┌─────────────────────────────────────────────────────────────────────┐
│ 组件/类              │ 功能描述                                     │
├─────────────────────────────────────────────────────────────────────┤
│ BusinessType         │ 业务类型枚举(控制信令/视频/监测)             │
│ QoSProfile           │ QoS配置数据类(速率/时延/丢包/优先级等)       │
│ QOS_PROFILES         │ 全局预定义配置字典(BusinessType → QoSProfile)│
│ BUSINESS_FEATURE_..  │ 业务特征生成参数(用于模拟流量)               │
└─────────────────────────────────────────────────────────────────────┘

【三种业务类型详解】

1. **控制信令** (CONTROL_SIGNAL, ID=0)
   
   ┌─────────────────────┬─────────────────────────────────────────────┐
 │ 特性                 │ 详细说明                                    │
 ├─────────────────────┼─────────────────────────────────────────────┤
 │ 典型应用            │ 遥控指令、状态上报、告警推送、飞行控制       │
 │ 5G映射              │ URLLC (Ultra-Reliable Low Latency Commun.)   │
 │ 带宽需求            │ 极低: 0.15-0.5 Mbps (150-500 kbps)          │
 │ 时延要求            │ 极严格: ≤20 ms (理想<10 ms)                  │
 │ 可靠性要求          │ 极高: 丢包率≤1% (99.999%可靠性)             │
 │ 优先级              │ 最高: 0.99 (接近1.0)                         │
 │ 关键性              │ 安全关键: criticality=1.0                    │
 │ 时延敏感度          │ 最高: latency_sensitivity=1.0               │
 │ 降级容忍度          │ 最低: 仅允许5%降级(95%-100%)                │
 │ 抖动容忍度          │ 极低: <2 ms                                  │
 │ 典型包大小          │ 小: 64-256 bytes (指令/ACK)                  │
 │ 发送模式            │ 突发性、低频、小包                           │
 └─────────────────────┴─────────────────────────────────────────────┘
 
 设计考量:
 - 无人机遥控指令丢失可能导致坠机，因此采用最严格的QoS保障
 - 即使在资源紧张时，也要优先保证控制信令的完整性
 - 切换算法应避免让控制信令UAV经历长时间断连

2. **视频回传** (VIDEO_STREAMING, ID=1)
 
   ┌─────────────────────┬─────────────────────────────────────────────┐
 │ 特性                 │ 详细说明                                    │
 ├─────────────────────┼─────────────────────────────────────────────┤
 │ 典型应用            │ 4K视频流、实时监控、AR/VR传输、图像识别     │
 │ 5G映射              │ eMBB (Enhanced Mobile Broadband)             │
 │ 带宽需求            │ 高: 25-100 Mbps (4K≈50Mbps, 8K≈200Mbps)    │
 │ 时延要求            │ 中等: ≤20-50 ms (实时交互要求)              │
 │ 可靠性要求          │ 中高: 丢包率≤5% (可接受偶尔卡顿)            │
 │ 优先级              │ 高: 0.75                                    │
 │ 关键性              │ 体验关键: criticality=0.7                   │
 │ 时延敏感度          │ 中高: latency_sensitivity=0.8               │
 │ 降级容忍度          │ 中等: 允许35%降级(65%-100%, 可降分辨率)     │
 │ 抖动容忍度          │ 低: <10 ms (影响画面流畅度)                 │
 │ 典型包大小          │ 大: 1400-1500 bytes (MTU大小)               │
 │ 发送模式            │ 持续性、高频、大包、恒定比特率(CBR)        │
 └─────────────────────┴─────────────────────────────────────────────┘
 
 设计考量:
 - 视频业务占用大量带宽，是系统负载的主要来源
 - 可以通过降低分辨率/帧率来适应带宽限制(自适应码率)
 - 对时延敏感但不如控制信令严格(人类感知有容限)
 - 是切换决策的重点优化对象(带宽敏感)

3. **环境监测** (ENVIRONMENT_MONITORING, ID=2)
 
   ┌─────────────────────┬─────────────────────────────────────────────┐
 │ 特性                 │ 详细说明                                    │
 ├─────────────────────┼─────────────────────────────────────────────┤
 │ 典型应用            │ 传感器数据采集、周期性巡检、日志上传、      │
 │                     │ 温湿度/气体浓度上报                        │
 │ 5G映射              │ mMTC (Massive Machine-Type Communications)  │
 │ 带宽需求            │ 低: 0.5-2 Mbps (传感器数据通常很小)         │
 │ 时延要求            │ 宽松: ≤500-1000 ms (非实时)                 │
 │ 可靠性要求          │ 中等: 丢包率≤5% (可重传)                    │
 │ 优先级              │ 低: 0.30                                    │
 │ 关键性              │ 非关键: criticality=0.3                      │
 │ 时延敏感度          │ 低: latency_sensitivity=0.2                 │
 │ 降级容忍度          │ 高: 允许75%降级(25%-100%, 尽力而为)         │
 │ 抖动容忍度          │ 高: <70 ms (批量数据传输)                   │
 │ 典型包大小          │ 中: 256-512 bytes (传感器读数)              │
 │ 发送模式            │ 周期性、低频、中包                          │
 └─────────────────────┴─────────────────────────────────────────────┘
 
 设计考量:
 - 环境监测数据量小但设备数量可能很大(mMTC场景)
 - 对时延和可靠性要求最低，可作为"缓冲池"吸收系统压力
 - 在资源竞争时可以优先牺牲环境监测的QoS
 - 适合作为被抢占对象(低优先级+高降级容忍)

【QoSProfile数据类详解】

属性列表:

┌──────────────────────┬──────────┬────────────────────────────────────────┐
│ 属性名                │ 类型      │ 说明                                │
├──────────────────────┼──────────┼────────────────────────────────────────┤
│ business_type        │ Enum      │ 所属业务类型                        │
│ min_rate             │ float     │ 最低保障速率(Mbps), 低于此值视为不满足│
│ ideal_rate           │ float     │ 理想目标速率(Mbps), 满分参考标准     │
│ max_delay            │ float     │ 最大容忍时延(ms), 超过则惩罚        │
│ max_loss_rate        │ float     │ 最大容忍丢包率(0-1), 超过则惩罚     │
│ priority             │ float     │ 业务优先级(0-1), 用于加权计算        │
│ downgrade_tolerance  │ float     │ 降级容忍度(0-1), 最小可接受比例      │
│ criticality          │ float     │ 关键性等级(0-1), ≥0.9触发特殊处理   │
│ latency_sensitivity  │ float     │ 时延敏感系数(0-1), 影响权重分配      │
└──────────────────────┴──────────┴────────────────────────────────────────┘

方法说明:

1. **calculate_satisfaction()** - 多维度满意度计算
 
 输入参数:
   - allocated_rate: 实际分配速率(Mbps)
   - estimated_delay: 估算时延(ms), 可选
   - loss_rate: 丢包率(0-1), 可选
 
 计算流程:
   a. 速率满意度(rate_sat):
      使用分段线性函数(按业务类型定制)
      - 控制信令: 三段式 [0, 0.7→0.3, 0.85→0.7, 0.95→1.0]
      - 视频回传: S形曲线(Smoothstep函数) + 线性段
      - 环境监测: 两段式 [0, 0.3→0.4, 0.8→1.0]
 
   b. 时延满意度(delay_sat):
      反比关系: delay_sat = max_delay / actual_delay
      截断到[0, 1]范围，分段线性化
 
   c. 丢包满意度(loss_sat):
      类似时延: loss_sat = max_loss / actual_loss
      更宽松的阈值(≥0.3即给0.5分)
 
   d. 加权综合:
      动态权重(同satisfaction.py模块):
      overall = w_rate × rate_sat + w_delay × delay_sat + w_loss × loss_sat
      
      惩罚机制: 任一维度<0.2时，overall×0.5(严厉惩罚)
 
 向后兼容:
   若delay/loss未提供，退化为纯速率评估(兼容旧接口)

2. **get_feasible_downgrade_ratios()** - 可行降级比例列表
 
 返回该业务类型可接受的降级序列(降序排列):
 - 控制信令: [1.0, 0.95, 0.9]     (仅3档,保守)
 - 视频回传: [1.0, 0.9, 0.8, 0.7, 0.6] (5档,中等)
 - 环境监测: [1.0, 0.8, 0.6, 0.4, 0.3] (5档,激进)
 
 用途:
 - EnhancedHandoverAlgorithm的降级搜索
 - 抢占时的最小保留资源计算
 - 资源分配的可行性判断

【全局配置字典】

QOS_PROFILES:
  类型: Dict[BusinessType, QoSProfile]
  
  用法:
  >>> from business import QOS_PROFILES, BusinessType
  >>> control_qos = QOS_PROFILES[BusinessType.CONTROL_SIGNAL]
  >>> print(f"最大时延: {control_qos.max_delay}ms")
  
  注意: 这是单例字典，程序启动时初始化，不应修改

BUSINESS_FEATURE_PARAMS:
  类型: Dict[BusinessType, Dict[str, tuple]]
  
  结构:
  {
    BusinessType.X: {
      'delay': (mean, std),          # 时延分布参数(正态分布)
      'bandwidth': (mean, std),      # 带宽分布参数(Mbps)
      'loss_beta': (alpha, beta),     # Beta分布形状参数(丢包率)
      'loss_scale': float,            # Beta分布尺度因子
      'jitter': (mean, std)           # 抖动分布参数(ms)
    }
  }
  
  用途:
  - recognition模块生成训练/测试数据
  - environment模块模拟UAV流量特征
  - MAPPO环境的观测空间构建
  
  参数选择依据:
  - delay/bandwidth: 基于QOS_PROFILES的理想值加合理波动
  - loss_beta: Beta分布适合模拟[0,1]区间的比率数据
    * α大β小 → 分布偏向0(低丢包, 如控制信令α=1, β=1000)
    * αβ接近均匀 → 中等丢包(如环境监测α=2, β=20)

【与其他模块的关系】

上游依赖:
  无(本模块是最底层的业务定义层)

下游调用:
  - satisfaction.py: 使用QOS_PROFILES计算满意度
  - recognition.py: 使用BUSINESS_FEATURE_PARAMS生成训练数据
  - algorithms.py: 使用priority/criticality进行切换决策
  - environment.py: 使用min_rate/ideal_rate初始化UAV
  - mappo_environment.py: 使用QoS需求构建奖励函数
  - experiments.py: 按BusinessType分组统计性能指标

【使用示例】

# 示例1: 查询业务类型的QoS需求
>>> from business import BusinessType, QOS_PROFILES
>>> video_qos = QOS_PROFILES[BusinessType.VIDEO_STREAMING]
>>> print(f"视频业务理想速率: {video_qos.ideal_rate} Mbps")
>>> print(f"最大容忍时延: {video_qos.max_delay} ms")

# 示例2: 计算特定分配下的满意度
>>> control_qos = QOS_PROFILES[BusinessType.CONTROL_SIGNAL]
>>> sat = control_qos.calculate_satisfaction(
...     allocated_rate=0.45,      # 分配450kbps
...     estimated_delay=15,      # 时延15ms
...     loss_rate=0.005          # 丢包率0.5%
... )
>>> print(f"控制信令满意度: {sat:.1%}")

# 示例3: 获取可行的降级选项
>>> for ratio in video_qos.get_feasible_downgrade_ratios():
...     rate = video_qos.ideal_rate * ratio
...     print(f"  {ratio*100:.0f}%: {rate:.1f} Mbps")

# 示例4: 生成模拟的业务特征数据
>>> from business import BUSINESS_FEATURE_PARAMS
>>> import numpy as np
>>> params = BUSINESS_FEATURE_PARAMS[BusinessType.VIDEO_STREAMING]
>>> delay = np.random.normal(*params['delay'])
>>> bandwidth = np.random.normal(*params['bandwidth'])
>>> print(f"模拟视频业务: 延迟={delay:.1f}ms, 带宽={bandwidth:.1f}Mbps")

# 示例5: 遍历所有业务类型
>>> for bt in BusinessType:
...     qos = QOS_PROFILES[bt]
...     print(f"{bt.name}: 优先级={qos.priority}, 关键性={qos.criticality}")

【已知限制】
  1. 固定3种业务类型，扩展需修改多处代码(Enum/配置/训练)
  2. QoS参数基于行业平均值，未考虑具体厂商实现差异
  3. 降级比例为离散档位，不支持连续细粒度调整
  4. calculate_satisfaction()中的分段函数需手动调优(非学习得到)
  5. 不支持动态QoS协商(如用户根据价格自选质量等级)

【版本历史】
  V1.0: 初始版本，定义3种业务类型和基础QoS配置
  V1.1: 添加QoSProfile数据类和calculate_satisfaction方法
  V1.2: 引入get_feasible_downgrade_ratios降级策略
  V1.3: 添加BUSINESS_FEATURE_PARAMS用于数据生成
  V1.4: 完善多维度满意度计算(时延/丢包加权)
  V1.5: 对齐3GPP 5G三大场景(URLLC/eMBB/mMTC)

【参考文献】
  1. Huawei: "5G Application Scenario White Paper" (2024)
  2. 3GPP TS 22.125: "Service requirements for the 5G system"
  3. ITU-T Y.3100: "Framework for supporting UAV-based applications"
  4. 3GPP TR 22.862: "Study on enhancement of Cyber Physical Mobile Robotics applications"
"""

from enum import Enum
from dataclasses import dataclass
from typing import List
import numpy as np


class BusinessType(Enum):
    """无人机业务类型枚举"""
    CONTROL_SIGNAL = 0           # 控制信令：高优先级，低时延，低带宽
    VIDEO_STREAMING = 1          # 视频回传：中优先级，高带宽，中时延
    ENVIRONMENT_MONITORING = 2   # 环境监测：低优先级，低带宽，高时延容忍


@dataclass
class QoSProfile:
    """
    QoS配置文件

    Attributes:
        business_type: 业务类型
        min_rate: 最低保障速率(Mbps)
        ideal_rate: 理想速率(Mbps)
        max_delay: 最大允许时延(ms)
        max_loss_rate: 最大允许丢包率
        priority: 业务优先级(0-1)
        downgrade_tolerance: 降级容忍度(0-1)
        criticality: 关键程度(0-1)
        latency_sensitivity: 时延敏感度(0-1)
    """
    business_type: BusinessType
    min_rate: float
    ideal_rate: float
    max_delay: float
    max_loss_rate: float
    priority: float
    downgrade_tolerance: float
    criticality: float = 0.5
    latency_sensitivity: float = 0.5

    def calculate_satisfaction(self, allocated_rate: float,
                              estimated_delay: float = None,
                              loss_rate: float = None) -> float:
        """
        根据分配速率、时延、丢包率综合计算满意度

        各业务类型的满意度由三个维度加权合成：
        - 速率满意度：分配速率与理想速率的比值
        - 时延满意度：估计时延与最大允许时延的关系
        - 丢包满意度：丢包率与最大允许丢包率的关系

        权重由 latency_sensitivity（时延敏感度）决定：
        - latency_sensitivity=1.0(控制信令): 时延权重高
        - latency_sensitivity=0.8(视频回传): 时延和丢包都有一定权重
        - latency_sensitivity=0.2(环境监测): 速率权重占主导

        向后兼容：estimated_delay/loss_rate 为 None 时退化为纯速率评估。
        """
        # ========== 速率满意度（保留原有逻辑） ==========
        rate_ratio = allocated_rate / self.ideal_rate

        if self.business_type == BusinessType.CONTROL_SIGNAL:
            if rate_ratio >= 0.95:
                rate_sat = 1.0
            elif rate_ratio >= 0.85:
                rate_sat = 0.7 + 0.3 * (rate_ratio - 0.85) / 0.1
            elif rate_ratio >= 0.7:
                rate_sat = 0.3 + 0.4 * (rate_ratio - 0.7) / 0.15
            else:
                rate_sat = 0.0

        elif self.business_type == BusinessType.VIDEO_STREAMING:
            if rate_ratio >= 0.9:
                rate_sat = 1.0
            elif rate_ratio >= 0.7:
                x = (rate_ratio - 0.7) / 0.2
                rate_sat = 0.5 + 0.5 * (3 * x**2 - 2 * x**3)
            elif rate_ratio >= 0.5:
                rate_sat = 0.2 + 0.3 * (rate_ratio - 0.5) / 0.2
            else:
                rate_sat = max(0.0, rate_ratio / 0.5 * 0.2)

        else:  # ENVIRONMENT_MONITORING
            if rate_ratio >= 0.8:
                rate_sat = 1.0
            elif rate_ratio >= 0.3:
                rate_sat = 0.4 + 0.6 * (rate_ratio - 0.3) / 0.5
            else:
                rate_sat = max(0.0, rate_ratio / 0.3 * 0.4)

        # ========== 向后兼容：无时延/丢包数据时退化为纯速率 ==========
        if estimated_delay is None and loss_rate is None:
            return rate_sat

        # ========== 时延满意度 ==========
        if estimated_delay is not None and self.max_delay > 0:
            delay_ratio = self.max_delay / (estimated_delay + 1e-6)
            if delay_ratio >= 1.0:
                delay_sat = 1.0
            elif delay_ratio >= 0.5:
                delay_sat = 0.5 + 0.5 * (delay_ratio - 0.5) / 0.5
            else:
                delay_sat = max(0.0, delay_ratio)
        else:
            delay_sat = 1.0  # 无时延数据时视为满足

        # ========== 丢包率满意度 ==========
        if loss_rate is not None and self.max_loss_rate > 0:
            loss_ratio = self.max_loss_rate / (loss_rate + 1e-6)
            if loss_ratio >= 1.0:
                loss_sat = 1.0
            elif loss_ratio >= 0.3:
                loss_sat = 0.5 + 0.5 * (loss_ratio - 0.3) / 0.7
            else:
                loss_sat = max(0.0, loss_ratio)
        else:
            loss_sat = 1.0  # 无丢包数据时视为满足

        # ========== 多维度加权综合 ==========
        ls = self.latency_sensitivity
        # 速率权重：时延敏感度越高，速率权重越低（但最低0.4）
        w_rate = max(0.4, 1.0 - ls * 0.4)
        # 时延权重：直接由 latency_sensitivity 决定
        w_delay = ls
        # 丢包权重：时延敏感度越高，丢包权重越高（反映业务对传输质量的要求）
        w_loss = 0.2 + ls * 0.3
        # 归一化
        w_total = w_rate + w_delay + w_loss
        w_rate /= w_total
        w_delay /= w_total
        w_loss /= w_total

        overall = w_rate * rate_sat + w_delay * delay_sat + w_loss * loss_sat

        # 任何维度严重不满足时，整体满意度惩罚
        if rate_sat < 0.2 or delay_sat < 0.2 or loss_sat < 0.2:
            overall *= 0.5

        return np.clip(overall, 0.0, 1.0)

    def get_feasible_downgrade_ratios(self) -> List[float]:
        """
        获取可行的降级比例列表

        控制信令容忍最低降级(5%)，环境监测容忍最大降级(70%)。
        """
        if self.business_type == BusinessType.CONTROL_SIGNAL:
            return [1.0, 0.95, 0.9]
        elif self.business_type == BusinessType.VIDEO_STREAMING:
            return [1.0, 0.9, 0.8, 0.7, 0.6]
        else:
            return [1.0, 0.8, 0.6, 0.4, 0.3]


# ==================== 预定义QoS配置 ====================
# 参数来源：华为白皮书、3GPP TS 22.125、论文KPI表格
# - 控制信令(URlLC): ≈200kbps上行, ≤20ms时延, 99.999%可靠性
# - 视频回传(eMBB): 25-100Mbps带宽, ≈20ms时延, 连续传输高可靠
# - 环境监测(mMTC): ≤1Mbps带宽, <1000ms时延, 容忍一定丢包
QOS_PROFILES = {
    BusinessType.CONTROL_SIGNAL: QoSProfile(
        business_type=BusinessType.CONTROL_SIGNAL,
        min_rate=0.15, ideal_rate=0.5, max_delay=20, max_loss_rate=0.01,
        priority=0.99, downgrade_tolerance=0.05,
        criticality=1.0, latency_sensitivity=1.0
    ),
    BusinessType.VIDEO_STREAMING: QoSProfile(
        business_type=BusinessType.VIDEO_STREAMING,
        min_rate=25, ideal_rate=50, max_delay=20, max_loss_rate=0.05,
        priority=0.75, downgrade_tolerance=0.35,
        criticality=0.7, latency_sensitivity=0.8
    ),
    BusinessType.ENVIRONMENT_MONITORING: QoSProfile(
        business_type=BusinessType.ENVIRONMENT_MONITORING,
        min_rate=0.5, ideal_rate=1.0, max_delay=1000, max_loss_rate=0.05,
        priority=0.30, downgrade_tolerance=0.75,
        criticality=0.3, latency_sensitivity=0.2
    )
}


# ==================== 业务特征生成参数 ====================
# 用于模拟各业务类型的网络流量特征（时延ms、带宽Mbps、丢包率、抖动ms）
# 与QOS_PROFILES对齐：控制信令低带宽低时延、视频高带宽中时延、监测低带宽高时延容忍
BUSINESS_FEATURE_PARAMS = {
    BusinessType.CONTROL_SIGNAL: {
        'delay': (10, 3),                # 10±3ms (≤20ms要求)
        'bandwidth': (0.5, 0.1),         # 500±100kbps
        'loss_beta': (1, 1000),          # 极低丢包率(99.999%可靠性)
        'loss_scale': 0.00001,
        'jitter': (1, 0.5)
    },
    BusinessType.VIDEO_STREAMING: {
        'delay': (15, 5),                # 15±5ms (≈20ms要求)
        'bandwidth': (50, 15),           # 50±15Mbps (25-100Mbps范围)
        'loss_beta': (5, 100),           # 低丢包(连续传输)
        'loss_scale': 0.001,
        'jitter': (3, 1)
    },
    BusinessType.ENVIRONMENT_MONITORING: {
        'delay': (500, 200),             # 500±200ms (<1000ms)
        'bandwidth': (1, 0.3),           # 1±0.3Mbps (≤1Mbps)
        'loss_beta': (2, 20),
        'loss_scale': 0.05,
        'jitter': (50, 20)
    }
}
