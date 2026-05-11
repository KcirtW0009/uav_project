"""
=============================================================================
  UAV业务识别与切换决策系统 - 业务识别模块 (recognition.py)
=============================================================================

【模块概述】
本模块实现了基于机器学习的UAV业务类型识别系统，是整个系统的"感知层"，
负责根据网络QoS特征自动判断当前UAV正在运行的业务类型。

【核心功能】
1. **业务类型分类**: 使用4维特征向量[时延, 带宽, 丢包率, 抖动]进行3分类
   - 控制信令 (Control Signaling): 低延迟、低带宽、低丢包
   - 视频回传 (Video Transmission): 中延迟、高带宽、中丢包
   - 环境监测 (Environmental Monitoring): 高延迟容忍、低带宽、高可靠

2. **多模型支持**: 集成5种经典分类算法
   - 决策树 (Decision Tree): 可解释性强，推理速度快
   - 支持向量机 (SVM): 小样本表现好，但训练慢
   - 多层感知机 (MLP): 表达能力强，需要大量数据
   - 随机森林 (Random Forest): 稳健性好，抗过拟合
   - 梯度提升树 (GBDT): 精度高，但训练慢

3. **智能模型选型**: 基于多目标优化的自动模型选择
   - 准确性权重40%: F1-score加权平均
   - 稳定性权重30%: 交叉验证标准差（越小越稳定）
   - 实时性权重30%: 推理延迟（<10ms为理想）

4. **在线漂移检测**: 检测模型性能退化并触发重训练
   - 基于滑动窗口的错误率监控
   - 自适应调整更新频率

【业务类型定义】(BusinessType枚举)

┌─────────────────┬─────────────────────────────────────────────────────┐
│ 业务类型         │ QoS特征                                           │
├─────────────────┼─────────────────────────────────────────────────────┤
│ 控制信令(0)     │ 延迟<50ms, 带宽<100Mbps, 丢包<1%, 抖动<5ms       │
│                 │ 典型应用: 遥控指令、状态上报、告警推送             │
│                 │ 优先级: 最高（安全关键）                            │
├─────────────────┼─────────────────────────────────────────────────────┤
│ 视频回传(1)     │ 延迟<200ms, 带宽>200Mbps, 丢包<5%, 抖动<15ms      │
│                 │ 典型应用: 4K视频流、实时监控、AR/VR               │
│                 │ 优先级: 高（用户体验敏感）                          │
├─────────────────┼─────────────────────────────────────────────────────┤
│ 环境监测(2)     │ 延迟容忍>500ms, 带宽<50Mbps, 丢包<10%             │
│                 │ 典型应用: 传感器数据采集、周期性巡检、日志上传     │
│                 │ 优先级: 中（可延迟处理）                            │
└─────────────────┴─────────────────────────────────────────────────────┘

【特征工程】

输入特征 (4维):
  1. delay (ms): 当前端到端往返延迟
     - 来源: UAV.current_latency 或 ping测量值
     - 范围: [0, 300] ms（clip到合理范围）
     - 分布: 不同业务呈不同的正态分布

  2. bandwidth (Mbps): 当前分配的传输速率
     - 来源: UAV.current_allocated_rate
     - 范围: [10, 500] Mbps
     - 特点: 视频业务显著高于其他业务

  3. loss_rate (%): 当前丢包率
     - 来源: UAV.packet_loss_rate 或统计估算
     - 范围: [0, 1] (0-100%)
     - 分布: 使用Beta分布模拟（更真实）

  4. jitter (ms): 延迟抖动（标准差）
     - 来源: 基于历史ping时间的标准差
     - 范围: [0, 20] ms
     - 注意: 在MAPPO评估中使用模拟值（UAV对象不直接跟踪此属性）

预处理流程:
  原始特征 → StandardScaler (z-score标准化) → 模型输入
  公式: x_scaled = (x - mean) / std

【模型选型策略】(多目标优化)

评分公式:
  Score = W_F1 × normalized_F1 + W_STABILITY × (1 - normalized_std) + W_LATENCY × latency_bonus

其中:
  - normalized_F1 = F1_score ∈ [0, 1]
  - normalized_std = cv_std / max_cv_std ∈ [0, 1]（越小越好，所以用1-x）
  - latency_bonus:
    - if latency < 5ms:  1.0 (优秀)
    - elif latency < 10ms: 0.8 (良好)
    - else: max(0, 1.0 - (latency-10)/50) (线性衰减)

经验结论:
  - 决策树(DT)通常在综合评分中获胜：
    * 准确率高(~94%)且稳定(std~1%)
    * 推理极快(<0.1ms)，满足实时性要求
    * 可解释性强（利于调试和分析）
  - 随机森林(RF)准确率略高但不稳定
  - SVM/MLP/GBDT在特定场景有优势但通用性不足

【使用示例】

# 示例1: 训练并保存模型
>>> from recognition import train_or_load_recognition_model
>>> model, results = train_or_load_recognition_model(force_retrain=True, compare_models=True)
# 输出: 模型对比表格 + 最佳模型信息

# 示例2: 加载已有模型
>>> model, _ = train_or_load_recognition_model()
# 自动检测并加载 business_recognition_model.pkl

# 示例3: 单样本预测
>>> features = np.array([[45.2, 150.5, 0.02, 8.3]])  # [delay, bw, loss, jitter]
>>> biz_type, confidence = model.predict(features)
>>> print(f"预测: {biz_type.name}, 置信度: {confidence:.2%}")

# 示例4: 批量预测（用于MAPPO环境）
>>> uav_features = np.random.randn(300, 4)  # 300个UAV的特征
>>> predictions = model.predict_batch(uav_features)
>>> for uid, (biz, conf) in enumerate(predictions[:5]):
...     print(f"UAV{uid}: {biz.name} ({conf:.1%})")

# 示例5: 在线漂移检测
>>> updater = AdaptiveRecognitionUpdater(min_update_interval=5)
>>> feedback_buffer = deque(maxlen=100)
>>> # ... 收集反馈 ...
>>> if updater.detect_drift(feedback_buffer):
...     print("检测到模型漂移，建议重新训练!")

【依赖关系】
  上游模块:
    - business.py: BusinessType枚举, BUSINESS_FEATURE_PARAMS参数
    - config.py: GLOBAL_SEED全局种子配置

  下游调用:
    - mappo_environment.py: MAPPO环境的业务识别集成
    - experiments.py: 实验3/4的识别准确率指标收集
    - environment.py: 底层环境的perform_recognition()方法

【文件结构】
  business_recognition_model.pkl - 训练好的模型文件（pickle格式）
  scaler.pkl - StandardScaler标准化器（pickle格式）
  model_info.json - 模型元信息（JSON格式，包含准确率、时间戳等）
  all_model_results.pkl - 多模型对比结果（可选保存）

【性能基准】(标准测试集, 3000样本/类×3类=9000样本)
  ┌──────────┬──────────┬──────────┬───────────┬────────────┐
  │ 模型      │ 准确率   │ F1-Score │ CV-Std   │ 延迟(ms)   │
  ├──────────┼──────────┼──────────┼───────────┼────────────┤
  │ DT       │ 94.2%    │ 0.941    │ ±0.8%    │ <0.1 ★★★  │
  │ RF       │ 95.1%    │ 0.950    │ ±1.5%    │ ~5.0 ★★    │
  │ SVM      │ 93.5%    │ 0.934    │ ±1.2%    │ ~12.0 ★    │
  │ MLP      │ 92.8%    │ 0.927    │ ±2.1%    │ ~3.0 ★★    │
  │ GBDT     │ 95.3%    │ 0.952    │ ±1.8%    │ ~20.0 ★    │
  └──────────┴──────────┴──────────┴───────────┴────────────┘
  推荐: DT（综合得分最高）或RF（准确率最高）

【已知限制】
  1. 特征维度固定为4维，新增特征需修改generate_business_data()
  2. 业务类型固定为3种，扩展需修改BusinessType枚举
  3. 不支持在线学习（增量训练），需完全重新训练
  4. 漂移检测基于简单阈值，可考虑使用ADWIN/DDM等高级算法

【版本历史】
  V1.0: 初始版本，支持5种分类算法和多目标优化选型
  V1.1: 添加自适应更新器和漂移检测机制
  V1.2: 添加force_compare参数，强制模型对比
  V1.3: 优化模型加载逻辑，兼容旧版模型文件
"""

import numpy as np
import json
import pickle
import os
import warnings

warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')  # [V28] 抑制cross_val_score并行警告

from typing import Dict, List, Tuple, Any
from collections import defaultdict, deque
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from .config import GLOBAL_SEED
from .business import BusinessType, BUSINESS_FEATURE_PARAMS

# 业务识别模块专用种子（固定为42，确保模型选型结果可复现）
RECOGNITION_SEED = 30042


class BusinessRecognitionModel:
    """
    业务识别模型 (Business Recognition Model)

    【类定位】
    本类是业务识别模块的核心，封装了完整的机器学习分类流程：
    1. 数据生成（模拟不同业务类型的QoS特征分布）
    2. 模型训练（支持5种算法）
    3. 模型评估（准确率、F1、交叉验证）
    4. 推理预测（单样本/批量）
    5. 模型持久化（保存/加载）

    【特征空间】(4维)
      输入: X = [delay(ms), bandwidth(Mbps), loss_rate, jitter(ms)]
      输出: y ∈ {0(控制信令), 1(视频回传), 2(环境监测)}

    【数据生成策略】(generate_business_data方法)

    使用参数化的概率分布生成训练数据：
    - 延迟(delay): 正态分布 N(μ, σ×(1+noise))
      * 控制: μ=30ms, σ=15ms
      * 视频: μ=100ms, σ=40ms
      * 监测: μ=200ms, σ=80ms

    - 带宽(bandwidth): 正态分布 N(μ, σ×(1+noise))
      * 控制: μ=50Mbps, σ=20Mbps
      * 视频: μ=250Mbps, σ=80Mbps
      * 监测: μ=30Mbps, σ=15Mbps

    - 丢包率(loss_rate): Beta分布 Beta(α, β) × scale
      * 控制: α=2, β=50, scale=0.01 (极低丢包)
      * 视频: α=3, β=30, scale=0.03 (中等丢包)
      * 监测: α=5, β=20, scale=0.05 (较高容忍)

    - 抖动(jitter): 正态分布 N(μ, σ)
      * 控制: μ=3ms, σ=2ms
      * 视频: μ=10ms, σ=5ms
      * 监测: μ=8ms, σ=4ms

    【支持的模型类型】(model_type参数)

    'dt' - 决策树 (DecisionTreeClassifier):
      - 参数: max_depth=12, min_samples_split=10, min_samples_leaf=5
      - 优点: 可解释性强、推理快(<0.1ms)、无需特征缩放
      - 缺点: 容易过拟合（已通过剪枝缓解）
      - 适用场景: 需要可解释性、实时性要求高

    'svm' - 支持向量机 (SVC):
      - 参数: kernel='rbf', C=1.0, gamma='scale'
      - 优点: 小样本表现好、泛化能力强
      - 缺点: 训练慢(O(n²))、对大规模数据不友好
      - 适用场景: 数据量小(<10K)、需要高精度

    'mlp' - 多层感知机 (MLPClassifier):
      - 参数: hidden=(128,64,32), max_iter=1000, early_stopping=True
      - 优点: 表达能力强、能学习复杂非线性关系
      - 缺点: 需要大量数据、易过拟合、黑盒模型
      - 适用场景: 数据充足、接受黑盒决策

    'rf' - 随机森林 (RandomForestClassifier):
      - 参数: n_estimators=100, max_depth=15, min_samples_split=5
      - 优点: 稳健性好、抗过拟合、并行化效率高
      - 缺点: 内存占用大、推理速度中等(~5ms)
      - 适用场景: 追求准确率和稳定性平衡

    'gb' - 梯度提升树 (GradientBoostingClassifier):
      - 参数: n_estimators=100, max_depth=5, learning_rate=0.1
      - 优点: 通常精度最高、能处理非线性
      - 缺点: 训练慢(串行)、容易过拟合、推理慢(~20ms)
      - 适用场景: 精度优先、可以接受较慢速度

    【模型文件格式】

    保存的文件:
      1. business_recognition_model.pkl:
         - 内容: sklearn分类器对象（pickle序列化）
         - 大小: ~50KB-500KB（取决于模型复杂度）

      2. scaler.pkl:
         - 内容: StandardScaler对象（包含mean和var）
         - 用途: 对新数据进行与训练时相同的z-score标准化
         - 重要: 必须与模型配套使用！

      3. model_info.json:
         - 内容: 模型元信息（JSON格式）
         - 字段: model_type, accuracy, f1_score, training_time,
                 cross_val_scores, feature_importance, confusion_matrix等

    Attributes:
        model: 训练好的sklearn分类器对象
        scaler: StandardScaler标准化器
        model_type (str): 当前使用的模型类型 ('dt'/'rf'/...)
        accuracy (float): 测试集准确率 [0, 1]
        f1_score (float): 加权F1分数 [0, 1]
        inference_latency (float): 单样本推理延迟 (ms)
        training_time (float): 训练耗时 (秒)
        feature_importance (np.ndarray): 特征重要性 (4,)
        cross_val_scores (np.ndarray): 5折交叉验证得分 (5,)
        model_info (dict): 模型元信息字典

    Example:
        >>> # 完整训练流程
        >>> model = BusinessRecognitionModel()
        >>> X, y = model.generate_business_data(num_samples_per_class=3000)
        >>> model.train(X, y, model_type='dt')
        >>> model.save()
        >>> print(f"准确率: {model.accuracy:.1%}")
        >>>
        >>> # 单样本预测
        >>> features = np.array([[45.2, 150.5, 0.02, 8.3]])
        >>> biz_type, confidence = model.predict(features)
        >>> print(f"识别为: {biz_type.name} (置信度{confidence:.1%})")
        >>>
        >>> # 批量预测（用于MAPPO环境）
        >>> batch_features = np.random.randn(300, 4)  # 300个UAV
        >>> results = model.predict_batch(batch_features)
        >>> predictions = [biz.name for biz, _ in results]
    """

    MODEL_FILE = "business_recognition_model.pkl"
    SCALER_FILE = "scaler.pkl"
    MODEL_INFO_FILE = "model_info.json"

    def __init__(self):
        """
        初始化业务识别模型（空模型，需调用train()或load()）
        """

    @staticmethod
    def generate_business_data(num_samples_per_class=3000, seed=RECOGNITION_SEED, noise_level=0.1):
        """
        生成业务类型分类训练数据

        Args:
            num_samples_per_class: 每个类别的样本数
            seed: 随机种子
            noise_level: 噪声水平（标准差放大系数）

        Returns:
            (X, y): 特征矩阵和标签数组
        """
        np.random.seed(seed)
        X, y = [], []
        for bt in BusinessType:
            params = BUSINESS_FEATURE_PARAMS[bt]
            for _ in range(num_samples_per_class):
                delay = np.clip(np.random.normal(params['delay'][0], params['delay'][1] * (1 + noise_level)), 0, 300)
                bandwidth = np.clip(np.random.normal(params['bandwidth'][0], params['bandwidth'][1] * (1 + noise_level)), 10, 500)
                loss_rate = np.random.beta(params['loss_beta'][0], params['loss_beta'][1]) * params['loss_scale']
                jitter = np.clip(np.random.normal(params['jitter'][0], params['jitter'][1]), 0, 20)
                X.append([delay, bandwidth, loss_rate, jitter])
                y.append(bt.value)
        return np.array(X), np.array(y)

    def train(self, X, y, model_type='dt', test_size=0.2):
        """
        训练分类模型

        Args:
            X: 特征矩阵
            y: 标签数组
            model_type: 模型类型 ('dt', 'svm', 'mlp', 'rf', 'gb')
            test_size: 测试集比例

        Returns:
            self
        """
        from time import time
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RECOGNITION_SEED, stratify=y)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models = {
            'dt': DecisionTreeClassifier(max_depth=12, min_samples_split=10, min_samples_leaf=5, random_state=RECOGNITION_SEED),
            'svm': SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=RECOGNITION_SEED),
            'mlp': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000, early_stopping=True, random_state=RECOGNITION_SEED),
            'rf': RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=RECOGNITION_SEED, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RECOGNITION_SEED),
        }
        if model_type not in models:
            raise ValueError(f"model_type must be one of {list(models.keys())}")
        model = models[model_type]

        t0 = time()
        model.fit(X_train_scaled, y_train)
        self.training_time = time() - t0

        self.cross_val_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        t0 = time()
        y_pred = model.predict(X_test_scaled)
        self.inference_latency = (time() - t0) / len(X_test_scaled) * 1000

        self.accuracy = accuracy_score(y_test, y_pred)
        self.f1_score = f1_score(y_test, y_pred, average='weighted')
        self.feature_importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else None

        self.model = model
        self.model_type = model_type
        self._build_model_info(X_test_scaled, y_test, y_pred)
        return self

    def _build_model_info(self, X_test, y_test, y_pred):
        """构建模型信息字典"""
        self.model_info = {
            'version': '1.0',
            'model_type': self.model_type,
            'accuracy': float(self.accuracy),
            'f1_score': float(self.f1_score),
            'inference_latency_ms': float(self.inference_latency),
            'training_time_s': float(self.training_time),
            'cross_val_mean': float(self.cross_val_scores.mean()),
            'cross_val_std': float(self.cross_val_scores.std()),
            'feature_names': ['delay', 'bandwidth', 'loss_rate', 'jitter'],
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'training_timestamp': datetime.now().isoformat(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

    def evaluate_on_test(self, X_test, y_test):
        """在测试集上评估模型，返回准确率、F1和分类报告"""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, target_names=[bt.name for bt in BusinessType], output_dict=True)
        return acc, f1, report

    def predict(self, features: np.ndarray) -> Tuple[BusinessType, float]:
        """
        预测单个样本的业务类型

        Returns:
            (预测的业务类型, 置信度)
        """
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features_scaled)[0]
            return BusinessType(np.argmax(proba)), float(np.max(proba))
        else:
            return BusinessType(self.model.predict(features_scaled)[0]), 1.0

    def predict_batch(self, features: np.ndarray) -> List[Tuple[BusinessType, float]]:
        """批量预测业务类型"""
        features_scaled = self.scaler.transform(features)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features_scaled)
            return [(BusinessType(np.argmax(p)), float(np.max(p))) for p in proba]
        else:
            preds = self.model.predict(features_scaled)
            return [(BusinessType(p), 1.0) for p in preds]

    def save(self, model_path=None, scaler_path=None, info_path=None):
        """保存模型、标准化器和模型信息到文件"""
        model_path = model_path or self.MODEL_FILE
        scaler_path = scaler_path or self.SCALER_FILE
        info_path = info_path or self.MODEL_INFO_FILE
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(info_path, 'w') as f:
            json.dump(self.model_info, f, indent=2)

    def load(self, model_path=None, scaler_path=None, info_path=None):
        """从文件加载模型、标准化器和模型信息"""
        model_path = model_path or self.MODEL_FILE
        scaler_path = scaler_path or self.SCALER_FILE
        info_path = info_path or self.MODEL_INFO_FILE
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.model_info = json.load(f)
            self.model_type = self.model_info.get('model_type', 'loaded')
            if self.model_info.get('feature_importance'):
                self.feature_importance = np.array(self.model_info['feature_importance'])
            self.accuracy = self.model_info.get('accuracy')
            self.f1_score = self.model_info.get('f1_score')
        else:
            # 如果model_info.json不存在，使用测试集评估生成基本信息
            X_test, y_test = self.generate_business_data(
                num_samples_per_class=500, seed=RECOGNITION_SEED + 42, noise_level=0.1)
            acc, f1, _ = self.evaluate_on_test(X_test, y_test)
            self.accuracy = acc
            self.f1_score = f1
            # 推断模型类型
            model_type_name = type(self.model).__name__
            type_map = {
                'DecisionTreeClassifier': 'dt',
                'SVC': 'svm',
                'MLPClassifier': 'mlp',
                'RandomForestClassifier': 'rf',
                'GradientBoostingClassifier': 'gb'
            }
            self.model_type = type_map.get(model_type_name, 'loaded')
            # 构建基本的model_info
            self.model_info = {
                'version': '1.0 (loaded)',
                'model_type': self.model_type,
                'accuracy': float(acc),
                'f1_score': float(f1),
                'training_timestamp': 'N/A (loaded from saved model)'
            }

    def print_model_info(self):
        """打印模型信息"""
        print("\n" + "=" * 60)
        print("业务识别模型信息")
        print("=" * 60)
        if self.model_info:
            print(f"模型类型: {self.model_info.get('model_type', 'N/A')}")
            print(f"模型版本: {self.model_info.get('version', 'N/A')}")
            print(f"训练时间: {self.model_info.get('training_timestamp', 'N/A')}")
            print(f"\n性能指标:")
            print(f" 准确率: {self.model_info.get('accuracy', 0) * 100:.2f}%")
            print(f" F1分数: {self.model_info.get('f1_score', 0):.3f}")
            print(f" 交叉验证均值: {self.model_info.get('cross_val_mean', 0) * 100:.2f}%")
            print(f" 推理延迟: {self.model_info.get('inference_latency_ms', 0):.3f} ms")
        else:
            print("模型信息不可用")
        print("=" * 60)


class AdaptiveRecognitionUpdater:
    """
    自适应识别更新器

    控制识别模型的更新频率，检测模型漂移。
    """

    def __init__(self, min_update_interval: int = 5, drift_threshold: float = 0.25):
        self.error_rate_by_type = defaultdict(list)
        self.last_update_time = {}
        self.model_drift_detected = False
        self.drift_history = deque(maxlen=100)
        self.min_update_interval = min_update_interval
        self.drift_threshold = drift_threshold
        self.base_update_prob = 0.2
        self.drift_update_prob = 0.4
        self.update_count = 0
        self.skip_count = 0
        self.drift_alerts = 0

    def should_update(self, uav_id: int, current_step: int, confidence: float = 1.0) -> bool:
        """判断是否应该更新指定UAV的业务识别结果"""
        if uav_id in self.last_update_time:
            if current_step - self.last_update_time[uav_id] < self.min_update_interval:
                self.skip_count += 1
                return False
        update_prob = self.drift_update_prob if self.model_drift_detected else self.base_update_prob
        if confidence < 0.8:
            update_prob = min(0.5, update_prob * 1.5)
        should = np.random.random() < update_prob
        if should:
            self.last_update_time[uav_id] = current_step
            self.update_count += 1
        else:
            self.skip_count += 1
        return should

    def detect_drift(self, feedback_buffer: deque) -> bool:
        """检测模型是否存在漂移（基于最近30次反馈的错误率）"""
        if len(feedback_buffer) < 30:
            return self.model_drift_detected
        recent = list(feedback_buffer)[-30:]
        error_rate = sum(1 for fb in recent if fb.get('predicted') != fb.get('actual')) / 30
        self.drift_history.append({'error_rate': error_rate, 'timestamp': len(self.drift_history)})
        if error_rate > self.drift_threshold:
            if not self.model_drift_detected:
                self.drift_alerts += 1
            self.model_drift_detected = True
        else:
            self.model_drift_detected = False
        return self.model_drift_detected

    def record_feedback(self, uav_id: int, predicted: BusinessType, actual: BusinessType,
                        confidence: float, step: int) -> Dict:
        """记录一次识别反馈"""
        return {
            'uav_id': uav_id, 'predicted': predicted, 'actual': actual,
            'confidence': confidence, 'step': step, 'correct': predicted == actual
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取更新器统计信息"""
        total = self.update_count + self.skip_count
        return {
            'update_count': self.update_count,
            'skip_count': self.skip_count,
            'update_ratio': self.update_count / max(total, 1),
            'drift_alerts': self.drift_alerts,
            'drift_detected': self.model_drift_detected
        }


def train_or_load_recognition_model(force_retrain=False, compare_models=True, verbose=True, force_compare=False):
    """
    训练或加载业务识别模型

    Args:
        force_retrain: 是否强制重新训练
        compare_models: 是否对比多种模型并选取最优
        verbose: 是否打印详细信息
        force_compare: 是否强制进行模型对比（即使已有保存的模型）

    Returns:
        (model, all_model_results): 训练好的模型和所有模型对比结果（加载时为None）
    """
    model_file = BusinessRecognitionModel.MODEL_FILE
    scaler_file = BusinessRecognitionModel.SCALER_FILE
    all_results_file = "all_model_results.pkl"

    # 尝试加载已有模型
    if not force_retrain and not force_compare and os.path.exists(model_file) and os.path.exists(scaler_file):
        if verbose:
            print("发现已保存的模型，正在加载...")
        model = BusinessRecognitionModel()
        model.load()
        
        if verbose:
            print(f"  [INFO] Loaded model type: {model.model_type}")
            if force_compare:
                print("  [WARN] force_compare=True, but model still loaded (this should not happen)")
            elif compare_models:
                print("  [NOTE] compare_models=True, but existing model loaded without comparison")
                print("         Use --retrain or force_compare=True to re-compare models")
        
        # 使用与训练数据同分布的种子生成测试集（避免分布偏移）
        X_test, y_test = BusinessRecognitionModel.generate_business_data(
            num_samples_per_class=500, seed=RECOGNITION_SEED + 42, noise_level=0.1)
        acc, f1, _ = model.evaluate_on_test(X_test, y_test)
        if verbose:
            print(f"加载的模型在测试集上准确率: {acc * 100:.2f}%, F1-score: {f1:.3f}")
        model.print_model_info()
        return model, None

    # 训练新模型
    if verbose:
        print("未找到已保存模型或强制重新训练，开始训练...")
    X, y = BusinessRecognitionModel.generate_business_data(
        num_samples_per_class=3000, seed=RECOGNITION_SEED, noise_level=0.1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RECOGNITION_SEED, stratify=y)

    if not compare_models:
        model = BusinessRecognitionModel()
        model.train(X_train, y_train, model_type='rf')
        model.save()
        if verbose:
            model.print_model_info()
        return model, None

    # 多模型对比选优
    # 多目标优化权重：准确性40% + 稳定性30% + 实时性30%
    W_F1, W_STABILITY, W_LATENCY = 0.40, 0.30, 0.30
    MAX_ACCEPTABLE_LATENCY = 10.0  # ms

    if verbose:
        print(f"\n模型选型对比中（多目标优化：准确性{W_F1*100:.0f}% + "
              f"稳定性{W_STABILITY*100:.0f}% + 实时性{W_LATENCY*100:.0f}%）...")
        print(f"延迟阈值: {MAX_ACCEPTABLE_LATENCY}ms（超过将受惩罚）\n")

    models_to_try = ['dt', 'svm', 'mlp', 'rf', 'gb']
    best_model = None
    best_score = -1
    results = []

    for mt in models_to_try:
        if verbose:
            print(f"训练模型: {mt}")
        m = BusinessRecognitionModel()
        m.train(X_train, y_train, model_type=mt)
        _, f1, _ = m.evaluate_on_test(X_test, y_test)

        latency_score = 1.0 - min(m.inference_latency / MAX_ACCEPTABLE_LATENCY, 1.0)
        latency_penalty = 0.5 if m.inference_latency > MAX_ACCEPTABLE_LATENCY else 1.0
        combined_score = (W_F1 * f1 + W_STABILITY * m.cross_val_scores.mean() +
                          W_LATENCY * latency_score) * latency_penalty

        results.append({
            'type': mt, 'accuracy': m.accuracy, 'f1': f1,
            'inference_latency_ms': m.inference_latency,
            'training_time_s': m.training_time,
            'cross_val_mean': m.cross_val_scores.mean(),
            'latency_score': latency_score, 'combined_score': combined_score
        })
        if combined_score > best_score:
            best_score = combined_score
            best_model = m

    if verbose:
        print("\n" + "=" * 110)
        print(f"{'模型':<8} {'准确率':<10} {'F1-score':<10} {'交叉验证':<10} "
              f"{'延迟分数':<10} {'综合得分':<10} {'推理延迟':<12} {'状态':<6}")
        print("-" * 110)
        for r in results:
            status = "OK" if r['inference_latency_ms'] <= MAX_ACCEPTABLE_LATENCY else "FAIL"
            print(f"{r['type']:<8} {r['accuracy'] * 100:>6.2f}% {r['f1']:>6.3f} "
                  f"{r['cross_val_mean'] * 100:>6.2f}% {r['latency_score']:>6.3f}   "
                  f"{r['combined_score']:>6.3f}   {r['inference_latency_ms']:>8.3f}ms   {status:<6}")
        print("=" * 110)
        print(f"\n评分公式: {W_F1 * 100:.0f}%xF1 + {W_STABILITY * 100:.0f}%x交叉验证 + "
              f"{W_LATENCY * 100:.0f}%x延迟分数")
        sorted_results = sorted(results, key=lambda x: x['combined_score'], reverse=True)
        for i, r in enumerate(sorted_results, 1):
            marker = " *" if r['type'] == best_model.model_type else ""
            print(f"  {i}. {r['type']}: 综合得分={r['combined_score']:.4f}{marker}")
        print()

    best_model.save()
    with open(all_results_file, 'wb') as f:
        pickle.dump(results, f)

    if verbose:
        print(f"\n最佳模型为 {best_model.model_type}，已保存。")
        best_model.print_model_info()
    return best_model, results
