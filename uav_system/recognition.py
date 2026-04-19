"""
业务识别模块

包含业务识别模型(BusinessRecognitionModel)和自适应识别更新器(AdaptiveRecognitionUpdater)。
支持多种分类算法（决策树、SVM、MLP、随机森林、GBDT），通过多目标优化选取最佳模型。
"""

import numpy as np
import json
import pickle
import os
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
    业务识别模型

    使用多维特征向量（时延、带宽、丢包率、抖动）进行业务类型分类。
    支持训练、保存、加载、预测等功能。

    特征维度: [delay(ms), bandwidth(Mbps), loss_rate, jitter(ms)]
    """

    MODEL_FILE = "business_recognition_model.pkl"
    SCALER_FILE = "scaler.pkl"
    MODEL_INFO_FILE = "model_info.json"

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = None
        self.accuracy = None
        self.f1_score = None
        self.inference_latency = None
        self.training_time = None
        self.feature_importance = None
        self.cross_val_scores = None
        self.model_info = {}

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


def train_or_load_recognition_model(force_retrain=False, compare_models=True, verbose=True):
    """
    训练或加载业务识别模型

    Args:
        force_retrain: 是否强制重新训练
        compare_models: 是否对比多种模型并选取最优
        verbose: 是否打印详细信息

    Returns:
        (model, all_model_results): 训练好的模型和所有模型对比结果（加载时为None）
    """
    model_file = BusinessRecognitionModel.MODEL_FILE
    scaler_file = BusinessRecognitionModel.SCALER_FILE
    all_results_file = "all_model_results.pkl"

    # 尝试加载已有模型
    if not force_retrain and os.path.exists(model_file) and os.path.exists(scaler_file):
        if verbose:
            print("发现已保存的模型，正在加载...")
        model = BusinessRecognitionModel()
        model.load()
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
