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

class BusinessRecognitionModel:
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
    def generate_business_data(num_samples_per_class=3000, seed=GLOBAL_SEED, noise_level=0.1):
        np.random.seed(seed)
        X, y = [], []
        for bt in BusinessType:
            params = BUSINESS_FEATURE_PARAMS[bt]
            for _ in range(num_samples_per_class):
                delay = np.random.normal(params['delay'][0], params['delay'][1] * (1 + noise_level))
                delay = np.clip(delay, 0, 300)
                bandwidth = np.random.normal(params['bandwidth'][0], params['bandwidth'][1] * (1 + noise_level))
                bandwidth = np.clip(bandwidth, 10, 500)
                loss_rate = np.random.beta(params['loss_beta'][0], params['loss_beta'][1])
                loss_rate = loss_rate * params['loss_scale']
                jitter = np.random.normal(params['jitter'][0], params['jitter'][1])
                jitter = np.clip(jitter, 0, 20)
                X.append([delay, bandwidth, loss_rate, jitter])
                y.append(bt.value)
        return np.array(X), np.array(y)

    def train(self, X, y, model_type='dt', test_size=0.2):
        from time import time
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=GLOBAL_SEED, stratify=y)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if model_type == 'dt':
            model = DecisionTreeClassifier(max_depth=12, min_samples_split=10,
                                           min_samples_leaf=5, random_state=GLOBAL_SEED)
        elif model_type == 'svm':
            model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=GLOBAL_SEED)
        elif model_type == 'mlp':
            model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000,
                                  early_stopping=True, random_state=GLOBAL_SEED)
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, max_depth=15,
                                           min_samples_split=5, random_state=GLOBAL_SEED, n_jobs=-1)
        elif model_type == 'gb':
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                               learning_rate=0.1, random_state=GLOBAL_SEED)
        else:
            raise ValueError("model_type must be 'dt', 'svm', 'mlp', 'rf', or 'gb'")

        t0 = time()
        model.fit(X_train_scaled, y_train)
        t1 = time()
        self.training_time = t1 - t0

        self.cross_val_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        t0 = time()
        y_pred = model.predict(X_test_scaled)
        t1 = time()
        self.inference_latency = (t1 - t0) / len(X_test_scaled) * 1000

        self.accuracy = accuracy_score(y_test, y_pred)
        self.f1_score = f1_score(y_test, y_pred, average='weighted')

        if hasattr(model, 'feature_importances_'):
            self.feature_importance = model.feature_importances_

        self.model = model
        self.model_type = model_type
        self._build_model_info(X_test_scaled, y_test, y_pred)
        return self

    def _build_model_info(self, X_test, y_test, y_pred):
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
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, target_names=[bt.name for bt in BusinessType],
                                       output_dict=True)
        return acc, f1, report

    def predict(self, features: np.ndarray) -> Tuple[BusinessType, float]:
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features_scaled)[0]
            pred_class = np.argmax(proba)
            confidence = np.max(proba)
        else:
            pred_class = self.model.predict(features_scaled)[0]
            confidence = 1.0
        return BusinessType(pred_class), confidence

    def predict_batch(self, features: np.ndarray) -> List[Tuple[BusinessType, float]]:
        features_scaled = self.scaler.transform(features)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features_scaled)
            pred_classes = np.argmax(proba, axis=1)
            confidences = np.max(proba, axis=1)
        else:
            pred_classes = self.model.predict(features_scaled)
            confidences = np.ones(len(pred_classes))
        return [(BusinessType(pc), conf) for pc, conf in zip(pred_classes, confidences)]

    def save(self, model_path=None, scaler_path=None, info_path=None):
        model_path = model_path or self.MODEL_FILE
        scaler_path = scaler_path or self.SCALER_FILE
        info_path = info_path or self.MODEL_INFO_FILE
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(info_path, 'w') as f:
            json.dump(self.model_info, f, indent=2)
        print(f"模型已保存至 {model_path}, {scaler_path}, {info_path}")

    def load(self, model_path=None, scaler_path=None, info_path=None):
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
            print(f"模型信息已加载: 类型={self.model_type}, 准确率={self.accuracy*100:.2f}%")
        else:
            self.model_type = 'loaded'
            print(f"模型已从 {model_path}, {scaler_path} 加载")

    def print_model_info(self):
        print("\n" + "="*60)
        print("业务识别模型信息")
        print("="*60)
        if self.model_info:
            print(f"模型类型: {self.model_info.get('model_type', 'N/A')}")
            print(f"模型版本: {self.model_info.get('version', 'N/A')}")
            print(f"训练时间: {self.model_info.get('training_timestamp', 'N/A')}")
            print(f"\n性能指标:")
            print(f" 准确率: {self.model_info.get('accuracy', 0)*100:.2f}%")
            print(f" F1分数: {self.model_info.get('f1_score', 0):.3f}")
            print(f" 交叉验证均值: {self.model_info.get('cross_val_mean', 0)*100:.2f}%")
            print(f" 推理延迟: {self.model_info.get('inference_latency_ms', 0):.3f} ms")
            print(f"\n特征重要性:")
            importance = self.model_info.get('feature_importance')
            if importance:
                for name, imp in zip(self.model_info.get('feature_names', []), importance):
                    print(f" {name}: {imp:.3f}")
        else:
            print("模型信息不可用")
        print("="*60)


class AdaptiveRecognitionUpdater:
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
        if uav_id in self.last_update_time:
            if current_step - self.last_update_time[uav_id] < self.min_update_interval:
                self.skip_count += 1
                return False
        update_prob = self.base_update_prob
        if self.model_drift_detected:
            update_prob = self.drift_update_prob
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
        if len(feedback_buffer) < 30:
            return self.model_drift_detected
        recent = list(feedback_buffer)[-30:]
        error_count = sum(1 for fb in recent if fb.get('predicted') != fb.get('actual'))
        error_rate = error_count / 30
        self.drift_history.append({'error_rate': error_rate, 'timestamp': len(self.drift_history)})
        if error_rate > self.drift_threshold:
            if not self.model_drift_detected:
                print(f"⚠️ 检测到模型漂移，错误率{error_rate*100:.1f}%")
                self.drift_alerts += 1
            self.model_drift_detected = True
        else:
            self.model_drift_detected = False
        return self.model_drift_detected

    def record_feedback(self, uav_id: int, predicted: BusinessType, actual: BusinessType,
                        confidence: float, step: int):
        return {
            'uav_id': uav_id,
            'predicted': predicted,
            'actual': actual,
            'confidence': confidence,
            'step': step,
            'correct': predicted == actual
        }

    def get_stats(self) -> Dict[str, Any]:
        total = self.update_count + self.skip_count
        return {
            'update_count': self.update_count,
            'skip_count': self.skip_count,
            'update_ratio': self.update_count / max(total, 1),
            'drift_alerts': self.drift_alerts,
            'drift_detected': self.model_drift_detected
        }


def train_or_load_recognition_model(force_retrain=False, compare_models=True, verbose=True):
    model_file = BusinessRecognitionModel.MODEL_FILE
    scaler_file = BusinessRecognitionModel.SCALER_FILE
    info_file = BusinessRecognitionModel.MODEL_INFO_FILE

    if not force_retrain and os.path.exists(model_file) and os.path.exists(scaler_file):
        if verbose:
            print("发现已保存的模型，正在加载...")
        model = BusinessRecognitionModel()
        model.load()
        X_test, y_test = BusinessRecognitionModel.generate_business_data(num_samples_per_class=500, seed=GLOBAL_SEED+999)
        acc, f1, report = model.evaluate_on_test(X_test, y_test)
        if verbose:
            print(f"加载的模型在测试集上准确率: {acc*100:.2f}%, F1-score: {f1:.3f}")
        model.print_model_info()
        return model

    if verbose:
        print("未找到已保存模型或强制重新训练，开始训练...")
    X, y = BusinessRecognitionModel.generate_business_data(num_samples_per_class=3000,
                                                           seed=GLOBAL_SEED, noise_level=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=GLOBAL_SEED, stratify=y)

    if compare_models:
        models_to_try = ['dt', 'svm', 'mlp', 'rf', 'gb']
        best_model = None
        best_score = -1
        results = []
        
        # 多目标优化权重配置
        W_F1 = 0.40           # 准确性权重
        W_STABILITY = 0.30    # 稳定性权重（交叉验证）
        W_LATENCY = 0.30      # 实时性权重
        MAX_ACCEPTABLE_LATENCY = 10.0  # 最大可接受延迟(ms)
        
        if verbose:
            print("\n模型选型对比中（多目标优化：准确性40% + 稳定性30% + 实时性30%）...")
            print(f"延迟阈值: {MAX_ACCEPTABLE_LATENCY}ms（超过将受惩罚）\n")
            
        for mt in models_to_try:
            if verbose:
                print(f"训练模型: {mt}")
            m = BusinessRecognitionModel()
            m.train(X_train, y_train, model_type=mt)
            acc, f1, report = m.evaluate_on_test(X_test, y_test)
            
            # 归一化延迟分数（越低越好）
            normalized_latency = min(m.inference_latency / MAX_ACCEPTABLE_LATENCY, 1.0)
            latency_score = 1.0 - normalized_latency  # 转换为分数（越高越好）
            
            # 延迟惩罚：如果超过阈值，大幅降低分数
            latency_penalty = 1.0
            if m.inference_latency > MAX_ACCEPTABLE_LATENCY:
                latency_penalty = 0.5
                if verbose:
                    print(f"  ⚠️ 延迟 {m.inference_latency:.2f}ms 超过阈值，应用惩罚")
            
            # 多目标综合评分
            combined_score = (
                W_F1 * f1 +
                W_STABILITY * m.cross_val_scores.mean() +
                W_LATENCY * latency_score
            ) * latency_penalty
            
            results.append({
                'type': mt,
                'accuracy': acc,
                'f1': f1,
                'inference_latency_ms': m.inference_latency,
                'training_time_s': m.training_time,
                'cross_val_mean': m.cross_val_scores.mean(),
                'latency_score': latency_score,
                'combined_score': combined_score
            })
            
            if combined_score > best_score:
                best_score = combined_score
                best_model = m
                
        if verbose:
            print("\n" + "="*110)
            print(f"{'模型':<8} {'准确率':<10} {'F1-score':<10} {'交叉验证':<10} {'延迟分数':<10} {'综合得分':<10} {'推理延迟':<12} {'状态':<6}")
            print("-"*110)
            for r in results:
                status = "✓" if r['inference_latency_ms'] <= MAX_ACCEPTABLE_LATENCY else "✗"
                print(f"{r['type']:<8} {r['accuracy']*100:>6.2f}% {r['f1']:>6.3f} "
                      f"{r['cross_val_mean']*100:>6.2f}% {r['latency_score']:>6.3f}   "
                      f"{r['combined_score']:>6.3f}   {r['inference_latency_ms']:>8.3f}ms   {status:<6}")
            print("="*110)
            print(f"\n评分公式: {W_F1*100:.0f}%×F1 + {W_STABILITY*100:.0f}%×交叉验证 + {W_LATENCY*100:.0f}%×延迟分数")
            print(f"延迟分数 = 1 - min(延迟/{MAX_ACCEPTABLE_LATENCY}ms, 1)，超过阈值×0.5惩罚\n")
            print("按综合得分排序：")
            sorted_results = sorted(results, key=lambda x: x['combined_score'], reverse=True)
            for i, r in enumerate(sorted_results, 1):
                marker = " ★最佳" if r['type'] == best_model.model_type else ""
                print(f"  {i}. {r['type']}: 综合得分={r['combined_score']:.4f}{marker}")
            print()
        best_model.save()
        if verbose:
            print(f"\n最佳模型为 {best_model.model_type}，已保存。")
            best_model.print_model_info()
        return best_model
    else:
        model = BusinessRecognitionModel()
        model.train(X_train, y_train, model_type='rf')
        model.save()
        if verbose:
            model.print_model_info()
        return model