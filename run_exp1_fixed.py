"""
运行修复后的实验1
"""

from uav_system.experiments import Experiment1
from uav_system.recognition import train_or_load_recognition_model
import joblib

# 加载识别模型和scaler
recognition_model = train_or_load_recognition_model()
scaler = joblib.load('scaler.pkl')

# 运行实验1
summary = Experiment1.run(recognition_model, scaler, num_steps=150, repeats=5)
