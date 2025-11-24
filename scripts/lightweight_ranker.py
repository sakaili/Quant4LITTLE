#!/usr/bin/env python3
"""
轻量级模型推理器 - 适用于资源受限的ECS服务器

特点:
1. 不依赖训练代码，只做推理
2. 内存占用小（200-300MB）
3. 使用ONNX优化（可选）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# 如果有ONNX，可以进一步优化
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class LightweightRanker:
    """轻量级排序模型（仅推理）"""

    def __init__(self, model_dir: Path = Path("models")):
        self.model_dir = model_dir
        self.model = None
        self.meta = None
        self.device = torch.device("cpu")

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载模型和元数据"""
        model_path = self.model_dir / "rank_model.pt"
        meta_path = self.model_dir / "rank_model_meta.json"

        # 加载元数据
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        # 尝试使用ONNX（更快，内存更小）
        onnx_path = self.model_dir / "rank_model.onnx"
        if HAS_ONNX and onnx_path.exists():
            print(f"使用ONNX模型: {onnx_path}")
            self.model = ort.InferenceSession(str(onnx_path))
            self.use_onnx = True
        else:
            # 使用PyTorch
            from scripts.modeling.model import TransformerRanker
            self.model = TransformerRanker(**self.meta["model_kwargs"])
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()
            self.use_onnx = False
            print(f"使用PyTorch模型: {model_path}")

    def predict(self, tabular: np.ndarray, sequence: np.ndarray) -> Tuple[float, int, np.ndarray]:
        """
        预测单个样本

        Returns:
            expected_value: 期望收益分数
            pred_class: 预测类别 (0/1/2)
            probs: 三个类别的概率
        """
        # 归一化
        feat_mean = np.array(self.meta["feature_mean"])
        feat_std = np.array(self.meta["feature_std"])
        seq_mean = np.array(self.meta["seq_mean"])
        seq_std = np.array(self.meta["seq_std"])

        tabular_norm = (tabular - feat_mean) / feat_std
        sequence_norm = (sequence - seq_mean) / seq_std

        if self.use_onnx:
            # ONNX推理（注意：输入名称需要与convert_to_onnx.py匹配）
            onnx_inputs = {
                "tabular": tabular_norm.astype(np.float32)[np.newaxis, :],
                "sequence": sequence_norm.astype(np.float32)[np.newaxis, :],
            }
            logits = self.model.run(None, onnx_inputs)[0][0]
            probs = self._softmax(logits)
        else:
            # PyTorch推理
            with torch.no_grad():
                tab_tensor = torch.from_numpy(tabular_norm).float().unsqueeze(0)
                seq_tensor = torch.from_numpy(sequence_norm).float().unsqueeze(0)
                logits = self.model(seq_tensor, tab_tensor)
                probs = torch.softmax(logits, dim=-1).squeeze().numpy()

        # 计算期望值
        class_values = np.array(self.meta["class_values"])
        expected_value = (probs * class_values).sum()
        pred_class = int(probs.argmax())

        return expected_value, pred_class, probs

    def _softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


def check_resources():
    """检查系统资源"""
    import psutil

    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()

    print("系统资源:")
    print(f"  CPU核心数: {cpu_count}")
    print(f"  总内存: {memory.total / 1024**3:.2f} GB")
    print(f"  可用内存: {memory.available / 1024**3:.2f} GB")
    print(f"  内存使用率: {memory.percent}%")

    # 检查是否满足最低要求
    min_memory_gb = 0.5
    if memory.available / 1024**3 < min_memory_gb:
        print(f"\n⚠️  警告: 可用内存不足 {min_memory_gb}GB")
        print("   建议关闭其他进程或升级服务器")
        return False

    return True


def main():
    """测试轻量级推理"""
    print("=" * 60)
    print("轻量级模型推理测试")
    print("=" * 60)

    # 检查资源
    if not check_resources():
        return 1

    # 加载模型
    ranker = LightweightRanker()

    # 模拟特征
    print("\n生成测试特征...")
    tabular = np.random.randn(15)
    sequence = np.random.randn(24, 5)

    # 预测
    print("执行推理...")
    expected_value, pred_class, probs = ranker.predict(tabular, sequence)

    print("\n预测结果:")
    print(f"  期望收益: {expected_value:.4f}")
    print(f"  预测类别: {pred_class}")
    print(f"  概率分布: {probs}")

    # 内存使用
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024**2
    print(f"\n当前进程内存: {memory_mb:.1f} MB")

    print("\n✓ 轻量级推理测试成功")


if __name__ == "__main__":
    import sys
    sys.exit(main())
