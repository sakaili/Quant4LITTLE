#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch模型转ONNX - 用于ECS低内存部署

将PyTorch Transformer模型转换为ONNX格式
内存占用: 800MB → 400MB
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
import json

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.onnx
except ImportError:
    print("❌ 请先安装PyTorch: pip install torch")
    sys.exit(1)

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("❌ 请先安装ONNX: pip install onnx onnxruntime")
    sys.exit(1)

from scripts.modeling.train_ranker import TransformerRanker


def convert_to_onnx(
    pytorch_model_path: Path = Path("models/rank_model.pt"),
    onnx_model_path: Path = Path("models/rank_model.onnx"),
    opset_version: int = 14
):
    """
    转换PyTorch模型为ONNX格式

    Args:
        pytorch_model_path: PyTorch模型路径
        onnx_model_path: 输出ONNX模型路径
        opset_version: ONNX opset版本
    """
    print(f"\n{'='*70}")
    print(f"  🔄 PyTorch → ONNX 模型转换")
    print(f"{'='*70}\n")

    # 1. 加载元数据
    print(f"[1/5] 加载模型和元数据")

    if not pytorch_model_path.exists():
        print(f"  ❌ 模型文件不存在: {pytorch_model_path}")
        return False

    meta_path = pytorch_model_path.parent / "rank_model_meta.json"
    if not meta_path.exists():
        print(f"  ❌ 元数据文件不存在: {meta_path}")
        return False

    try:
        # 加载元数据
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # 加载模型state_dict
        state_dict = torch.load(pytorch_model_path, map_location='cpu')

        # 从元数据中提取模型配置
        seq_len = meta['seq_len']
        seq_dim = meta['model_kwargs']['seq_dim']
        tab_dim = meta['model_kwargs']['feature_dim']
        num_classes = len(meta['class_values'])

        print(f"  ✅ 模型加载成功")
        print(f"  配置: tab_dim={tab_dim}, seq_len={seq_len}, "
              f"seq_dim={seq_dim}, num_classes={num_classes}")

    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. 重建模型
    print(f"\n[2/5] 重建模型结构")

    try:
        model = TransformerRanker(
            feature_dim=tab_dim,  # 注意：参数名是feature_dim不是tab_dim
            seq_len=seq_len,
            seq_dim=seq_dim,
            num_classes=num_classes,
            d_model=64,  # 从错误信息看出是64不是128
            nhead=4,
            num_layers=2,
            dropout=0.1
        )

        model.load_state_dict(state_dict)
        model.eval()

        print(f"  ✅ 模型结构重建成功")

    except Exception as e:
        print(f"  ❌ 重建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. 创建虚拟输入
    print(f"\n[3/5] 创建虚拟输入")

    dummy_tabular = torch.randn(1, tab_dim)
    dummy_sequence = torch.randn(1, seq_len, seq_dim)

    print(f"  Tabular: {dummy_tabular.shape}")
    print(f"  Sequence: {dummy_sequence.shape}")

    # 4. 导出ONNX
    print(f"\n[4/5] 导出ONNX模型")

    try:
        torch.onnx.export(
            model,
            (dummy_tabular, dummy_sequence),
            str(onnx_model_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['tabular', 'sequence'],
            output_names=['output'],
            dynamic_axes={
                'tabular': {0: 'batch_size'},
                'sequence': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        print(f"  ✅ ONNX模型导出成功: {onnx_model_path}")

    except Exception as e:
        print(f"  ❌ 导出失败: {e}")
        return False

    # 5. 验证ONNX模型
    print(f"\n[5/5] 验证ONNX模型")

    try:
        # 检查模型
        onnx_model = onnx.load(str(onnx_model_path))
        onnx.checker.check_model(onnx_model)
        print(f"  ✅ ONNX模型结构验证通过")

        # 测试推理
        ort_session = ort.InferenceSession(
            str(onnx_model_path),
            providers=['CPUExecutionProvider']
        )

        ort_inputs = {
            'tabular': dummy_tabular.numpy(),
            'sequence': dummy_sequence.numpy()
        }

        ort_outputs = ort_session.run(None, ort_inputs)
        pytorch_outputs = model(dummy_tabular, dummy_sequence).detach().numpy()

        # 对比输出
        import numpy as np
        diff = np.abs(ort_outputs[0] - pytorch_outputs).max()

        print(f"  PyTorch vs ONNX 最大误差: {diff:.6f}")

        if diff < 1e-4:
            print(f"  ✅ 推理结果一致（误差 < 1e-4）")
        else:
            print(f"  ⚠️  推理结果存在差异（误差 {diff:.6f}）")

    except Exception as e:
        print(f"  ❌ 验证失败: {e}")
        return False

    # 6. 保存元数据
    meta_path = onnx_model_path.parent / "rank_model_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  ✅ 元数据已保存: {meta_path}")

    # 7. 文件大小对比
    print(f"\n{'='*70}")
    print(f"  📊 文件大小对比")
    print(f"{'='*70}")

    pytorch_size = pytorch_model_path.stat().st_size / 1024 / 1024
    onnx_size = onnx_model_path.stat().st_size / 1024 / 1024

    print(f"  PyTorch: {pytorch_size:.2f} MB")
    print(f"  ONNX:    {onnx_size:.2f} MB")
    print(f"  节省:    {pytorch_size - onnx_size:.2f} MB ({(1 - onnx_size/pytorch_size)*100:.1f}%)")

    print(f"\n{'='*70}")
    print(f"  ✅ 转换完成!")
    print(f"{'='*70}\n")

    print(f"使用方法:")
    print(f"  1. 将 {onnx_model_path} 上传到ECS服务器")
    print(f"  2. 将 {meta_path} 上传到ECS服务器")
    print(f"  3. 使用 requirements_onnx.txt 安装依赖")
    print(f"  4. 运行 paper_trader.py 时会自动使用ONNX模型")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch模型转ONNX")
    parser.add_argument(
        "--pytorch-model",
        type=Path,
        default=Path("models/rank_model.pt"),
        help="PyTorch模型路径"
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=Path("models/rank_model.onnx"),
        help="输出ONNX模型路径"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset版本"
    )

    args = parser.parse_args()

    success = convert_to_onnx(
        pytorch_model_path=args.pytorch_model,
        onnx_model_path=args.onnx_model,
        opset_version=args.opset_version
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
