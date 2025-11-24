#!/usr/bin/env python3
"""
使用训练好的排序模型对候选标的进行打分和排序

用法示例:
    # 对今天的候选进行打分
    python scripts/use_ranker_model.py --date 2025-01-20

    # 指定模型路径
    python scripts/use_ranker_model.py --date 2025-01-20 --model-dir models
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from scripts.indicator_utils import add_indicators
from scripts.modeling.features import SEQ_COLUMNS, TABULAR_FEATURE_NAMES, build_tabular_features
from scripts.modeling.model import TransformerRanker, expected_value_from_logits


def load_model(model_dir: Path) -> Tuple[TransformerRanker, dict]:
    """加载模型和元数据"""
    model_path = model_dir / "rank_model.pt"
    meta_path = model_dir / "rank_model_meta.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Model files not found in {model_dir}")

    # 加载元数据
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # 构建模型
    model = TransformerRanker(**meta["model_kwargs"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print(f"✓ 模型加载成功: {model_path}")
    print(f"  特征维度: {meta['model_kwargs']['feature_dim']}")
    print(f"  序列长度: {meta['model_kwargs']['seq_len']}")
    print(f"  类别阈值: {meta['value_thresholds']}")

    return model, meta


def load_candidate_data(date: str, daily_dir: Path, hourly_dir: Path) -> pd.DataFrame:
    """加载候选标的的K线数据"""
    candidates_file = Path("data/daily_scans") / f"candidates_{date}.csv"

    if not candidates_file.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_file}")

    candidates = pd.read_csv(candidates_file)
    print(f"✓ 读取候选文件: {candidates_file}")
    print(f"  候选数量: {len(candidates)}")

    return candidates


def prepare_features(
    symbol: str, signal_date: str, daily_dir: Path, hourly_dir: Path, seq_len: int = 24
) -> Tuple[np.ndarray, np.ndarray] | None:
    """为单个标的准备特征"""
    daily_file = daily_dir / f"{symbol}.csv"
    hourly_file = hourly_dir / f"{symbol}.csv"

    if not daily_file.exists() or not hourly_file.exists():
        return None

    # 读取K线数据
    daily_df = pd.read_csv(daily_file, parse_dates=["open_time"])
    hourly_df = pd.read_csv(hourly_file, parse_dates=["open_time"])

    # 添加指标
    daily_df = add_indicators(daily_df)
    hourly_df = add_indicators(hourly_df)

    # 找到信号时间点
    signal_ts = pd.to_datetime(signal_date)
    daily_mask = daily_df["open_time"] <= signal_ts
    hourly_mask = hourly_df["open_time"] <= signal_ts

    if daily_mask.sum() < 30 or hourly_mask.sum() < seq_len:
        return None

    signal_daily = daily_df[daily_mask].iloc[-1]
    hist_hourly = hourly_df[hourly_mask].iloc[-seq_len:]

    # 构建表格特征
    tabular = build_tabular_features(signal_daily, daily_df[daily_mask])

    # 构建序列特征
    sequence = hist_hourly[SEQ_COLUMNS].values

    return tabular, sequence


def score_candidates(
    candidates: pd.DataFrame,
    model: TransformerRanker,
    meta: dict,
    daily_dir: Path,
    hourly_dir: Path,
    signal_date: str,
) -> pd.DataFrame:
    """对所有候选标的进行打分"""
    # 准备归一化参数
    feat_mean = torch.tensor(meta["feature_mean"], dtype=torch.float32)
    feat_std = torch.tensor(meta["feature_std"], dtype=torch.float32)
    seq_mean = torch.tensor(meta["seq_mean"], dtype=torch.float32)
    seq_std = torch.tensor(meta["seq_std"], dtype=torch.float32)
    class_values = torch.tensor(meta["class_values"], dtype=torch.float32)

    results = []
    seq_len = meta["model_kwargs"]["seq_len"]

    print("\n正在对候选标的进行打分...")
    for idx, row in candidates.iterrows():
        symbol = row["symbol"]

        # 准备特征
        features_tuple = prepare_features(symbol, signal_date, daily_dir, hourly_dir, seq_len)
        if features_tuple is None:
            print(f"  ✗ {symbol}: 数据不足")
            continue

        tabular, sequence = features_tuple

        # 归一化
        tabular_norm = (tabular - feat_mean.numpy()) / feat_std.numpy()
        sequence_norm = (sequence - seq_mean.numpy()) / seq_std.numpy()

        # 转为tensor
        tab_tensor = torch.from_numpy(tabular_norm).float().unsqueeze(0)
        seq_tensor = torch.from_numpy(sequence_norm).float().unsqueeze(0)

        # 模型推理
        with torch.no_grad():
            logits = model(seq_tensor, tab_tensor)
            exp_value = expected_value_from_logits(logits, class_values).item()
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()
            pred_class = logits.argmax(dim=-1).item()

        results.append(
            {
                "symbol": symbol,
                "expected_value": exp_value,
                "pred_class": pred_class,
                "prob_class0": probs[0],
                "prob_class1": probs[1],
                "prob_class2": probs[2],
                **{k: row[k] for k in row.index if k != "symbol"},
            }
        )

        print(f"  ✓ {symbol}: score={exp_value:.4f}, class={pred_class}, probs={probs}")

    results_df = pd.DataFrame(results)
    return results_df


def main():
    parser = argparse.ArgumentParser(description="使用训练好的模型对候选标的打分")
    parser.add_argument("--date", type=str, required=True, help="信号日期 (YYYYMMDD)")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--daily-dir", type=Path, default=Path("data/daily_klines"))
    parser.add_argument("--hourly-dir", type=Path, default=Path("data/hourly_klines"))
    parser.add_argument("--top-k", type=int, default=20, help="选择排名前K的标的")
    parser.add_argument("--output", type=Path, help="输出文件路径（可选）")
    args = parser.parse_args()

    print("=" * 60)
    print("排序模型使用工具")
    print("=" * 60)

    # 1. 加载模型
    model, meta = load_model(args.model_dir)

    # 2. 加载候选数据
    candidates = load_candidate_data(args.date, args.daily_dir, args.hourly_dir)

    # 3. 对候选打分
    results = score_candidates(candidates, model, meta, args.daily_dir, args.hourly_dir, args.date)

    if len(results) == 0:
        print("\n✗ 没有成功打分的标的")
        return 1

    # 4. 按expected_value排序
    results = results.sort_values("expected_value", ascending=False)

    # 5. 输出结果
    print("\n" + "=" * 60)
    print(f"排序结果 (前{args.top_k}名)")
    print("=" * 60)
    print(
        results[["symbol", "expected_value", "pred_class", "prob_class0", "prob_class1", "prob_class2"]].head(
            args.top_k
        )
    )

    # 6. 保存到文件（可选）
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\n✓ 结果已保存到: {args.output}")
    else:
        # 默认保存到data/ranked_candidates/
        output_dir = Path("data/ranked_candidates")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"ranked_{args.date}.csv"
        results.to_csv(output_path, index=False)
        print(f"\n✓ 结果已保存到: {output_path}")

    # 7. 统计信息
    print("\n统计信息:")
    print(f"  总候选数: {len(candidates)}")
    print(f"  成功打分: {len(results)}")
    print(f"  预测Class 0 (差): {(results['pred_class'] == 0).sum()}")
    print(f"  预测Class 1 (中): {(results['pred_class'] == 1).sum()}")
    print(f"  预测Class 2 (优): {(results['pred_class'] == 2).sum()}")
    print(f"  平均期望值: {results['expected_value'].mean():.4f}")
    print(f"  Top{args.top_k}平均期望值: {results.head(args.top_k)['expected_value'].mean():.4f}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
