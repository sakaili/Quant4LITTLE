#!/usr/bin/env python3
"""
诊断数据划分问题的脚本
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.modeling.data_builder import build_samples
from scripts.modeling.dataset import split_by_date


def main():
    print("=" * 60)
    print("数据划分诊断")
    print("=" * 60)

    # 构建样本
    print("\n正在加载样本...")
    samples = build_samples(
        candidates_dir=Path("data/daily_scans"),
        backtest_csv=Path("data/backtest_trades.csv"),
        daily_dir=Path("data/daily_klines"),
        hourly_dir=Path("data/hourly_klines"),
    )
    print(f"总样本数: {len(samples)}")

    # 统计类别分布
    print("\n总体类别分布:")
    class_counts = Counter(s.class_label for s in samples)
    for label, count in sorted(class_counts.items()):
        pct = count / len(samples) * 100
        print(f"  类别 {label}: {count} ({pct:.1f}%)")

    # 统计日期分布
    dates = sorted({s.signal_date for s in samples})
    print(f"\n日期范围: {dates[0]} 到 {dates[-1]}")
    print(f"总日期数: {len(dates)}")

    # 当前的划分方式
    print("\n" + "=" * 60)
    print("当前划分方式 (70/15/15)")
    print("=" * 60)

    train, val, test = split_by_date(samples, train_ratio=0.7, val_ratio=0.15)

    def print_split_info(name: str, split):
        print(f"\n{name}集:")
        print(f"  样本数: {len(split)}")
        if len(split) > 0:
            split_dates = sorted({s.signal_date for s in split})
            print(f"  日期数: {len(split_dates)}")
            print(f"  日期范围: {split_dates[0]} 到 {split_dates[-1]}")

            # 类别分布
            split_classes = Counter(s.class_label for s in split)
            print(f"  类别分布:")
            for label, count in sorted(split_classes.items()):
                pct = count / len(split) * 100
                print(f"    类别 {label}: {count} ({pct:.1f}%)")

            # 每个日期的样本数
            samples_per_date = Counter(s.signal_date for s in split)
            avg_per_date = len(split) / len(split_dates)
            print(f"  平均每日样本数: {avg_per_date:.1f}")

    print_split_info("训练", train)
    print_split_info("验证", val)
    print_split_info("测试", test)

    # 检查是否存在问题
    print("\n" + "=" * 60)
    print("潜在问题检测")
    print("=" * 60)

    issues = []

    # 检查验证集大小
    if len(val) < 50:
        issues.append(f"⚠️ 验证集样本过少 ({len(val)} 个)")

    # 检查类别不平衡
    if len(val) > 0:
        val_classes = Counter(s.class_label for s in val)
        max_class_pct = max(val_classes.values()) / len(val) * 100
        if max_class_pct > 80:
            issues.append(f"⚠️ 验证集类别严重不平衡 (最大类占 {max_class_pct:.1f}%)")

    # 检查是否有日期只有很少样本
    if len(val) > 0:
        val_dates = Counter(s.signal_date for s in val)
        single_sample_dates = sum(1 for count in val_dates.values() if count == 1)
        if single_sample_dates > len(val_dates) * 0.3:
            issues.append(f"⚠️ 验证集中 {single_sample_dates} 个日期只有1个样本")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✓ 未发现明显问题")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
