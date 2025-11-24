#!/usr/bin/env python3
"""
持续学习系统：基于实盘数据持续进化模型

核心功能：
1. 从实盘交易结果收集新样本
2. 增量更新模型（避免灾难性遗忘）
3. 定期评估模型性能
4. 自动触发重训练
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.modeling.data_builder_v2 import build_samples_from_backtest
from scripts.modeling.dataset import FeatureStats, SymbolDataset, compute_feature_stats, split_by_date
from scripts.modeling.features import SEQ_COLUMNS, TABULAR_FEATURE_NAMES
from scripts.modeling.model import TransformerRanker, expected_value_from_logits


class ContinualLearner:
    """持续学习管理器"""

    def __init__(
        self,
        model_dir: Path = Path("models"),
        live_trades_csv: Path = Path("data/live_trades.csv"),
        daily_dir: Path = Path("data/daily_klines"),
        hourly_dir: Path = Path("data/hourly_klines"),
        backup_dir: Path = Path("models/backups"),
    ):
        self.model_dir = model_dir
        self.live_trades_csv = live_trades_csv
        self.daily_dir = daily_dir
        self.hourly_dir = hourly_dir
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # 加载当前模型
        self.model, self.meta = self._load_model()
        self.device = torch.device("cpu")
        self.model.to(self.device)

        # 持续学习配置
        self.min_new_samples = 10  # 触发增量学习的最小新样本数
        self.retrain_interval_days = 30  # 完全重训练的间隔天数
        self.learning_rate = 1e-4  # 增量学习使用更小的学习率

    def _load_model(self) -> Tuple[TransformerRanker, dict]:
        """加载模型和元数据"""
        model_path = self.model_dir / "rank_model.pt"
        meta_path = self.model_dir / "rank_model_meta.json"

        with open(meta_path, "r") as f:
            meta = json.load(f)

        model = TransformerRanker(**meta["model_kwargs"])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        return model, meta

    def _backup_current_model(self, tag: str = ""):
        """备份当前模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / f"{timestamp}_{tag}"
        backup_subdir.mkdir(parents=True, exist_ok=True)

        shutil.copy(self.model_dir / "rank_model.pt", backup_subdir / "rank_model.pt")
        shutil.copy(self.model_dir / "rank_model_meta.json", backup_subdir / "rank_model_meta.json")

        print(f"✓ 模型已备份到: {backup_subdir}")
        return backup_subdir

    def collect_new_samples(self, since_date: str = None) -> List:
        """从实盘交易记录中收集新样本"""
        if not self.live_trades_csv.exists():
            print(f"✗ 实盘交易记录不存在: {self.live_trades_csv}")
            return []

        # 读取实盘交易记录
        live_trades = pd.read_csv(self.live_trades_csv)

        # 过滤出已平仓且有收益数据的交易
        closed_trades = live_trades[live_trades["status"] == "closed"].copy()

        if since_date:
            closed_trades = closed_trades[closed_trades["entry_time"] >= since_date]

        print(f"✓ 发现 {len(closed_trades)} 笔已平仓交易")

        # 转换为训练样本格式
        # 假设live_trades.csv包含: symbol, entry_time, exit_time, pnl_pct 等字段
        samples = build_samples_from_backtest(
            backtest_csv=self.live_trades_csv,
            daily_dir=self.daily_dir,
            hourly_dir=self.hourly_dir,
            seq_len=self.meta["seq_len"],
            value_thresholds=tuple(self.meta["value_thresholds"]),
        )

        return samples

    def incremental_update(self, new_samples: List, epochs: int = 5) -> dict:
        """增量更新模型（使用新样本微调）"""
        print(f"\n开始增量学习 ({len(new_samples)} 个新样本)...")

        # 备份当前模型
        self._backup_current_model(tag="before_incremental")

        # 计算新样本的归一化统计（使用原有统计 + EMA更新）
        old_stats = FeatureStats(
            feature_mean=np.array(self.meta["feature_mean"]),
            feature_std=np.array(self.meta["feature_std"]),
            seq_mean=np.array(self.meta["seq_mean"]),
            seq_std=np.array(self.meta["seq_std"]),
        )

        # 创建数据集
        dataset = SymbolDataset(new_samples, old_stats)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # 使用更小的学习率微调
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        ce = nn.CrossEntropyLoss()
        class_values = torch.tensor(self.meta["class_values"], dtype=torch.float32)

        self.model.train()
        metrics = {"loss": [], "accuracy": []}

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_correct = 0
            total = 0

            for features, sequences, labels, _, _, _ in loader:
                features = features.to(self.device)
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(sequences, features)
                loss = ce(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()
                total += labels.size(0)

            avg_loss = total_loss / max(total, 1)
            accuracy = total_correct / max(total, 1)
            metrics["loss"].append(avg_loss)
            metrics["accuracy"].append(accuracy)

            print(f"  Epoch {epoch}: loss={avg_loss:.4f}, acc={accuracy:.3f}")

        # 保存更新后的模型
        self._save_updated_model()

        print(f"✓ 增量学习完成")
        return metrics

    def full_retrain(self, all_trades_csv: Path, epochs: int = 50) -> dict:
        """完全重训练模型（使用所有历史+新数据）"""
        print(f"\n开始完全重训练...")

        # 备份当前模型
        self._backup_current_model(tag="before_retrain")

        # 构建所有样本
        samples = build_samples_from_backtest(
            backtest_csv=all_trades_csv,
            daily_dir=self.daily_dir,
            hourly_dir=self.hourly_dir,
            seq_len=self.meta["seq_len"],
            value_thresholds=tuple(self.meta["value_thresholds"]),
        )

        print(f"✓ 总样本数: {len(samples)}")

        # 划分数据集
        train_samples, val_samples, test_samples = split_by_date(samples, train_ratio=0.5, val_ratio=0.3)
        print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

        # 重新计算归一化统计
        stats = compute_feature_stats(train_samples)
        train_dataset = SymbolDataset(train_samples, stats)
        val_dataset = SymbolDataset(val_samples, stats)

        # 重新初始化模型
        self.model = TransformerRanker(**self.meta["model_kwargs"]).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()
        class_values = torch.tensor(self.meta["class_values"], dtype=torch.float32)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            # 训练
            self.model.train()
            train_loss = 0.0
            train_total = 0

            for features, sequences, labels, _, _, _ in train_loader:
                features = features.to(self.device)
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(sequences, features)
                loss = ce(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * labels.size(0)
                train_total += labels.size(0)

            avg_train_loss = train_loss / max(train_total, 1)

            # 验证
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for features, sequences, labels, _, _, _ in val_loader:
                    features = features.to(self.device)
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.model(sequences, features)
                    loss = ce(logits, labels)

                    val_loss += loss.item() * labels.size(0)
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            print(f"  Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_acc={val_acc:.3f}")

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = self.model.state_dict()

        # 恢复最佳模型
        if best_state:
            self.model.load_state_dict(best_state)

        # 更新元数据
        self.meta["feature_mean"] = stats.feature_mean.tolist()
        self.meta["feature_std"] = stats.feature_std.tolist()
        self.meta["seq_mean"] = stats.seq_mean.tolist()
        self.meta["seq_std"] = stats.seq_std.tolist()

        # 保存模型
        self._save_updated_model()

        print(f"✓ 完全重训练完成，最佳验证损失: {best_val_loss:.4f}")
        return {"best_val_loss": best_val_loss, "total_samples": len(samples)}

    def _save_updated_model(self):
        """保存更新后的模型"""
        model_path = self.model_dir / "rank_model.pt"
        meta_path = self.model_dir / "rank_model_meta.json"

        torch.save(self.model.state_dict(), model_path)

        # 添加更新时间戳
        self.meta["last_updated"] = datetime.now().isoformat()

        with open(meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)

        print(f"✓ 模型已保存: {model_path}")

    def evaluate_on_recent_trades(self, days: int = 7) -> dict:
        """在最近的交易上评估模型性能"""
        since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        recent_samples = self.collect_new_samples(since_date=since_date)

        if len(recent_samples) < 3:
            return {"error": "样本不足，无法评估"}

        # 使用当前统计创建数据集
        stats = FeatureStats(
            feature_mean=np.array(self.meta["feature_mean"]),
            feature_std=np.array(self.meta["feature_std"]),
            seq_mean=np.array(self.meta["seq_mean"]),
            seq_std=np.array(self.meta["seq_std"]),
        )
        dataset = SymbolDataset(recent_samples, stats)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        # 评估
        self.model.eval()
        ce = nn.CrossEntropyLoss()
        class_values = torch.tensor(self.meta["class_values"], dtype=torch.float32)

        total_loss = 0.0
        total_correct = 0
        total = 0

        with torch.no_grad():
            for features, sequences, labels, _, _, _ in loader:
                features = features.to(self.device)
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(sequences, features)
                loss = ce(logits, labels)

                total_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()
                total += labels.size(0)

        return {
            "samples": total,
            "loss": total_loss / max(total, 1),
            "accuracy": total_correct / max(total, 1),
            "period_days": days,
        }

    def auto_update_workflow(self):
        """自动更新工作流"""
        print("=" * 60)
        print("自动持续学习工作流")
        print("=" * 60)

        # 1. 收集新样本
        print("\n[1/4] 收集实盘新样本...")
        # 获取上次更新时间
        last_updated = self.meta.get("last_updated")
        if last_updated:
            last_updated_dt = datetime.fromisoformat(last_updated)
            days_since_update = (datetime.now() - last_updated_dt).days
            print(f"  距离上次更新: {days_since_update} 天")
        else:
            last_updated = None
            days_since_update = 999

        new_samples = self.collect_new_samples(since_date=last_updated)

        if len(new_samples) == 0:
            print("  ✗ 没有新样本，跳过更新")
            return

        # 2. 评估当前模型
        print("\n[2/4] 评估当前模型性能...")
        recent_metrics = self.evaluate_on_recent_trades(days=7)
        print(f"  最近7天: {recent_metrics}")

        # 3. 决定更新策略
        print("\n[3/4] 决定更新策略...")
        if days_since_update >= self.retrain_interval_days:
            print(f"  → 距上次更新 {days_since_update} 天 >= {self.retrain_interval_days} 天")
            print(f"  → 触发完全重训练")
            strategy = "full_retrain"
        elif len(new_samples) >= self.min_new_samples:
            print(f"  → 新样本数 {len(new_samples)} >= {self.min_new_samples}")
            print(f"  → 触发增量更新")
            strategy = "incremental"
        else:
            print(f"  → 新样本不足 ({len(new_samples)} < {self.min_new_samples})")
            print(f"  → 跳过更新")
            return

        # 4. 执行更新
        print("\n[4/4] 执行模型更新...")
        if strategy == "incremental":
            self.incremental_update(new_samples, epochs=10)
        else:
            # 合并所有交易数据
            all_trades_csv = Path("data/all_trades.csv")
            if not all_trades_csv.exists():
                print("  ✗ 缺少完整交易记录文件: data/all_trades.csv")
                print("  → 改用增量更新")
                self.incremental_update(new_samples, epochs=10)
            else:
                self.full_retrain(all_trades_csv, epochs=50)

        # 5. 重新评估
        print("\n[5/5] 评估更新后模型...")
        new_metrics = self.evaluate_on_recent_trades(days=7)
        print(f"  更新后性能: {new_metrics}")

        if "accuracy" in recent_metrics and "accuracy" in new_metrics:
            acc_change = new_metrics["accuracy"] - recent_metrics["accuracy"]
            print(f"  准确率变化: {acc_change:+.3f}")

        print("\n✓ 持续学习工作流完成")


def main():
    parser = argparse.ArgumentParser(description="持续学习系统")
    parser.add_argument("--mode", choices=["auto", "incremental", "retrain", "eval"], default="auto")
    parser.add_argument("--live-trades", type=Path, default=Path("data/live_trades.csv"))
    parser.add_argument("--all-trades", type=Path, default=Path("data/all_trades.csv"))
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    learner = ContinualLearner(live_trades_csv=args.live_trades)

    if args.mode == "auto":
        # 自动工作流
        learner.auto_update_workflow()

    elif args.mode == "eval":
        # 仅评估
        metrics = learner.evaluate_on_recent_trades(days=7)
        print("最近7天模型性能:")
        print(json.dumps(metrics, indent=2))

    elif args.mode == "incremental":
        # 增量学习
        new_samples = learner.collect_new_samples()
        if new_samples:
            learner.incremental_update(new_samples, epochs=args.epochs)
        else:
            print("没有新样本")

    elif args.mode == "retrain":
        # 完全重训练
        if args.all_trades.exists():
            learner.full_retrain(args.all_trades, epochs=args.epochs)
        else:
            print(f"错误: 找不到 {args.all_trades}")


if __name__ == "__main__":
    main()
