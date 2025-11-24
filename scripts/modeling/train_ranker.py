#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.modeling.data_builder_v2 import Sample, build_samples_from_backtest
from scripts.modeling.dataset import FeatureStats, SymbolDataset, compute_feature_stats, split_by_date
from scripts.modeling.features import SEQ_COLUMNS, TABULAR_FEATURE_NAMES
from scripts.modeling.model import TransformerRanker, expected_value_from_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer ranker for candidate selection.")
    parser.add_argument("--candidates-dir", type=Path, default=Path("data/daily_scans"), help="(Unused - kept for compatibility)")
    parser.add_argument("--backtest-csv", type=Path, default=Path("data/backtest_trades.csv"))
    parser.add_argument("--daily-dir", type=Path, default=Path("data/daily_klines"))
    parser.add_argument("--hourly-dir", type=Path, default=Path("data/hourly_klines"))
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--value-threshold-low", type=float, default=-0.2)
    parser.add_argument("--value-threshold-high", type=float, default=0.2)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_model(
    model: TransformerRanker,
    loader: DataLoader,
    class_values: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    ce = nn.CrossEntropyLoss()
    records: List[Dict[str, object]] = []
    with torch.no_grad():
        for features, sequences, labels, symbols, dates, value_targets in loader:
            features = features.to(device)
            sequences = sequences.to(device)
            labels = labels.to(device)
            logits = model(sequences, features)
            loss = ce(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
            exp_val = expected_value_from_logits(logits, class_values.to(device)).cpu().numpy()
            for sym, dt, label, val, score in zip(symbols, dates, labels.cpu().numpy(), value_targets.numpy(), exp_val):
                records.append(
                    {
                        "symbol": sym,
                        "date": dt,
                        "label": int(label),
                        "actual": float(val),
                        "score": float(score),
                    }
                )
    metrics = {"loss": total_loss / max(total, 1), "accuracy": total_correct / max(total, 1)}
    if records:
        metrics.update(_topk_metrics(records))
    return metrics


def _topk_metrics(records: List[Dict[str, object]], k: int = 20) -> Dict[str, float]:
    from collections import defaultdict

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec["date"])].append(rec)
    topk_actual = []
    topk_tp_ratio = []
    for rows in grouped.values():
        rows.sort(key=lambda r: r["score"], reverse=True)
        top = rows[:k]
        if not top:
            continue
        topk_actual.append(np.mean([r["actual"] for r in top]))
        tp_ratio = sum(1 for r in top if r["label"] == 2) / len(top)
        topk_tp_ratio.append(tp_ratio)
    return {
        "topk_actual_mean": float(np.mean(topk_actual)) if topk_actual else 0.0,
        "topk_tp_ratio": float(np.mean(topk_tp_ratio)) if topk_tp_ratio else 0.0,
    }


def train_one_epoch(
    model: TransformerRanker,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_values: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    for features, sequences, labels, _, _, _ in loader:
        features = features.to(device)
        sequences = sequences.to(device)
        labels = labels.to(device)
        logits = model(sequences, features)
        loss = ce(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / max(total, 1)


def save_model(
    *,
    model: TransformerRanker,
    stats: FeatureStats,
    class_values: torch.Tensor,
    output_dir: Path,
    value_thresholds: tuple = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "rank_model.pt"
    meta_path = output_dir / "rank_model_meta.json"
    torch.save(model.state_dict(), model_path)
    feature_mean, feature_std, seq_mean, seq_std = stats.to_tensors()
    meta = {
        "feature_names": TABULAR_FEATURE_NAMES,
        "seq_feature_names": SEQ_COLUMNS,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "seq_mean": seq_mean.tolist(),
        "seq_std": seq_std.tolist(),
        "seq_len": model.seq_len,
        "class_values": class_values.tolist(),
        "value_thresholds": list(value_thresholds) if value_thresholds else None,
        "model_kwargs": {
            "seq_len": model.seq_len,
            "seq_dim": model.seq_dim,
            "feature_dim": model.feature_dim,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved model to {model_path}")


def main() -> None:
    args = parse_args()
    set_seed()
    print(f"Building samples from backtest trades: {args.backtest_csv}")
    samples = build_samples_from_backtest(
        backtest_csv=args.backtest_csv,
        daily_dir=args.daily_dir,
        hourly_dir=args.hourly_dir,
        seq_len=args.seq_len,
        value_thresholds=(args.value_threshold_low, args.value_threshold_high),
    )
    if not samples:
        raise SystemExit("No training samples collected. Ensure backtests and K-line data are present.")

    print(f"\nTotal samples: {len(samples)}")
    print(f"Splitting by date (70/15/15)...")
    train_samples, val_samples, test_samples = split_by_date(samples)
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    stats = compute_feature_stats(train_samples)
    train_dataset = SymbolDataset(train_samples, stats)
    val_dataset = SymbolDataset(val_samples, stats)
    test_dataset = SymbolDataset(test_samples, stats)

    device = torch.device(args.device)
    model = TransformerRanker(
        seq_len=args.seq_len,
        seq_dim=len(SEQ_COLUMNS),
        feature_dim=len(TABULAR_FEATURE_NAMES),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    class_values = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, class_values, device)
        val_metrics = evaluate_model(model, val_loader, class_values, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.3f}")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = model.state_dict()
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, class_values, device)
    print("Test metrics:", test_metrics)
    save_model(
        model=model,
        stats=stats,
        class_values=class_values,
        output_dir=args.output_dir,
        value_thresholds=(args.value_threshold_low, args.value_threshold_high),
    )


if __name__ == "__main__":
    main()
