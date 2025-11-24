from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.modeling.data_builder import Sample


@dataclass
class FeatureStats:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    seq_mean: np.ndarray
    seq_std: np.ndarray

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.feature_mean),
            torch.from_numpy(self.feature_std),
            torch.from_numpy(self.seq_mean),
            torch.from_numpy(self.seq_std),
        )


def compute_feature_stats(samples: Sequence[Sample]) -> FeatureStats:
    feature_matrix = np.stack([s.tabular for s in samples], axis=0)
    seq_tensor = np.stack([s.sequence for s in samples], axis=0)
    feat_mean = feature_matrix.mean(axis=0)
    feat_std = feature_matrix.std(axis=0)
    seq_mean = seq_tensor.reshape(-1, seq_tensor.shape[-1]).mean(axis=0)
    seq_std = seq_tensor.reshape(-1, seq_tensor.shape[-1]).std(axis=0)
    feat_std[feat_std < 1e-6] = 1e-6
    seq_std[seq_std < 1e-6] = 1e-6
    return FeatureStats(feat_mean, feat_std, seq_mean, seq_std)


class SymbolDataset(Dataset):
    def __init__(self, samples: List[Sample], stats: FeatureStats):
        self.samples = samples
        self.stats = stats

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        features = (sample.tabular - self.stats.feature_mean) / self.stats.feature_std
        sequence = (sample.sequence - self.stats.seq_mean) / self.stats.seq_std
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(sequence).float(),
            torch.tensor(sample.class_label, dtype=torch.long),
            sample.symbol,
            sample.signal_date.isoformat(),  # 转换为字符串以便 collate
            torch.tensor(sample.value_score, dtype=torch.float32),
        )


def split_by_date(
    samples: List[Sample],
    train_ratio: float = 0.5,
    val_ratio: float = 0.3,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    按日期划分训练/验证/测试集

    默认比例: 5:3:2 (train:val:test)
    这样可以给验证集和测试集更多样本，提高评估的可靠性
    """
    samples = sorted(samples, key=lambda s: s.signal_date)
    dates = sorted({s.signal_date for s in samples})
    n_dates = len(dates)
    train_cut = int(n_dates * train_ratio)
    val_cut = int(n_dates * (train_ratio + val_ratio))
    train_dates = set(dates[:train_cut])
    val_dates = set(dates[train_cut:val_cut])
    train = [s for s in samples if s.signal_date in train_dates]
    val = [s for s in samples if s.signal_date in val_dates]
    test = [s for s in samples if s.signal_date not in train_dates and s.signal_date not in val_dates]
    return train, val, test


__all__ = ["FeatureStats", "compute_feature_stats", "SymbolDataset", "split_by_date"]
