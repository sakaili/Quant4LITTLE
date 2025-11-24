from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        return x + self.pe[:, :length]


class TransformerRanker(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        seq_dim: int,
        feature_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        mlp_dim: int = 128,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.feature_dim = feature_dim

        self.input_proj = nn.Linear(seq_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional = PositionalEncoding(d_model, max_len=seq_len)

        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(d_model + mlp_dim, num_classes)

    def forward(self, sequence: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        seq_embed = self.input_proj(sequence)
        seq_embed = self.positional(seq_embed)
        seq_out = self.transformer(seq_embed)
        seq_repr = seq_out.mean(dim=1)

        feat_repr = self.feature_mlp(features)
        combined = torch.cat([seq_repr, feat_repr], dim=-1)
        logits = self.classifier(combined)
        return logits


@dataclass
class NormalizationStats:
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    seq_mean: torch.Tensor
    seq_std: torch.Tensor

    def normalize_features(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feature_mean) / self.feature_std

    def normalize_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.seq_mean) / self.seq_std


def build_norm_stats(feature_matrix: torch.Tensor, seq_tensor: torch.Tensor) -> NormalizationStats:
    feat_mean = feature_matrix.mean(dim=0)
    feat_std = feature_matrix.std(dim=0).clamp_min(1e-6)
    seq_mean = seq_tensor.view(-1, seq_tensor.size(-1)).mean(dim=0)
    seq_std = seq_tensor.view(-1, seq_tensor.size(-1)).std(dim=0).clamp_min(1e-6)
    return NormalizationStats(feat_mean, feat_std, seq_mean, seq_std)


def expected_value_from_logits(logits: torch.Tensor, class_values: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return (probs * class_values).sum(dim=-1)


__all__ = ["TransformerRanker", "NormalizationStats", "build_norm_stats", "expected_value_from_logits"]
