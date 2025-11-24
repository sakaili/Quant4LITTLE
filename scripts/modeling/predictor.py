from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from scripts.modeling.features import build_hourly_sequence, build_tabular_vector
from scripts.modeling.model import TransformerRanker, expected_value_from_logits


@dataclass
class CandidateInput:
    symbol: str
    signal_date: date
    tabular: np.ndarray
    sequence: np.ndarray


class RankerPredictor:
    def __init__(self, model_path: Path, meta_path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.meta = json.loads(meta_path.read_text())
        self.feature_mean = torch.tensor(self.meta["feature_mean"], dtype=torch.float32, device=self.device)
        self.feature_std = torch.tensor(self.meta["feature_std"], dtype=torch.float32, device=self.device).clamp_min(1e-6)
        self.seq_mean = torch.tensor(self.meta["seq_mean"], dtype=torch.float32, device=self.device)
        self.seq_std = torch.tensor(self.meta["seq_std"], dtype=torch.float32, device=self.device).clamp_min(1e-6)
        self.class_values = torch.tensor(self.meta["class_values"], dtype=torch.float32, device=self.device)
        model_kwargs = self.meta.get("model_kwargs", {})
        model_kwargs.setdefault("seq_len", self.meta.get("seq_len", 24))
        model_kwargs.setdefault("seq_dim", len(self.meta["seq_mean"]))
        model_kwargs.setdefault("feature_dim", len(self.meta["feature_mean"]))
        self.seq_len = model_kwargs["seq_len"]
        self.model = TransformerRanker(**model_kwargs).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def score(self, candidates: List[CandidateInput]) -> List[Dict[str, object]]:
        if not candidates:
            return []
        feats = torch.from_numpy(np.stack([c.tabular for c in candidates])).to(self.device)
        seqs = torch.from_numpy(np.stack([c.sequence for c in candidates])).to(self.device)
        feats = (feats - self.feature_mean) / self.feature_std
        seqs = (seqs - self.seq_mean) / self.seq_std
        with torch.no_grad():
            logits = self.model(seqs, feats)
            probs = torch.softmax(logits, dim=-1)
            exp_scores = expected_value_from_logits(logits, self.class_values)
        results = []
        for candidate, score, prob in zip(candidates, exp_scores.cpu().tolist(), probs.cpu().tolist()):
            results.append(
                {
                    "symbol": candidate.symbol,
                    "score": score,
                    "probs": prob,
                    "signal_date": candidate.signal_date.isoformat(),
                }
            )
        return results


def prepare_candidate_input(
    *,
    symbol: str,
    signal_date: date,
    entry_cutoff: datetime,
    daily_history,
    hourly_history,
    funding_rate: float,
    quote_volume: float,
    market_cap: float,
    seq_len: int = 24,
) -> Optional[CandidateInput]:
    tabular = build_tabular_vector(
        daily_history,
        signal_date,
        funding_rate=funding_rate,
        quote_volume=quote_volume,
        market_cap=market_cap,
    )
    if tabular is None:
        return None
    sequence = build_hourly_sequence(hourly_history, entry_cutoff, seq_len=seq_len)
    return CandidateInput(symbol=symbol, signal_date=signal_date, tabular=tabular, sequence=sequence)


__all__ = ["RankerPredictor", "CandidateInput", "prepare_candidate_input"]
