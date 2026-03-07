"""
PyTorch Dataset for coral bleaching LSTM sequences.

Loads split parquet files produced by build_sequences, parses JSON sequence
columns into tensors, and applies z-score normalization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as C

logger = logging.getLogger(__name__)


def compute_norm_stats(
    df: pd.DataFrame,
    max_sample: int = 100_000,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute per-feature mean/std from a training DataFrame.

    Returns dict mapping feature name to (mean, std).
    Sentinel values (-1) are excluded from the computation.
    If the DataFrame has more than *max_sample* rows, a random subsample
    is used to keep memory bounded.
    """
    if len(df) > max_sample:
        df = df.sample(n=max_sample, random_state=42)

    stats: Dict[str, Tuple[float, float]] = {}

    for col in C.SEQUENCE_FEATURES:
        seq_col = f"x_{col}_seq"
        if seq_col not in df.columns:
            continue
        values = []
        for raw in df[seq_col]:
            arr = json.loads(raw) if isinstance(raw, str) else raw
            values.extend(v for v in arr if v != -1.0)
        vals = np.array(values, dtype=np.float32)
        stats[col] = (float(vals.mean()), float(vals.std() + 1e-8))

    for col in C.STATIC_FEATURES:
        if col in df.columns:
            vals = df[col].values.astype(np.float32)
            stats[col] = (float(vals.mean()), float(vals.std() + 1e-8))

    return stats


class CoralSequenceDataset(Dataset):
    """
    PyTorch dataset that yields (x_seq, x_static, y) tuples.

    x_seq:    (lookback, n_seq_features) float tensor
    x_static: (n_static_features,) float tensor
    y:        scalar long tensor (BAA class 0–4)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        norm_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self.labels = df["y_baa_cat"].values.astype(np.int64)

        # Parse sequence columns
        seq_arrays = []
        for col in C.SEQUENCE_FEATURES:
            seq_col = f"x_{col}_seq"
            if seq_col not in df.columns:
                continue
            parsed = df[seq_col].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
            seq_arrays.append(np.stack(parsed.values).astype(np.float32))

        # (N, lookback, n_features)
        self.x_seq = np.stack(seq_arrays, axis=-1)

        # Static features
        static_cols = [c for c in C.STATIC_FEATURES if c in df.columns]
        self.x_static = df[static_cols].values.astype(np.float32)

        # Normalize
        if norm_stats is not None:
            self._normalize(norm_stats)

    def _normalize(self, stats: Dict[str, Tuple[float, float]]) -> None:
        for i, col in enumerate(C.SEQUENCE_FEATURES):
            if col not in stats:
                continue
            mean, std = stats[col]
            mask = self.x_seq[:, :, i] != -1.0
            self.x_seq[:, :, i][mask] = (self.x_seq[:, :, i][mask] - mean) / std
            # Replace -1 sentinels with 0 post-normalization
            self.x_seq[:, :, i][~mask] = 0.0

        for i, col in enumerate(C.STATIC_FEATURES):
            if col not in stats:
                continue
            mean, std = stats[col]
            self.x_static[:, i] = (self.x_static[:, i] - mean) / std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.x_seq[idx]),
            torch.from_numpy(self.x_static[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
