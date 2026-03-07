"""
LSTM model for coral bleaching BAA classification.

Architecture:
    x_seq (B,60,3) → LSTM → h_n[-1] (B,H) → cat(h_last, x_static) → FC → logits
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C


class CoralLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = len(C.SEQUENCE_FEATURES),
        static_size: int = len(C.STATIC_FEATURES),
        hidden_size: int = C.TRAIN_DEFAULTS["hidden_size"],
        num_layers: int = C.TRAIN_DEFAULTS["num_layers"],
        dropout: float = C.TRAIN_DEFAULTS["dropout"],
        num_classes: int = C.NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        fc_in = hidden_size + static_size
        self.head = nn.Sequential(
            nn.Linear(fc_in, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, C), x_static: (B, S)
        _, (h_n, _) = self.lstm(x_seq)
        h_last = h_n[-1]  # (B, H)
        combined = torch.cat([h_last, x_static], dim=1)
        return self.head(combined)
