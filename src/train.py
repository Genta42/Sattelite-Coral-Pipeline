"""
Training loop for the CoralLSTM model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .build_sequences import inverse_class_weights
from .dataset import CoralSequenceDataset, compute_norm_stats
from .model import CoralLSTM

logger = logging.getLogger(__name__)


def _get_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _macro_f1(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1s.append(f1)
    return float(np.mean(f1s))


def train_model(
    seq_dir: str | Path,
    out_dir: str | Path,
    **hparams: Any,
) -> Path:
    """
    Train the CoralLSTM model and return the path to the best checkpoint.
    """
    seq_dir = Path(seq_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge defaults with overrides
    cfg: Dict[str, Any] = {**C.TRAIN_DEFAULTS, **hparams}
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = _get_device(cfg.get("device"))
    logger.info("Training on device: %s", device)

    # Load data
    train_df = pd.read_parquet(seq_dir / "sequences_train.parquet")
    val_df = pd.read_parquet(seq_dir / "sequences_val.parquet")

    max_samples = cfg.pop("max_samples", None)
    if max_samples and len(train_df) > max_samples:
        logger.info("Subsampling training data: %d → %d", len(train_df), max_samples)
        train_df = train_df.sample(n=max_samples, random_state=seed).reset_index(
            drop=True
        )
    if max_samples and len(val_df) > max_samples // 5:
        val_cap = max(max_samples // 5, 10_000)
        logger.info("Subsampling val data: %d → %d", len(val_df), val_cap)
        val_df = val_df.sample(n=val_cap, random_state=seed).reset_index(drop=True)

    logger.info("Train: %d samples, Val: %d samples", len(train_df), len(val_df))

    # Normalization stats from training data only
    norm_stats = compute_norm_stats(train_df)

    train_ds = CoralSequenceDataset(train_df, norm_stats)
    val_ds = CoralSequenceDataset(val_df, norm_stats)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    # Class weights
    weights = inverse_class_weights(train_df)
    weight_tensor = torch.zeros(C.NUM_CLASSES, dtype=torch.float32)
    for cls, w in weights.items():
        weight_tensor[cls] = w
    weight_tensor = weight_tensor.to(device)

    # Model
    model = CoralLSTM(
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = out_dir / "best_model.pt"

    for epoch in range(1, cfg["max_epochs"] + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_batches = 0
        for x_seq, x_static, y in train_loader:
            x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x_seq, x_static)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x_seq, x_static, y in val_loader:
                x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)
                logits = model(x_seq, x_static)
                loss = criterion(logits, y)
                val_loss += loss.item()
                val_batches += 1
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(y.cpu().numpy())

        avg_val_loss = val_loss / max(val_batches, 1)
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        val_acc = (preds == labels).mean()
        val_f1 = _macro_f1(preds, labels, C.NUM_CLASSES)

        scheduler.step(avg_val_loss)

        logger.info(
            "Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_f1=%.4f",
            epoch,
            avg_train_loss,
            avg_val_loss,
            val_acc,
            val_f1,
        )

        # Early stopping + checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "norm_stats": norm_stats,
                    "config": cfg,
                    "metrics": {
                        "val_loss": avg_val_loss,
                        "val_acc": float(val_acc),
                        "val_f1": val_f1,
                    },
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            logger.info("  → Saved best checkpoint (val_loss=%.4f)", avg_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                logger.info("Early stopping at epoch %d", epoch)
                break

    logger.info("Training complete. Best checkpoint: %s", checkpoint_path)
    return checkpoint_path
