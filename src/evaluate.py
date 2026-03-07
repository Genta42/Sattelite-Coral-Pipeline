"""
Evaluation module for the CoralLSTM model.

Loads a saved checkpoint, runs inference on a data split, computes
classification metrics, and writes eval_metrics.json + confusion_matrix.png
to the specified output directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from . import config as C
from .dataset import CoralSequenceDataset
from .model import CoralLSTM

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _plot_confusion_matrix(
    cm: np.ndarray,
    out_path: Path,
) -> None:
    class_labels = [str(i) for i in range(C.NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(C.NUM_CLASSES),
        yticks=np.arange(C.NUM_CLASSES),
        xticklabels=class_labels,
        yticklabels=class_labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix",
    )
    thresh = cm.max() / 2.0
    for i in range(C.NUM_CLASSES):
        for j in range(C.NUM_CLASSES):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix: %s", out_path)


def evaluate_model(
    checkpoint_path: str | Path,
    seq_dir: str | Path,
    out_dir: str | Path,
    split: str = "test",
) -> Dict:
    """
    Evaluate a saved CoralLSTM checkpoint on a data split.

    Writes eval_metrics.json and confusion_matrix.png to out_dir and returns
    the metrics dict.
    """
    checkpoint_path = Path(checkpoint_path)
    seq_dir = Path(seq_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    norm_stats = ckpt["norm_stats"]
    saved_epoch = ckpt.get("epoch", "?")
    logger.info("Checkpoint from epoch %s", saved_epoch)

    # Reconstruct model
    model = CoralLSTM(
        hidden_size=cfg.get("hidden_size", C.TRAIN_DEFAULTS["hidden_size"]),
        num_layers=cfg.get("num_layers", C.TRAIN_DEFAULTS["num_layers"]),
        dropout=cfg.get("dropout", C.TRAIN_DEFAULTS["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])

    device = _get_device()
    logger.info("Evaluating on device: %s", device)
    model = model.to(device)
    model.eval()

    # Load split data
    parquet_path = seq_dir / f"sequences_{split}.parquet"
    logger.info("Loading %s split: %s", split, parquet_path)
    df = pd.read_parquet(parquet_path)
    logger.info("Samples: %d", len(df))

    dataset = CoralSequenceDataset(df, norm_stats)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    # Inference
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_seq, x_static, y in loader:
            x_seq = x_seq.to(device)
            x_static = x_static.to(device)
            logits = model(x_seq, x_static)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    # Metrics
    report: Dict = classification_report(
        labels,
        preds,
        labels=list(range(C.NUM_CLASSES)),
        output_dict=True,
        zero_division=0,
    )
    accuracy = float((preds == labels).mean())
    macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0))

    metrics: Dict = {
        **report,
        "overall_accuracy": accuracy,
        "macro_f1": macro_f1,
    }

    logger.info("Accuracy: %.4f  Macro-F1: %.4f", accuracy, macro_f1)

    # Save metrics JSON
    metrics_path = out_dir / "eval_metrics.json"
    with metrics_path.open("w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved eval metrics: %s", metrics_path)

    # Confusion matrix plot
    cm = confusion_matrix(labels, preds, labels=list(range(C.NUM_CLASSES)))
    _plot_confusion_matrix(cm, out_dir / "confusion_matrix.png")

    return metrics
