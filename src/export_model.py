"""
export_model – export a trained CoralLSTM checkpoint to CoreML (.mlpackage).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import coremltools as ct
import torch

from . import config as C
from .model import CoralLSTM

logger = logging.getLogger(__name__)

CLASS_LABELS = ["No Stress", "Watch", "Warning", "Alert Level 1", "Alert Level 2", "Alert Level 3"]


def export_to_coreml(checkpoint_path: str | Path, out_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    out_path = Path(out_path)

    logger.info("Loading checkpoint from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    cfg = ckpt["config"]
    model = CoralLSTM(
        input_size=cfg.get("input_size", len(C.SEQUENCE_FEATURES)),
        static_size=cfg.get("static_size", len(C.STATIC_FEATURES)),
        hidden_size=cfg.get("hidden_size"),
        num_layers=cfg.get("num_layers"),
        dropout=cfg.get("dropout", 0.0),
        num_classes=cfg.get("num_classes", C.NUM_CLASSES),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model reconstructed (epoch %s)", ckpt.get("epoch"))

    x_seq = torch.randn(1, C.DEFAULT_LOOKBACK_DAYS, len(C.SEQUENCE_FEATURES))
    x_static = torch.randn(1, len(C.STATIC_FEATURES))

    logger.info("Tracing model with JIT")
    traced = torch.jit.trace(model, (x_seq, x_static))

    logger.info("Converting to CoreML")
    ml_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="x_seq",
                shape=(1, C.DEFAULT_LOOKBACK_DAYS, len(C.SEQUENCE_FEATURES)),
            ),
            ct.TensorType(name="x_static", shape=(1, len(C.STATIC_FEATURES))),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ml_model.save(str(out_path))
    logger.info("Saved mlpackage to %s", out_path)

    preprocessing = {
        "norm_stats": ckpt["norm_stats"],
        "class_labels": CLASS_LABELS,
        "lookback": C.DEFAULT_LOOKBACK_DAYS,
        "horizon": C.DEFAULT_HORIZON_DAYS,
        "sequence_features": list(C.SEQUENCE_FEATURES),
        "static_features": list(C.STATIC_FEATURES),
    }
    json_path = out_path.parent / "preprocessing.json"
    json_path.write_text(json.dumps(preprocessing, indent=2))
    logger.info("Saved preprocessing metadata to %s", json_path)

    return out_path
