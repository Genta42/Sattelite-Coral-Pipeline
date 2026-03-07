#!/usr/bin/env python3
"""
coral_pipeline CLI – NOAA CRW BAA scraping & LSTM preprocessing pipeline.

Subcommands
-----------
  discover     Query ERDDAP metadata (dataset variables + time bounds)
  fetch        Download raw data from ERDDAP griddap
  build_table  Merge cached CSVs into a clean long table (Format 1)
  build_sequences  Create LSTM-ready sequence samples
  qa           Generate QA reports and sanity plots
  export       Write final per-continent + global files
  train        Train LSTM model on sequence data
  evaluate     Evaluate trained model on a split
  export_model Export trained model to CoreML
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("coral_pipeline")

# Ensure src package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ═══════════════════════════════════════════════════════════════════════════
# discover
# ═══════════════════════════════════════════════════════════════════════════


def cmd_discover(args: argparse.Namespace) -> None:
    from src.fetch import run_discovery, run_time_discovery
    from src import config as C

    dataset_id = args.dataset_id or C.DATASET_ID
    print(f"Dataset: {dataset_id}")
    print()

    var_map = run_discovery(dataset_id)
    print("Resolved variables:")
    for logical, actual in var_map.items():
        print(f"  {logical:12s} → {actual}")
    print()

    t_min, t_max = run_time_discovery(dataset_id)
    print(f"Time range: {t_min}  →  {t_max}")


# ═══════════════════════════════════════════════════════════════════════════
# fetch
# ═══════════════════════════════════════════════════════════════════════════


def cmd_fetch(args: argparse.Namespace) -> None:
    from src.fetch import run_discovery, fetch_continent
    from src import config as C

    dataset_id = args.dataset_id or C.DATASET_ID
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Discover available variables
    var_map = run_discovery(dataset_id)
    requested = [v.lower() for v in args.variables]
    variables: List[str] = []
    for req in requested:
        if req in var_map:
            variables.append(var_map[req])
        else:
            logger.warning("Variable group '%s' not available – skipping.", req)

    if not variables:
        logger.error("No variables resolved. Available: %s", list(var_map.keys()))
        sys.exit(1)

    logger.info("Fetching variables: %s", variables)

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    continents = args.continents if args.continents else list(C.CONTINENT_BOUNDS.keys())
    # Remove 'world' if specific continents are listed (avoid double-fetch)
    if len(continents) > 1 and "world" in continents:
        continents.remove("world")

    for continent in continents:
        logger.info("═══ Fetching %s ═══", continent)
        paths = fetch_continent(
            continent=continent,
            start_date=start,
            end_date=end,
            variables=variables,
            stride=args.stride,
            cache_dir=cache_dir,
            dataset_id=dataset_id,
        )
        logger.info("  → %d chunks cached for %s", len(paths), continent)


# ═══════════════════════════════════════════════════════════════════════════
# build_table
# ═══════════════════════════════════════════════════════════════════════════


def cmd_build_table(args: argparse.Namespace) -> None:
    from src.fetch import run_discovery
    from src.build_table import build_long_table
    from src import config as C

    dataset_id = args.dataset_id or C.DATASET_ID
    var_map = run_discovery(dataset_id)

    cache_dir = Path(args.cache_dir)
    csvs = sorted(cache_dir.glob("*.csv"))
    if not csvs:
        logger.error("No cached CSVs found in %s", cache_dir)
        sys.exit(1)

    logger.info("Found %d cached CSV files", len(csvs))
    out_path = Path(args.out_path)
    build_long_table(
        csvs,
        var_map,
        out_path,
        parquet=not args.no_parquet,
        n_shards=args.n_shards,
    )


# ═══════════════════════════════════════════════════════════════════════════
# build_sequences
# ═══════════════════════════════════════════════════════════════════════════


def cmd_build_sequences(args: argparse.Namespace) -> None:
    shard_dir = Path(args.shard_dir) if args.shard_dir else None
    out_dir = Path(args.out_dir)

    if shard_dir and shard_dir.exists() and list(shard_dir.glob("shard_*.parquet")):
        from src.build_sequences import build_sequences_from_shards

        logger.info("Using streaming shard mode from %s", shard_dir)
        build_sequences_from_shards(
            shard_dir=shard_dir,
            out_dir=out_dir,
            lookback=args.lookback,
            horizon=args.horizon,
            serialization=args.serialization,
        )
    else:
        from src.build_sequences import build_sequences
        import pandas as pd

        table_path = Path(args.table_path)
        if table_path.suffix == ".parquet":
            long_table = pd.read_parquet(table_path)
        else:
            long_table = pd.read_csv(table_path)

        build_sequences(
            long_table=long_table,
            lookback=args.lookback,
            horizon=args.horizon,
            serialization=args.serialization,
            out_dir=out_dir,
            split=not args.no_split,
            parquet=not args.no_parquet,
        )


# ═══════════════════════════════════════════════════════════════════════════
# qa
# ═══════════════════════════════════════════════════════════════════════════


def cmd_qa(args: argparse.Namespace) -> None:
    from src.qa import run_qa
    import pandas as pd

    table_path = Path(args.table_path)
    if table_path.suffix == ".parquet":
        long_table = pd.read_parquet(table_path)
    else:
        long_table = pd.read_csv(table_path)

    seq_df = None
    if args.seq_path:
        sp = Path(args.seq_path)
        seq_df = pd.read_parquet(sp) if sp.suffix == ".parquet" else pd.read_csv(sp)

    produced: List[Path] = []
    if args.manifest_dir:
        md = Path(args.manifest_dir)
        produced = list(md.rglob("*.csv")) + list(md.rglob("*.parquet"))

    reports_dir = Path(args.reports_dir)
    run_qa(long_table, reports_dir, seq_df=seq_df, produced_files=produced or None)


# ═══════════════════════════════════════════════════════════════════════════
# export
# ═══════════════════════════════════════════════════════════════════════════


def cmd_export(args: argparse.Namespace) -> None:
    from src.export import export_by_continent
    import pandas as pd

    in_path = Path(args.input_path)
    df = (
        pd.read_parquet(in_path)
        if in_path.suffix == ".parquet"
        else pd.read_csv(in_path)
    )

    out_dir = Path(args.out_dir)
    export_by_continent(df, out_dir, prefix=args.prefix, parquet=not args.no_parquet)


# ═══════════════════════════════════════════════════════════════════════════
# train
# ═══════════════════════════════════════════════════════════════════════════


def cmd_train(args: argparse.Namespace) -> None:
    from src.train import train_model

    train_model(
        seq_dir=args.seq_dir,
        out_dir=args.out_dir,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        max_samples=args.max_samples,
    )


# ═══════════════════════════════════════════════════════════════════════════
# evaluate
# ═══════════════════════════════════════════════════════════════════════════


def cmd_evaluate(args: argparse.Namespace) -> None:
    from src.evaluate import evaluate_model

    metrics = evaluate_model(
        checkpoint_path=args.checkpoint,
        seq_dir=args.seq_dir,
        out_dir=args.out_dir,
        split=args.split,
    )
    print(f"Accuracy: {metrics.get('accuracy', 'N/A')}")
    print(f"Macro F1: {metrics.get('macro avg', {}).get('f1-score', 'N/A')}")


# ═══════════════════════════════════════════════════════════════════════════
# export_model
# ═══════════════════════════════════════════════════════════════════════════


def cmd_export_model(args: argparse.Namespace) -> None:
    from src.export_model import export_to_coreml

    out = export_to_coreml(
        checkpoint_path=args.checkpoint,
        out_path=args.out_path,
    )
    print(f"CoreML model exported to: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coral_pipeline",
        description="NOAA CRW BAA data pipeline for per-cell LSTM training.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- discover ---
    d = sub.add_parser("discover", help="Query ERDDAP metadata")
    d.add_argument("--dataset-id", default=None)
    d.set_defaults(func=cmd_discover)

    # --- fetch ---
    f = sub.add_parser("fetch", help="Download raw data from ERDDAP")
    f.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    f.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    f.add_argument(
        "--continents",
        nargs="*",
        default=None,
        help="Continent names (default: all). E.g. asia australia",
    )
    f.add_argument(
        "--variables",
        nargs="*",
        default=["baa", "dhw"],
        help="Variable groups to fetch: baa dhw hotspot sst_anomaly",
    )
    f.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Spatial stride (1=full 5km, 2=10km, 4=20km …)",
    )
    f.add_argument("--cache-dir", default="coral_pipeline/data/cache")
    f.add_argument("--dataset-id", default=None)
    f.set_defaults(func=cmd_fetch)

    # --- build_table ---
    bt = sub.add_parser("build_table", help="Merge cached CSVs into long table")
    bt.add_argument("--cache-dir", default="coral_pipeline/data/cache")
    bt.add_argument("--out-path", default="coral_pipeline/data/processed/long_table")
    bt.add_argument("--no-parquet", action="store_true")
    bt.add_argument("--dataset-id", default=None)
    bt.add_argument(
        "--n-shards",
        type=int,
        default=0,
        help="Create N cell-based parquet shards for memory-efficient sequence building (0=off)",
    )
    bt.set_defaults(func=cmd_build_table)

    # --- build_sequences ---
    bs = sub.add_parser("build_sequences", help="Build LSTM-ready samples")
    bs.add_argument(
        "--table-path",
        default=None,
        help="Path to long_table CSV or Parquet (for in-memory mode)",
    )
    bs.add_argument(
        "--shard-dir",
        default=None,
        help="Path to cell shards directory (for streaming mode, preferred for large data)",
    )
    bs.add_argument("--out-dir", default="coral_pipeline/data/sequences")
    bs.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Number of past days in input sequence (default: 60)",
    )
    bs.add_argument(
        "--horizon", type=int, default=7, help="Days ahead to predict (default: 7)"
    )
    bs.add_argument(
        "--serialization",
        choices=["json", "flat"],
        default="json",
        help="Sequence serialization mode",
    )
    bs.add_argument("--no-split", action="store_true", help="Skip train/val/test split")
    bs.add_argument("--no-parquet", action="store_true")
    bs.set_defaults(func=cmd_build_sequences)

    # --- qa ---
    qa = sub.add_parser("qa", help="Generate QA reports and plots")
    qa.add_argument(
        "--table-path", required=True, help="Path to long_table CSV or Parquet"
    )
    qa.add_argument(
        "--seq-path",
        default=None,
        help="Optional path to sequences file for class dist",
    )
    qa.add_argument("--reports-dir", default="coral_pipeline/reports")
    qa.add_argument(
        "--manifest-dir", default=None, help="Directory to scan for checksum manifest"
    )
    qa.set_defaults(func=cmd_qa)

    # --- export ---
    ex = sub.add_parser("export", help="Export per-continent + global files")
    ex.add_argument(
        "--input-path", required=True, help="Sequences or long-table CSV/Parquet"
    )
    ex.add_argument("--out-dir", default="coral_pipeline/data/exported")
    ex.add_argument("--prefix", default="sequences")
    ex.add_argument("--no-parquet", action="store_true")
    ex.set_defaults(func=cmd_export)

    # --- train ---
    tr = sub.add_parser("train", help="Train LSTM model")
    tr.add_argument("--seq-dir", default="coral_pipeline/data/sequences")
    tr.add_argument("--out-dir", default="coral_pipeline/models")
    tr.add_argument("--hidden-size", type=int, default=64)
    tr.add_argument("--num-layers", type=int, default=2)
    tr.add_argument("--dropout", type=float, default=0.3)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--max-epochs", type=int, default=100)
    tr.add_argument("--patience", type=int, default=10)
    tr.add_argument("--device", default=None, help="Force device (cpu/cuda/mps)")
    tr.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap training samples to limit memory (e.g. 2000000)",
    )
    tr.set_defaults(func=cmd_train)

    # --- evaluate ---
    ev = sub.add_parser("evaluate", help="Evaluate trained model")
    ev.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    ev.add_argument("--seq-dir", default="coral_pipeline/data/sequences")
    ev.add_argument("--out-dir", default="coral_pipeline/models")
    ev.add_argument("--split", default="test", choices=["train", "val", "test"])
    ev.set_defaults(func=cmd_evaluate)

    # --- export_model ---
    em = sub.add_parser("export_model", help="Export model to CoreML")
    em.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    em.add_argument(
        "--out-path", default="coral_pipeline/models/CoralBleaching.mlpackage"
    )
    em.set_defaults(func=cmd_export_model)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
