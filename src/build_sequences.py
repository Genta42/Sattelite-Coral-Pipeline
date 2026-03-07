"""
build_sequences – create LSTM-ready (sequence → one) samples from the long table.

Each sample:
    cell_id, lat, lon, target_date, horizon_days, y_baa_cat,
    x_baa_seq, x_dhw_seq (optional), x_hotspot_seq (optional)

Two serialisation modes:
  A) "json"     – sequences stored as JSON arrays in a single column
  B) "flat"     – sequences exploded into x_baa_t-59, x_baa_t-58, … x_baa_t0 columns

Also produces temporal + optional spatial splits.

Streaming mode: when given a shard directory (from build_table), processes
one shard at a time to stay within memory limits.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config as C

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


def temporal_split(
    df: pd.DataFrame,
    train_end: str = C.SPLIT_TRAIN_END,
    val_end: str = C.SPLIT_VAL_END,
    date_col: str = "target_date",
) -> Dict[str, pd.DataFrame]:
    """
    Split by time:
      train: earliest … train_end
      val:   train_end+1 … val_end
      test:  val_end+1 … latest
    No leakage because splits are strictly non-overlapping in time.
    """
    from datetime import date as _date

    te = _date.fromisoformat(train_end) if isinstance(train_end, str) else train_end
    ve = _date.fromisoformat(val_end) if isinstance(val_end, str) else val_end

    dates = pd.to_datetime(df[date_col]).dt.date
    return {
        "train": df[dates <= te].copy(),
        "val": df[(dates > te) & (dates <= ve)].copy(),
        "test": df[dates > ve].copy(),
    }


def spatial_holdout(
    df: pd.DataFrame,
    holdout_continent: str,
) -> Dict[str, pd.DataFrame]:
    """
    Hold out all cells whose (lat, lon) fall inside *holdout_continent* bounds.
    Returns {"main": …, "holdout": …}.
    """
    bounds = C.CONTINENT_BOUNDS.get(holdout_continent)
    if bounds is None:
        raise ValueError(f"Unknown continent for holdout: {holdout_continent}")
    lat_lo, lat_hi, lon_lo, lon_hi = bounds
    mask = (
        (df["lat"] >= lat_lo)
        & (df["lat"] <= lat_hi)
        & (df["lon"] >= lon_lo)
        & (df["lon"] <= lon_hi)
    )
    return {
        "main": df[~mask].copy(),
        "holdout": df[mask].copy(),
    }


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------


def _build_cell_sequences(
    cell_df: pd.DataFrame,
    lookback: int,
    horizon: int,
    feature_cols: List[str],
) -> List[Dict]:
    """
    For one cell (already sorted by date_utc), create all valid samples.
    A sample is valid when we have *lookback* consecutive days before
    (target_date - horizon) and the target_date itself has a BAA value.
    """
    cell_df = cell_df.sort_values("date_utc").reset_index(drop=True)
    dates = pd.to_datetime(cell_df["date_utc"])
    records: List[Dict] = []

    # Build a date → row-index map for O(1) lookup
    date_to_idx = {
        d.date() if hasattr(d, "date") else d: i for i, d in enumerate(dates)
    }

    n = len(cell_df)
    cell_id = cell_df.iloc[0]["cell_id"]
    lat = cell_df.iloc[0]["lat"]
    lon = cell_df.iloc[0]["lon"]

    for target_idx in range(lookback + horizon - 1, n):
        target_date = dates.iloc[target_idx]
        target_date_d = (
            target_date.date() if hasattr(target_date, "date") else target_date
        )

        # The last observation we use as input is at (target_date - horizon)
        from datetime import timedelta

        input_end_date = target_date_d - timedelta(days=horizon)
        input_start_date = input_end_date - timedelta(days=lookback - 1)

        # Check all lookback dates are present
        input_dates = [input_start_date + timedelta(days=d) for d in range(lookback)]
        idxs = [date_to_idx.get(d) for d in input_dates]
        if any(i is None for i in idxs):
            continue  # gap in time series – skip this sample

        # Target BAA
        target_baa = cell_df.iloc[target_idx]["baa_cat"]
        if pd.isna(target_baa):
            continue

        rec: Dict = {
            "cell_id": cell_id,
            "lat": lat,
            "lon": lon,
            "target_date": str(target_date_d),
            "horizon_days": horizon,
            "y_baa_cat": int(target_baa),
        }

        # Feature sequences
        for col in feature_cols:
            seq = [cell_df.iloc[i][col] for i in idxs]
            # Replace NaN with -1 sentinel (downstream model handles it)
            seq = [float(v) if pd.notna(v) else -1.0 for v in seq]
            rec[f"x_{col}_seq"] = seq

        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# In-memory build (for small datasets)
# ---------------------------------------------------------------------------


def build_sequences(
    long_table: pd.DataFrame,
    lookback: int = C.DEFAULT_LOOKBACK_DAYS,
    horizon: int = C.DEFAULT_HORIZON_DAYS,
    serialization: str = "json",  # "json" | "flat"
    out_dir: Optional[Path] = None,
    split: bool = True,
    parquet: bool = True,
) -> pd.DataFrame:
    """
    Build the full sequence dataset from the long table (in-memory).
    Use build_sequences_from_shards for large datasets.
    """
    # Determine which feature columns to include in sequences
    feature_cols = ["baa_cat"]
    for optional in ["dhw", "hotspot"]:
        if optional in long_table.columns:
            feature_cols.append(optional)

    logger.info(
        "Building sequences: lookback=%d, horizon=%d, features=%s, serialization=%s",
        lookback,
        horizon,
        feature_cols,
        serialization,
    )

    # Group by cell and build sequences
    all_records: List[Dict] = []
    cells = long_table.groupby("cell_id")
    total_cells = len(cells)
    for i, (cid, cell_df) in enumerate(cells):
        if len(cell_df) < lookback + horizon:
            continue
        recs = _build_cell_sequences(cell_df, lookback, horizon, feature_cols)
        all_records.extend(recs)
        if (i + 1) % 5000 == 0:
            logger.info(
                "Processed %d / %d cells (%d samples so far)",
                i + 1,
                total_cells,
                len(all_records),
            )

    if not all_records:
        raise RuntimeError(
            "No valid sequences produced. Check data coverage and lookback/horizon settings."
        )

    logger.info("Total samples: %d from %d cells", len(all_records), total_cells)

    # Convert to DataFrame
    seq_df = pd.DataFrame(all_records)

    # Serialise sequences
    seq_df = _serialize_sequences(seq_df, serialization, lookback)

    # --- Write outputs ---
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Full dataset
        _write(seq_df, out_dir / "sequences_all", parquet)

        # Temporal splits
        if split:
            splits = temporal_split(seq_df)
            for name, sdf in splits.items():
                _write(sdf, out_dir / f"sequences_{name}", parquet)
                logger.info("Split '%s': %d samples", name, len(sdf))

    return seq_df


def _serialize_sequences(
    seq_df: pd.DataFrame, serialization: str, lookback: int
) -> pd.DataFrame:
    """Apply json or flat serialization to sequence columns."""
    seq_cols = [c for c in seq_df.columns if c.startswith("x_") and c.endswith("_seq")]

    if serialization == "json":
        for col in seq_cols:
            seq_df[col] = seq_df[col].apply(json.dumps)
    elif serialization == "flat":
        for col in seq_cols:
            base = col.replace("_seq", "")
            expanded = pd.DataFrame(
                seq_df[col].tolist(),
                columns=[
                    f"{base}_t-{lookback - 1 - j}" if j < lookback - 1 else f"{base}_t0"
                    for j in range(lookback)
                ],
                index=seq_df.index,
            )
            seq_df = pd.concat([seq_df.drop(columns=[col]), expanded], axis=1)
    else:
        raise ValueError(f"Unknown serialization: {serialization!r}")

    return seq_df


# ---------------------------------------------------------------------------
# Streaming build from shards (for large datasets)
# ---------------------------------------------------------------------------


def build_sequences_from_shards(
    shard_dir: Path,
    out_dir: Path,
    lookback: int = C.DEFAULT_LOOKBACK_DAYS,
    horizon: int = C.DEFAULT_HORIZON_DAYS,
    serialization: str = "json",
) -> int:
    """
    Build sequences from cell-sharded parquet files, one shard at a time.
    Memory usage is bounded by the largest shard (~total_data / n_shards).

    Writes sequences_all.parquet and split files to out_dir.
    Returns total number of samples.
    """
    shard_dir = Path(shard_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(shard_dir.glob("shard_*.parquet"))
    if not shard_files:
        raise RuntimeError(f"No shard files found in {shard_dir}")

    all_writer: Optional[pq.ParquetWriter] = None
    split_writers: Dict[str, pq.ParquetWriter] = {}
    schema: Optional[pa.Schema] = None
    total_samples = 0
    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    from datetime import date as _date

    te = _date.fromisoformat(C.SPLIT_TRAIN_END)
    ve = _date.fromisoformat(C.SPLIT_VAL_END)

    for si, shard_path in enumerate(shard_files):
        logger.info(
            "Processing shard %d / %d: %s",
            si + 1,
            len(shard_files),
            shard_path.name,
        )

        df = pd.read_parquet(shard_path)

        feature_cols = ["baa_cat"]
        for optional in ["dhw", "hotspot"]:
            if optional in df.columns:
                feature_cols.append(optional)

        records: List[Dict] = []
        cells = df.groupby("cell_id")
        for cid, cell_df in cells:
            if len(cell_df) < lookback + horizon:
                continue
            recs = _build_cell_sequences(cell_df, lookback, horizon, feature_cols)
            records.extend(recs)

        if not records:
            del df
            gc.collect()
            continue

        shard_seq_df = pd.DataFrame(records)
        shard_seq_df = _serialize_sequences(shard_seq_df, serialization, lookback)

        table = pa.Table.from_pandas(shard_seq_df, preserve_index=False)

        if all_writer is None:
            schema = table.schema
            all_writer = pq.ParquetWriter(out_dir / "sequences_all.parquet", schema)

        table = table.cast(schema)
        all_writer.write_table(table)
        total_samples += len(shard_seq_df)

        # Write to split files
        dates = pd.to_datetime(shard_seq_df["target_date"]).dt.date
        for split_name, mask in [
            ("train", dates <= te),
            ("val", (dates > te) & (dates <= ve)),
            ("test", dates > ve),
        ]:
            split_df = shard_seq_df[mask]
            if split_df.empty:
                continue
            split_table = pa.Table.from_pandas(split_df, preserve_index=False)
            split_table = split_table.cast(schema)
            if split_name not in split_writers:
                split_writers[split_name] = pq.ParquetWriter(
                    out_dir / f"sequences_{split_name}.parquet", schema
                )
            split_writers[split_name].write_table(split_table)
            split_counts[split_name] += len(split_df)

        logger.info(
            "  Shard %d: %d samples (total: %d)",
            si + 1,
            len(records),
            total_samples,
        )

        del df, records, shard_seq_df, table
        gc.collect()

    if all_writer:
        all_writer.close()
    for w in split_writers.values():
        w.close()

    if total_samples == 0:
        raise RuntimeError("No valid sequences produced from shards.")

    for name, count in split_counts.items():
        logger.info("Split '%s': %d samples", name, count)

    logger.info("Total: %d samples written to %s", total_samples, out_dir)
    return total_samples


def _write(df: pd.DataFrame, stem: Path, parquet: bool) -> None:
    csv_path = stem.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s (%d rows)", csv_path, len(df))
    if parquet:
        pq_path = stem.with_suffix(".parquet")
        df.to_parquet(pq_path, index=False, engine="pyarrow")


# ---------------------------------------------------------------------------
# Class-imbalance utilities
# ---------------------------------------------------------------------------


def class_distribution(seq_df: pd.DataFrame) -> pd.DataFrame:
    """Return BAA class counts and proportions."""
    counts = seq_df["y_baa_cat"].value_counts().sort_index()
    props = counts / counts.sum()
    return pd.DataFrame({"count": counts, "proportion": props})


def inverse_class_weights(seq_df: pd.DataFrame) -> Dict[int, float]:
    """Compute inverse-frequency weights for each BAA class."""
    counts = seq_df["y_baa_cat"].value_counts()
    total = counts.sum()
    n_classes = len(counts)
    return {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}


def oversampling_indices(seq_df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """
    Return an array of row indices that oversample minority classes
    so all classes have equal representation.
    """
    rng = np.random.RandomState(seed)
    max_count = seq_df["y_baa_cat"].value_counts().max()
    indices: List[np.ndarray] = []
    for cls in seq_df["y_baa_cat"].unique():
        cls_idx = seq_df.index[seq_df["y_baa_cat"] == cls].values
        resampled = rng.choice(cls_idx, size=max_count, replace=True)
        indices.append(resampled)
    all_idx = np.concatenate(indices)
    rng.shuffle(all_idx)
    return all_idx
