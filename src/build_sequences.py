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
import pyarrow.compute as pc
import pyarrow.parquet as pq

import sys

try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

def _should_use_rich() -> bool:
    if not HAS_RICH or not sys.stdout.isatty():
        return False
    try:
        Console(force_terminal=True).print("", end="")
        return True
    except (UnicodeEncodeError, OSError):
        return False

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

    Vectorised: finds contiguous daily runs, then uses sliding windows
    within each run — no per-sample date lookups.
    """
    cell_df = cell_df.sort_values("date_utc").reset_index(drop=True)
    n = len(cell_df)
    min_len = lookback + horizon
    if n < min_len:
        return []

    cell_id = cell_df.iloc[0]["cell_id"]
    lat = float(cell_df.iloc[0]["lat"])
    lon = float(cell_df.iloc[0]["lon"])

    # Timestamps as int64 days for fast diff
    dates_ns = pd.to_datetime(cell_df["date_utc"]).values.astype("datetime64[D]").astype(np.int64)
    day_diffs = np.diff(dates_ns)

    # Feature arrays (float64)
    feat_arrays = {}
    for col in feature_cols:
        arr = cell_df[col].values.astype(np.float64)
        # Replace NaN with -1 sentinel upfront
        np.putmask(arr, np.isnan(arr), -1.0)
        feat_arrays[col] = arr

    baa_arr = feat_arrays["baa_cat"]

    # Date strings for target_date output
    date_strs = pd.to_datetime(cell_df["date_utc"]).dt.strftime("%Y-%m-%d").values

    # Find contiguous daily runs: split where diff != 1
    breaks = np.where(day_diffs != 1)[0]  # indices where a gap occurs
    run_starts = np.concatenate([[0], breaks + 1])
    run_ends = np.concatenate([breaks + 1, [n]])

    records: List[Dict] = []

    for rs, re in zip(run_starts, run_ends):
        run_len = re - rs
        if run_len < min_len:
            continue

        # Within this contiguous run, every window of (lookback + horizon) days
        # is valid. target = input_start + lookback - 1 + horizon
        # i.e. target offset within run: from (min_len - 1) to (run_len - 1)
        # input window starts at: target_offset - lookback - horizon + 1
        for t_off in range(min_len - 1, run_len):
            target_idx = rs + t_off
            target_baa = baa_arr[target_idx]
            if target_baa == -1.0:
                continue

            # Input window: lookback days ending at (target - horizon)
            input_end_idx = rs + t_off - horizon
            input_start_idx = input_end_idx - lookback + 1
            # Slice [input_start_idx : input_end_idx + 1] is always valid
            # because we're within a contiguous run of sufficient length

            rec: Dict = {
                "cell_id": cell_id,
                "lat": lat,
                "lon": lon,
                "target_date": date_strs[target_idx],
                "horizon_days": horizon,
                "y_baa_cat": int(target_baa),
            }

            for col in feature_cols:
                rec[f"x_{col}_seq"] = feat_arrays[col][input_start_idx:input_end_idx + 1].tolist()

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

    n_shards = len(shard_files)

    try:
        from notify import ProgressTracker
        ntfy_tracker = ProgressTracker("Build Sequences", total=n_shards, unit="shards")
    except ImportError:
        ntfy_tracker = None

    if _should_use_rich():
        progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("samples: {task.fields[samples]}"),
            TimeRemainingColumn(),
        )
        shard_task = progress.add_task("Shard", total=n_shards, samples=0)
        cell_task = progress.add_task("Cells", total=0, samples=0)
        progress.start()
    else:
        progress = None

    def _flush_records(
        records: List[Dict],
        serialization: str,
        lookback: int,
        all_writer,
        schema,
        split_writers: Dict,
        split_counts: Dict,
        te,
        ve,
        out_dir: Path,
    ) -> Tuple[pq.ParquetWriter, pa.Schema]:
        """Flush accumulated records to parquet writers; return (all_writer, schema)."""
        batch_df = pd.DataFrame(records)
        batch_df = _serialize_sequences(batch_df, serialization, lookback)
        table = pa.Table.from_pandas(batch_df, preserve_index=False)

        if all_writer is None:
            schema = table.schema
            all_writer = pq.ParquetWriter(out_dir / "sequences_all.parquet", schema)

        table = table.cast(schema)
        all_writer.write_table(table)

        # Write to split files
        dates = pd.to_datetime(batch_df["target_date"]).dt.date
        for split_name, mask in [
            ("train", dates <= te),
            ("val", (dates > te) & (dates <= ve)),
            ("test", dates > ve),
        ]:
            split_chunk = batch_df[mask]
            if split_chunk.empty:
                continue
            split_table = pa.Table.from_pandas(split_chunk, preserve_index=False).cast(schema)
            if split_name not in split_writers:
                split_writers[split_name] = pq.ParquetWriter(
                    out_dir / f"sequences_{split_name}.parquet", schema)
            split_writers[split_name].write_table(split_table)
            split_counts[split_name] += len(split_chunk)

        del batch_df, table
        return all_writer, schema

    FLUSH_THRESHOLD = 100_000  # flush to parquet every N records

    try:
        for si, shard_path in enumerate(shard_files):
            logger.info(
                "Processing shard %d / %d: %s",
                si + 1,
                n_shards,
                shard_path.name,
            )

            # Determine feature columns from schema
            shard_schema = pq.read_schema(shard_path)
            shard_col_names = set(shard_schema.names)
            feature_cols = ["baa_cat"]
            for optional in ["dhw", "hotspot"]:
                if optional in shard_col_names:
                    feature_cols.append(optional)
            read_cols = ["cell_id", "date_utc", "lat", "lon"] + feature_cols

            # Read shard in row-group batches to stay under memory limit.
            # Windows Store Python has ~8 GB process limit, so we accumulate
            # per-cell data from batches of row groups (~4M rows at a time).
            pf = pq.ParquetFile(shard_path)
            n_row_groups = pf.metadata.num_row_groups
            RG_BATCH = 500  # ~4M rows per batch

            # First pass: collect all rows per cell using batched reads
            cell_data: Dict[str, List[pd.DataFrame]] = {}
            for rg_start in range(0, n_row_groups, RG_BATCH):
                rg_end = min(rg_start + RG_BATCH, n_row_groups)
                rg_indices = list(range(rg_start, rg_end))
                batch_table = pf.read_row_groups(rg_indices, columns=read_cols)
                batch_df = batch_table.to_pandas()
                del batch_table

                for cid, grp in batch_df.groupby("cell_id", sort=False):
                    if cid not in cell_data:
                        cell_data[cid] = []
                    cell_data[cid].append(grp)

                del batch_df
                gc.collect()

            cell_names = sorted(cell_data.keys())
            total_cells = len(cell_names)

            if progress is not None:
                progress.reset(cell_task, total=total_cells, completed=0, samples=0)

            shard_samples = 0
            records: List[Dict] = []

            # Process each cell from accumulated chunks
            for ci, cid in enumerate(cell_names):
                chunks = cell_data.pop(cid)
                cell_df = pd.concat(chunks, ignore_index=True) if len(chunks) > 1 else chunks[0].reset_index(drop=True)
                del chunks
                if len(cell_df) < lookback + horizon:
                    if progress is not None:
                        progress.update(cell_task, advance=1)
                    continue
                recs = _build_cell_sequences(cell_df, lookback, horizon, feature_cols)
                records.extend(recs)
                if progress is not None:
                    progress.update(cell_task, advance=1, samples=shard_samples + len(records))
                elif (ci + 1) % 5000 == 0:
                    logger.info(
                        "  Shard %d: processed %d / %d cells (%d samples)",
                        si + 1, ci + 1, total_cells, shard_samples + len(records),
                    )

                # Flush records when threshold reached
                if len(records) >= FLUSH_THRESHOLD:
                    all_writer, schema = _flush_records(
                        records, serialization, lookback,
                        all_writer, schema, split_writers, split_counts,
                        te, ve, out_dir,
                    )
                    shard_samples += len(records)
                    total_samples += len(records)
                    records = []
                    gc.collect()

            # Flush remaining records for this shard
            if records:
                all_writer, schema = _flush_records(
                    records, serialization, lookback,
                    all_writer, schema, split_writers, split_counts,
                    te, ve, out_dir,
                )
                shard_samples += len(records)
                total_samples += len(records)
                records = []
                gc.collect()

            if progress is not None:
                progress.update(shard_task, advance=1, samples=total_samples)
            else:
                logger.info(
                    "  Shard %d: %d samples (total: %d)",
                    si + 1,
                    shard_samples,
                    total_samples,
                )

            if ntfy_tracker:
                ntfy_tracker.update(si + 1, extra={"samples": total_samples})

            del cell_data
            gc.collect()
    finally:
        if progress is not None:
            progress.stop()
        if ntfy_tracker:
            ntfy_tracker.finish(extra={"samples": total_samples})

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
