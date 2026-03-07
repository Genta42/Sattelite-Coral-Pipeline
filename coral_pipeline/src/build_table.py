"""
build_table – merge raw cached CSVs into a clean per-(cell, date) long table.

Output columns (Format 1):
    cell_id, lat, lon, date_utc, baa_cat, dhw, hotspot

Where cell_id = deterministic hash of (lat, lon) so it is stable across runs.

Streaming mode: processes one CSV at a time via pyarrow ParquetWriter
to avoid loading all data into memory at once.
"""

from __future__ import annotations

import gc
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config as C

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cell ID
# ---------------------------------------------------------------------------


def make_cell_id(lat: float, lon: float) -> str:
    """Deterministic 12-char hex id from (lat, lon) rounded to grid."""
    key = f"{round(lat, 4)}:{round(lon, 4)}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Read one cached CSV
# ---------------------------------------------------------------------------


def _read_cached_csv(path: Path, var_map: Dict[str, str]) -> Optional[pd.DataFrame]:
    """
    Read an ERDDAP CSV (first row = col names, second row = units → skip).
    Rename columns to logical names.  Return cleaned DataFrame or None.
    """
    try:
        # ERDDAP CSVs: row 0 = header, row 1 = units
        df = pd.read_csv(path, skiprows=[1], low_memory=False)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None

    if df.empty:
        return None

    # Rename ERDDAP columns → logical names
    rename: Dict[str, str] = {}
    if "time" in df.columns:
        rename["time"] = "date_utc"
    if "latitude" in df.columns:
        rename["latitude"] = "lat"
    if "longitude" in df.columns:
        rename["longitude"] = "lon"

    # Variable renames: actual ERDDAP name → our logical name
    inv_map = {v: k for k, v in var_map.items()}
    for col in df.columns:
        if col in inv_map:
            rename[col] = inv_map[col]

    df.rename(columns=rename, inplace=True)

    # Parse date
    if "date_utc" in df.columns:
        df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True).dt.date

    return df


# ---------------------------------------------------------------------------
# Clean + normalise BAA
# ---------------------------------------------------------------------------


def _clean_baa(series: pd.Series) -> pd.Series:
    """Round to int, clamp to [0, BAA_MAX], NaN → BAA_FILL_VALUE."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.round().astype("Int64")  # nullable int
    s = s.clip(lower=C.BAA_MIN, upper=C.BAA_MAX)
    return s


def _clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a single CSV chunk in-place: BAA, covariates, NaN filter, cell_id."""
    if "baa" in df.columns:
        df["baa_cat"] = _clean_baa(df["baa"])
        df.drop(columns=["baa"], inplace=True, errors="ignore")

    for col in ["dhw", "hotspot"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "baa_cat" in df.columns:
        df = df[df["baa_cat"].notna()].copy()

    df.dropna(subset=["lat", "lon", "date_utc"], inplace=True)

    if df.empty:
        return df

    df["cell_id"] = df.apply(lambda r: make_cell_id(r["lat"], r["lon"]), axis=1)

    keep = ["cell_id", "lat", "lon", "date_utc", "baa_cat"]
    for optional in ["dhw", "hotspot"]:
        if optional in df.columns:
            keep.append(optional)

    return df[keep]


# ---------------------------------------------------------------------------
# Streaming build (one CSV at a time → parquet)
# ---------------------------------------------------------------------------


def build_long_table(
    cached_csvs: List[Path],
    var_map: Dict[str, str],
    out_path: Path,
    parquet: bool = True,
    n_shards: int = 0,
) -> Path:
    """
    Merge all cached chunk CSVs into one long table, streaming one CSV at a time.
    Never holds more than one CSV's data in memory.

    Returns path to the output parquet file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq_path = out_path.with_suffix(".parquet")

    writer: Optional[pq.ParquetWriter] = None
    schema: Optional[pa.Schema] = None
    total_rows = 0

    for i, p in enumerate(sorted(cached_csvs)):
        df = _read_cached_csv(p, var_map)
        if df is None or df.empty:
            continue

        df = _clean_chunk(df)
        if df.empty:
            continue

        # Ensure date_utc is string for consistent parquet schema
        df["date_utc"] = df["date_utc"].astype(str)

        table = pa.Table.from_pandas(df, preserve_index=False)

        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(pq_path, schema)
        else:
            table = table.cast(schema)

        writer.write_table(table)
        total_rows += len(df)

        if (i + 1) % 50 == 0:
            logger.info(
                "Processed %d / %d cached files (%d rows so far)",
                i + 1,
                len(cached_csvs),
                total_rows,
            )

        del df, table
        gc.collect()

    if writer:
        writer.close()

    if total_rows == 0:
        raise RuntimeError("No data frames produced from cached CSVs.")

    logger.info("Wrote long table: %s (%d rows)", pq_path, total_rows)

    if n_shards > 0:
        shard_dir = out_path.parent / "shards"
        shard_by_cell(pq_path, shard_dir, n_shards)

    return pq_path


# ---------------------------------------------------------------------------
# Cell sharding (for memory-efficient sequence building)
# ---------------------------------------------------------------------------


def shard_by_cell(
    pq_path: Path,
    shard_dir: Path,
    n_shards: int = C.N_CELL_SHARDS,
) -> Path:
    """
    Read a long-table parquet in batches and split rows into N parquet shards
    based on cell_id hash. Each shard contains all data for its subset of cells.
    Memory usage is bounded by batch_size (~500K rows at a time).
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(pq_path)

    writers: Dict[int, pq.ParquetWriter] = {}
    shard_schema: Optional[pa.Schema] = None

    for batch in pf.iter_batches(batch_size=500_000):
        df = batch.to_pandas()
        df["_shard"] = df["cell_id"].apply(lambda x: hash(x) % n_shards)

        if shard_schema is None:
            shard_schema = pa.Schema.from_pandas(
                df.drop(columns=["_shard"]), preserve_index=False
            )

        for shard_id, shard_df in df.groupby("_shard"):
            shard_df = shard_df.drop(columns=["_shard"])
            table = pa.Table.from_pandas(shard_df, preserve_index=False)
            table = table.cast(shard_schema)

            if shard_id not in writers:
                shard_path = shard_dir / f"shard_{shard_id:04d}.parquet"
                writers[shard_id] = pq.ParquetWriter(shard_path, shard_schema)

            writers[shard_id].write_table(table)

        del df
        gc.collect()

    for w in writers.values():
        w.close()

    logger.info("Created %d cell shards in %s", len(writers), shard_dir)
    return shard_dir
