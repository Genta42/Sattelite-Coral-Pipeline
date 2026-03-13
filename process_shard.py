#!/usr/bin/env python3
"""
Process a single shard into sequences. Designed to be run as a standalone
process so multiple shards can be processed in parallel.

Usage: python process_shard.py <shard_path> <out_dir> [--lookback 60] [--horizon 7]
"""
import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from src.build_sequences import _build_cell_sequences, _serialize_sequences
from src import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_one_shard(
    shard_path: Path,
    out_dir: Path,
    lookback: int,
    horizon: int,
    serialization: str = "json",
) -> int:
    """Process a single shard file and write sequences to per-shard parquet."""
    from datetime import date as _date

    te = _date.fromisoformat(C.SPLIT_TRAIN_END)
    ve = _date.fromisoformat(C.SPLIT_VAL_END)

    shard_name = shard_path.stem  # e.g., "shard_0000"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine feature columns
    schema = pq.read_schema(shard_path)
    col_names = set(schema.names)
    feature_cols = ["baa_cat"]
    for optional in ["dhw", "hotspot"]:
        if optional in col_names:
            feature_cols.append(optional)
    read_cols = ["cell_id", "date_utc", "lat", "lon"] + feature_cols

    # Read in row-group batches to stay under memory limit
    pf = pq.ParquetFile(shard_path)
    n_row_groups = pf.metadata.num_row_groups
    RG_BATCH = 500

    logger.info("Reading %d row groups from %s...", n_row_groups, shard_path.name)
    t0 = time.time()

    cell_data: Dict[str, List[pd.DataFrame]] = {}
    for rg_start in range(0, n_row_groups, RG_BATCH):
        rg_end = min(rg_start + RG_BATCH, n_row_groups)
        batch = pf.read_row_groups(
            list(range(rg_start, rg_end)), columns=read_cols
        ).to_pandas()
        for cid, grp in batch.groupby("cell_id", sort=False):
            if cid not in cell_data:
                cell_data[cid] = []
            cell_data[cid].append(grp)
        del batch
        gc.collect()

    total_cells = len(cell_data)
    logger.info("Accumulated %d cells in %.1fs", total_cells, time.time() - t0)

    # Process cells and write sequences
    FLUSH_THRESHOLD = 100_000
    records: List[Dict] = []
    total_samples = 0

    writer_all = None
    split_writers: Dict[str, pq.ParquetWriter] = {}
    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    pa_schema = None

    cell_names = sorted(cell_data.keys())
    for ci, cid in enumerate(cell_names):
        chunks = cell_data.pop(cid)
        cell_df = (
            pd.concat(chunks, ignore_index=True)
            if len(chunks) > 1
            else chunks[0].reset_index(drop=True)
        )
        del chunks

        if len(cell_df) < lookback + horizon:
            continue

        recs = _build_cell_sequences(cell_df, lookback, horizon, feature_cols)
        records.extend(recs)
        del cell_df

        if (ci + 1) % 5000 == 0:
            logger.info(
                "  %d / %d cells (%d samples)", ci + 1, total_cells, total_samples + len(records)
            )

        # Flush when threshold reached or last cell
        if len(records) >= FLUSH_THRESHOLD or ci == len(cell_names) - 1:
            if not records:
                continue

            batch_df = pd.DataFrame(records)
            batch_df = _serialize_sequences(batch_df, serialization, lookback)
            table = pa.Table.from_pandas(batch_df, preserve_index=False)

            if writer_all is None:
                pa_schema = table.schema
                writer_all = pq.ParquetWriter(
                    out_dir / f"{shard_name}_all.parquet", pa_schema
                )

            table = table.cast(pa_schema)
            writer_all.write_table(table)

            # Split files
            dates = pd.to_datetime(batch_df["target_date"]).dt.date
            for split_name, mask in [
                ("train", dates <= te),
                ("val", (dates > te) & (dates <= ve)),
                ("test", dates > ve),
            ]:
                split_chunk = batch_df[mask]
                if split_chunk.empty:
                    continue
                split_table = (
                    pa.Table.from_pandas(split_chunk, preserve_index=False)
                    .cast(pa_schema)
                )
                if split_name not in split_writers:
                    split_writers[split_name] = pq.ParquetWriter(
                        out_dir / f"{shard_name}_{split_name}.parquet", pa_schema
                    )
                split_writers[split_name].write_table(split_table)
                split_counts[split_name] += len(split_chunk)

            total_samples += len(records)
            del batch_df, table, records
            records = []
            gc.collect()

    # Close writers
    if writer_all:
        writer_all.close()
    for w in split_writers.values():
        w.close()

    elapsed = time.time() - t0
    logger.info(
        "Done %s: %d samples in %.1fs (%.1f min)",
        shard_name, total_samples, elapsed, elapsed / 60,
    )
    return total_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("shard_path", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--serialization", default="json")
    args = p.parse_args()

    n = process_one_shard(
        args.shard_path, args.out_dir, args.lookback, args.horizon, args.serialization
    )
    print(f"RESULT:{n}")


if __name__ == "__main__":
    main()
