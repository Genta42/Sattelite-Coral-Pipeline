"""
export – write final CSV/Parquet outputs per continent and combined global.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import config as C

logger = logging.getLogger(__name__)


def export_by_continent(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str = "sequences",
    parquet: bool = True,
) -> List[Path]:
    """
    Split *df* by continent bounding boxes and write one file per continent.
    Also writes a combined global file.
    Returns list of all written file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for name, (lat_lo, lat_hi, lon_lo, lon_hi) in C.CONTINENT_BOUNDS.items():
        if name == "world":
            continue  # handled separately as "global"
        mask = (
            (df["lat"] >= lat_lo) & (df["lat"] <= lat_hi) &
            (df["lon"] >= lon_lo) & (df["lon"] <= lon_hi)
        )
        sub = df[mask]
        if sub.empty:
            logger.info("No data for continent %s – skipping.", name)
            continue

        csv_path = out_dir / f"{prefix}_{name}.csv"
        sub.to_csv(csv_path, index=False)
        written.append(csv_path)
        logger.info("Exported %s: %d rows → %s", name, len(sub), csv_path)

        if parquet:
            pq_path = csv_path.with_suffix(".parquet")
            sub.to_parquet(pq_path, index=False, engine="pyarrow")
            written.append(pq_path)

    # Global combined
    csv_path = out_dir / f"{prefix}_global.csv"
    df.to_csv(csv_path, index=False)
    written.append(csv_path)
    logger.info("Exported global: %d rows → %s", len(df), csv_path)

    if parquet:
        pq_path = csv_path.with_suffix(".parquet")
        df.to_parquet(pq_path, index=False, engine="pyarrow")
        written.append(pq_path)

    return written
