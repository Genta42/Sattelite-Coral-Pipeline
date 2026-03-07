"""
qa – quality-assurance reports, coverage stats, missingness, class distribution,
     checksum manifest, and basic sanity plots (matplotlib only, no seaborn).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def coverage_report(
    long_table: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """
    Per-continent and global coverage stats:
      - number of cells, dates, total rows
      - date range
      - lat/lon extent
    """
    rows: List[Dict] = []
    for name, (lat_lo, lat_hi, lon_lo, lon_hi) in C.CONTINENT_BOUNDS.items():
        mask = (
            (long_table["lat"] >= lat_lo) & (long_table["lat"] <= lat_hi) &
            (long_table["lon"] >= lon_lo) & (long_table["lon"] <= lon_hi)
        )
        sub = long_table[mask]
        if sub.empty:
            rows.append({"region": name, "n_cells": 0, "n_dates": 0, "n_rows": 0})
            continue
        rows.append({
            "region": name,
            "n_cells": sub["cell_id"].nunique(),
            "n_dates": sub["date_utc"].nunique(),
            "n_rows": len(sub),
            "date_min": str(sub["date_utc"].min()),
            "date_max": str(sub["date_utc"].max()),
            "lat_min": sub["lat"].min(),
            "lat_max": sub["lat"].max(),
            "lon_min": sub["lon"].min(),
            "lon_max": sub["lon"].max(),
        })

    report = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    logger.info("Coverage report → %s", out_path)
    return report


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------

def missingness_report(
    long_table: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """Column-level missingness counts and proportions."""
    total = len(long_table)
    records = []
    for col in long_table.columns:
        n_miss = int(long_table[col].isna().sum())
        records.append({
            "column": col,
            "n_missing": n_miss,
            "pct_missing": round(n_miss / total * 100, 4) if total else 0,
        })
    report = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    logger.info("Missingness report → %s", out_path)
    return report


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------

def class_distribution_report(
    long_table: pd.DataFrame,
    out_path: Path,
    seq_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """BAA class distribution (from long table and optionally from sequence targets)."""
    records = []

    # Long table
    if "baa_cat" in long_table.columns:
        counts = long_table["baa_cat"].value_counts().sort_index()
        total = counts.sum()
        for cls, cnt in counts.items():
            records.append({
                "source": "long_table",
                "baa_class": int(cls),
                "count": int(cnt),
                "proportion": round(cnt / total, 6),
            })

    # Sequence targets
    if seq_df is not None and "y_baa_cat" in seq_df.columns:
        counts = seq_df["y_baa_cat"].value_counts().sort_index()
        total = counts.sum()
        for cls, cnt in counts.items():
            records.append({
                "source": "sequences",
                "baa_class": int(cls),
                "count": int(cnt),
                "proportion": round(cnt / total, 6),
            })

    report = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    logger.info("Class distribution report → %s", out_path)
    return report


# ---------------------------------------------------------------------------
# Checksum manifest
# ---------------------------------------------------------------------------

def checksum_manifest(
    file_paths: List[Path],
    out_path: Path,
) -> pd.DataFrame:
    """SHA-256 checksums for all produced files."""
    records = []
    for fp in sorted(file_paths):
        if fp.is_file():
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            records.append({
                "file": str(fp),
                "size_bytes": fp.stat().st_size,
                "sha256": h,
            })
    manifest = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    logger.info("Checksum manifest → %s (%d files)", out_path, len(manifest))
    return manifest


# ---------------------------------------------------------------------------
# Sanity plots (matplotlib only)
# ---------------------------------------------------------------------------

def plot_class_distribution(
    long_table: pd.DataFrame,
    out_path: Path,
) -> None:
    """Bar chart of BAA class counts."""
    if "baa_cat" not in long_table.columns:
        return
    counts = long_table["baa_cat"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index.astype(str), counts.values, color="#2196F3", edgecolor="black")
    ax.set_xlabel("BAA Category")
    ax.set_ylabel("Count")
    ax.set_title("BAA Class Distribution (Long Table)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Plot → %s", out_path)


def plot_temporal_coverage(
    long_table: pd.DataFrame,
    out_path: Path,
) -> None:
    """Line chart: number of cells observed per date."""
    dates = pd.to_datetime(long_table["date_utc"])
    daily_counts = dates.groupby(dates.dt.to_period("M")).count()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily_counts.index.to_timestamp(), daily_counts.values, linewidth=0.5, color="#FF5722")
    ax.set_xlabel("Month")
    ax.set_ylabel("Observations")
    ax.set_title("Monthly Observation Count")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Plot → %s", out_path)


def plot_spatial_snapshot(
    long_table: pd.DataFrame,
    out_path: Path,
    sample_date: Optional[str] = None,
) -> None:
    """Scatter map of BAA values for one date (or last available date)."""
    if sample_date:
        sub = long_table[long_table["date_utc"].astype(str) == sample_date]
    else:
        last = long_table["date_utc"].max()
        sub = long_table[long_table["date_utc"] == last]
        sample_date = str(last)

    if sub.empty:
        logger.warning("No data for spatial snapshot on %s", sample_date)
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    sc = ax.scatter(
        sub["lon"], sub["lat"],
        c=sub["baa_cat"].astype(float),
        cmap="YlOrRd", s=0.2, vmin=0, vmax=C.BAA_MAX,
    )
    fig.colorbar(sc, ax=ax, label="BAA Category")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"BAA Spatial Snapshot – {sample_date}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Plot → %s", out_path)


# ---------------------------------------------------------------------------
# All-in-one QA runner
# ---------------------------------------------------------------------------

def run_qa(
    long_table: pd.DataFrame,
    reports_dir: Path,
    seq_df: Optional[pd.DataFrame] = None,
    produced_files: Optional[List[Path]] = None,
) -> None:
    """Run all QA reports and plots."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    coverage_report(long_table, reports_dir / "coverage.csv")
    missingness_report(long_table, reports_dir / "missingness.csv")
    class_distribution_report(long_table, reports_dir / "class_distribution.csv", seq_df)

    if produced_files:
        checksum_manifest(produced_files, reports_dir / "checksums.csv")

    plot_class_distribution(long_table, reports_dir / "plot_class_dist.png")
    plot_temporal_coverage(long_table, reports_dir / "plot_temporal.png")
    plot_spatial_snapshot(long_table, reports_dir / "plot_spatial.png")

    logger.info("All QA reports written to %s", reports_dir)
