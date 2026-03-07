"""
Global configuration, continent bounds, ERDDAP endpoints, and defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# ERDDAP endpoint
# ---------------------------------------------------------------------------
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
DATASET_ID = "NOAA_DHW"

# Candidate variable names (tried in order; first found wins)
BAA_CANDIDATES: List[str] = ["CRW_BAA_7D_MAX", "CRW_BAA"]
DHW_CANDIDATES: List[str] = ["CRW_DHW"]
HOTSPOT_CANDIDATES: List[str] = ["CRW_HOTSPOT"]
SST_ANOMALY_CANDIDATES: List[str] = ["CRW_SSTANOMALY"]

# All variables we *might* request (order matters for fallback)
VARIABLE_GROUPS: Dict[str, List[str]] = {
    "baa": BAA_CANDIDATES,
    "dhw": DHW_CANDIDATES,
    "hotspot": HOTSPOT_CANDIDATES,
    "sst_anomaly": SST_ANOMALY_CANDIDATES,
}

# Dataset time bounds (discovered from ERDDAP metadata)
DATASET_TIME_MIN = "1985-04-01"
DATASET_TIME_MAX = "2025-12-31"  # pipeline upper bound

# Spatial grid resolution (degrees)
GRID_RESOLUTION = 0.05

# ---------------------------------------------------------------------------
# Continent bounding boxes: (lat_min, lat_max, lon_min, lon_max)
# ---------------------------------------------------------------------------
CONTINENT_BOUNDS: Dict[str, Tuple[float, float, float, float]] = {
    "world": (-90.0, 90.0, -180.0, 180.0),
    "africa": (-35.0, 38.0, -20.0, 52.0),
    "asia": (1.0, 77.0, 26.0, 180.0),
    "australia": (-45.0, -10.0, 110.0, 155.0),
    "europe": (35.0, 72.0, -10.0, 60.0),
    "northAmerica": (5.0, 72.0, -170.0, -50.0),
    "southAmerica": (-55.0, 15.0, -82.0, -34.0),
}

# Reef-belt latitude mask (tropical + subtropical where reefs exist)
REEF_BELT_LAT = (-35.0, 35.0)

# ---------------------------------------------------------------------------
# HTTP / concurrency
# ---------------------------------------------------------------------------
MAX_CONCURRENT_REQUESTS = 3
HTTP_TIMEOUT_S = 120
MAX_RETRIES = 5
BACKOFF_BASE_S = 2.0  # exponential: 2, 4, 8, 16, 32 …
RATE_LIMIT_PAUSE_S = 60  # pause on 429

# ---------------------------------------------------------------------------
# Fetch chunking
# ---------------------------------------------------------------------------
# Max days per ERDDAP request (to keep response size manageable)
FETCH_CHUNK_DAYS = 30

# ---------------------------------------------------------------------------
# BAA category mapping
# ---------------------------------------------------------------------------
BAA_MIN = 0
BAA_MAX = 4  # CRW_BAA uses 0-4 (No Stress, Watch, Warning, Alert1, Alert2)
BAA_FILL_VALUE = -1  # what we write for missing/land

# ---------------------------------------------------------------------------
# Sequence / modelling defaults
# ---------------------------------------------------------------------------
DEFAULT_LOOKBACK_DAYS = 60
DEFAULT_HORIZON_DAYS = 7

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
TRAIN_DEFAULTS = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "max_epochs": 100,
    "patience": 10,
    "seed": 42,
}
NUM_CLASSES = 5
SEQUENCE_FEATURES = ["baa_cat", "dhw", "hotspot"]
STATIC_FEATURES = ["lat", "lon"]

# Default temporal splits
SPLIT_TRAIN_END = "2023-12-31"
SPLIT_VAL_END = "2024-12-31"
# Everything after val end → test (2025)


# ---------------------------------------------------------------------------
# SSD storage (for full global pipeline)
# ---------------------------------------------------------------------------
SSD_BASE = Path("/Volumes/New Volume/Coral-Sattelite")

# Spatial tile size for sub-chunking large fetch bounding boxes (degrees)
SPATIAL_TILE_DEG = 10.0

# Number of cell-based parquet shards for memory-efficient processing
N_CELL_SHARDS = 32


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass
class PipelinePaths:
    base: Path = field(default_factory=lambda: Path("coral_pipeline"))

    @property
    def raw(self) -> Path:
        return self.base / "data" / "raw"

    @property
    def cache(self) -> Path:
        return self.base / "data" / "cache"

    @property
    def processed(self) -> Path:
        return self.base / "data" / "processed"

    @property
    def sequences(self) -> Path:
        return self.base / "data" / "sequences"

    @property
    def reports(self) -> Path:
        return self.base / "reports"

    @property
    def models(self) -> Path:
        return self.base / "models"

    def ensure(self) -> None:
        for p in [
            self.raw,
            self.cache,
            self.processed,
            self.sequences,
            self.reports,
            self.models,
        ]:
            p.mkdir(parents=True, exist_ok=True)
