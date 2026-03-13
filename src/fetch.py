"""
ERDDAP griddap fetcher with:
  - automatic dataset/variable discovery & fallback
  - spatial batching by continent
  - temporal chunking (month-sized requests)
  - robust HTTP: retries, exponential backoff, 429 handling
  - local CSV cache (skip already-downloaded chunks)
  - concurrency limiter (asyncio semaphore)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import aiohttp
import pandas as pd

import sys

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

def _rich_usable() -> bool:
    """Check if rich progress bars can safely render."""
    if not HAS_RICH:
        return False
    if not sys.stdout.isatty():
        return False
    try:
        Console(force_terminal=True).print("", end="")
        return True
    except (UnicodeEncodeError, OSError):
        return False

_USE_RICH = None

def _should_use_rich() -> bool:
    global _USE_RICH
    if _USE_RICH is None:
        _USE_RICH = _rich_usable()
    return _USE_RICH

from . import config as C

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snap_coord(val: float, resolution: float = C.GRID_RESOLUTION) -> float:
    """Snap a coordinate to the nearest grid centre."""
    return round(round(val / resolution) * resolution, 4)


def _chunk_date_range(
    start: date, end: date, chunk_days: int = C.FETCH_CHUNK_DAYS
) -> List[Tuple[date, date]]:
    """Split [start, end] into sub-ranges of at most *chunk_days*."""
    chunks: List[Tuple[date, date]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def _cache_key(
    dataset_id: str,
    variables: List[str],
    t0: date,
    t1: date,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    stride: int,
) -> str:
    """Deterministic hash for a request → used as cache filename."""
    blob = json.dumps(
        {
            "ds": dataset_id,
            "vars": sorted(variables),
            "t0": str(t0),
            "t1": str(t1),
            "lat": lat_range,
            "lon": lon_range,
            "stride": stride,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _tile_bbox(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    tile_deg: float = C.SPATIAL_TILE_DEG,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Split a large bounding box into sub-tiles of at most *tile_deg* degrees."""
    tiles = []
    lat = lat_range[0]
    while lat < lat_range[1]:
        lat_end = min(lat + tile_deg, lat_range[1])
        lon = lon_range[0]
        while lon < lon_range[1]:
            lon_end = min(lon + tile_deg, lon_range[1])
            tiles.append(((lat, lat_end), (lon, lon_end)))
            lon = lon_end
        lat = lat_end
    return tiles


# ---------------------------------------------------------------------------
# Variable discovery
# ---------------------------------------------------------------------------


async def discover_variables(
    session: aiohttp.ClientSession,
    dataset_id: str = C.DATASET_ID,
) -> Dict[str, str]:
    """
    Query ERDDAP info endpoint and return a map of
    logical-name → actual-variable-name for everything available.
    E.g. {"baa": "CRW_BAA", "dhw": "CRW_DHW", ...}
    """
    url = f"{C.ERDDAP_BASE.rsplit('/griddap', 1)[0]}/info/{dataset_id}/index.json"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        resp.raise_for_status()
        payload = await resp.json()

    # Extract variable names from the info table
    col_names = payload["table"]["columnNames"]
    rows = payload["table"]["rows"]
    row_type_idx = col_names.index("Row Type")
    var_name_idx = col_names.index("Variable Name")

    available_vars = {r[var_name_idx] for r in rows if r[row_type_idx] == "variable"}

    resolved: Dict[str, str] = {}
    for logical, candidates in C.VARIABLE_GROUPS.items():
        for cand in candidates:
            if cand in available_vars:
                resolved[logical] = cand
                logger.info("Resolved %s → %s", logical, cand)
                break
        else:
            logger.warning(
                "No variable found for group '%s' (tried %s)", logical, candidates
            )

    if "baa" not in resolved:
        raise RuntimeError(
            f"Could not find any BAA variable in dataset {dataset_id}. "
            f"Tried: {C.BAA_CANDIDATES}"
        )
    return resolved


# ---------------------------------------------------------------------------
# Time bounds discovery
# ---------------------------------------------------------------------------


async def discover_time_bounds(
    session: aiohttp.ClientSession,
    dataset_id: str = C.DATASET_ID,
) -> Tuple[str, str]:
    """Return (min_time, max_time) ISO strings from ERDDAP metadata."""
    url = f"{C.ERDDAP_BASE.rsplit('/griddap', 1)[0]}/info/{dataset_id}/index.json"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        resp.raise_for_status()
        payload = await resp.json()

    col_names = payload["table"]["columnNames"]
    rows = payload["table"]["rows"]
    var_idx = col_names.index("Variable Name")
    attr_idx = col_names.index("Attribute Name")
    val_idx = col_names.index("Value")

    time_min = time_max = None
    for r in rows:
        if r[var_idx] == "time":
            if r[attr_idx] == "actual_range":
                parts = r[val_idx].split(",")
                time_min = parts[0].strip()
                time_max = parts[1].strip()
                break

    if not time_min:
        # fallback: use configured defaults
        time_min = C.DATASET_TIME_MIN + "T12:00:00Z"
        time_max = C.DATASET_TIME_MAX + "T12:00:00Z"
    else:
        # ERDDAP may return epoch seconds as floats – convert to ISO
        from datetime import datetime, timezone

        def _to_iso(val: str) -> str:
            try:
                epoch = float(val)
                return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            except ValueError:
                return val  # already ISO

        time_min = _to_iso(time_min)
        time_max = _to_iso(time_max)

    logger.info("Dataset time bounds: %s → %s", time_min, time_max)
    return time_min, time_max


# ---------------------------------------------------------------------------
# Single-chunk fetcher
# ---------------------------------------------------------------------------


async def _fetch_chunk(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    dataset_id: str,
    variables: List[str],
    t0: date,
    t1: date,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    stride: int,
    cache_dir: Path,
) -> Optional[Path]:
    """
    Fetch one temporal/spatial chunk from ERDDAP griddap as CSV.
    Returns path to cached CSV, or None on unrecoverable failure.
    """
    cache_name = _cache_key(dataset_id, variables, t0, t1, lat_range, lon_range, stride)
    cache_path = cache_dir / f"{cache_name}.csv"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        logger.debug("Cache hit: %s", cache_path.name)
        return cache_path

    # Build ERDDAP griddap query
    lat_lo = _snap_coord(lat_range[0])
    lat_hi = _snap_coord(lat_range[1])
    lon_lo = _snap_coord(lon_range[0])
    lon_hi = _snap_coord(lon_range[1])

    time_start = f"{t0}T12:00:00Z"
    time_end = f"{t1}T12:00:00Z"

    stride_str = str(stride)
    constraint = (
        f"[({time_start}):1:({time_end})]"
        f"[({lat_lo}):{stride_str}:({lat_hi})]"
        f"[({lon_lo}):{stride_str}:({lon_hi})]"
    )

    var_query = ",".join(f"{v}{constraint}" for v in variables)
    url = f"{C.ERDDAP_BASE}/{dataset_id}.csv?{var_query}"

    for attempt in range(1, C.MAX_RETRIES + 1):
        async with sem:
            try:
                logger.info(
                    "Fetch [%d/%d] %s→%s lat(%.1f,%.1f) lon(%.1f,%.1f) stride=%d",
                    attempt,
                    C.MAX_RETRIES,
                    t0,
                    t1,
                    lat_lo,
                    lat_hi,
                    lon_lo,
                    lon_hi,
                    stride,
                )
                timeout = aiohttp.ClientTimeout(total=C.HTTP_TIMEOUT_S)
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status == 429:
                        logger.warning(
                            "Rate limited (429). Pausing %ds …", C.RATE_LIMIT_PAUSE_S
                        )
                        await asyncio.sleep(C.RATE_LIMIT_PAUSE_S)
                        continue

                    if resp.status >= 500:
                        wait = C.BACKOFF_BASE_S**attempt
                        logger.warning(
                            "Server error %d. Backoff %.0fs …", resp.status, wait
                        )
                        await asyncio.sleep(wait)
                        continue

                    if resp.status == 404:
                        logger.warning(
                            "404 for chunk %s→%s – possibly no data in range.", t0, t1
                        )
                        return None

                    resp.raise_for_status()
                    data = await resp.read()

                    # Write to cache atomically
                    tmp = cache_path.with_suffix(".tmp")
                    tmp.write_bytes(data)
                    tmp.replace(cache_path)
                    logger.info("Cached %s (%d KB)", cache_path.name, len(data) // 1024)
                    return cache_path

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                wait = C.BACKOFF_BASE_S**attempt
                logger.warning("Request error: %s. Backoff %.0fs …", exc, wait)
                await asyncio.sleep(wait)

    logger.error("All %d attempts failed for chunk %s→%s", C.MAX_RETRIES, t0, t1)
    return None


# ---------------------------------------------------------------------------
# Public fetch orchestrator
# ---------------------------------------------------------------------------


async def fetch_region(
    start_date: date,
    end_date: date,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    variables: List[str],
    stride: int = 1,
    cache_dir: Path = Path("data/cache"),
    dataset_id: str = C.DATASET_ID,
    out_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Fetch all temporal chunks for one spatial region.
    Large bounding boxes are automatically split into spatial tiles
    to keep individual ERDDAP requests at a manageable size.
    Returns list of cached CSV paths.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporal_chunks = _chunk_date_range(start_date, end_date, C.FETCH_CHUNK_DAYS)

    lat_span = lat_range[1] - lat_range[0]
    lon_span = lon_range[1] - lon_range[0]
    if lat_span > C.SPATIAL_TILE_DEG or lon_span > C.SPATIAL_TILE_DEG:
        tiles = _tile_bbox(lat_range, lon_range, C.SPATIAL_TILE_DEG)
        logger.info(
            "Split bbox into %d spatial tiles (%.0f° each)",
            len(tiles),
            C.SPATIAL_TILE_DEG,
        )
    else:
        tiles = [(lat_range, lon_range)]

    all_fetches = [
        (t0, t1, lat_r, lon_r) for t0, t1 in temporal_chunks for lat_r, lon_r in tiles
    ]
    logger.info(
        "Total fetch units: %d (%d temporal × %d spatial)",
        len(all_fetches),
        len(temporal_chunks),
        len(tiles),
    )

    BATCH = 30
    sem = asyncio.Semaphore(C.MAX_CONCURRENT_REQUESTS)
    paths: List[Path] = []
    cached_count = 0
    total = len(all_fetches)

    try:
        from notify import ProgressTracker
        ntfy_tracker = ProgressTracker("Fetch", total=total, unit="chunks")
    except ImportError:
        ntfy_tracker = None

    if _should_use_rich():
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("cached: {task.fields[cached]}"),
            TimeRemainingColumn(),
            console=Console(force_terminal=True),
        )
        task_id = progress.add_task("Fetching", total=total, cached=0)
        progress.start()
    else:
        progress = None

    try:
        async with aiohttp.ClientSession() as session:
            for batch_start in range(0, total, BATCH):
                batch = all_fetches[batch_start : batch_start + BATCH]
                tasks = [
                    _fetch_chunk(
                        session,
                        sem,
                        dataset_id,
                        variables,
                        t0,
                        t1,
                        lat_r,
                        lon_r,
                        stride,
                        cache_dir,
                    )
                    for t0, t1, lat_r, lon_r in batch
                ]
                results = await asyncio.gather(*tasks)
                for p in results:
                    if p is not None:
                        paths.append(p)
                        cached_count = len(paths)

                done = min(batch_start + BATCH, total)
                if progress is not None:
                    progress.update(task_id, completed=done, cached=cached_count)
                else:
                    logger.info("Fetch progress: %d / %d", done, total)

                if ntfy_tracker:
                    ntfy_tracker.update(done, extra={"cached": cached_count})
    finally:
        if progress is not None:
            progress.stop()
        if ntfy_tracker:
            ntfy_tracker.finish(extra={"cached": cached_count})

    logger.info(
        "Fetched %d / %d total chunks for region",
        len(paths),
        total,
    )
    return paths


def fetch_continent(
    continent: str,
    start_date: date,
    end_date: date,
    variables: List[str],
    stride: int = 1,
    cache_dir: Path = Path("data/cache"),
    dataset_id: str = C.DATASET_ID,
) -> List[Path]:
    """Synchronous wrapper: fetch all data for a named continent."""
    bounds = C.CONTINENT_BOUNDS.get(continent)
    if bounds is None:
        raise ValueError(
            f"Unknown continent '{continent}'. Choose from: {list(C.CONTINENT_BOUNDS)}"
        )

    lat_range = (bounds[0], bounds[1])
    lon_range = (bounds[2], bounds[3])

    return asyncio.run(
        fetch_region(
            start_date=start_date,
            end_date=end_date,
            lat_range=lat_range,
            lon_range=lon_range,
            variables=variables,
            stride=stride,
            cache_dir=cache_dir,
            dataset_id=dataset_id,
        )
    )


def run_discovery(dataset_id: str = C.DATASET_ID) -> Dict[str, str]:
    """Synchronous wrapper for variable discovery."""

    async def _inner():
        async with aiohttp.ClientSession() as session:
            return await discover_variables(session, dataset_id)

    return asyncio.run(_inner())


def run_time_discovery(dataset_id: str = C.DATASET_ID) -> Tuple[str, str]:
    """Synchronous wrapper for time-bounds discovery."""

    async def _inner():
        async with aiohttp.ClientSession() as session:
            return await discover_time_bounds(session, dataset_id)

    return asyncio.run(_inner())
