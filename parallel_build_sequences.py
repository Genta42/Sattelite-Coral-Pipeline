#!/usr/bin/env python3
"""
Parallel sequence builder — launches independent subprocess per shard.
Each worker is a fully separate OS process with its own memory space.

Usage: python parallel_build_sequences.py [--workers 8]
"""
import argparse
import gc
import logging
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread, Lock
from queue import Queue

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from notify import notify, notify_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE = Path("D:/genta coral/coral_pipeline")
SHARD_DIR = BASE / "data/processed/shards"
OUT_DIR = BASE / "data/sequences"
WORKER_SCRIPT = Path(__file__).parent / "process_shard.py"


def worker_thread(task_queue: Queue, results: list, results_lock: Lock,
                  out_dir: Path, lookback: int, horizon: int):
    """Thread that pulls shard paths from queue and runs subprocess workers."""
    while True:
        item = task_queue.get()
        if item is None:
            break

        shard_path = item
        shard_name = shard_path.stem
        start = time.time()

        try:
            result = subprocess.run(
                [
                    sys.executable, str(WORKER_SCRIPT),
                    str(shard_path), str(out_dir),
                    "--lookback", str(lookback),
                    "--horizon", str(horizon),
                ],
                capture_output=True,
                text=True,
                timeout=7200,
            )
            elapsed = time.time() - start

            if result.returncode == 0:
                samples = 0
                for line in result.stdout.strip().splitlines():
                    if line.startswith("RESULT:"):
                        samples = int(line.split(":")[1])
                with results_lock:
                    results.append((shard_name, samples, elapsed, True))
            else:
                stderr_tail = result.stderr[-1500:] if result.stderr else "no stderr"
                logger.error("Worker %s failed (exit %d):\n%s",
                             shard_name, result.returncode, stderr_tail)
                with results_lock:
                    results.append((shard_name, 0, elapsed, False))

        except subprocess.TimeoutExpired:
            with results_lock:
                results.append((shard_name, 0, time.time() - start, False))
        except Exception as e:
            logger.error("Worker %s exception: %s", shard_name, e)
            with results_lock:
                results.append((shard_name, 0, time.time() - start, False))

        task_queue.task_done()


def merge_shard_outputs(out_dir: Path):
    """Merge per-shard parquet files into final combined files."""
    logger.info("Merging shard outputs...")

    for suffix in ["all", "train", "val", "test"]:
        shard_files = sorted(out_dir.glob(f"shard_*_{suffix}.parquet"))
        if not shard_files:
            continue

        out_path = out_dir / f"sequences_{suffix}.parquet"
        writer = None
        schema = None
        total_rows = 0

        for sf in shard_files:
            try:
                table = pq.read_table(sf)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(out_path, schema)
                table = table.cast(schema)
                writer.write_table(table)
                total_rows += len(table)
                del table
                gc.collect()
                sf.unlink()  # delete shard immediately after writing to free disk
            except Exception as e:
                logger.warning("Skip %s: %s", sf.name, e)

        if writer:
            writer.close()
            logger.info("  %s: %d rows", out_path.name, total_rows)


def main():
    p = argparse.ArgumentParser(description="Parallel shard sequence builder")
    p.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--horizon", type=int, default=7)
    args = p.parse_args()

    shard_files = sorted(SHARD_DIR.glob("shard_*.parquet"))
    n_shards = len(shard_files)
    if not shard_files:
        logger.error("No shard files found in %s", SHARD_DIR)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    notify(
        f"Parallel build sequences: {n_shards} shards, {args.workers} workers",
        title="Build Sequences",
    )

    try:
        from notify import ProgressTracker
        tracker = ProgressTracker("Build Sequences", total=n_shards, unit="shards")
    except ImportError:
        tracker = None

    results = []
    results_lock = Lock()
    task_queue = Queue()

    # Start worker threads (each launches subprocess.run — separate OS processes)
    threads = []
    for _ in range(args.workers):
        t = Thread(
            target=worker_thread,
            args=(task_queue, results, results_lock, OUT_DIR, args.lookback, args.horizon),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Enqueue all shards
    for sf in shard_files:
        task_queue.put(sf)

    # Monitor progress
    start_time = time.time()
    total_samples = 0
    completed = 0
    failed = []

    while completed < n_shards:
        time.sleep(5)
        with results_lock:
            new_results = results[completed:]

        for shard_name, samples, elapsed, success in new_results:
            completed += 1
            if success:
                total_samples += samples
                logger.info(
                    "[%d/%d] %s: %d samples in %.1fm (total: %d)",
                    completed, n_shards, shard_name, samples, elapsed / 60, total_samples,
                )
            else:
                failed.append(shard_name)
                logger.error("[%d/%d] %s: FAILED", completed, n_shards, shard_name)

            if tracker:
                tracker.update(completed, extra={"samples": total_samples})

    # Signal threads to exit
    for _ in threads:
        task_queue.put(None)
    for t in threads:
        t.join(timeout=10)

    total_elapsed = time.time() - start_time

    if failed:
        notify_error(f"Build sequences: {len(failed)}/{n_shards} shards failed: {failed[:10]}")
    else:
        logger.info("All %d shards completed. Merging...", n_shards)

    # Merge per-shard files into final outputs
    merge_shard_outputs(OUT_DIR)

    if tracker:
        tracker.finish(extra={"samples": total_samples})

    elapsed_str = f"{total_elapsed/3600:.1f}h"
    notify(
        f"Build sequences done: {total_samples:,} samples in {elapsed_str} ({args.workers} workers)",
        title="Build Sequences Complete",
    )

    logger.info(
        "DONE: %d samples from %d shards in %s (%d workers)",
        total_samples, n_shards, elapsed_str, args.workers,
    )

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
