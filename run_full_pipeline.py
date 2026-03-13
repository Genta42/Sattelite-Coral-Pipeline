#!/usr/bin/env python3
"""
Full historical pipeline runner (1985-2025) with Telegram notifications.
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from notify import notify, notify_error, notify_and_wait

BASE = "D:/genta coral/coral_pipeline"
CLI = str(Path(__file__).parent / "cli.py")


def run_stage(name: str, cmd: list[str]) -> bool:
    print(f"\n{'='*60}")
    print(f"  STAGE: {name}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    elapsed = time.time() - start
    elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}m"

    if result.returncode == 0:
        notify(f"Done: {name} ({elapsed_str})", title="Stage Complete")
        return True
    else:
        notify_error(f"{name} failed (exit {result.returncode}) after {elapsed_str}")
        try:
            if sys.stdin.isatty():
                response = notify_and_wait(f"Retry {name}? [y/n]")
                if response.strip().lower() in ("y", "yes"):
                    return run_stage(name, cmd)
        except EOFError:
            notify("No TTY — cannot prompt for retry. Halting.", title="Pipeline")
        return False


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run coral pipeline stages")
    p.add_argument("--skip-to", type=int, default=1,
                    help="Start from stage N (1-indexed), skipping earlier stages")
    args = p.parse_args()

    stages = [
        (
            "Fetch (1985-2025, reef-belt)",
            [sys.executable, CLI, "fetch",
             "--start-date", "1985-04-01",
             "--end-date", "2025-03-08",
             "--variables", "baa", "dhw", "hotspot",
             "--stride", "4",
             "--reef-belt",
             "--cache-dir", f"{BASE}/data/cache"],
        ),
        (
            "Build table (64 shards)",
            [sys.executable, CLI, "build_table",
             "--cache-dir", f"{BASE}/data/cache",
             "--out-path", f"{BASE}/data/processed/long_table",
             "--n-shards", "64"],
        ),
        (
            "Build sequences (parallel)",
            [sys.executable,
             str(Path(__file__).parent / "parallel_build_sequences.py"),
             "--workers", "8",
             "--lookback", "60",
             "--horizon", "7"],
        ),
        (
            "Train LSTM",
            [sys.executable, CLI, "train",
             "--seq-dir", f"{BASE}/data/sequences",
             "--out-dir", f"{BASE}/models",
             "--max-samples", "500000"],
        ),
        (
            "Evaluate (test split)",
            [sys.executable, CLI, "evaluate",
             "--checkpoint", f"{BASE}/models/best_model.pt",
             "--seq-dir", f"{BASE}/data/sequences",
             "--out-dir", f"{BASE}/models",
             "--split", "test",
             "--max-samples", "500000"],
        ),
    ]

    notify(f"Full pipeline starting (1985-2025) from stage {args.skip_to}", title="Coral Pipeline")
    total_start = time.time()

    for i, (name, cmd) in enumerate(stages):
        if i + 1 < args.skip_to:
            print(f"Skipping stage {i+1}: {name}")
            continue
        if not run_stage(name, cmd):
            notify_error(f"Pipeline halted at: {name}")
            sys.exit(1)

    total_elapsed = (time.time() - total_start) / 3600
    notify(f"All 5 stages complete! Total: {total_elapsed:.1f}h", title="Pipeline FINISHED")


if __name__ == "__main__":
    main()
