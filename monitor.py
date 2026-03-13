#!/usr/bin/env python3
"""
Live progress monitor for the running pipeline (stage 3).
Polls output file sizes + process stats since parquet metadata
is unavailable while the writer is open.
"""
import subprocess
import sys
import time
from pathlib import Path

try:
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("pip install rich")
    sys.exit(1)

SEQ_DIR = Path("D:/genta coral/coral_pipeline/data/sequences")
SHARD_DIR = Path("D:/genta coral/coral_pipeline/data/processed/shards")
LOG_FILE = Path(__file__).parent / "pipeline.log"  # if logging to file
TOTAL_SHARDS = 64
POLL_SEC = 5

# Pipeline started at this time (from process start)
PIPELINE_START_STR = "2026-03-10 22:48:07"


def _file_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _fmt_size(b: int) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.2f} GB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.1f} MB"
    return f"{b / (1 << 10):.0f} KB"


def _fmt_elapsed(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _get_process_info() -> dict | None:
    """Get the main pipeline python process info."""
    try:
        out = subprocess.check_output(
            ["powershell", "-Command",
             "Get-Process python* | Where-Object {$_.WorkingSet64 -gt 1GB} "
             "| Select-Object Id,CPU,@{N='MemGB';E={[math]::Round($_.WorkingSet64/1GB,2)}} "
             "| ConvertTo-Json"],
            text=True, timeout=5
        )
        import json
        data = json.loads(out)
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data:
            return max(data, key=lambda x: x.get("MemGB", 0))
    except Exception:
        pass
    return None


def _get_pipeline_log_tail(n: int = 5) -> list[str]:
    """Get last N lines from pipeline task output."""
    task_output = Path(
        "C:/Users/USER/AppData/Local/Temp/claude/C--Users-USER-Desktop-genta/tasks/bmigl4emj.output"
    )
    try:
        text = task_output.read_text(encoding="utf-8", errors="replace")
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        return lines[-n:]
    except Exception:
        return []


def _estimate_progress(all_size: int, elapsed: float, sizes_history: list) -> tuple[str, str]:
    """Estimate progress from file size growth rate."""
    if not sizes_history or elapsed < 120:
        return "calculating...", "calculating..."

    # Growth rate from last few samples
    if len(sizes_history) >= 2:
        recent = sizes_history[-1]
        older = sizes_history[max(0, len(sizes_history) - 12)]  # ~1 min ago
        dt = recent[0] - older[0]
        ds = recent[1] - older[1]
        if dt > 0 and ds > 0:
            rate = ds / dt  # bytes per second
            rate_str = f"{_fmt_size(int(rate))}/s"
            return rate_str, ""

    return "", ""


def build_display(
    start_time: float,
    proc_info: dict | None,
    sizes_history: list,
    prev_all_size: int,
) -> Table:
    now = time.time()
    elapsed = now - start_time

    files = {
        "sequences_all.parquet": SEQ_DIR / "sequences_all.parquet",
        "sequences_train.parquet": SEQ_DIR / "sequences_train.parquet",
        "sequences_val.parquet": SEQ_DIR / "sequences_val.parquet",
        "sequences_test.parquet": SEQ_DIR / "sequences_test.parquet",
    }

    all_size = _file_size(files["sequences_all.parquet"])

    table = Table(
        title=f"[bold]Stage 3: Build Sequences[/bold]  |  Elapsed: {_fmt_elapsed(elapsed)}",
        expand=True,
        border_style="cyan",
    )
    table.add_column("", style="bold cyan", width=28)
    table.add_column("Value", style="green", justify="right", width=20)

    # File sizes
    table.add_row("[bold]Output Files[/bold]", "")
    for name, path in files.items():
        sz = _file_size(path)
        table.add_row(f"  {name}", _fmt_size(sz) if sz else "not yet")

    total_size = sum(_file_size(p) for p in files.values())
    table.add_row("  [bold]Total output[/bold]", f"[bold]{_fmt_size(total_size)}[/bold]")

    # Growth rate
    table.add_section()
    table.add_row("[bold]Throughput[/bold]", "")
    if len(sizes_history) >= 6:
        recent = sizes_history[-1]
        older = sizes_history[max(0, len(sizes_history) - 12)]
        dt = recent[0] - older[0]
        ds = recent[1] - older[1]
        if dt > 30 and ds > 0:
            rate = ds / dt
            table.add_row("  Write rate", f"{_fmt_size(int(rate))}/s")

            # Very rough ETA: assume ~10-15 GB total output (based on 14GB input with sequence compression)
            # Better: use proportion of shard input processed
            shard_total_size = sum(
                _file_size(f) for f in SHARD_DIR.glob("shard_*.parquet")
            )
            # Output is roughly proportional to input; estimate total output
            if all_size > 0 and elapsed > 300:
                # project total based on current rate and elapsed
                est_total_time = elapsed * (shard_total_size / (all_size * 0.7))  # rough ratio
                remaining = max(0, est_total_time - elapsed)
                table.add_row("  Est. remaining", f"~{_fmt_elapsed(remaining)}")
        else:
            table.add_row("  Write rate", "measuring...")
    else:
        table.add_row("  Write rate", "measuring...")

    # Delta since last check
    delta = all_size - prev_all_size
    if delta > 0:
        table.add_row("  Last 5s delta", f"+{_fmt_size(delta)}")

    # Process info
    table.add_section()
    table.add_row("[bold]Process[/bold]", "")
    if proc_info:
        table.add_row("  PID", str(proc_info.get("Id", "?")))
        table.add_row("  Memory", f"{proc_info.get('MemGB', '?')} GB")
        cpu_s = proc_info.get("CPU", 0)
        table.add_row("  CPU time", _fmt_elapsed(cpu_s))
    else:
        table.add_row("  Status", "[red]Process not found![/red]")

    # Log tail
    table.add_section()
    table.add_row("[bold]Last log lines[/bold]", "")
    for line in _get_pipeline_log_tail(3):
        # Truncate long lines
        if len(line) > 70:
            line = line[:67] + "..."
        table.add_row(f"  {line}", "")

    return table


def main():
    # Use actual pipeline start time
    from datetime import datetime
    start_time = datetime.strptime(PIPELINE_START_STR, "%Y-%m-%d %H:%M:%S").timestamp()

    sizes_history: list[tuple[float, int]] = []
    proc_info = None
    proc_check_interval = 15
    last_proc_check = 0.0
    prev_all_size = 0

    with Live(Text("Starting monitor..."), refresh_per_second=0.3, console=None) as live:
        while True:
            now = time.time()

            all_size = _file_size(SEQ_DIR / "sequences_all.parquet")
            sizes_history.append((now, all_size))
            # Keep last 5 min of history
            cutoff = now - 300
            sizes_history = [(t, s) for t, s in sizes_history if t > cutoff]

            if now - last_proc_check > proc_check_interval:
                proc_info = _get_process_info()
                last_proc_check = now

            display = build_display(start_time, proc_info, sizes_history, prev_all_size)
            live.update(display)

            prev_all_size = all_size
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
