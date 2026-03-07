#!/usr/bin/env bash
#
# run_pipeline.sh — Full global coral bleaching training pipeline
#
# Designed for long-running unattended operation:
#   - Resumable: each step writes a completion marker; re-run skips done steps
#   - Low priority: uses 'nice' to avoid hogging CPU during normal use
#   - Internet-resilient: fetch retries on network loss with exponential backoff
#   - Memory-safe: uses streaming/sharded modes for 18 GB RAM systems
#   - Fetch cache is atomic (write .tmp → rename), safe to kill anytime
#
# Usage:
#   ./run_pipeline.sh          # run full pipeline (resume from last completed step)
#   ./run_pipeline.sh --reset  # delete markers and re-run everything
#
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────
SSD="/Volumes/New Volume/Coral-Sattelite"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLI="$SCRIPT_DIR/cli.py"
LOG="$SSD/pipeline.log"
MARKERS="$SSD/.markers"
MIN_FREE_GB=20
N_SHARDS=32

# ─── Helpers ──────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

check_ssd() {
    if [ ! -d "$SSD" ]; then
        echo "ERROR: SSD not mounted at $SSD"
        exit 1
    fi
}

check_disk_space() {
    local free_gb
    free_gb=$(df -g "$SSD" | tail -1 | awk '{print $4}')
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        log "ERROR: Only ${free_gb} GB free (minimum: ${MIN_FREE_GB} GB). Aborting."
        exit 1
    fi
    log "Disk: ${free_gb} GB free"
}

disk_usage() {
    log "--- Disk usage ---"
    du -sh "$SSD"/data/cache/ 2>/dev/null   | tee -a "$LOG" || true
    du -sh "$SSD"/data/processed/ 2>/dev/null | tee -a "$LOG" || true
    du -sh "$SSD"/data/sequences/ 2>/dev/null | tee -a "$LOG" || true
    du -sh "$SSD"/models/ 2>/dev/null         | tee -a "$LOG" || true
    log "--- end ---"
}

step_done() { [ -f "$MARKERS/$1.done" ]; }
mark_done() { touch "$MARKERS/$1.done"; log "Step '$1' marked complete."; }

wait_for_internet() {
    local wait=10
    while ! curl -s --max-time 5 https://coastwatch.pfeg.noaa.gov >/dev/null 2>&1; do
        log "No internet. Retrying in ${wait}s..."
        sleep "$wait"
        wait=$((wait < 300 ? wait * 2 : 300))
    done
}

# ─── Parse args ───────────────────────────────────────────────────────────
if [ "${1:-}" = "--reset" ]; then
    rm -rf "$MARKERS"
    echo "Markers reset. Will re-run all steps."
fi

# ─── Pre-flight ───────────────────────────────────────────────────────────

check_ssd

mkdir -p "$SSD/data/cache" \
         "$SSD/data/processed" \
         "$SSD/data/processed/shards" \
         "$SSD/data/sequences" \
         "$SSD/models" \
         "$MARKERS"

log "=========================================="
log "Pipeline starting (resumable)"
log "=========================================="

# ─── Step 1: Fetch global data ────────────────────────────────────────────
if ! step_done "01_fetch"; then
    log "STEP 1/6: Fetching global ERDDAP data..."
    check_disk_space
    wait_for_internet

    nice -n 15 python "$CLI" fetch \
        --start-date 1985-04-01 \
        --end-date   2025-12-31 \
        --continents world \
        --variables  baa dhw hotspot \
        --stride     1 \
        --cache-dir  "$SSD/data/cache" \
        2>&1 | tee -a "$LOG"

    mark_done "01_fetch"
    disk_usage
else
    log "STEP 1/6: Fetch already complete, skipping."
fi

# ─── Step 2: Build long table + shards ────────────────────────────────────
if ! step_done "02_build_table"; then
    log "STEP 2/6: Building long table (streaming) + cell shards..."
    check_disk_space

    nice -n 15 python "$CLI" build_table \
        --cache-dir  "$SSD/data/cache" \
        --out-path   "$SSD/data/processed/long_table" \
        --n-shards   "$N_SHARDS" \
        2>&1 | tee -a "$LOG"

    mark_done "02_build_table"
    disk_usage
else
    log "STEP 2/6: Build table already complete, skipping."
fi

# ─── Step 3: Build sequences (from shards) ───────────────────────────────
if ! step_done "03_build_sequences"; then
    log "STEP 3/6: Building LSTM-ready sequences (streaming from shards)..."
    check_disk_space

    nice -n 15 python "$CLI" build_sequences \
        --shard-dir  "$SSD/data/processed/shards" \
        --out-dir    "$SSD/data/sequences" \
        --lookback   60 \
        --horizon    7 \
        --serialization json \
        2>&1 | tee -a "$LOG"

    mark_done "03_build_sequences"
    disk_usage
else
    log "STEP 3/6: Build sequences already complete, skipping."
fi

# ─── Step 4: Train ────────────────────────────────────────────────────────
if ! step_done "04_train"; then
    log "STEP 4/6: Training LSTM model..."
    check_disk_space

    nice -n 10 python "$CLI" train \
        --seq-dir     "$SSD/data/sequences" \
        --out-dir     "$SSD/models" \
        --max-samples 2000000 \
        2>&1 | tee -a "$LOG"

    mark_done "04_train"
else
    log "STEP 4/6: Training already complete, skipping."
fi

# ─── Step 5: Evaluate ────────────────────────────────────────────────────
if ! step_done "05_evaluate"; then
    log "STEP 5/6: Evaluating on test split..."

    nice -n 10 python "$CLI" evaluate \
        --checkpoint "$SSD/models/best_model.pt" \
        --seq-dir    "$SSD/data/sequences" \
        --out-dir    "$SSD/models" \
        2>&1 | tee -a "$LOG"

    mark_done "05_evaluate"
else
    log "STEP 5/6: Evaluation already complete, skipping."
fi

# ─── Step 6: Export CoreML ────────────────────────────────────────────────
if ! step_done "06_export"; then
    log "STEP 6/6: Exporting to CoreML..."

    python "$CLI" export_model \
        --checkpoint "$SSD/models/best_model.pt" \
        --out-path   "$SSD/models/CoralBleaching.mlpackage" \
        2>&1 | tee -a "$LOG"

    mark_done "06_export"
else
    log "STEP 6/6: Export already complete, skipping."
fi

# ─── Summary ─────────────────────────────────────────────────────────────
log "=========================================="
log "Pipeline finished successfully!"
log "=========================================="
disk_usage

echo ""
echo "Outputs:"
echo "  Long table:  $SSD/data/processed/long_table.parquet"
echo "  Sequences:   $SSD/data/sequences/"
echo "  Model:       $SSD/models/best_model.pt"
echo "  CoreML:      $SSD/models/CoralBleaching.mlpackage"
echo "  Eval:        $SSD/models/eval_metrics.json"
echo "  Log:         $LOG"
