#
# run_pipeline.ps1 — Full global coral bleaching training pipeline (Windows)
#
# Resumable: each step writes a completion marker; re-run skips done steps
# Internet-resilient: fetch retries on network loss
# Memory-safe: uses streaming/sharded modes
#
# Usage:
#   .\run_pipeline.ps1              # run full pipeline
#   .\run_pipeline.ps1 -Reset       # delete markers and re-run everything
#
param(
    [switch]$Reset
)

$ErrorActionPreference = "Stop"

# ─── Configuration ────────────────────────────────────────────────────────
# UPDATE THIS to your data drive path
$SSD = "D:\Coral-Satellite"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CLI = Join-Path $ScriptDir "cli.py"
$LOG = Join-Path $SSD "pipeline.log"
$MARKERS = Join-Path $SSD ".markers"
$MinFreeGB = 20
$NShards = 32

# ─── Helpers ──────────────────────────────────────────────────────────────

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LOG -Value $line
}

function Check-DiskSpace {
    $drive = (Get-Item $SSD).PSDrive.Name
    $freeGB = [math]::Floor((Get-PSDrive $drive).Free / 1GB)
    if ($freeGB -lt $MinFreeGB) {
        Log "ERROR: Only $freeGB GB free (minimum: $MinFreeGB GB). Aborting."
        exit 1
    }
    Log "Disk: $freeGB GB free"
}

function Step-Done($name) {
    return Test-Path (Join-Path $MARKERS "$name.done")
}

function Mark-Done($name) {
    New-Item -ItemType File -Path (Join-Path $MARKERS "$name.done") -Force | Out-Null
    Log "Step '$name' marked complete."
}

function Wait-ForInternet {
    $wait = 10
    while ($true) {
        try {
            $null = Invoke-WebRequest -Uri "https://coastwatch.pfeg.noaa.gov" -TimeoutSec 5 -UseBasicParsing
            return
        } catch {
            Log "No internet. Retrying in ${wait}s..."
            Start-Sleep -Seconds $wait
            $wait = [math]::Min($wait * 2, 300)
        }
    }
}

function Disk-Usage {
    Log "--- Disk usage ---"
    foreach ($sub in @("data\cache", "data\processed", "data\sequences", "models")) {
        $p = Join-Path $SSD $sub
        if (Test-Path $p) {
            $size = (Get-ChildItem -Path $p -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
            $sizeGB = [math]::Round($size / 1GB, 2)
            Log "  $sub : $sizeGB GB"
        }
    }
    Log "--- end ---"
}

# ─── Parse args ───────────────────────────────────────────────────────────
if ($Reset) {
    Remove-Item -Path $MARKERS -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Markers reset. Will re-run all steps."
}

# ─── Pre-flight ───────────────────────────────────────────────────────────
if (-not (Test-Path $SSD)) {
    Write-Host "ERROR: Data directory not found at $SSD"
    Write-Host "Update `$SSD in this script to point to your data drive."
    exit 1
}

foreach ($sub in @("data\cache", "data\processed", "data\processed\shards", "data\sequences", "models")) {
    New-Item -ItemType Directory -Path (Join-Path $SSD $sub) -Force | Out-Null
}
New-Item -ItemType Directory -Path $MARKERS -Force | Out-Null

Log "=========================================="
Log "Pipeline starting (resumable)"
Log "=========================================="

# ─── Step 1: Fetch global data ────────────────────────────────────────────
if (-not (Step-Done "01_fetch")) {
    Log "STEP 1/6: Fetching global ERDDAP data..."
    Check-DiskSpace
    Wait-ForInternet

    python $CLI fetch `
        --start-date 1985-04-01 `
        --end-date   2025-12-31 `
        --continents world `
        --variables  baa dhw hotspot `
        --stride     1 `
        --cache-dir  "$SSD\data\cache" `
        2>&1 | Tee-Object -Append -FilePath $LOG

    if ($LASTEXITCODE -ne 0) { Log "STEP 1 FAILED"; exit 1 }
    Mark-Done "01_fetch"
    Disk-Usage
} else {
    Log "STEP 1/6: Fetch already complete, skipping."
}

# ─── Step 2: Build long table + shards ────────────────────────────────────
if (-not (Step-Done "02_build_table")) {
    Log "STEP 2/6: Building long table (streaming) + cell shards..."
    Check-DiskSpace

    python $CLI build_table `
        --cache-dir  "$SSD\data\cache" `
        --out-path   "$SSD\data\processed\long_table" `
        --n-shards   $NShards `
        2>&1 | Tee-Object -Append -FilePath $LOG

    if ($LASTEXITCODE -ne 0) { Log "STEP 2 FAILED"; exit 1 }
    Mark-Done "02_build_table"
    Disk-Usage
} else {
    Log "STEP 2/6: Build table already complete, skipping."
}

# ─── Step 3: Build sequences (from shards) ───────────────────────────────
if (-not (Step-Done "03_build_sequences")) {
    Log "STEP 3/6: Building LSTM-ready sequences (streaming from shards)..."
    Check-DiskSpace

    python $CLI build_sequences `
        --shard-dir  "$SSD\data\processed\shards" `
        --out-dir    "$SSD\data\sequences" `
        --lookback   60 `
        --horizon    7 `
        --serialization json `
        2>&1 | Tee-Object -Append -FilePath $LOG

    if ($LASTEXITCODE -ne 0) { Log "STEP 3 FAILED"; exit 1 }
    Mark-Done "03_build_sequences"
    Disk-Usage
} else {
    Log "STEP 3/6: Build sequences already complete, skipping."
}

# ─── Step 4: Train ────────────────────────────────────────────────────────
if (-not (Step-Done "04_train")) {
    Log "STEP 4/6: Training LSTM model..."
    Check-DiskSpace

    python $CLI train `
        --seq-dir     "$SSD\data\sequences" `
        --out-dir     "$SSD\models" `
        2>&1 | Tee-Object -Append -FilePath $LOG

    if ($LASTEXITCODE -ne 0) { Log "STEP 4 FAILED"; exit 1 }
    Mark-Done "04_train"
} else {
    Log "STEP 4/6: Training already complete, skipping."
}

# ─── Step 5: Evaluate ────────────────────────────────────────────────────
if (-not (Step-Done "05_evaluate")) {
    Log "STEP 5/6: Evaluating on test split..."

    python $CLI evaluate `
        --checkpoint "$SSD\models\best_model.pt" `
        --seq-dir    "$SSD\data\sequences" `
        --out-dir    "$SSD\models" `
        2>&1 | Tee-Object -Append -FilePath $LOG

    if ($LASTEXITCODE -ne 0) { Log "STEP 5 FAILED"; exit 1 }
    Mark-Done "05_evaluate"
} else {
    Log "STEP 5/6: Evaluation already complete, skipping."
}

# ─── Step 6: Export CoreML ────────────────────────────────────────────────
if (-not (Step-Done "06_export")) {
    Log "STEP 6/6: Exporting to CoreML..."

    python $CLI export_model `
        --checkpoint "$SSD\models\best_model.pt" `
        --out-path   "$SSD\models\CoralBleaching.mlpackage" `
        2>&1 | Tee-Object -Append -FilePath $LOG

    if ($LASTEXITCODE -ne 0) { Log "STEP 6 FAILED"; exit 1 }
    Mark-Done "06_export"
} else {
    Log "STEP 6/6: Export already complete, skipping."
}

# ─── Summary ─────────────────────────────────────────────────────────────
Log "=========================================="
Log "Pipeline finished successfully!"
Log "=========================================="
Disk-Usage

Write-Host ""
Write-Host "Outputs:"
Write-Host "  Long table:  $SSD\data\processed\long_table.parquet"
Write-Host "  Sequences:   $SSD\data\sequences\"
Write-Host "  Model:       $SSD\models\best_model.pt"
Write-Host "  CoreML:      $SSD\models\CoralBleaching.mlpackage"
Write-Host "  Eval:        $SSD\models\eval_metrics.json"
Write-Host "  Log:         $LOG"
