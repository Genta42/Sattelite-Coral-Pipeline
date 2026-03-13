# Coral Bleaching Prediction Pipeline

**NOAA CRW Satellite Data → LSTM Classification Model**

A 5-stage pipeline that ingests 40 years (1985–2025) of NOAA Coral Reef Watch satellite data, constructs time-series sequences for every reef-belt grid cell on Earth, and trains an LSTM classifier to predict Bleaching Alert Area (BAA) categories 7 days ahead.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Stage 1: Fetch](#stage-1-fetch)
3. [Stage 2: Build Table](#stage-2-build-table)
4. [Stage 3: Build Sequences](#stage-3-build-sequences)
5. [Stage 4: Train LSTM](#stage-4-train-lstm)
6. [Stage 5: Evaluate](#stage-5-evaluate)
7. [Configuration Reference](#configuration-reference)
8. [Running the Pipeline](#running-the-pipeline)
9. [File Layout](#file-layout)

---

## Architecture Overview

```
NOAA ERDDAP API                    D:/genta coral/coral_pipeline/
     │                                  │
     ▼                                  ▼
┌──────────┐   ┌──────────────┐   ┌───────────────┐   ┌───────────┐   ┌──────────┐
│  Stage 1  │──▶│   Stage 2    │──▶│    Stage 3    │──▶│  Stage 4  │──▶│ Stage 5  │
│   Fetch   │   │ Build Table  │   │Build Sequences│   │   Train   │   │ Evaluate │
│           │   │              │   │  (parallel)   │   │   LSTM    │   │          │
└──────────┘   └──────────────┘   └───────────────┘   └───────────┘   └──────────┘
  CSV cache       64 shards         ~3B sequences      best_model.pt   metrics +
  (~220 GB)     (parquet, ~80GB)   (parquet, ~420GB)                   confusion
```

**Data flow:** Raw satellite CSVs → cell-sharded parquet → sliding-window sequences → PyTorch LSTM → evaluation metrics.

**Key numbers (1985–2025 run):**
- **Time span:** 40 years, daily resolution
- **Spatial coverage:** Global reef belt (35°S–35°N), ~0.05° grid (5 km)
- **Variables:** BAA (7-day max), Degree Heating Weeks, HotSpot
- **Sequences produced:** ~2.99 billion samples
- **Train/Val/Test split:** Through 2022 / 2023 / 2024+

---

## Stage 1: Fetch

**Source:** `src/fetch.py` | **CLI:** `python cli.py fetch`

Downloads raw satellite observations from NOAA CoastWatch ERDDAP griddap service.

### What it does
- Queries `https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW` for variables: `CRW_BAA_7D_MAX`, `CRW_DHW`, `CRW_HOTSPOT`
- Clips latitude to reef-belt bounds (35°S to 35°N) to skip polar regions
- Splits requests by continent bounding boxes and 30-day temporal chunks
- Uses spatial stride of 4 (≈20 km resolution) to manage data volume
- Async HTTP with retry logic: exponential backoff, 429 rate-limit handling, 3 concurrent requests max
- Caches each chunk as a CSV — already-downloaded chunks are skipped on retry

### Parameters
| Flag | Default | Description |
|------|---------|-------------|
| `--start-date` | required | Start date (YYYY-MM-DD) |
| `--end-date` | required | End date (YYYY-MM-DD) |
| `--variables` | `baa dhw` | Variable groups to fetch |
| `--stride` | `1` | Spatial stride (4 = 20 km) |
| `--reef-belt` | off | Clip to 35°S–35°N |
| `--cache-dir` | `coral_pipeline/data/cache` | Where to store CSVs |

### Output
```
D:/genta coral/coral_pipeline/data/cache/
  ├── africa_baa_2000-01-01_2000-01-30.csv
  ├── asia_dhw_1990-06-01_1990-06-30.csv
  └── ... (~220 GB total, thousands of files)
```

---

## Stage 2: Build Table

**Source:** `src/build_table.py` | **CLI:** `python cli.py build_table`

Merges all cached CSVs into a single normalized long table, then partitions by cell into shards.

### What it does
1. Reads all CSVs from the cache directory
2. Standardizes column names, adds `cell_id` (lat/lon hash), converts dates
3. Assigns `baa_cat` integer label (0–5: No Stress, Watch, Warning, Alert Level 1, Alert Level 2, Alert Level 3)
4. Partitions rows by `cell_id` hash into N parquet shards (default: 64)
5. Each shard contains all time-series data for a subset of grid cells

### Parameters
| Flag | Default | Description |
|------|---------|-------------|
| `--cache-dir` | `coral_pipeline/data/cache` | Input CSV directory |
| `--out-path` | `coral_pipeline/data/processed/long_table` | Output directory |
| `--n-shards` | `0` | Number of cell shards (0 = single file) |

### Output
```
D:/genta coral/coral_pipeline/data/processed/shards/
  ├── shard_0000.parquet
  ├── shard_0001.parquet
  └── ... (64 files, ~80 GB total)
```

Each shard schema: `cell_id | date_utc | lat | lon | baa_cat | dhw | hotspot`

---

## Stage 3: Build Sequences

**Source:** `parallel_build_sequences.py`, `process_shard.py`, `src/build_sequences.py`

Transforms the long table shards into LSTM-ready sliding-window sequences. This is the most compute-intensive stage.

### Architecture: Parallel Subprocess Workers

```
parallel_build_sequences.py (orchestrator)
  ├── Thread 1 → subprocess: python process_shard.py shard_0000.parquet
  ├── Thread 2 → subprocess: python process_shard.py shard_0001.parquet
  ├── ...
  └── Thread 8 → subprocess: python process_shard.py shard_XXXX.parquet
      ↓
  merge_shard_outputs() → sequences_{all,train,val,test}.parquet
```

Each worker runs as an **independent OS process** (via `subprocess.run`), giving it its own memory space. This is critical on Windows where Python's per-process memory ceiling is ~8 GB.

### Sequence Construction Algorithm

For each grid cell's time series:
1. **Sort** by date, find contiguous daily runs (no gaps)
2. For each run of length ≥ `lookback + horizon`:
   - Slide a window: input = `[t - lookback - horizon + 1 ... t - horizon]`, target = BAA at day `t`
   - Skip samples where target BAA is missing (-1)
3. **Serialize** feature sequences as JSON arrays
4. **Flush** to parquet every 100K records to bound memory

### Temporal Splits
| Split | Date Range | Purpose |
|-------|-----------|---------|
| Train | ≤ 2022-12-31 | Model training |
| Val | 2023-01-01 – 2023-12-31 | Hyperparameter tuning |
| Test | ≥ 2024-01-01 | Final evaluation |

### Parameters
| Flag | Default | Description |
|------|---------|-------------|
| `--workers` | `8` | Number of parallel subprocess workers |
| `--lookback` | `60` | Input sequence length (days) |
| `--horizon` | `7` | Prediction horizon (days ahead) |

### Output
```
D:/genta coral/coral_pipeline/data/sequences/
  ├── sequences_all.parquet    (209 GB, ~2.99B rows)
  ├── sequences_train.parquet  (201 GB)
  ├── sequences_val.parquet    (5.9 GB)
  └── sequences_test.parquet   (1.6 GB)
```

Sequence schema: `cell_id | lat | lon | target_date | horizon_days | y_baa_cat | x_baa_cat_seq | x_dhw_seq | x_hotspot_seq`

---

## Stage 4: Train LSTM

**Source:** `src/train.py`, `src/model.py`, `src/dataset.py` | **CLI:** `python cli.py train`

Trains a 2-layer LSTM classifier on the sequence data.

### Model Architecture — `CoralLSTM`

```
Input:
  x_seq:    (B, 60, 3)  — 60-day sequences of [baa_cat, dhw, hotspot]
  x_static: (B, 2)      — [lat, lon]

Architecture:
  x_seq → LSTM(3→64, 2 layers, dropout=0.3) → h_last (B, 64)
  cat(h_last, x_static) → (B, 66)
  → Linear(66→32) → ReLU → Dropout(0.3) → Linear(32→6) → logits

Output:
  logits (B, 6)  — BAA categories 0–5
```

### Training Details
- **Loss:** CrossEntropyLoss with inverse-frequency class weights (handles class imbalance)
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-5)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)
- **Early stopping:** Patience of 10 epochs on validation loss
- **Data sampling:** When `--max-samples` is set, reads a strided subset of the parquet file to limit memory

### Parameters
| Flag | Default | Description |
|------|---------|-------------|
| `--seq-dir` | `coral_pipeline/data/sequences` | Sequence parquet directory |
| `--out-dir` | `coral_pipeline/models` | Where to save checkpoint |
| `--hidden-size` | `64` | LSTM hidden dimension |
| `--num-layers` | `2` | LSTM layers |
| `--dropout` | `0.3` | Dropout rate |
| `--batch-size` | `256` | Training batch size |
| `--lr` | `1e-3` | Learning rate |
| `--max-epochs` | `100` | Maximum training epochs |
| `--patience` | `10` | Early stopping patience |
| `--max-samples` | `None` | Cap training samples |

### Results

**Training run:** 2026-03-14 | 500K training samples from 2.89B rows | 100K validation samples from 83.6M rows

| Epoch | Train Loss | Val Loss | Val Acc | Val F1 |
|-------|-----------|----------|---------|--------|
| 1     | 0.5649    | 0.2587   | 84.4%   | 0.720  |
| 6     | 0.4010    | 0.2063   | 84.8%   | 0.748  |
| 18    | 0.3780    | 0.2040   | 85.0%   | 0.758  |
| 25    | 0.3700    | 0.2030   | 85.3%   | 0.763  |
| 28    | 0.3650    | **0.2027**| 85.5%  | 0.765  |
| 38    | 0.3580    | 0.2095   | 85.1%   | 0.760  |

Early stopping at epoch 38 (patience=10), best checkpoint at epoch 28 (val_loss=0.2027).

### Output
```
D:/genta coral/coral_pipeline/models/
  └── best_model.pt   — checkpoint with model weights, norm_stats, config, metrics (213 KB)
```

---

## Stage 5: Evaluate

**Source:** `src/evaluate.py` | **CLI:** `python cli.py evaluate`

Loads the best checkpoint and evaluates on the test split.

### What it does
1. Loads `best_model.pt` and reconstructs the `CoralLSTM` model
2. Reads `sequences_test.parquet` (or a sampled subset)
3. Runs inference on all test samples
4. Computes per-class precision/recall/F1, overall accuracy, macro F1
5. Generates confusion matrix visualization
6. Writes metrics to JSON

### Parameters
| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Path to `best_model.pt` |
| `--seq-dir` | `coral_pipeline/data/sequences` | Sequence directory |
| `--split` | `test` | Which split to evaluate |
| `--max-samples` | `None` | Cap evaluation samples |

### Results

**Evaluation run:** 2026-03-14 | Test split: 2024+ data, sampled to 500K rows

| BAA | Meaning | Precision | Recall | F1 | Support |
|-----|---------|-----------|--------|------|---------|
| 0 | No Stress | 99.3% | 96.6% | 0.979 | 454,855 |
| 1 | Watch | 47.2% | 43.2% | 0.451 | 26,863 |
| 2 | Warning | 35.4% | 62.9% | 0.453 | 2,492 |
| 3 | Alert Level 1 | 24.1% | 86.2% | 0.376 | 1,459 |
| 4 | Alert Level 2 | 60.5% | 97.1% | 0.746 | 14,331 |
| 5 | Alert Level 3 | 0% | 0% | 0.000 | 0 |

**Overall: 93.6% accuracy | Macro F1: 0.501 | Weighted F1: 0.940**

Class 5 (Alert Level 3) has zero test samples — no grid cell in the 2024+ test data reached this severity. This pulls macro F1 down artificially; weighted F1 (0.940) better reflects real-world performance.

### Output
```
D:/genta coral/coral_pipeline/models/
  ├── eval_metrics.json     — full classification report
  └── confusion_matrix.png  — 6×6 BAA confusion matrix
```

### BAA Categories
| Code | Meaning | Description |
|------|---------|-------------|
| 0 | No Stress | No thermal stress |
| 1 | Watch | Possible bleaching |
| 2 | Warning | Likely bleaching |
| 3 | Alert Level 1 | Bleaching expected |
| 4 | Alert Level 2 | Severe bleaching / mortality expected |
| 5 | Alert Level 3 | Unprecedented bleaching / mortality |

---

## Configuration Reference

All defaults are in `src/config.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `ERDDAP_BASE` | `coastwatch.pfeg.noaa.gov/erddap/griddap` | Data source |
| `DATASET_ID` | `NOAA_DHW` | ERDDAP dataset ID |
| `GRID_RESOLUTION` | 0.05° | ~5 km native grid |
| `REEF_BELT_LAT` | (-35, 35) | Tropical + subtropical |
| `DEFAULT_LOOKBACK_DAYS` | 60 | Input window |
| `DEFAULT_HORIZON_DAYS` | 7 | Prediction horizon |
| `SPLIT_TRAIN_END` | 2022-12-31 | Train/val boundary |
| `SPLIT_VAL_END` | 2023-12-31 | Val/test boundary |
| `NUM_CLASSES` | 6 | BAA categories 0–5 |

---

## Running the Pipeline

### Full run (all 5 stages)
```bash
python run_full_pipeline.py
```

### Resume from a specific stage
```bash
# Skip stages 1-2 (data already fetched and tabled)
python run_full_pipeline.py --skip-to 3
```

### Run individual stages
```bash
# Stage 1
python cli.py fetch --start-date 1985-04-01 --end-date 2025-03-08 \
  --variables baa dhw hotspot --stride 4 --reef-belt \
  --cache-dir "D:/genta coral/coral_pipeline/data/cache"

# Stage 2
python cli.py build_table --cache-dir "D:/genta coral/coral_pipeline/data/cache" \
  --out-path "D:/genta coral/coral_pipeline/data/processed/long_table" --n-shards 64

# Stage 3 (parallel)
python parallel_build_sequences.py --workers 8 --lookback 60 --horizon 7

# Stage 4
python cli.py train --seq-dir "D:/genta coral/coral_pipeline/data/sequences" \
  --out-dir "D:/genta coral/coral_pipeline/models" --max-samples 500000

# Stage 5
python cli.py evaluate --checkpoint "D:/genta coral/coral_pipeline/models/best_model.pt" \
  --seq-dir "D:/genta coral/coral_pipeline/data/sequences" --split test --max-samples 500000
```

### Monitoring
- **Telegram notifications:** Automatic via `notify.py` (progress updates at each stage)
- **Rich progress bars:** Rendered in terminal when `rich` is installed and TTY is detected
- **Monitor script:** `python monitor.py` shows live write rates and memory usage

---

## File Layout

```
Sattelite-Coral-Pipeline/
├── cli.py                          # Main CLI entry point
├── run_full_pipeline.py            # 5-stage orchestrator with --skip-to
├── parallel_build_sequences.py     # Stage 3 parallel orchestrator
├── process_shard.py                # Stage 3 single-shard worker
├── monitor.py                      # Rich live progress monitor
├── notify.py                       # Telegram bot notifications
├── requirements.txt
├── docs/
│   ├── DOCUMENTATION.md            # This file
│   └── DEVLOG.md                   # Development log
└── src/
    ├── config.py                   # All constants and defaults
    ├── fetch.py                    # Stage 1: ERDDAP data fetcher
    ├── build_table.py              # Stage 2: CSV → sharded parquet
    ├── build_sequences.py          # Stage 3: sequence construction
    ├── dataset.py                  # PyTorch Dataset for sequences
    ├── model.py                    # CoralLSTM architecture
    ├── train.py                    # Stage 4: training loop
    ├── evaluate.py                 # Stage 5: evaluation + metrics
    ├── qa.py                       # QA reports and plots
    ├── export.py                   # Per-continent export
    └── export_model.py             # CoreML export
```

### Data Directory (D: drive)
```
D:/genta coral/coral_pipeline/
├── data/
│   ├── cache/                      # Stage 1 output: raw CSVs (~220 GB)
│   ├── processed/
│   │   ├── long_table/             # Stage 2 intermediate
│   │   └── shards/                 # Stage 2 output: 64 parquet shards
│   └── sequences/                  # Stage 3 output: sequence parquets
│       ├── sequences_all.parquet   # 209 GB
│       ├── sequences_train.parquet # 201 GB
│       ├── sequences_val.parquet   # 5.9 GB
│       └── sequences_test.parquet  # 1.6 GB
└── models/                         # Stage 4-5 output
    ├── best_model.pt
    ├── eval_metrics.json
    └── confusion_matrix.png
```
