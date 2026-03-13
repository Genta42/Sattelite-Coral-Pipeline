# Coral Pipeline – NOAA CRW BAA Scraping & LSTM Preprocessing

Production-grade pipeline for downloading NOAA Coral Reef Watch Bleaching Alert Area (BAA) data and preparing per-cell LSTM training sequences.

## Data Source

- **ERDDAP endpoint**: `https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW`
- **Primary target**: `CRW_BAA` (Bleaching Alert Area, categories 0–5)
- **Covariates**: `CRW_DHW` (Degree Heating Weeks), `CRW_HOTSPOT`
- **Resolution**: 5 km global, daily
- **Available range**: 1985-04-01 → present

## Install

```bash
cd coral_pipeline
pip install -r requirements.txt
```

## Pipeline Steps

### 1. Discover metadata

```bash
python cli.py discover
```

Prints resolved variable names and time bounds from ERDDAP.

### 2. Fetch raw data

```bash
# Global, full resolution, 1985–2025
python cli.py fetch --start-date 1985-04-01 --end-date 2025-12-31

# Asia only, preview stride (20 km)
python cli.py fetch --start-date 2020-01-01 --end-date 2025-12-31 \
    --continents asia --stride 4

# BAA + DHW + HotSpot
python cli.py fetch --start-date 2000-01-01 --end-date 2025-12-31 \
    --variables baa dhw hotspot
```

Cached as individual CSV chunks in `data/cache/`. Re-runs skip already-downloaded chunks.

### 3. Build long table (Format 1)

```bash
python cli.py build_table
```

Merges all cached CSVs → `data/processed/long_table.csv` (and `.parquet`).

Columns: `cell_id, lat, lon, date_utc, baa_cat, dhw, hotspot`

### 4. Build LSTM sequences

```bash
# Default: lookback=60 days, horizon=7 days, JSON serialization
python cli.py build_sequences --table-path data/processed/long_table.parquet

# Custom: lookback=90, horizon=14, flat columns
python cli.py build_sequences \
    --table-path data/processed/long_table.parquet \
    --lookback 90 --horizon 14 --serialization flat
```

Outputs to `data/sequences/`:
- `sequences_all.csv` / `.parquet`
- `sequences_train.csv` / `.parquet` (earliest–2023)
- `sequences_val.csv` / `.parquet` (2024)
- `sequences_test.csv` / `.parquet` (2025)

### 5. QA reports

```bash
python cli.py qa --table-path data/processed/long_table.parquet \
    --seq-path data/sequences/sequences_all.parquet \
    --manifest-dir data/
```

Generates in `reports/`:
- `coverage.csv` – per-continent cell/date counts
- `missingness.csv` – column-level missing stats
- `class_distribution.csv` – BAA class counts
- `checksums.csv` – SHA-256 manifest
- `plot_class_dist.png`, `plot_temporal.png`, `plot_spatial.png`

### 6. Export per-continent files

```bash
python cli.py export --input-path data/sequences/sequences_all.parquet
```

Writes `data/exported/sequences_{continent}.csv` + global combined.

## Disk Usage Estimates

| Scope | Stride | Approx raw cache | Long table |
|-------|--------|------------------|------------|
| Global, 40 years | 1 (5 km) | ~500 GB+ | ~200 GB |
| Global, 40 years | 4 (20 km) | ~30 GB | ~12 GB |
| Australia, 5 years | 1 | ~2 GB | ~800 MB |
| Asia, 1 year | 4 | ~200 MB | ~80 MB |

Sequence files are typically 2–5× the long table size depending on lookback.

## Changing N (lookback) and H (horizon)

Pass `--lookback` and `--horizon` to `build_sequences`:

```bash
python cli.py build_sequences --table-path ... --lookback 30 --horizon 14
```

- **lookback** controls how many past days go into each input sequence
- **horizon** controls how far ahead the target BAA is predicted
- The sequence builder only creates a sample when all lookback days have data (no gaps) and the target day exists

## Model Performance (1985–2025 Full Run, 6 Classes)

| Metric | Value |
|--------|-------|
| Test Accuracy | 93.6% |
| Macro F1 | 0.501 |
| Weighted F1 | 0.940 |
| Best Val Loss | 0.203 (epoch 28/38) |

The 6-class model (BAA 0–5) achieves strong performance on "No Stress" (F1=0.979) and "Alert Level 2" (F1=0.746, 97.1% recall). Rare middle classes (Watch/Warning/Alert 1) have lower F1 due to class imbalance. Class 5 (Alert Level 3) has zero test samples in the 2024+ data, which pulls macro F1 down artificially — weighted F1 (0.940) better reflects real-world performance.

See `docs/DOCUMENTATION.md` for full per-class breakdown and `docs/DEVLOG.md` for the development journey.

## Data Leakage Prevention

1. **Strict temporal split**: train ≤ 2023-12-31, val = 2024, test = 2025. No future data ever leaks into training.
2. **Horizon gap**: input sequences end at `target_date - horizon_days`. The model never sees any data within the forecast horizon window.
3. **Per-cell sequences**: each sequence belongs to exactly one grid cell. No cross-cell information leaks.
4. **Optional spatial holdout**: hold out an entire continent's cells for geographic generalization testing.

## Extending Features

To add new variables (e.g., SST anomaly, winds):

1. **config.py**: add to `VARIABLE_GROUPS`:
   ```python
   "sst_anomaly": ["CRW_SSTANOMALY"],
   "wind_speed": ["WIND_SPEED"],  # from another ERDDAP dataset
   ```

2. **fetch**: pass `--variables baa dhw sst_anomaly` to include in download

3. **build_table**: the merge logic auto-detects all resolved variables

4. **build_sequences**: feature columns are auto-detected from the long table

For variables from a **different ERDDAP dataset**, add the dataset ID to config and extend the fetch module to query multiple datasets, then join on (lat, lon, date) during `build_table`.

## Known Limitations

- `CRW_BAA_7D_MAX` is **not available** on CoastWatch ERDDAP; pipeline falls back to `CRW_BAA` (daily max).
- BAA categories are 0–5 (No Stress through Alert Level 3); the pipeline clamps to this range.
- Full global fetch at 5 km for 40 years is very large (~14,000+ requests). Use stride and date subsetting for development.
- ERDDAP may rate-limit under heavy load; the pipeline backs off automatically but full global runs can take days.
- Dateline handling: Asia bbox extends to lon=180. ERDDAP grid ends at 179.975, so no wrap-around is needed.

## Repo Layout

```
coral_pipeline/
├── cli.py                  # CLI entry point
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py           # Constants, bounds, defaults
│   ├── fetch.py            # Async ERDDAP downloader
│   ├── build_table.py      # Raw → long table (Format 1)
│   ├── build_sequences.py  # Long table → LSTM sequences
│   ├── qa.py               # Reports, plots, checksums
│   └── export.py           # Per-continent + global export
├── data/
│   ├── raw/
│   ├── cache/              # Chunk-level download cache
│   ├── processed/          # Long table
│   ├── sequences/          # LSTM-ready files
│   └── exported/           # Per-continent final outputs
└── reports/                # QA reports and plots
```
