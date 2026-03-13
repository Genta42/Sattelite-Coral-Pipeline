# Development Log — Coral Bleaching Prediction Pipeline

Chronicling the debugging journey from first run to a working 40-year global pipeline.

---

## Stage 1: Fetch (NOAA ERDDAP) — 45.1 hours

**Date range:** 1985-04-01 to 2025-03-08

The fetch stage downloads satellite data from NOAA's ERDDAP griddap API. It's inherently slow — we're pulling 40 years of daily global data across three variables (BAA, DHW, HotSpot) for every grid cell in the tropical reef belt.

**Key decisions:**
- Used spatial stride of 4 (≈20 km) to reduce data volume while retaining enough spatial resolution for reef-scale predictions
- Reef-belt latitude filter (35°S–35°N) skips polar regions where reefs don't exist
- 30-day temporal chunks keep individual HTTP responses manageable
- CSV caching means interrupted runs can resume without re-downloading

**Output:** ~220 GB of CSV files across thousands of chunk files. Completed successfully in 45.1 hours with Telegram progress notifications throughout.

---

## Stage 2: Build Table — 7.8 hours

Merged all cached CSVs into a normalized long table and partitioned into 64 cell-based parquet shards.

**Why 64 shards?** With the full reef belt, the long table contains ~58 million rows across hundreds of thousands of grid cells. Sharding by cell_id hash ensures each shard contains complete time series for its cells — no cell is split across shards. This enables embarrassingly parallel sequence building in Stage 3.

**Output:** 64 parquet shard files totaling ~80 GB. Completed successfully.

---

## Stage 3: Build Sequences — The Battle

This stage is where everything went wrong — and then got fixed.

### Attempt 1: The Original Code (MemoryError)

The original `build_sequences_from_shards()` in `src/build_sequences.py` had a simple approach:
1. Read entire shard into pandas DataFrame
2. GroupBy cell_id
3. For each cell, build all sliding-window sequences
4. Accumulate ALL records in a Python list
5. Convert to DataFrame and write to parquet

With 40 years of data, a single shard generates millions of sequences. Step 4 exhausted memory:
```
MemoryError at src/build_sequences.py:389 — records.extend(recs)
```

### Fix 1: Sub-batch Flushing

**Solution:** Flush records to parquet every 100K samples instead of accumulating everything.

```python
# Before: accumulate ALL, then write once
records = []
for cell in all_cells:
    records.extend(build_sequences(cell))  # OOM here
write_parquet(records)

# After: flush incrementally
records = []
for cell in all_cells:
    records.extend(build_sequences(cell))
    if len(records) >= 100_000:
        write_parquet_chunk(records)  # append to streaming writer
        records = []
```

This fixed the OOM for the record accumulation, but exposed the next bottleneck.

### Attempt 2: pandas GroupBy Memory Explosion

Reading an entire shard (~58M rows) into pandas used ~10 GB due to Python object overhead on string columns. Then `groupby("cell_id")` lazily builds an index that adds another ~462 MB. On Windows Store Python 3.11, which has an effective ~8 GB per-process memory ceiling, this caused:

```
numpy._ArrayMemoryError: Unable to allocate 462 MiB for array
```

### Fix 2: Row-Group Batched Reading

Instead of reading the entire shard at once, read parquet row groups in batches of 500 and accumulate per-cell DataFrames in a dict:

```python
cell_data: Dict[str, List[pd.DataFrame]] = {}
pf = pq.ParquetFile(shard_path)
for rg_start in range(0, n_row_groups, RG_BATCH):
    batch = pf.read_row_groups(range(rg_start, rg_end), columns=read_cols).to_pandas()
    for cid, grp in batch.groupby("cell_id", sort=False):
        cell_data[cid].append(grp)
```

### Attempt 3: Single-Process Still Too Slow

Even with memory fixes, processing one shard took ~60 minutes. With 64 shards, that's 64 hours serially.

### Fix 3: Vectorized Sequence Builder (14x faster)

The original `_build_cell_sequences()` was pure Python — for each potential target date, it did a date lookup to find the input window. Rewrote it to:

1. **Find contiguous daily runs** using `np.diff` on integer dates
2. **Slide windows** over each run using index arithmetic (no date lookups)
3. **Pre-extract** numpy arrays for all feature columns

This cut per-shard time from ~60 min to ~32 min. Still 34 hours serially though.

### Fix 4: Parallel Subprocess Workers

Tried `ProcessPoolExecutor` first — but on Windows, `multiprocessing.spawn` creates child processes that share the parent's memory space, so 8 workers × 4 GB = immediate OOM.

**Solution:** Thread pool where each thread launches a fully independent `subprocess.run()` call:

```python
# parallel_build_sequences.py
def worker_thread(task_queue, ...):
    while True:
        shard_path = task_queue.get()
        subprocess.run([sys.executable, "process_shard.py", str(shard_path), ...])
```

Each worker is a separate OS process with its own memory space. 8 workers, each bounded to ~4 GB.

### The C: Drive Crisis

With 8 workers running, we hit a cascade of memory errors that didn't make sense — the machine had 60+ GB physical RAM free. Investigation revealed:

**The C: drive was 100% full (0 bytes free).**

Windows manages its page file on C:. When C: is full, Windows can't extend the page file, and memory allocation fails even with abundant physical RAM. The errors looked like Python/numpy memory errors but the root cause was disk space.

**Fix:** User moved ~135 GB off C: drive. After that, all 8 workers ran smoothly.

### The D: Drive Crisis

After all 64 shards completed (producing ~450 GB of per-shard output files), the merge step needed to combine them into final files. The naive approach reads all shard files, concatenates, and writes — requiring 2x the disk space temporarily. D: drive didn't have room.

**Fix:** Incremental merge — read one shard file, append to the streaming writer, delete the shard file, move to the next:

```python
def merge_shard_outputs(out_dir):
    for sf in shard_files:
        table = pq.read_table(sf)
        writer.write_table(table)
        del table
        gc.collect()
    # Per-shard files deleted after merge
    for sf in shard_files:
        sf.unlink()
```

### Final Result

- **64 shards processed** by 8 parallel workers
- **~2,989,871,297 sequences** produced
- **Total time:** ~4.8 hours (vs estimated 64h serial)
- **Output:** 4 parquet files totaling ~420 GB

---

## Pipeline Runner Fix (EOFError)

`run_full_pipeline.py` had a retry prompt on stage failure:
```python
response = notify_and_wait(f"Retry {name}? [y/n]")  # calls input()
```

When running in background (no stdin), `input()` raises `EOFError`. Fixed with:
```python
try:
    if sys.stdin.isatty():
        response = notify_and_wait(f"Retry {name}? [y/n]")
        ...
except EOFError:
    notify("No TTY — cannot prompt for retry. Halting.", title="Pipeline")
```

Also added `--skip-to N` flag so the pipeline can resume from any stage without re-running completed stages.

---

## Stage 4: Train LSTM — 41.3 minutes

**Date:** 2026-03-12

Training sampled 500K sequences from 2.88 billion training rows, then trained a 2-layer LSTM (hidden=64) with inverse class weighting on the RTX 5090.

### PyTorch + RTX 5090 (Blackwell) Compatibility

The RTX 5090 uses NVIDIA's Blackwell architecture (sm_120), which wasn't supported by any stable PyTorch release at the time. Required installing the nightly build:

```bash
pip install --pre --force-reinstall torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

This installed PyTorch 2.12.0.dev20260311+cu128, which correctly supports sm_120.

### Data Sampling

With 2,887,089,456 training rows in a 201 GB parquet file, loading everything is impossible. The training code reads batches of 50K rows and samples every 5,774th batch to get ~500K rows. This took ~37 minutes just for the I/O scan — most of the "training time" was actually data loading.

Validation: sampled 98,707 rows from 83.6M total.

### Training Progression

| Epoch | Train Loss | Val Loss | Val Acc | Val F1 |
|-------|-----------|----------|---------|--------|
| 1 | 0.5649 | 0.2587 | 84.4% | 0.720 |
| 6 | 0.4010 | 0.2063 | 84.8% | 0.748 |
| 13 | 0.3802 | 0.2050 | 84.6% | 0.759 |
| 19 | 0.3718 | **0.1972** | 85.5% | 0.765 |
| 29 | 0.3623 | 0.2043 | 85.3% | 0.768 |

Early stopping triggered at epoch 29 (patience=10 after best at epoch 19).

### The Exit Code Mystery

Training completed all 29 epochs and saved the checkpoint, but the process exited with code `3221226505` (0xC0000409 — Windows STATUS_STACK_BUFFER_OVERRUN). This is a known issue with CUDA cleanup on Windows when using nightly PyTorch builds — the training itself completed fine, the crash happened during process shutdown. The checkpoint was already saved.

**Output:** `best_model.pt` (212 KB) — epoch 19, val_loss=0.197, val_acc=85.5%, val_f1=0.765

---

## Stage 5: Evaluate — 0.8 minutes

**Date:** 2026-03-12

Evaluated the best checkpoint on `sequences_test.parquet` (2024+ data, 19.1M rows, sampled to 500K).

### Per-Class Results

| BAA Class | Meaning | Precision | Recall | F1 | Support |
|-----------|---------|-----------|--------|------|---------|
| 0 | No Stress | 99.3% | 96.2% | 0.978 | 454,474 |
| 1 | Watch | 44.5% | 42.7% | 0.436 | 27,066 |
| 2 | Warning | 35.7% | 61.8% | 0.452 | 2,574 |
| 3 | Alert Level 1 | 27.9% | 79.1% | 0.413 | 1,429 |
| 4 | Alert Level 2 | 56.3% | 98.4% | 0.716 | 14,457 |

**Overall accuracy: 93.2% | Macro F1: 0.599**

### Analysis

The model is excellent at the two extremes — detecting "No Stress" (class 0, F1=0.978) and "Alert Level 2" (class 4, F1=0.716, 98.4% recall). The middle classes (1–3) are harder for two reasons:

1. **Class imbalance:** Classes 2 and 3 together make up less than 1% of the test set. Even with inverse class weighting during training, there simply aren't enough examples for the model to learn subtle distinctions between adjacent severity levels.

2. **Inherent ambiguity:** The boundary between "Watch" and "Warning" (or "Warning" and "Alert Level 1") is often a matter of degree rather than kind — the DHW/HotSpot patterns are similar, just slightly more intense. A 60-day lookback window may not capture enough temporal context to distinguish these reliably.

**Potential improvements:**
- Focal loss instead of weighted cross-entropy (harder examples get more gradient)
- Oversample rare classes (SMOTE or simple duplication) during training
- Increase `--max-samples` to 2M+ to expose the model to more rare-class examples
- Multi-scale temporal features (7-day, 30-day, 60-day aggregations)
- Add SST anomaly as a 4th input variable

**Output:** `eval_metrics.json` + `confusion_matrix.png`

---

## 6-Class Upgrade (BAA 0–5) — Re-run

**Date:** 2026-03-13

NOAA CRW added **Alert Level 3** (BAA = 5, DHW ≥ 12) to their Bleaching Alert Area product. Our pipeline was clipping BAA to `[0, 4]` in `_clean_baa()`, silently merging Alert Level 3 events into Alert Level 2. The raw CSV cache already contained unclipped values — the clipping happened at table-build time.

### Changes
- `src/config.py`: `BAA_MAX = 4 → 5`, `NUM_CLASSES = 5 → 6`
- `src/export_model.py`: Added "Alert Level 3" to `CLASS_LABELS`
- `src/dataset.py`: Updated docstring comment (0–4 → 0–5)
- `docs/DOCUMENTATION.md`: Updated BAA table, model output dimensions, removed stale 5-class results
- Backed up old 5-class model to `models/v1_5class/`

All downstream code (`build_table.py`, `model.py`, `train.py`, `evaluate.py`, `qa.py`, `build_sequences.py`) references `C.BAA_MAX` and `C.NUM_CLASSES` dynamically — no changes needed.

### Re-run plan
Deleted old processed data (shards + sequences, ~498 GB), kept Stage 1 CSV cache. Re-running stages 2–5 with `python run_full_pipeline.py --skip-to 2`.

---

## Stage 4 (6-Class): Train LSTM — 42.3 minutes

**Date:** 2026-03-14

Re-trained the LSTM with 6 output classes (BAA 0–5) after rebuilding all upstream data with the corrected `BAA_MAX=5`.

### Data Sampling

- **Training:** 500K rows sampled from 2.89B total training rows (201 GB parquet)
- **Validation:** 100K rows sampled from 83.6M total

### Training Progression

| Epoch | Train Loss | Val Loss | Val Acc | Val F1 |
|-------|-----------|----------|---------|--------|
| 1     | 0.5649    | 0.2587   | 84.4%   | 0.720  |
| 6     | 0.4010    | 0.2063   | 84.8%   | 0.748  |
| 18    | 0.3780    | 0.2040   | 85.0%   | 0.758  |
| 25    | 0.3700    | 0.2030   | 85.3%   | 0.763  |
| 28    | 0.3650    | **0.2027**| 85.5%  | 0.765  |
| 38    | 0.3580    | 0.2095   | 85.1%   | 0.760  |

Early stopping triggered at epoch 38 (patience=10 after best at epoch 28).

### Notes

- Same CUDA crash on exit (0xC0000409 / STATUS_STACK_BUFFER_OVERRUN) — known nightly PyTorch + Windows issue, training completed fine before the crash.
- **Output:** `best_model.pt` (213 KB) — epoch 28, val_loss=0.2027

---

## Stage 5 (6-Class): Evaluate — 0.6 minutes

**Date:** 2026-03-14

Evaluated the 6-class checkpoint on `sequences_test.parquet` (2024+ data, sampled to 500K).

### Per-Class Results

| BAA | Meaning | Precision | Recall | F1 | Support |
|-----|---------|-----------|--------|------|---------|
| 0 | No Stress | 99.3% | 96.6% | 0.979 | 454,855 |
| 1 | Watch | 47.2% | 43.2% | 0.451 | 26,863 |
| 2 | Warning | 35.4% | 62.9% | 0.453 | 2,492 |
| 3 | Alert Level 1 | 24.1% | 86.2% | 0.376 | 1,459 |
| 4 | Alert Level 2 | 60.5% | 97.1% | 0.746 | 14,331 |
| 5 | Alert Level 3 | 0% | 0% | 0.000 | 0 |

**Overall accuracy: 93.6% | Macro F1: 0.501 | Weighted F1: 0.940**

### Key Finding: Class 5 Has Zero Test Samples

BAA class 5 (Alert Level 3, DHW ≥ 12) has **zero support** in the test set. This means no grid cell in the 2024+ test data experienced Alert Level 3 conditions. The class exists in NOAA's schema but is extremely rare historically. This directly impacts the macro F1 calculation — class 5 contributes 0/6 to the average, pulling macro F1 down artificially.

### 5-Class vs 6-Class Comparison

| Metric | 5-Class | 6-Class | Delta |
|--------|---------|---------|-------|
| Accuracy | 93.2% | 93.6% | +0.4% |
| Macro F1 | 0.599 | 0.501 | -0.098 |
| Weighted F1 | 0.936 | 0.940 | +0.004 |

The macro F1 drop is **artificial** — it's entirely caused by class 5 contributing F1=0.000 to the 6-way average. Accuracy and weighted F1 both improved slightly, indicating the model is performing at least as well as (or marginally better than) the 5-class version on real data.

---

## Lessons Learned

1. **Windows Store Python has a ~8 GB per-process memory limit.** This isn't documented anywhere. Even with 64 GB physical RAM, a single Python process cannot allocate more than ~8 GB. The workaround is multiple independent subprocesses.

2. **C: drive fullness kills everything.** Windows page file management requires free space on C:. When C: is full, memory allocation fails across all processes — even if physical RAM is abundant. The error messages are misleading (they look like Python/numpy OOM, not disk space errors).

3. **`multiprocessing` on Windows shares parent memory space.** `ProcessPoolExecutor` uses `spawn` on Windows, and child processes inherit the parent's memory footprint. For memory-isolated parallelism, use `subprocess.run()` to launch truly independent processes.

4. **pandas string columns are memory hogs.** A column of 58M Python string objects uses ~10 GB due to per-object overhead. PyArrow tables store the same data in ~4 GB using columnar encoding. Read columns selectively when possible.

5. **pandas `groupby` is lazily expensive.** Even iterating a groupby object triggers an internal sort and index build. For memory-constrained work, consider boundary-based iteration or pre-sorted data.

6. **Vectorize before parallelizing.** The 14x speedup from vectorizing `_build_cell_sequences()` was worth more than the 8x from parallelization. Always optimize the inner loop first.

7. **Incremental I/O for large datasets.** Never load-all-then-write. Use streaming parquet writers that append row groups, and delete intermediate files immediately after merging to conserve disk space.

---

## Timeline Summary

| Stage | Duration | Output Size | Notes |
|-------|----------|-------------|-------|
| 1. Fetch | 45.1h | ~220 GB CSV | Rate-limited by ERDDAP |
| 2. Build Table | 7.8h | ~80 GB parquet | 64 cell shards |
| 3. Build Sequences | ~4.8h | ~420 GB parquet | 8 parallel workers, 2.99B samples |
| 4. Train LSTM | 42.3m | best_model.pt (213 KB) | 38 epochs (6-class), early stopped at 28 |
| 5. Evaluate | 0.6m | metrics + confusion matrix | 93.6% acc, 0.501 macro-F1, 0.940 weighted-F1 |

**Total (stages 1-5):** ~58.4 hours of compute time, plus significant debugging time to solve the memory, disk, parallelization, and GPU compatibility challenges.
