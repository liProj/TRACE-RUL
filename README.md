# TRACE-RUL

**Age-Conditioned and Trajectory-Consistent Expert Ensembling for Aero-Engine Remaining Useful Life Prediction**

This repository contains the experimental implementation of TRACE-RUL, together with training and evaluation commands and archived result summaries. TRACE-RUL combines operating age, sensor degradation features, heterogeneous expert predictions, and end-of-life consistency across successive observations. Its compact student model, TRACE-S, learns from both ground-truth labels and out-of-fold teacher predictions.

## Method and implementation

| Component | Implementation | Purpose |
| --- | --- | --- |
| C-MAPSS preprocessing | `code/cmapss.py`, `code/trace_data.py` | Operating-condition normalization, age encoding, drift from the early operating state, sliding windows, and tabular features |
| N-CMAPSS preprocessing | `code/ncmapss.py` | Unit splits, subsampling, normalization, and window construction |
| Expert pool | `code/experts.py`, `code/aidgn.py` | AIDGN-A, 1D-CNN, GRU, LightGBM, and optional TabPFN v2 |
| Engine-level cross-fitting | `code/run_trace_cmapss.py`, `code/run_trace_ncmapss.py` | Out-of-fold training predictions and test predictions averaged across folds |
| Fusion and trajectory consistency | `code/trace_post.py` | Nonnegative least-squares stacking, recursive end-of-life estimation, and blending with the current prediction |
| Evaluation | `code/eval_cmapss.py`, `code/eval_ncmapss.py` | RMSE, Score, fusion weights, and trajectory parameters |
| TRACE-S | `code/distill.py` | Student training using ground-truth labels and out-of-fold teacher predictions |
| Baseline experiments | `code/train_aidgn.py`, `code/train_aidgn_n.py`, `code/age_only.py`, `code/noise_aidgn.py` | Baseline training, age-only prediction, and noise evaluation |
| Efficiency and visualization | `code/efficiency.py`, `code/collect.py`, `code/make_figs.py` | Parameter counts, inference latency, result aggregation, and experimental plots |

### Benchmark configurations

| Setting | C-MAPSS | N-CMAPSS |
| --- | --- | --- |
| Main subsets | FD001–FD004 | DS01–DS07 |
| Seeds available in the archived main results | 0–2 | 0–2 |
| Input channels | 14 sensors + 14 drift channels + 1 age channel | 20 observation channels + 1 age channel |
| Window length | 50 cycles | 50 sampled rows |
| Main expert pool | AIDGN-A, CNN, GRU, LightGBM, TabPFN v2 | AIDGN-A, CNN, GRU, LightGBM |
| Cross-fitting | 5 folds grouped by engine | Leave-one-development-unit-out (`--louo`) |
| Evaluation samples | The final window of each test engine | All windows of the test units |
| Score aggregation | Sum | Mean |
| TRACE-S parameters | **19,265** | **17,985** |

The 19,265-parameter count applies to the C-MAPSS student with 29 input channels. The N-CMAPSS student uses the same architecture with 21 input channels, which changes the size of its first convolutional layer. The archived FD003 timing experiment records approximately 0.49 ms of GPU latency for a batch of 100 engines using the C-MAPSS configuration. Timing depends on the hardware and runtime environment. The recorded values are available in [`reference_results/efficiency.json`](reference_results/efficiency.json).

## Installation

Create a separate Python environment, install PyTorch for the intended CPU or CUDA platform, and install the core dependencies:

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# On Windows, use .venv\Scripts\activate
python -m pip install torch
python -m pip install -r requirements.txt
```

The full five-expert C-MAPSS configuration also requires a TabPFN v2-compatible package and the corresponding pretrained weights:

```bash
python -m pip install -r requirements-tabpfn.txt
```

The dependency lists were reconstructed from source imports and are not a lockfile for the original training environment. TabPFN loads local v2 weights, so check API compatibility when installing a newer package version. The archived TabPFN implementation explicitly uses CUDA. A four-expert configuration can be run without TabPFN, but its results correspond to a different expert pool from the five-expert C-MAPSS main experiment.

## Data preparation

Datasets and pretrained weights are obtained separately. Place the data in the default locations below, or configure the corresponding environment variables:

```text
repo/AIDGN-RUL/CMAPSS/
  train_FD001.txt  test_FD001.txt  RUL_FD001.txt
  ... FD002 / FD003 / FD004
data/ncmapss/h5/
  N-CMAPSS_DS01*.h5
  ... DS02 / DS03 / DS04 / DS05 / DS06 / DS07
```

| Environment variable | Location |
| --- | --- |
| `TRACE_CMAPSS_DIR` | Directory containing `train_FD001.txt` and the other C-MAPSS files |
| `TRACE_NCMAPSS_DIR` | Directory containing the N-CMAPSS HDF5 files; nested directories are supported |
| `TRACE_TABPFN_CKPT` | Path to `tabpfn-v2-regressor.ckpt` |

The default TabPFN checkpoint location is `~/.cache/tabpfn/tabpfn-v2-regressor.ckpt`. N-CMAPSS preprocessing caches are written to `data/ncmapss/cache`. Regenerate the cache when changing the data or preprocessing settings.

## Running the main experiments

Run the following commands from the repository root. Training time depends on the dataset, hardware, and selected experts.

### C-MAPSS: one subset and one seed

```bash
python code/run_trace_cmapss.py FD001 0 --experts aidgna,cnn,gru,lgbm,tabpfn --tag trace
python code/eval_cmapss.py trace FD001 0
python code/distill.py cmapss FD001 0 --tag trace
```

The experiment runner covers all four subsets and three seeds, executing expert training, evaluation, and student training in sequence. Use `--dry-run` to inspect the commands before execution:

```bash
python tools/run_experiments.py cmapss --dry-run
python tools/run_experiments.py cmapss
```

### N-CMAPSS: one subset and one seed

```bash
python code/run_trace_ncmapss.py DS01 0 --experts aidgna,cnn,gru,lgbm --tag trace_louo --louo
python code/eval_ncmapss.py trace_louo DS01 0
python code/distill.py ncmapss DS01 0 --tag trace_louo
```

Run all seven subsets and three seeds with:

```bash
python tools/run_experiments.py ncmapss --dry-run
python tools/run_experiments.py ncmapss
```

The runner skips expert training and student training when their corresponding outputs already exist. Evaluation is recomputed from the saved predictions. To start a fresh run, supply a new `--tag`. Expert training saves out-of-fold and test predictions, but does not export the model weights for every fold. Student weights are saved separately as `_student.pt`.

Five-fold cross-fitting partitions engines into five groups within a run. Random seeds repeat the entire experiment. The original `q_*.sh` scripts retain queues that scheduled additional seeds, whereas the archived main results for both benchmarks contain seeds 0, 1, and 2. The portable runner uses these three seeds by default.

### Component and baseline experiments

```bash
# C-MAPSS: compare input variants using the same four-expert pool
python code/run_trace_cmapss.py FD001 0 --experts aidgna,cnn,gru,lgbm --tag trace4
python code/run_trace_cmapss.py FD001 0 --experts aidgna,cnn,gru,lgbm --tag trace_noage --no_age
python code/run_trace_cmapss.py FD001 0 --experts aidgna,cnn,gru,lgbm --tag trace_nodrift --no_drift
python code/eval_cmapss.py trace4 FD001 0
python code/eval_cmapss.py trace_noage FD001 0
python code/eval_cmapss.py trace_nodrift FD001 0

# AIDGN baseline training
python code/train_aidgn.py FD001 0
python code/train_aidgn_n.py DS01 0
```

The evaluation scripts produce raw and trajectory-processed (TT) results for individual experts, uniform averaging, and NNLS stacking. For the N-CMAPSS age ablation, use the four-expert pool with `--louo --no_age` and a separate tag. Keep the expert pool fixed when comparing input components. The student training script targets the main input configurations and does not implement age or drift ablations for the student.

## Archived results and validation

[`reference_results/`](reference_results/) contains the source project's existing JSON metrics for individual subsets, seeds, configurations, and aggregate summaries. New runs write to `results/`. The archive includes both main and exploratory configurations, so interpret each directory together with its run settings.

```bash
python tools/summarize_results.py
python tools/check_release.py
```

Release checks cover Python syntax, archived JSON parsing, command generation, source-file checksums, and numerical behavior of the trajectory recursion. Full model training was not repeated during repository preparation. The legacy `code/make_tables.py`, `make_prose.py`, and `make_report.py` scripts generate early experimental reports and do not represent the final manuscript text or table layouts.

## Attribution and licensing

The AIDGN architecture and parts of the benchmark preprocessing follow [TCCofWANG/AIDGN-RUL](https://github.com/TCCofWANG/AIDGN-RUL), at source revision `d1ac173b13659af1854cf500a9b1cba8367353fe`. The `code/aidgn_official` directory retains the baseline code present in the source project, with its original attribution. The inspected upstream snapshot did not include a root LICENSE file. This repository does not add a new license grant for third-party code. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The source-file checksum inventory is available in [`docs/source_manifest.json`](docs/source_manifest.json). Release preparation added configurable data paths and a portable experiment runner while preserving the original training, fusion, and evaluation algorithms.
