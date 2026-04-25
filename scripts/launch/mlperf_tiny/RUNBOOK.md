# MLPerf Tiny IC Benchmark — Runbook

Runs the MLPerf Tiny v1.1 Image Classification benchmark (ResNet v1 int8 on CIFAR-10) using TFLite on Azure A100 GPU nodes. Results are stored in the standard MLPerf Tiny submission directory format.

## Prerequisites

- Azure A100 GPU node with NVIDIA drivers
- Docker (optional — only needed for the reference Dockerfile)
- Python 3.10+
- `nvidia-smi` accessible in PATH

## End-to-End Workflow

```bash
# 1. Create dedicated Python venv with TF 2.14 (separate from main .venv — conflicts with torch)
make setup-mlperf-tiny           # ~2 min

# 2. Download CIFAR-10 test set (~170 MB) + ResNet TFLite model (~1 MB)
make prepare-mlperf-tiny-data    # ~5 min

# 3. Run inference benchmark (performance + accuracy phases)
make run-mlperf-tiny             # ~10 min (GPU)

# 4. Verify accuracy meets Open Division target (>= 85%)
make verify-mlperf-tiny          # ~5 sec
```

## CPU Mode (local dev, no GPU required)

```bash
make run-mlperf-tiny-cpu
```

This uses `DELEGATE=cpu` and runs inference on the CPU. Latency numbers will be
different from A100 but accuracy results are identical.

## Output

### Submission tree (committed to repo)
```
results/mlperf_tiny_v1.1/open/ml-infra-profiler/
  systems/linux_a100_tflite.json   ← auto-generated hardware description
  results/ic/
    performance/
      result.txt    ← p90 latency, VALID/INVALID
      log.txt       ← per-sample latencies (JSON-lines)
    accuracy/
      result.txt    ← top-1 %, VALID/INVALID
      log.txt       ← per-sample predictions (JSON-lines)
```

### Raw run logs (gitignored — lives in logs/)
```
logs/mlperf_tiny_YYYYMMDD_HHMMSS/
  run.log                 ← full stdout/stderr
  result_performance.txt  ← copy before moving to submission tree
  result_accuracy.txt
  log_performance.txt
  log_accuracy.txt
```

## Configuration

`scripts/launch/mlperf_tiny/config/ic_resnet_gpu.yaml` controls all benchmark parameters:

| Key | Default | Description |
|---|---|---|
| `benchmark` | `ic` | Benchmark type |
| `model_path` | `data/mlperf_tiny/models/resnet_v1_int8.tflite` | TFLite model file |
| `data_path` | `data/mlperf_tiny/ic/cifar10_test.npz` | CIFAR-10 test npz |
| `delegate` | `gpu` | `gpu` or `cpu` |
| `num_threads` | `1` | CPU threads (ignored for GPU delegate) |
| `performance_samples` | `1024` | Samples for latency measurement |
| `accuracy_target` | `85.0` | Minimum top-1 % (Open Division) |
| `scenario` | `single_stream` | MLPerf scenario |
| `system_id` | `linux_a100_tflite` | System identifier in submission JSON |

## Experiment Matrix

| Config | Hardware | Delegate | p90 Latency | Top-1 Acc | Submission |
|---|---|---|---|---|---|
| ic_resnet_gpu.yaml | Azure A100 80GB | TFLite GPU | TBD | TBD | results/mlperf_tiny_v1.1/ |

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `tensorflow` import error | Wrong venv — run `source .venv-tiny/bin/activate` |
| GPU delegate fails, falls back to CPU | TFLite GPU delegate `.so` not found; benchmark proceeds on CPU |
| `data/mlperf_tiny/models/` missing | Run `make prepare-mlperf-tiny-data` |
| Accuracy < 85% | Model file corrupted — re-run `prepare-mlperf-tiny-data` |
| `result.txt` not found in verify step | Run `make run-mlperf-tiny` first |

## Design Notes

- **No W&B** — consistent with the existing MLPerf training benchmark. Submission files are the source of truth.
- **Separate `.venv-tiny`** — TF 2.14 conflicts with torch in the main `.venv`.
- **Batch=1 enforced** — MLPerf Tiny Single Stream scenario mandates it; config `performance_samples` controls how many Single Stream queries are issued.
- **Open Division** — no reference model constraints; matches the spirit of an infra profiling repo.
- **GPU/network metrics** — covered separately by the ml-netprof agent → Prometheus → Grafana (see `infra/prometheus/`).
