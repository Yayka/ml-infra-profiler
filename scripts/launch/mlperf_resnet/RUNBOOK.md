# MLPerf ResNet-50 v1.5 Image Classification — Runbook

Runs the MLPerf Training ResNet-50 v1.5 image classification benchmark on ImageNet
(ILSVRC2012) using TensorFlow 2.x. This is a **non-transformer, CNN-based** benchmark
that measures how quickly a system can train ResNet-50 to 75.9% top-1 accuracy on
ImageNet — a classic infrastructure stress test for GPU throughput and data pipeline
efficiency.

Based on the retired MLPerf Training reference implementation:
https://github.com/mlcommons/training/tree/master/retired_benchmarks/resnet-tf2

**Cluster:** Single node, 4x GPU (A100 80GB recommended).
**Expected runtime:** ~6–12 hours depending on hardware.
**Convergence target:** Top-1 validation accuracy >= 75.9%.

---

## Prerequisites

- NVIDIA GPU node with drivers and CUDA 11.8+
- Python 3.10+
- TensorFlow 2.14 (installed in a dedicated venv)
- ImageNet ILSVRC2012 dataset in TFRecord format (~150 GB)
- `nvidia-smi` accessible in PATH

---

## End-to-End Workflow

```bash
# 1. Create dedicated Python venv with TF 2.14 (separate from main .venv)
make setup-mlperf-resnet           # ~2 min

# 2. Download ImageNet ILSVRC2012 and convert to TFRecord (manual — see below)
make prepare-mlperf-resnet-data    # prints instructions

# 3. Run training benchmark (4 GPUs)
make run-mlperf-resnet             # ~6–12 hours

# 4. Verify convergence (top-1 accuracy >= 75.9%)
make verify-mlperf-resnet          # ~5 sec
```

---

## Data Preparation

ImageNet ILSVRC2012 requires manual download (academic license). The data must be
converted to TFRecord format for efficient TF2 data loading.

1. Register at https://image-net.org and download:
   - `ILSVRC2012_img_train.tar` (~138 GB)
   - `ILSVRC2012_img_val.tar` (~6.3 GB)

2. Extract and convert to TFRecords:
```bash
# Extract
mkdir -p data/imagenet/raw
tar -xf ILSVRC2012_img_train.tar -C data/imagenet/raw/train/
tar -xf ILSVRC2012_img_val.tar -C data/imagenet/raw/val/

# Convert to TFRecord (uses TF's imagenet_to_gcs.py or similar)
make prepare-mlperf-resnet-data
```

Expected layout after conversion:
```
data/imagenet/tfrecord/
  train-00000-of-01024
  train-00001-of-01024
  ...
  validation-00000-of-00128
  validation-00001-of-00128
  ...
```

---

## Output

### Results directory (committed to repo)
```
results/mlperf_resnet/
  result.txt            <- final top-1/top-5 accuracy, PASS/FAIL
  training_log.jsonl    <- per-epoch metrics (JSON-lines)
```

### Raw run logs (gitignored — lives in logs/)
```
logs/mlperf_resnet_YYYYMMDD_HHMMSS/
  run.log               <- full stdout/stderr
  result.txt            <- copy before moving to results/
  training_log.jsonl
```

---

## Configuration

`scripts/launch/mlperf_resnet/config/resnet_4gpu.yaml` controls all benchmark parameters:

| Key | Default | Description |
|---|---|---|
| `model` | `resnet50_v1.5` | Model architecture |
| `num_classes` | `1000` | ImageNet classes |
| `data_dir` | `data/imagenet/tfrecord` | Path to TFRecord files |
| `batch_size_per_gpu` | `128` | Per-GPU batch size |
| `num_gpus` | `4` | Number of GPUs |
| `epochs` | `90` | Training epochs |
| `base_lr` | `0.1` | Base learning rate (scaled by total batch size / 256) |
| `warmup_epochs` | `5` | LR warmup epochs |
| `weight_decay` | `0.0001` | L2 regularization |
| `momentum` | `0.9` | SGD momentum |
| `label_smoothing` | `0.1` | Label smoothing factor |
| `target_top1` | `75.9` | Convergence target (%) |
| `wandb_project` | `mlperf-resnet` | W&B project name |
| `wandb_enabled` | `false` | Enable W&B logging |

---

## Experiment Matrix

| Config | Hardware | GPUs | Expected Time | Target Top-1 | W&B Run |
|---|---|---|---|---|---|
| resnet_4gpu.yaml | Azure A100 80GB | 4 | ~8h | >= 75.9% | TBD |

---

## Design Notes

- **Non-transformer benchmark** — ResNet-50 v1.5 is a pure CNN. This complements the
  Llama3.1 and MoE transformer benchmarks in this repo, providing a compute-bound
  (vs. memory-bound) workload for infrastructure profiling.
- **Separate `.venv-resnet`** — TF 2.14 conflicts with torch in the main `.venv`.
- **No W&B by default** — consistent with the other MLPerf benchmarks. Set `wandb_enabled: true`
  in the config to enable. Submission files are the primary source of truth.
- **Retired MLPerf benchmark** — ResNet-50 was retired from the MLPerf Training suite but
  remains a valid and widely-used reference for GPU training throughput measurement.
- **GPU/network metrics** — covered separately by the ml-netprof agent -> Prometheus -> Grafana
  (see `infra/prometheus/`).

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `tensorflow` import error | Wrong venv — run `source .venv-resnet/bin/activate` |
| `data/imagenet/tfrecord/` missing | Run `make prepare-mlperf-resnet-data` and follow instructions |
| OOM on GPU | Reduce `batch_size_per_gpu` in config (try 64) |
| Training stuck at epoch 0 | XLA compilation on first step; wait 2-3 minutes |
| Accuracy < 75.9% after 90 epochs | Check LR schedule; ensure label smoothing is enabled |
| `result.txt` not found in verify step | Run `make run-mlperf-resnet` first |
