# MLPerf Inference v5.0 — Llama3.1-8B Runbook

Benchmark: **MLPerf Inference v5.0, Datacenter, Open Division**
Model: Meta-Llama-3.1-8B-Instruct
Scenarios: Offline (throughput) + Server (latency)
Hardware: Single node, 2× A100 80GB (tensor_parallel_size=2)
Backend: vLLM 0.6.3

## Prerequisites

- Docker with NVIDIA runtime (`--gpus` support)
- 2× A100 80GB on a single node
- `nvidia-docker2` or `nvidia-container-toolkit` installed
- ~20 GB free disk space for model + dataset
- HuggingFace account with license accepted for `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `HF_TOKEN` set in `.env`

## End-to-End Workflow

```bash
# 1. Build Docker image (~10 min — downloads vLLM + reference scripts)
make build-mlperf-inference

# 2. Download CNN/DM dataset + Llama 3.1 8B model (~20 min, ~15 GB)
make prepare-mlperf-inference-data

# 3. Run Offline + Server scenarios + accuracy (~40 min total)
make run-mlperf-inference

# 4. Check ROUGE scores meet 99% of reference targets (~5 sec)
make verify-mlperf-inference
```

## Run Individual Scenarios

```bash
# Offline only
make run-mlperf-inference-offline

# Server only
SCENARIO=server bash scripts/launch/mlperf_inference/run_mlperf_inference.sh
```

## Accuracy Targets (Open Division, BF16)

| Metric   | 99% of Reference |
|----------|-----------------|
| ROUGE-1  | ≥ 38.78         |
| ROUGE-2  | ≥ 15.91         |
| ROUGE-L  | ≥ 24.50         |
| gen_len  | ≥ 90% of 8.17M tokens |

## Output Locations

| File | Description |
|------|-------------|
| `logs/mlperf_inference_<timestamp>_offline/` | Raw LoadGen logs, performance + accuracy |
| `logs/mlperf_inference_<timestamp>_server/`  | Raw LoadGen logs, performance + accuracy |
| `results/mlperf_inference_v5.0/open/ml-infra-profiler/systems/linux_a100_vllm.json` | System description (auto-generated) |
| `results/mlperf_inference_v5.0/open/ml-infra-profiler/results/llama3.1-8b/offline/result.txt` | Offline throughput summary |
| `results/mlperf_inference_v5.0/open/ml-infra-profiler/results/llama3.1-8b/server/result.txt`  | Server latency summary |

## Configuration

Config files live in `scripts/launch/mlperf_inference/config/`:

- `llama3_8b_offline.yaml` — Offline scenario (10 min min duration, batch_size=16)
- `llama3_8b_server.yaml`  — Server scenario (2 min min duration, target_qps=0.5)

Key parameters:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `tensor_parallel_size` | 2 | Both A100s on the node |
| `dtype` | bfloat16 | Matches reference baseline |
| `max_output_tokens` | 128 | MLPerf v5.0 requirement |
| `total_sample_count` | 13368 | Full CNN/DM eval set |

## Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| `docker: unknown flag: --gpus` | `nvidia-docker2` not installed; install `nvidia-container-toolkit` |
| `CUDA out of memory` | Both GPUs required — check `NVIDIA_VISIBLE_DEVICES` is not restricting to 1 GPU |
| `model path not found` | Run `make prepare-mlperf-inference-data` first |
| `cnn_eval.json not found` | Dataset download failed; re-run `make prepare-mlperf-inference-data` |
| `HF_TOKEN` error | Set `HF_TOKEN=hf_...` in `.env`; accept model license on HuggingFace |
| ROUGE scores below target | Check accuracy log for truncated outputs; `max_output_tokens` must be 128 |
| Server scenario fails QPS | Lower `target_qps` in `llama3_8b_server.yaml` (default 0.5 req/s) |

## Design Notes

- **Docker-based**: vLLM + CUDA dependencies are complex; Docker ensures a reproducible environment
- **Reference scripts**: `main.py` and `evaluate-accuracy.py` are mounted from the cloned MLCommons repo inside the container — no reimplementing LoadGen integration
- **No W&B**: consistent with all MLPerf benchmarks in this repo
- **Separate results dir**: `results/mlperf_inference_v5.0/` follows the same gitignore exception pattern as `mlperf_tiny_v1.1/`
