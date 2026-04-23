# MLPerf Small LLM MoE Pretraining (GPT-OSS-20B) — Runbook

Runs the MLPerf Training v5.1 Small LLM MoE Pretraining benchmark using NeMo on NVIDIA GPUs. The model is a 20B-parameter GPT with 32 Mixture-of-Experts (top-4 routing). Goal is infrastructure profiling (GPU utilization, expert parallelism communication, memory bandwidth), not hitting the MLPerf convergence target.

**Reference:** [mlcommons/training/small_llm_moe_pretraining/primus](https://github.com/mlcommons/training/tree/master/small_llm_moe_pretraining/primus)

**Cluster:** 1 node x 4 A100 80GB GPUs (expert parallelism over NVLink/PCIe).
**Expected runtime:** ~3 hours (200 steps, infra profiling mode).
**Parallelism:** EP=4 (expert parallel), TP=1, PP=1, DP=1, GBS=16.

---

## Prerequisites

- NVIDIA A100 80GB GPUs (4x minimum)
- Docker installed, user in `docker` group
- NeMo container: `nvcr.io/nvidia/nemo:24.12-rc0` (see Step 1)
- C4 data at `/data/c4/` (same dataset as MLPerf Llama3.1 8B benchmark)
- `ml-netprof-agent` running as systemd service (for Prometheus scraping)

---

## End-to-End Workflow

```bash
# 1. Pull NeMo container (~25 GB, one-time)
make pull-nemo

# 2. Download/prepare C4 dataset (~80 GB, one-time)
make prepare-mlperf-moe-data

# 3. Create Python venv with NeMo deps (or use container directly)
make setup-mlperf-moe

# 4. Run MoE pretraining benchmark
make run-mlperf-moe

# 5. Verify convergence (val_loss <= 3.34)
make verify-mlperf-moe
```

---

## Model Architecture

| Parameter | Value | Notes |
|---|---|---|
| Total parameters | ~20B | Sparse MoE |
| `num_layers` | 24 | Transformer layers |
| `hidden_size` | 2880 | Hidden dimension |
| `ffn_hidden_size` | 2880 | FFN hidden dimension |
| `num_attention_heads` | 64 | Multi-head attention |
| `num_query_groups` | 8 | Grouped-query attention |
| `num_moe_experts` | 32 | Total experts per layer |
| `moe_router_topk` | 4 | Top-4 expert routing |
| `seq_length` | 8192 | Sequence length |
| `position_embedding_type` | RoPE | Rotary base=150000 |

---

## Configuration

`scripts/launch/mlperf_moe/config/moe_4gpu.yaml` controls all benchmark parameters:

| Key | Default | Description |
|---|---|---|
| `trainer.max_steps` | 200 | Steps for infra profiling; set to 1200000 for convergence |
| `model.global_batch_size` | 16 | Matches MLPerf reference |
| `model.micro_batch_size` | 2 | Per-GPU batch size |
| `model.expert_model_parallel_size` | 4 | Expert parallelism across GPUs |
| `model.num_moe_experts` | 32 | Experts per MoE layer |
| `model.moe_router_topk` | 4 | Top-k routing |
| `trainer.val_check_interval` | 50 | Validation every 50 steps |
| `convergence.target_val_loss` | 3.34 | MLPerf convergence target |

---

## Output

### Results (committed to repo)
```
results/mlperf_moe/
  summary.json         <- run metadata, final val_loss, convergence status
  metrics.csv          <- per-step metrics (step, loss, tokens_per_sec, step_time)
```

### Raw run logs (gitignored — lives in logs/)
```
logs/mlperf_moe_YYYYMMDD_HHMMSS/
  run.log              <- full stdout/stderr
  wandb/               <- W&B local sync dir (if enabled)
```

---

## Convergence Target

Per MLPerf spec:
- **Metric:** Validation loss (log perplexity)
- **Target:** val_loss <= 3.34
- **Eval frequency:** Every 12,288 samples (768 iterations at GBS=16)
- **Eval subset:** First 1,024 validation samples

The 200-step profiling config will **not** reach convergence. Set `trainer.max_steps: 1200000` for a full convergence run (~6.5 hours on 8x GPUs).

---

## Experiment Matrix

| Config | Hardware | GPUs | EP | Steps | Expected Runtime | Target val_loss | W&B Run |
|---|---|---|---|---|---|---|---|
| moe_4gpu.yaml | 4x A100 80GB | 4 | 4 | 200 | ~3h | 3.34 (informational) | TBD |

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| OOM on A100 40GB | MoE 20B requires 80GB GPUs; reduce `micro_batch_size` to 1 |
| `expert_model_parallel_size` error | EP must evenly divide `num_moe_experts` (32) and match GPU count |
| Slow step time | Expert parallelism communication bottleneck; check NVLink bandwidth |
| `FileNotFoundError` for C4 data | Run `make prepare-mlperf-moe-data` first |
| NeMo container pull fails | Run `docker login nvcr.io` with NGC API key first |

## Design Notes

- **Same C4 dataset** as the MLPerf Llama3.1 8B benchmark — no separate download needed if already prepared.
- **W&B logging** enabled by default for infra profiling metrics. Submission files also saved locally.
- **Open Division** — 4-GPU config with reduced EP; matches the spirit of an infra profiling repo.
- **GPU/network metrics** — covered separately by the ml-netprof agent -> Prometheus -> Grafana (see `infra/prometheus/`).
