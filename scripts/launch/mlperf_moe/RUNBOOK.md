# MoE Pretraining — Network Profiling Runbook

A 2-node x 2-GPU FSDP training run of a Mixtral-style MoE model, designed
to generate a sustained mix of cross-node FSDP all-gather/reduce-scatter
and intra-node NCCL traffic for ~2-3 hours.

**Reference inspiration:** [mlcommons/training/small_llm_moe_pretraining](https://github.com/mlcommons/training/tree/master/small_llm_moe_pretraining/primus). We diverge from the Primus reference (AMD-only) in favor of a pure PyTorch + HuggingFace + FSDP path that's easier to debug and matches the rest of this repo's profiling scripts.

**Goal:** capture network traffic (`ml-netprof-agent` -> Prometheus), not hit a convergence target.

**Cluster:** 2 Azure A100 80GB PCIe nodes, 2 GPUs each, NCCL over Ethernet by default. Internal IPs `172.21.0.4` (rank 0) and `172.21.0.5` (rank 1).

**Model:** Mixtral-style MoE, random-init, ~3.7B total / ~1.3B active per token (top-2 of 8 experts, 12 layers, hidden 2048, GQA 16/4).

---

## How it works

`train_moe.py` is a single PyTorch script:

- Builds `MixtralForCausalLM` with a random-initialized config.
- Wraps each `MixtralDecoderLayer` (containing the 8 experts) with FSDP `FULL_SHARD`.
- Trains with bf16 mixed precision, AdamW, cosine LR schedule.
- Reads either fake random tokens (`--fake-data`) or streams `allenai/c4`.

`run_moe_multinode.sh` SSHes to each node and runs the script inside the
existing `ml-netprof/diloco:latest` Docker image (PyTorch 2.4 +
transformers 4.45.2). The script is bind-mounted into the container, so
no rebuild is needed.

---

## Smoke test

```bash
NODES="20.29.43.19 172.212.226.225" \
INTERNAL_IPS="172.21.0.4 172.21.0.5" \
GPUS_PER_NODE=2 \
SSH_KEY=~/.ssh/gpu-ib_key.pem \
SSH_USER=azureuser \
MAX_STEPS=10 FAKE_DATA=1 RUN_TAG=smoke MODEL_SIZE=moe-tiny \
LOG_EVERY_N_STEPS=1 WARMUP_STEPS=2 \
bash scripts/launch/mlperf_moe/run_moe_multinode.sh
```

A successful smoke test prints ~10 lines like:
```
step=    1  loss=10.4012  step_time=4.31s  tokens/sec=1900  lr=1.50e-04
...
Training complete.
```

---

## Production run (~2-3 hours)

```bash
NODES="20.29.43.19 172.212.226.225" \
INTERNAL_IPS="172.21.0.4 172.21.0.5" \
GPUS_PER_NODE=2 \
SSH_KEY=~/.ssh/gpu-ib_key.pem \
SSH_USER=azureuser \
MAX_STEPS=2000 FAKE_DATA=1 RUN_TAG=prod \
bash scripts/launch/mlperf_moe/run_moe_multinode.sh
```

`FAKE_DATA=1` is fine for a network-profiling run — step time and
network traffic are dominated by model size and FSDP collectives, not
data quality. To use real C4 instead, drop `FAKE_DATA=1` and let the
script stream `allenai/c4` (requires HF Hub access).

---

## Configuration knobs (env vars)

| Env var | Default | Notes |
|---|---|---|
| `NODES` | (required) | Space-separated public IPs for SSH |
| `INTERNAL_IPS` | = NODES | Used as MASTER_ADDR / NCCL endpoints |
| `GPUS_PER_NODE` | 2 | All A100s used |
| `MODEL_SIZE` | `moe-3b` | `moe-tiny` for smoke (~150M) |
| `MAX_STEPS` | 2000 | Stop after N optimizer steps |
| `TOTAL_STEPS` | 2000 | Cosine LR horizon |
| `WARMUP_STEPS` | 50 | Linear warmup |
| `SEQ_LENGTH` | 2048 | Per-sample tokens |
| `PER_DEVICE_BATCH_SIZE` | 1 | Micro batch per GPU |
| `TOTAL_BATCH_SIZE` | 8 | Effective global batch (grad accum auto-derived) |
| `LR` | 3e-4 | Peak |
| `FAKE_DATA` | 0 | 1 = synthetic random tokens (no HF download) |
| `NCCL_TRANSPORT` | tcp | `tcp` or `ib` (Azure A100 default has IB but no IPoIB; tcp is safer) |
| `RUN_TAG` | run | Subdir tag in `logs/` and `results/` |
| `MASTER_PORT` | 29501 | Different from mlperf llama3 (29500) so they can co-exist |

---

## Outputs

```
logs/mlperf_moe_<tag>_<timestamp>/
  node_0.log           <- full stdout of rank-0 node
  node_1.log           <- full stdout of rank-1 node

# On each remote node:
~/ml-infra-profiler/results/mlperf_moe_<tag>_<timestamp>/
  metrics.csv          <- per-step loss, step_time, tokens/sec
  hf_cache/            <- HuggingFace download cache (only if not fake-data)
```

Network/GPU metrics are scraped separately by `ml-netprof-agent` ->
Prometheus (`infra/prometheus/`); export them from Grafana as CSV after
the run finishes.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| NCCL hang at startup | `MASTER_ADDR` not reachable — must be the internal/VNet IP, not public |
| OOM | Drop `PER_DEVICE_BATCH_SIZE`, or set `--no-gradient-checkpointing` to off (off by default — already enabled) |
| `NCCL error: unhandled cuda error` | Often a torch/CUDA mismatch; verify `nvidia-smi` and the diloco image's `python -c "import torch; print(torch.cuda.is_available())"` |
| Step time wildly variable | Expected with FAKE_DATA across nodes (no fixed sync barrier other than collectives) |
| Container fails to start with `docker: Error response: failed to set up gpus` | Re-check `nvidia-smi` works on the host |
