# MLPerf Text-to-Image (Flux.1-schnell) — Infrastructure Profiling Runbook

Runs a 2-node x 2-GPU (4 GPU total) Flux.1-schnell text-to-image training job using
PyTorch FSDP and the diffusers/transformers libraries. Goal is infrastructure profiling
(GPU utilization, NCCL traffic, memory bandwidth), with convergence target of
validation loss <= 0.586 (MSE over latents).

**Model:** Flux.1-schnell (~11.9B params) — Multimodal Diffusion Transformer (MMDiT)
**Dataset:** CC12M subset (1.1M images, 256x256) for training, COCO-2014 (29,696 images) for validation
**Cluster:** 2 Azure A100 80GB nodes connected over Ethernet (TCP NCCL).
**Expected runtime:** ~24-48 hours (depends on network bandwidth between nodes).
**Parallelism:** FSDP across 4 GPUs, gradient accumulation = 2 (effective batch = 32).

---

## Prerequisites

On **both** GPU nodes:
- NVIDIA drivers + CUDA 12.1+
- Python 3.10+ with venv support
- CC12M data at `/data/mlperf_t2i/cc12m_256/` (see Step 2)
- COCO-2014 validation data at `/data/mlperf_t2i/coco2014_256/` (see Step 2)
- Flux.1-schnell model weights at `/data/mlperf_t2i/models/flux1-schnell/` (see Step 2)
- `ml-netprof-agent` running as systemd service (for Prometheus scraping)
- Nodes can SSH to each other without a password prompt

On your **local machine**:
- This repo cloned, `.env` populated (see `.env.example`)
- SSH key for the nodes at the path set in `.env` (`SSH_KEY`)

---

## .env Setup

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```bash
NODES="<node0-public-ip> <node1-public-ip>"
INTERNAL_IPS="<node0-private-ip> <node1-private-ip>"
SSH_USER=azureuser
SSH_KEY=gpu-ib_key.pem
HF_TOKEN=<your-huggingface-token>     # for downloading Flux.1-schnell + T5-XXL
WANDB_API_KEY=<your-key>              # optional, W&B disabled by default
```

---

## Step 1 — Setup Python Environment (both nodes)

```bash
make setup-mlperf-t2i
```

This creates `.venv-t2i` with PyTorch, diffusers, transformers, accelerate, and evaluation
dependencies. Run on each node, or SSH and run remotely.

---

## Step 2 — Prepare Data and Model Weights (one-time, both nodes)

```bash
make prepare-mlperf-t2i-data
```

This downloads:
- **CC12M subset** (~1.1M images, resized to 256x256) to `/data/mlperf_t2i/cc12m_256/`
- **COCO-2014 validation** (29,696 images, resized to 256x256) to `/data/mlperf_t2i/coco2014_256/`
- **Flux.1-schnell** model weights to `/data/mlperf_t2i/models/flux1-schnell/`
- **Frozen encoders**: CLIP ViT-L, T5-XXL, VAE (cached by HuggingFace)

Data download is ~200 GB total; takes 1-3 hours depending on bandwidth.

If nodes share NFS, download once. Otherwise run on each node:

```bash
ssh $SSH_OPTS azureuser@<node> "cd ~/ml-infra-profiler && make prepare-mlperf-t2i-data"
```

---

## Step 3 — Clone Repo on Both Nodes

```bash
SSH_OPTS=(-o StrictHostKeyChecking=no -i gpu-ib_key.pem)
for NODE in <node0-public-ip> <node1-public-ip>; do
    ssh "${SSH_OPTS[@]}" azureuser@$NODE \
        "git clone https://github.com/Yayka/ml-infra-profiler.git && \
         cd ml-infra-profiler && git checkout mlperf-t2i-benchmark" &
done
wait
```

---

## Step 4 — Run the Benchmark

From your **local machine**, in the repo root:

```bash
source .env && \
NODES="$NODES" \
INTERNAL_IPS="$INTERNAL_IPS" \
GPUS_PER_NODE=2 \
SSH_KEY=$SSH_KEY \
SSH_USER=$SSH_USER \
make run-mlperf-t2i
```

Logs are written to `logs/mlperf_t2i_<timestamp>/node_{0,1}.log`.

---

## Step 5 — Verify Convergence

```bash
make verify-mlperf-t2i
```

Checks that the final validation loss in `results/mlperf_t2i/` is <= 0.586.

---

## Monitoring Progress

**Training steps:**
```bash
tail -f $(ls -td ~/ml-infra-profiler/logs/mlperf_t2i_*/node_0.log | head -1)
```

Look for lines like:
```
Step 100/20000 | loss=0.842 | val_loss=0.712 | img_per_sec=12.4 | step_time=8.2s
```

**GPU utilization (healthy = ~95-100%, ~60-70 GB used):**
```bash
ssh -i gpu-ib_key.pem azureuser@<node-ip> \
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
```

**Prometheus metrics (if ml-netprof-agent is running):**
```bash
curl http://<node-private-ip>:9100/metrics | grep ml_
```

---

## Configuration

Key parameters in `scripts/launch/mlperf_t2i/config/t2i_4gpu.yaml`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `training.max_steps` | 20000 | Increase for longer profiling |
| `training.batch_size_per_gpu` | 4 | Per-GPU micro batch; 80GB A100 fits 4 |
| `training.gradient_accumulation_steps` | 2 | Effective GBS = 4 * 4 * 2 = 32 |
| `evaluation.eval_every_n_samples` | 262144 | Per MLPerf spec |
| `evaluation.convergence_target` | 0.586 | Validation loss target |
| `optimizer.lr` | 1e-4 | AdamW learning rate |

**Memory per GPU:** ~60-70 GB (of 80 GB A100) with FSDP + gradient checkpointing.
**Step time:** ~8-12s/step steady-state (depends on network).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM on A100 80GB | batch_size_per_gpu too large | Reduce to 2 and increase grad accum to 4 |
| `EADDRINUSE: port 29500` | Previous process still running | Kill stale torchrun processes on both nodes |
| `ncclRemoteError` on node 0 | Node 1 crashed first | Check `node_1.log` for root cause |
| Slow step time (>30s) | Ethernet NCCL bottleneck | Expected for TCP; check network bandwidth |
| `HfHubHTTPError` downloading model | Missing or invalid HF_TOKEN | Set HF_TOKEN in .env |
| Val loss not decreasing | LR too high or data issue | Check data paths, try lr=5e-5 |
| `torch.cuda.OutOfMemoryError` | Frozen encoders not offloaded | Ensure gradient_checkpointing=true |

## Experiment Matrix

| Config | Hardware | GPUs | Parallelism | Expected Runtime | Target Val Loss | W&B Run |
|--------|----------|------|-------------|------------------|-----------------|---------|
| t2i_4gpu.yaml | 2x Azure A100 80GB | 4 | FSDP | ~24-48h | <= 0.586 | TBD |
