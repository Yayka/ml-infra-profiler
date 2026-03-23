# MLPerf Llama3.1 8B — Infrastructure Profiling Runbook

Runs a 2-node × 2-GPU (4 GPU total) Llama3.1 8B pretraining job using the NeMo framework.
Goal is infrastructure profiling (GPU utilization, NCCL traffic, PCIe bandwidth), not
hitting the MLPerf perplexity target.

**Cluster:** 2 Azure A100 80GB nodes connected over Ethernet (TCP NCCL).
**Expected runtime:** ~3 hours (200 steps × ~50s/step).
**Parallelism:** TP=2 (intra-node), PP=2 (inter-node), DP=1, GBS=64.

---

## Prerequisites

On **both** GPU nodes:
- Docker installed, user in `docker` group
- NeMo image pulled (see Step 1)
- C4 data at `/data/c4/` (see Step 2)
- `ml-netprof-agent` running as systemd service (for Prometheus scraping)
- Nodes can SSH to each other without a password prompt (same VNet, key at `~/.ssh/id_rsa`)

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
NODES="<node0-public-ip> <node1-public-ip>"   # used for SSH
INTERNAL_IPS="<node0-private-ip> <node1-private-ip>"  # used for NCCL
SSH_USER=azureuser
SSH_KEY=gpu-ib_key.pem        # path relative to repo root
NGC_API_KEY=<your-ngc-key>    # for pulling NeMo image
WANDB_API_KEY=<your-key>      # optional, W&B disabled by default
```

---

## Step 1 — Pull NeMo Image (one-time, both nodes)

```bash
# From your local machine — pulls on both nodes in parallel
SSH_OPTS="-o StrictHostKeyChecking=no -i gpu-ib_key.pem"
for NODE in <node0-public-ip> <node1-public-ip>; do
    ssh $SSH_OPTS azureuser@$NODE "
        echo '<your-ngc-key>' | docker login nvcr.io -u '\$oauthtoken' --password-stdin &&
        docker pull nvcr.io/nvidia/nemo:24.12-rc0
    " &
done
wait
```

Or using the Makefile (pulls on the local machine only — run on each node separately):
```bash
make pull-nemo
```

Image is ~25 GB; takes 5–10 minutes on a fast connection.

---

## Step 2 — Prepare C4 Data (one-time, both nodes)

Data must exist at `/data/c4/` on **both** nodes. Either:

**Option A — shared NFS:** Mount the same NFS volume at `/data/c4` on both nodes. Run once:
```bash
DATA_DIR=/data/c4 bash scripts/data/prepare_c4_mlperf.sh
```

**Option B — replicated local disks:** Run on each node independently:
```bash
ssh $SSH_OPTS azureuser@<node0> "cd ~/ml-infra-profiler && DATA_DIR=/data/c4 bash scripts/data/prepare_c4_mlperf.sh"
ssh $SSH_OPTS azureuser@<node1> "cd ~/ml-infra-profiler && DATA_DIR=/data/c4 bash scripts/data/prepare_c4_mlperf.sh"
```

Data download is ~80 GB; takes 30–60 minutes depending on bandwidth.

---

## Step 3 — Clone Repo on Both Nodes

```bash
SSH_OPTS=(-o StrictHostKeyChecking=no -i gpu-ib_key.pem)
for NODE in <node0-public-ip> <node1-public-ip>; do
    ssh "${SSH_OPTS[@]}" azureuser@$NODE \
        "git clone https://github.com/Yayka/ml-infra-profiler.git && \
         cd ml-infra-profiler && git checkout mlperf-llama3-benchmark" &
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
NCCL_TRANSPORT=tcp \
make run-mlperf
```

Logs are written to `logs/mlperf_<timestamp>/node_{0,1}.log`.
The terminal tails node 0's log automatically. Press `Ctrl+C` to stop tailing (training keeps running on the nodes).

---

## Step 5 — Distribute Index Mapping Files (first run only)

Megatron builds dataset index files on rank 0 (node 0) and writes them to
`~/ml-infra-profiler/results/mlperf/index_mapping/`. Node 1 needs the same files.

**Node 1 will crash on the first run** with `FileNotFoundError: ... GPTDataset ... .npy`.
This is expected. After the first crash:

```bash
# From your local machine
ssh -o StrictHostKeyChecking=no -i gpu-ib_key.pem azureuser@<node0-public-ip> \
    "scp -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa \
     ~/ml-infra-profiler/results/mlperf/index_mapping/* \
     azureuser@<node1-private-ip>:~/ml-infra-profiler/results/mlperf/index_mapping/"
```

Then rerun `make run-mlperf`. The files are reused on subsequent runs **as long as you
don't change `global_batch_size`** — changing GBS generates new index files and you'll
need to repeat this step.

---

## Monitoring Progress

**Training steps (all NeMo logs come from node 0):**
```bash
tail -f $(ls -td ~/ml-infra-profiler/logs/mlperf_*/node_0.log | head -1)
```

Look for lines like:
```
Epoch 0:  10%| 20/200 [16:40, train_loss=9.8, global_step=19, train_step_timing=50s]
```

**Step summary only:**
```bash
grep "reduced_train_loss" $(ls -td ~/ml-infra-profiler/logs/mlperf_*/node_0.log | head -1)
```

**GPU utilization (healthy = ~100%, ~48–61 GB used):**
```bash
# Node 0
ssh -i gpu-ib_key.pem azureuser@<node0-public-ip> \
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"

# Node 1
ssh -i gpu-ib_key.pem azureuser@<node1-public-ip> \
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
```

**Prometheus metrics (if ml-netprof-agent is running):**
```bash
curl http://<node0-private-ip>:9100/metrics | grep ml_
```

---

## Configuration

Key parameters in `scripts/launch/mlperf/nemo_entrypoint.sh`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `trainer.max_steps` | 200 | ~3h total; increase for longer profiling |
| `model.global_batch_size` | 64 | Controls gradient accumulation (GAS=64); change requires re-copying index files |
| `model.tensor_model_parallel_size` | 2 | Splits model within each node over NVLink/PCIe |
| `model.pipeline_model_parallel_size` | 2 | Splits layers across nodes (0–15 on node 0, 16–31 on node 1) |
| `trainer.val_check_interval` | 50 | Validation every 50 steps |
| `trainer.log_every_n_steps` | 5 | Log every 5 steps |

**Memory per GPU:** ~48–61 GB (of 80 GB A100).
**Step time:** ~50s/step steady-state.

To run longer (e.g. overnight):
```bash
# Edit nemo_entrypoint.sh:
trainer.max_steps=500  # ~7h
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: GPTDataset...npy` on node 1 | Index files not synced | See Step 5 |
| `EADDRINUSE: port 29500` | Previous container still running | `docker ps -q --filter ancestor=nvcr.io/nvidia/nemo:24.12-rc0 \| xargs -r docker stop` on both nodes |
| `ncclRemoteError: remote process exited` on node 0 | Node 1 crashed first | Check `node_1.log` for the root cause |
| GPU utilization 0% after 5 min | Container failed to start | Check `node_1.log` for errors |
| Training stuck at step 0 for >15 min | CUDA kernel compilation on first step | Normal; wait it out |
| Node 1 log has very few lines | Expected — NeMo only logs from rank 0 (node 0) | Check node 0 log for progress |
