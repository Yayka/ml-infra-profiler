# Llama 3.1 8B — FSDP + DiLoCo Training

Single training script (`train_llama8b.py`) that supports two modes for interconnect profiling experiments:

- **Baseline**: Standard FSDP FULL_SHARD across all GPUs (cross-node gradient sync every step)
- **DiLoCo**: Per-node FSDP + periodic cross-node pseudo-gradient averaging (communication only every N steps)

Both modes use the same model, optimizer, data pipeline, and launch command — the only difference is the communication pattern, which is exactly what we want to measure.

## Requirements

- 2+ nodes with NVIDIA GPUs (tested on 2x A100 80GB PCIe per node)
- PyTorch >= 2.2 (tested with 2.4.0+cu121)
- `transformers`, `datasets`, `torch` installed in the container/environment
- Nodes must be able to reach each other on the network (for NCCL and torchrun rendezvous)

## Docker

A `Dockerfile.diloco` is provided at `infra/docker/Dockerfile.diloco`. Build it from the repo root:

```bash
docker build -f infra/docker/Dockerfile.diloco -t ml-netprof/diloco:latest .
```

The image is based on `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel` and includes `transformers` and `datasets`. The training script is baked in at `/workspace/train_llama8b.py` and the image `ENTRYPOINT` is `torchrun`, so pass torchrun flags directly to `docker run`.

**Multi-node example** (run on each node, adjust `--node-rank` and `MASTER_ADDR`):

```bash
docker run --rm --gpus all --ipc=host --network=host \
  ml-netprof/diloco:latest \
  --nnodes=2 --nproc_per_node=2 --node-rank=0 \
  --master-addr=$MASTER_ADDR --master-port=29500 \
  /workspace/train_llama8b.py \
  --fake-data --max-steps 10 --total-batch-size 8 \
  --per-device-train-batch-size 1 --seq-length 512
```

`--network=host` is required so NCCL and the torchrun rendezvous can reach the other node directly over the cluster network. Use each node's internal (VNet) IP as `MASTER_ADDR`, not the public IP.

## Quick Start — Smoke Test

Verify the script works on your cluster using fake data (no dataset download needed).

### Single node (2 GPUs)

```bash
# Baseline
torchrun --nproc_per_node=2 train_llama8b.py \
    --fake-data --max-steps 5 --total-batch-size 4 \
    --per-device-train-batch-size 1 --seq-length 512 \
    --lr 1e-3 --warmup-steps 2 --log-every-n-steps 5

# DiLoCo (single island — outer step is a no-op, but validates the code path)
torchrun --nproc_per_node=2 train_llama8b.py \
    --fake-data --max-steps 10 --total-batch-size 4 \
    --per-device-train-batch-size 1 --seq-length 512 \
    --lr 1e-3 --warmup-steps 2 --log-every-n-steps 5 \
    --diloco --diloco-local-steps 5
```

### Multi-node (2 nodes x 2 GPUs)

Run on **both nodes simultaneously**. Replace `MASTER_ADDR` with node 0's IP.

```bash
# Node 0:
torchrun --nnodes=2 --nproc_per_node=2 --node-rank=0 \
    --master-addr=$MASTER_ADDR --master-port=29500 \
    train_llama8b.py \
    --fake-data --max-steps 10 --total-batch-size 8 \
    --per-device-train-batch-size 1 --seq-length 512 \
    --lr 1e-3 --warmup-steps 2 --log-every-n-steps 5

# Node 1 (same command, different --node-rank):
torchrun --nnodes=2 --nproc_per_node=2 --node-rank=1 \
    --master-addr=$MASTER_ADDR --master-port=29500 \
    train_llama8b.py \
    --fake-data --max-steps 10 --total-batch-size 8 \
    --per-device-train-batch-size 1 --seq-length 512 \
    --lr 1e-3 --warmup-steps 2 --log-every-n-steps 5
```

Add `--diloco --diloco-local-steps 5` to both commands for a DiLoCo smoke test.

## Production Profiling Runs

For actual interconnect profiling with the full 8B model and C4 dataset.

A real tokenizer is required (the script will fail fast if you forget). The
official `meta-llama/Llama-3.1-8B` repo on HF Hub is gated, so either:

- accept the Llama license and pass `-e HF_TOKEN=$HF_TOKEN` to `docker run`
  with `--tokenizer meta-llama/Llama-3.1-8B`, or
- use the open mirror `--tokenizer NousResearch/Meta-Llama-3.1-8B` (same
  vocab, no auth required) — this is what the examples below use.

### Baseline (FSDP across all GPUs)

```bash
# On each node (adjust --node-rank):
torchrun --nnodes=2 --nproc_per_node=2 --node-rank=$RANK \
    --master-addr=$MASTER_ADDR --master-port=29500 \
    train_llama8b.py \
    --tokenizer NousResearch/Meta-Llama-3.1-8B \
    --total-batch-size 512 --per-device-train-batch-size 1 \
    --seq-length 2048 --lr 4e-4 --warmup-steps 1000 \
    --max-steps 500
```

### DiLoCo (H=500 inner steps between syncs)

```bash
# Same launch, add --diloco flags:
torchrun --nnodes=2 --nproc_per_node=2 --node-rank=$RANK \
    --master-addr=$MASTER_ADDR --master-port=29500 \
    train_llama8b.py \
    --tokenizer NousResearch/Meta-Llama-3.1-8B \
    --total-batch-size 512 --per-device-train-batch-size 1 \
    --seq-length 2048 --lr 4e-4 --warmup-steps 1000 \
    --max-steps 500 \
    --diloco --diloco-local-steps 500 --diloco-outer-lr 0.7
```

## How It Works

### Baseline mode

All GPUs form one FSDP group with `FULL_SHARD` (ZeRO-3). Every forward pass triggers all-gather of parameters across all nodes; every backward pass triggers reduce-scatter of gradients across all nodes.

### DiLoCo mode

A 2D `device_mesh` splits GPUs into per-node "islands":

```
device_mesh = (nnodes, gpus_per_node) = (2, 2)
  "shard" dimension   → per-node FSDP groups: {GPU0, GPU1} and {GPU2, GPU3}
  "replicate" dimension → cross-node pairs: {GPU0, GPU2} and {GPU1, GPU3}
```

- **Inner loop** (every step): FSDP all-gather/reduce-scatter stays within each node. No cross-node traffic.
- **Outer loop** (every `diloco-local-steps` steps): All ranks gather full params to CPU, compute pseudo-gradients `(saved_params - current_params)`, all-reduce via gloo (CPU), apply SGD+Nesterov outer optimizer, write back to FSDP shards.

This means cross-node communication is reduced from every-step to every-N-steps, at the cost of slightly different optimization dynamics (local SGD with periodic averaging).

## Memory Usage (2 nodes x 2 A100 80GB)

Measured on our test cluster:

| Mode     | Peak GPU | Peak CPU (per node) | Notes                                |
| -------- | -------- | ------------------- | ------------------------------------ |
| Baseline | ~70 GB   | ~60 GB              | FSDP/4 GPUs — comfortable fit        |
| DiLoCo   | ~78 GB   | ~200 GB             | FSDP/2 GPUs + outer optimizer on CPU |

DiLoCo uses more GPU memory (model sharded across 2 GPUs instead of 4) and significantly more CPU memory (each rank holds full fp32 params + momentum + snapshot for the outer optimizer, ~96 GB per rank).

## Key Flags

| Flag                            | Default | Description                                                           |
| ------------------------------- | ------- | --------------------------------------------------------------------- |
| `--fake-data`                   | off     | Use random tokens instead of C4 (for smoke tests)                     |
| `--diloco`                      | off     | Enable DiLoCo mode                                                    |
| `--diloco-local-steps`          | 500     | Inner steps between outer syncs                                       |
| `--diloco-outer-lr`             | 0.7     | Outer optimizer learning rate (SGD + Nesterov)                        |
| `--total-batch-size`            | 512     | Global batch size (tokens per optimizer step)                         |
| `--per-device-train-batch-size` | 1       | Micro-batch size per GPU                                              |
| `--seq-length`                  | 2048    | Sequence length                                                       |
| `--gradient-checkpointing`      | on      | Activation checkpointing (disable with `--no-gradient-checkpointing`) |
| `--max-steps`                   | None    | Stop after N optimizer steps                                          |
| `--path-model`                  | None    | HF model path (if None, creates random-init 8B)                       |
| `--tokenizer`                   | None    | Tokenizer path (defaults to `--path-model`)                           |

## Design Decisions

- **Pure PyTorch, no hivemind**: The original OpenDiloco uses hivemind for fault tolerance and NAT traversal. We don't need those on a controlled cluster, and pure PyTorch gives us direct control over communication primitives for profiling.
- **All ranks participate in outer step**: PyTorch 2.4 doesn't support `summon_full_params(writeback=True, rank0_only=True)`. Instead, all ranks compute identical outer steps and write back consistent shards.
- **Gloo for outer step**: Pseudo-gradients live on CPU (offloaded from GPU to save memory). NCCL requires CUDA tensors, so we use a separate gloo process group for the CPU all-reduce.
- **`dist.barrier()` before teardown**: Prevents NCCL watchdog errors when nodes finish at slightly different times.
