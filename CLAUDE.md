# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# ml-netprof

## Project Overview
ml-netprof is a reusable, open-source template for profiling network and system metrics during ML training runs. It uses Weights & Biases (W&B) for experiment tracking and provides instrumentation to measure infrastructure behavior (network traffic, GPU utilization, system resources) alongside standard training metrics. The goal is reproducibility and visibility into how infrastructure affects training performance.

License: MIT

## Architecture

### W&B Integration
- **Client** (open-source, MIT, `wandb/wandb` on GitHub): instruments training code, collects system metrics locally every ~15s, syncs to server over HTTPS. Must run on every training node.
- **Server** (closed-source, `wandb/local` Docker image): stores data (MySQL + S3-compatible object storage), serves web UI/API, manages auth/teams. Deployed as SaaS, Dedicated Cloud, or Self-Managed (K8s via Helm/Operator).
- Server is **not modifiable** — no custom UI panels, schema changes, API extensions, or server-side aggregation. All extensibility lives in the client.

### Custom Metrics Collection
Three approaches for adding metrics like per-port network traffic:
1. **Manual logging**: `wandb.log({"network/port_8080_sent": bytes})` — bounded metric sets only
2. **OpenMetrics/Prometheus scraping**: W&B client scrapes Prometheus endpoints with regex filters. Histogram buckets come in as flat scalars (no native histogram aggregation). Pre-compute p50/p95/p99 client-side.
3. **Fork open-source client**: Add custom collectors to the system monitor module

### W&B Metric Constraints
- Naming: `/^[_a-zA-Z][_a-zA-Z0-9]*$/`, one nesting level via `/` (e.g., `train/loss`, `network/port_8080_sent`)
- No spaces, commas, hyphens, special characters in metric names
- Single value: <1 MB; single `run.log()` call: <25 MB
- Log frequency: not more than a few times/second
- Thousands of unique metrics degrade UI performance
- Scalar metrics only (int/float); lists auto-convert to histograms (lossy on export)

### Data Export Constraints
- CSV from UI: ~100k rows per metric
- `run.history()`: downsampled (default 500 samples)
- `run.scan_history()`: full resolution for custom metrics only (NOT system metrics)
- Full export: use Parquet via Artifacts API
- SaaS API is rate-limited

## W&B Launch (Workload Orchestration)
- **Jobs**: blueprints (git repo, local dir, Docker image, or existing run)
- **Queues**: FIFO, targeting compute backends (Docker, K8s, SageMaker, Vertex AI, AKS, CoreWeave)
- **Agents**: poll queues, build/pull images, execute jobs
- Framework agnostic (container-based), only requires `wandb.init()`
- No native pre-run hooks — use K8s initContainers or entrypoint wrapper scripts for GPU health checks (DCGM, GPU Fryer)
- Limitations: no cluster provisioning, simple FIFO only, no DAG/workflow orchestration, multi-node requires external setup (Volcano)

## Repo Structure
```
src/              # Training code and metric collection
scripts/
  preflight/      # GPU health checks (GPU Fryer, DCGM, network checks)
  launch/
    nanochat/     # Nanochat launcher + configs/
  analysis/       # Export results from W&B API
infra/
  docker/         # Dockerfile with pinned versions (base image, CUDA, pip)
  kubernetes/     # Job templates, queue configs, initContainers for preflight
environments/     # requirements.txt (pinned), environment.yaml
results/          # Exported CSV summaries, reference figures
```

## Key Design Principles
1. **Config-driven**: every experiment fully defined by a YAML file
2. **Pin everything**: base image, CUDA version, pip versions
3. **Document hardware-result relationships**: InfiniBand bandwidth, GPU memory, multi-node topology matter
4. **Export reference results to CSV**: don't rely solely on W&B links for reproducibility
5. **Infrastructure is code**: K8s specs, Dockerfiles, queue configs are reproducibility artifacts
6. **Separation of concerns**: experiment repo assumes W&B is available and documents how to connect; W&B server deployment lives in a separate infra repo

## What Belongs in This Repo vs. Elsewhere
- **In repo**: configs, source code, metric collection code, Dockerfiles, launch scripts, preflight scripts, K8s specs, queue configs, exported summary CSVs, reference figures, experiment matrix, Makefile
- **Not in repo**: API keys/secrets, large datasets (use W&B Artifacts/DVC pointers), model checkpoints (use Artifacts/HF Hub pointers), Terraform state, `wandb/` local dir, compiled artifacts
- **Separate infra repo**: W&B server deployment (K8s operator, Helm, Terraform, licensing, backups)

## Experiment Matrix Format
| Experiment     | Config                 | Hardware          | GPUs | Training Time | Final Loss | W&B Run |
| -------------- | ---------------------- | ----------------- | ---- | ------------- | ---------- | ------- |
| Small baseline | small_local_1gpu.yaml  | RTX 4090          | 1    | 2h            | 0.42       | [link]  |
| Medium scale   | medium_cloud_4gpu.yaml | 4x A100 80GB      | 4    | 6h            | 0.31       | [link]  |
| Large scale    | large_cloud_8gpu.yaml  | 8x A100 (2 nodes) | 8    | 12h           | 0.28       | [link]  |

## Commands

### One-Time Setup

Prerequisites: Docker Desktop running, Python 3.10+, W&B trial license key.

```bash
git clone --recurse-submodules <repo>
make setup              # creates .venv, installs deps, copies .env.example → .env
# Edit .env: set WANDB_LICENSE
make start-wandb        # starts W&B at http://localhost:8080
# Visit http://localhost:8080 → create account → Settings → copy API key → paste into .env as WANDB_API_KEY
make prepare-data       # downloads TinyStories → data/base_data/{train,val}_00.parquet
make tokenizer          # trains BPE tokenizer → data/tokenizer/tokenizer.pkl (run once)
make run                # trains 2-layer model on MPS, logs to local W&B
```

### Makefile Targets

| Target | Description |
| ------ | ----------- |
| `make setup` | Init submodule, create `.venv`, install all deps, copy `.env.example` → `.env` |
| `make start-wandb` | Start W&B local server (Docker) at http://localhost:8080 |
| `make stop-wandb` | Stop W&B server (data persists in Docker volume) |
| `make prepare-data` | Download TinyStories (50k train / 5k val rows) to `data/base_data/` |
| `make tokenizer` | Train BPE tokenizer on downloaded data → `data/tokenizer/tokenizer.pkl` (run once) |
| `make run` | Train nanochat 2-layer model on Apple Silicon MPS, log to local W&B |

### nanochat Data Location

nanochat reads parquet files from `$NANOCHAT_BASE_DIR/base_data/`. `run_local.sh` sets
`NANOCHAT_BASE_DIR=<repo>/data`, so files go in `data/base_data/`. The last parquet file
(alphabetically) is used as the validation split; all others are training.