# ml-netprof

A reusable template for profiling network and system metrics during ML training runs. Uses [Weights & Biases](https://wandb.ai) for experiment tracking and instruments infrastructure behavior (GPU utilization, system resources, network traffic) alongside standard training metrics.

Built on top of [nanochat](https://github.com/smalber/nanochat) as the training harness.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3) — default config targets MPS
- Python 3.10+
- Docker Desktop (running)
- A W&B account — either [cloud](https://wandb.ai) (free tier, easier) or self-hosted local server (requires a trial license key)

## Quick Start

### 1. Clone

```bash
git clone --recurse-submodules <repo-url>
cd ml-infra-profiler
```

> If you already cloned without `--recurse-submodules`, run: `git submodule update --init --recursive`

### 2. Install dependencies

```bash
make setup
```

This creates `.venv`, installs all Python deps, and copies `.env.example` → `.env`.

### 3. Configure W&B

**Option A — W&B Cloud (recommended, no Docker needed):**

1. Sign up at https://wandb.ai and get your API key from Settings
2. Edit `.env`:
   ```
   WANDB_API_KEY=<your-key>
   # Comment out or remove WANDB_BASE_URL and WANDB_LICENSE
   ```

**Option B — Local W&B server (requires a [trial license](https://wandb.ai/site/self-hosted)):**

> Note: the `wandb/local` image is amd64-only. On Apple Silicon it runs under Rosetta emulation, which may be slow or unstable.

1. Edit `.env` and set `WANDB_LICENSE=<your-license-key>`
2. Start the server:
   ```bash
   make start-wandb
   ```
3. Visit http://localhost:8080, create an account, go to Settings, copy your API key, and paste it into `.env` as `WANDB_API_KEY`

### 4. Download training data

```bash
make prepare-data
```

Downloads 50k train / 5k val rows from TinyStories to `data/base_data/`.

### 5. Train the tokenizer

```bash
make tokenizer
```

Trains a BPE tokenizer on the downloaded data and saves it to `data/tokenizer/`. Only needs to run once.

### 6. Run training

```bash
make run
```

Trains a 2-layer ~1M parameter model on Apple Silicon MPS for 300 iterations, logging metrics to W&B. View results at the run URL printed in the terminal.

## Project Structure

```
scripts/
  data/           # Dataset download and preparation
  launch/
    nanochat/     # Nanochat launcher and its configs
    <your-job>/   # Your custom training script goes here
      run_local.py
      config/
        your_config.yaml
infra/
  docker/         # docker-compose for local W&B server
environments/     # requirements.txt (pinned), environment.yaml
nanochat/         # Training harness (git submodule)
data/             # Local data directory (gitignored)
  base_data/      # Downloaded parquet files
  tokenizer/      # Trained tokenizer artifacts
```

## Makefile Reference

| Target                    | Description                                                     |
| ------------------------- | --------------------------------------------------------------- |
| `make setup`              | Create `.venv`, install deps, copy `.env.example` → `.env`      |
| `make start-wandb`        | Start local W&B server at http://localhost:8080                 |
| `make stop-wandb`         | Stop local W&B server (data persists in Docker volume)          |
| `make prepare-data`       | Download TinyStories to `data/base_data/`                       |
| `make tokenizer`          | Train BPE tokenizer → `data/tokenizer/tokenizer.pkl` (run once) |
| `make agent-build-darwin` | Build the metrics agent binary (macOS arm64)                    |
| `make agent-start`        | Start the agent in the background on `:9100`                    |
| `make run`                | Train tiny model on MPS, log metrics to W&B                     |


## Using the agent with your own training script

The ml-netprof agent is a standalone binary
To profile any Python script:

**Step 1 — Start the agent:**
```bash
./agent/bin/agent-darwin -config agent/configs/agent_default.yaml &
```

**Step 2 — Add these lines before `wandb.init()` in your script:**
```python
import urllib.request, wandb
try:
    urllib.request.urlopen("http://localhost:9100/healthz", timeout=1)
    wandb.setup(settings=wandb.Settings(
        x_stats_open_metrics_endpoints={"agent": "http://localhost:9100/metrics"}
    ))
except Exception:
    pass  # agent not running; metrics collection skipped
```

That's it. Your existing `wandb.init()` / training loop continues unchanged.
Network metrics will appear in W&B under the system metrics panel.

## Configuration

Training is fully config-driven. The default config is `scripts/launch/nanochat/config/local_mac_tiny.yaml` — a minimal 2-layer model tuned for local Apple Silicon runs. Pass a different config as the first argument:

```bash
python scripts/launch/nanochat/run_local.py scripts/launch/nanochat/config/your_config.yaml
```

Key parameters in `local_mac_tiny.yaml`:

| Parameter        | Value | Notes                          |
| ---------------- | ----- | ------------------------------ |
| `depth`          | 2     | ~1M params, fast on MPS        |
| `max_seq_len`    | 512   | Reduced memory vs default 2048 |
| `device_type`    | mps   | Apple Silicon GPU              |
| `num_iterations` | 300   | Short run for local testing    |

## Troubleshooting

**White screen at localhost:8080** — The W&B backend may have failed to start under Rosetta. Check `docker logs docker-wandb-1`. Consider using W&B cloud instead.

**`tokenizer.pkl` not found** — Run `make tokenizer` before `make run`.

**`torch` install fails** — nanochat pins `torch==2.9.1`. If pip can't resolve it: `.venv/bin/pip install torch==2.9.1` first, then re-run `make setup`.

## License

MIT
