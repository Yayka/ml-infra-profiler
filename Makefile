.PHONY: setup start-wandb stop-wandb prepare-data run-mac run-linux launch-job launch-agent \
        prepare-mlperf-data pull-nemo run-mlperf verify-mlperf

# One-time setup: submodule, venv, deps
# Note: nanochat pins torch==2.9.1; if pip can't resolve it from PyPI on macOS arm64,
# install torch manually first: .venv/bin/pip install torch==2.9.1
setup:
	git submodule update --init --recursive
	python3 -m venv .venv
	.venv/bin/pip install -r environments/requirements.txt
	# nanochat has a flat layout that setuptools can't auto-discover; install deps only.
	# The nanochat package is importable at runtime because run_local.py cds into nanochat/.
	.venv/bin/python scripts/launch/nanochat/install_deps.py
	cp -n .env.example .env || true
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit .env — add your WANDB_LICENSE key"
	@echo "  2. Run: make start-wandb"
	@echo "  3. Visit http://localhost:8080 — create account, copy API key to .env"
	@echo "  4. Run: make prepare-data && make run"

# Start W&B local server
start-wandb:
	docker compose -f infra/docker/docker-compose.wandb.yml --env-file .env up -d
	@echo "W&B server starting at http://localhost:8080"
	@echo "Create an account, go to Settings, copy your API key, and paste it into .env"

# Stop W&B local server (data persists in Docker volume)
stop-wandb:
	docker compose -f infra/docker/docker-compose.wandb.yml down

# Download TinyStories → data/base_data/train_00.parquet + val_00.parquet
prepare-data:
	.venv/bin/python scripts/data/prepare_tiny_dataset.py

# Train BPE tokenizer on downloaded data (run once after prepare-data)
tokenizer:
	cd nanochat && NANOCHAT_BASE_DIR=$(CURDIR)/data \
		$(CURDIR)/.venv/bin/python -m scripts.tok_train --max-chars 50000000

# Train tiny 2-layer model on Apple Silicon MPS, logging to local W&B
run-mac:
	.venv/bin/python scripts/launch/nanochat/run_local.py

# Train tiny 2-layer model on Linux (CPU or CUDA), logging to local W&B
run-linux:
	.venv/bin/python scripts/launch/nanochat/run_local.py scripts/launch/nanochat/config/local_linux_tiny.yaml

# Train on multiple nodes via SSH + torchrun (edit NODES to match your Azure VMs)
# Example: NODES="10.0.0.10 10.0.0.11" GPUS_PER_NODE=2 make run-multinode
run-multinode:
	./scripts/launch/nanochat/run_multinode.sh

# --- MLPerf Llama3.1 8B benchmark ---

# Prepare C4 v3.0.1 data for MLPerf (runs tokenization inside NeMo container).
# Path A (preferred): copy pre-tokenized files from MLCommons to DATA_DIR first, then verify.
# Path B (self-tokenize): set HF_TOKEN and TOKENIZE_PATH=B.
# DATA_DIR defaults to /data/c4 on the node running this command.
prepare-mlperf-data:
	bash scripts/data/prepare_c4_mlperf.sh

# Pull the NeMo container image (~25GB). Requires NGC credentials:
#   docker login nvcr.io  (username: $$oauthtoken, password: NGC_API_KEY from .env)
pull-nemo:
	set -a && . ./.env && set +a && \
	echo "$$NGC_API_KEY" | docker login nvcr.io -u '$$oauthtoken' --password-stdin
	docker pull nvcr.io/nvidia/nemo:24.12-rc0

# Launch MLPerf Llama3.1 8B training on 2 nodes × 2 GPUs via SSH + Docker.
# Required env vars: NODES, INTERNAL_IPS, GPUS_PER_NODE, SSH_KEY, SSH_USER
# Example:
#   NODES="<pub0> <pub1>" INTERNAL_IPS="<priv0> <priv1>" \
#   GPUS_PER_NODE=2 SSH_KEY=~/gpu-key.pem SSH_USER=azureuser make run-mlperf
run-mlperf:
	bash scripts/launch/mlperf/run_mlperf_multinode.sh

# Check whether the completed MLPerf run met the val perplexity target (≤ 3.3).
# Reads final val_loss from W&B. Override run with: WANDB_RUN_PATH=entity/project/run_id
verify-mlperf:
	bash scripts/launch/mlperf/verify_run.sh

# --- ml-netprof monitoring agent ---

# Build native macOS binary (for local dev/testing on macOS)
agent-build-darwin:
	CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -C agent -o bin/agent-darwin ./cmd/agent

agent-start:
	./agent/bin/agent-darwin -config agent/configs/agent_default.yaml &
	@echo "Agent running at http://localhost:9100/metrics"

agent-test:
	go test -C agent ./...

# --- W&B Launch ---

# Submit nanochat job to the W&B Launch queue (run from local or W&B-server VM)
launch-job:
	set -a && . ./.env && set +a && \
	.venv/bin/wandb launch \
		--uri . \
		--dockerfile infra/docker/Dockerfile.nanochat \
		--entry-point "bash scripts/launch/nanochat/entrypoint.sh" \
		--queue nanochat-gpu \
		--entity $$WANDB_ENTITY \
		--project ml-netprof \
		--job-name nanochat-train

# Start the W&B launch agent (run ON the GPU VM, not locally)
launch-agent:
	wandb launch-agent \
		--queue nanochat-gpu \
		--entity $$WANDB_ENTITY \
		--config infra/launch/launch-config.yaml
