.PHONY: setup start-wandb stop-wandb prepare-data run-mac run-linux launch-job launch-agent \
        prepare-mlperf-data pull-nemo run-mlperf verify-mlperf \
        setup-mlperf-tiny prepare-mlperf-tiny-data run-mlperf-tiny run-mlperf-tiny-cpu verify-mlperf-tiny \
        build-mlperf-inference prepare-mlperf-inference-data run-mlperf-inference run-mlperf-inference-offline verify-mlperf-inference \
        build-mlperf-llama2 prepare-mlperf-llama2-data run-mlperf-llama2 run-mlperf-llama2-client \
        setup-mlperf-sdxl prepare-mlperf-sdxl-data run-mlperf-sdxl verify-mlperf-sdxl \
        setup-mlperf-t2v prepare-mlperf-t2v-data run-mlperf-t2v verify-mlperf-t2v \
        run-moe-smoke run-moe

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

# --- MLPerf Tiny v1.1 IC benchmark (TFLite, Single Stream) ---

# Create .venv-tiny with TFLite + deps (separate from main .venv — TF 2.14 conflicts with torch)
setup-mlperf-tiny:
	python3 -m venv .venv-tiny
	.venv-tiny/bin/pip install --upgrade pip
	.venv-tiny/bin/pip install -r environments/requirements-mlperf-tiny.txt

# Download CIFAR-10 test set + ResNet TFLite model (~170 MB)
prepare-mlperf-tiny-data:
	bash scripts/data/prepare_mlperf_tiny_data.sh

# Run IC benchmark with TFLite GPU delegate on A100
run-mlperf-tiny:
	bash scripts/launch/mlperf_tiny/run_mlperf_tiny.sh

# Run IC benchmark on CPU (local dev / no GPU)
run-mlperf-tiny-cpu:
	DELEGATE=cpu bash scripts/launch/mlperf_tiny/run_mlperf_tiny.sh

# Check IC accuracy >= 85% from submission result.txt
verify-mlperf-tiny:
	bash scripts/launch/mlperf_tiny/verify_tiny_run.sh

# --- MLPerf Inference v5.0 Llama3.1-8B benchmark (Datacenter, vLLM) ---

# Build Docker image with vLLM + LoadGen + MLCommons reference scripts
build-mlperf-inference:
	docker build -f infra/docker/Dockerfile.mlperf-inference \
		-t ml-netprof/mlperf-inference:latest .

# Download CNN/DM eval dataset + Llama3.1-8B-Instruct model (~15 GB). Requires HF_TOKEN in .env.
prepare-mlperf-inference-data:
	MODEL_DOWNLOAD_PATH=hf bash scripts/data/prepare_mlperf_inference_data.sh

# Run Offline + Server benchmark (vLLM, 2x A100, ~40 min total)
run-mlperf-inference:
	bash scripts/launch/mlperf_inference/run_mlperf_inference.sh

# Run Offline scenario only
run-mlperf-inference-offline:
	SCENARIO=offline bash scripts/launch/mlperf_inference/run_mlperf_inference.sh

# Check ROUGE scores meet 99% of reference targets (ROUGE-1 >= 38.78, ROUGE-2 >= 15.91, ROUGE-L >= 24.50)
verify-mlperf-inference:
	bash scripts/launch/mlperf_inference/verify_inference_run.sh

# --- MLPerf Inference v5.0 Llama2-70B benchmark (Datacenter, vLLM, TP=2) ---

# Build Docker image with vLLM + LoadGen + MLCommons reference scripts (llama2-70b workdir)
build-mlperf-llama2:
	docker build -f infra/docker/Dockerfile.mlperf-llama2 \
		-t ml-netprof/mlperf-llama2:latest .

# Download OpenOrca dataset + Llama-2-70b-chat-hf model (~140 GB). Requires HF_TOKEN in .env.
prepare-mlperf-llama2-data:
	bash scripts/data/prepare_mlperf_llama2_data.sh

# Run vLLM OpenAI API server (Node A — has GPUs + model weights). Blocks until Ctrl-C.
# Node B then runs: SERVER_URL=http://<node-a-ip>:8000 make run-mlperf-llama2-client
run-mlperf-llama2:
	bash scripts/launch/mlperf_llama2/run_server.sh

# Run LoadGen benchmark client (Node B — sends HTTP requests to Node A).
# Requires SERVER_URL env var pointing to the server started by run-mlperf-llama2.
run-mlperf-llama2-client:
	bash scripts/launch/mlperf_llama2/run_client.sh

# --- ml-netprof monitoring agent ---

# Build native macOS binary (for local dev/testing on macOS)
agent-build-darwin:
	CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -C agent -o bin/agent-darwin ./cmd/agent

agent-start:
	./agent/bin/agent-darwin -config agent/configs/agent_default.yaml &
	@echo "Agent running at http://localhost:9100/metrics"

agent-test:
	go test -C agent ./...

# --- MoE pretraining (FSDP, multi-node) ---

# Smoke test: tiny model, fake data, 5 steps. Should finish in <2 minutes.
# Override NODES/INTERNAL_IPS if your cluster differs.
run-moe-smoke:
	NODES="$${NODES:-20.29.43.19 172.212.226.225}" \
	INTERNAL_IPS="$${INTERNAL_IPS:-172.21.0.4 172.21.0.5}" \
	GPUS_PER_NODE=2 \
	SSH_KEY=$${SSH_KEY:-$$HOME/.ssh/gpu-ib_key.pem} \
	SSH_USER=$${SSH_USER:-azureuser} \
	MAX_STEPS=5 FAKE_DATA=1 RUN_TAG=smoke MODEL_SIZE=moe-tiny \
	LOG_EVERY_N_STEPS=1 WARMUP_STEPS=2 \
	bash scripts/launch/mlperf_moe/run_moe_multinode.sh

# Production run: 2-3h, scaled for network profiling
run-moe:
	NODES="$${NODES:-20.29.43.19 172.212.226.225}" \
	INTERNAL_IPS="$${INTERNAL_IPS:-172.21.0.4 172.21.0.5}" \
	GPUS_PER_NODE=2 \
	SSH_KEY=$${SSH_KEY:-$$HOME/.ssh/gpu-ib_key.pem} \
	SSH_USER=$${SSH_USER:-azureuser} \
	MAX_STEPS=$${MAX_STEPS:-2000} FAKE_DATA=1 RUN_TAG=$${RUN_TAG:-prod} \
	bash scripts/launch/mlperf_moe/run_moe_multinode.sh

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

# --- MLPerf SDXL T2I Inference Benchmark ---

SDXL_VENV ?= /data/.venv-sdxl

# Install Python deps for the SDXL inference benchmark (server + client + accuracy)
setup-mlperf-sdxl:
	python3 -m venv $(SDXL_VENV)
	$(SDXL_VENV)/bin/pip install --upgrade pip
	$(SDXL_VENV)/bin/pip install \
		torch diffusers transformers accelerate \
		fastapi uvicorn httpx Pillow \
		open-clip-torch "torchmetrics[image]" \
		pyyaml mlperf_loadgen pycocotools

# Download COCO 2014 val annotations + SDXL-base-1.0 model weights
prepare-mlperf-sdxl-data:
	bash scripts/data/prepare_mlperf_sdxl_data.sh

# Launch 2-node benchmark: starts server on SERVER_IP, runs LoadGen locally.
# Required: SERVER_IP=<gpu-node-ip>
# Optional: SCENARIO=SingleStream|Offline|all  MAX_QUERY_COUNT=100
run-mlperf-sdxl:
	bash scripts/launch/mlperf_sdxl/run_mlperf_sdxl.sh

# Evaluate FID + CLIP scores on saved output images
verify-mlperf-sdxl:
	$(SDXL_VENV)/bin/python3 scripts/launch/mlperf_sdxl/accuracy.py \
		--output-dir /data/mlperf_sdxl/output \
		--annotations /data/mlperf_sdxl/annotations/captions_val2014.json

# --- MLPerf Wan2.2-T2V-A14B Inference Benchmark (v6.0) ---

T2V_VENV ?= /data/.venv-t2v

# Install Python deps for the T2V inference benchmark
setup-mlperf-t2v:
	python3 -m venv $(T2V_VENV)
	$(T2V_VENV)/bin/pip install --upgrade pip
	$(T2V_VENV)/bin/pip install \
		"torch>=2.4" transformers accelerate \
		fastapi uvicorn httpx Pillow imageio imageio-ffmpeg \
		open-clip-torch "torchmetrics[image]" \
		pyyaml pycocotools
	$(T2V_VENV)/bin/pip install git+https://github.com/Wan-AI/Wan2.1.git

# Download COCO 2014 val annotations + Wan2.2-T2V-A14B model weights
# Check https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B for gating status;
# if gated: add HF_TOKEN=hf_... to .env before running.
prepare-mlperf-t2v-data:
	bash scripts/data/prepare_mlperf_t2v_data.sh

# Launch 2-node benchmark: starts server on SERVER_IP, runs LoadGen locally.
# Required: SERVER_IP=<gpu-node-ip>
# Optional: SCENARIO=SingleStream|Offline|all  MAX_QUERY_COUNT=5000
run-mlperf-t2v:
	bash scripts/launch/mlperf_t2v/run_mlperf_t2v.sh

# Evaluate CLIP score on saved output videos
verify-mlperf-t2v:
	$(T2V_VENV)/bin/python3 scripts/launch/mlperf_t2v/accuracy.py \
		--output-dir /data/mlperf_t2v/output \
		--annotations /data/mlperf_t2v/annotations/captions_val2014.json
