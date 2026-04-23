.PHONY: setup start-wandb stop-wandb prepare-data run-mac run-linux launch-job launch-agent \
        prepare-mlperf-data pull-nemo run-mlperf verify-mlperf \
        setup-mlperf-tiny prepare-mlperf-tiny-data run-mlperf-tiny run-mlperf-tiny-cpu verify-mlperf-tiny \
        build-mlperf-inference prepare-mlperf-inference-data run-mlperf-inference run-mlperf-inference-offline verify-mlperf-inference \
        build-mlperf-llama2 prepare-mlperf-llama2-data run-mlperf-llama2 \
        setup-mlperf-moe prepare-mlperf-moe-data run-mlperf-moe verify-mlperf-moe \
        setup-mlperf-resnet prepare-mlperf-resnet-data run-mlperf-resnet verify-mlperf-resnet

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

# Run Server performance benchmark (vLLM, 2x A100)
run-mlperf-llama2:
	bash scripts/launch/mlperf_llama2/run_mlperf_llama2.sh

# --- MLPerf Small LLM MoE Pretraining (GPT-OSS-20B) ---

# Create .venv-moe with NeMo deps (or use nvcr.io/nvidia/nemo:24.12-rc0 container directly)
setup-mlperf-moe:
	python3 -m venv .venv-moe
	.venv-moe/bin/pip install --upgrade pip
	.venv-moe/bin/pip install pyyaml wandb
	@echo ""
	@echo "NOTE: NeMo is best run inside the official container."
	@echo "  docker pull nvcr.io/nvidia/nemo:24.12-rc0"
	@echo "  Or install NeMo in venv: .venv-moe/bin/pip install nemo_toolkit[all]"

# Download/prepare C4 dataset for MoE pretraining (~80 GB, same as MLPerf Llama3.1 8B).
# Reuses prepare_c4_mlperf.sh — skip if you already ran make prepare-mlperf-data.
prepare-mlperf-moe-data:
	bash scripts/data/prepare_c4_mlperf.sh

# Run MoE pretraining benchmark (4 GPUs, EP=4, 200 steps for infra profiling)
run-mlperf-moe:
	@mkdir -p logs results/mlperf_moe
	python scripts/launch/mlperf_moe/benchmark_runner.py \
		--config scripts/launch/mlperf_moe/config/moe_4gpu.yaml \
		--log-dir logs/mlperf_moe_$$(date +%Y%m%d_%H%M%S) \
		--results-dir results/mlperf_moe

# Check whether MoE run met val_loss <= 3.34 convergence target
verify-mlperf-moe:
	bash scripts/launch/mlperf_moe/verify_run.sh

# --- MLPerf Training ResNet-50 v1.5 (TF2, ImageNet, multi-GPU) ---

# Create .venv-resnet with TF 2.14 + deps (separate from main .venv — TF conflicts with torch)
setup-mlperf-resnet:
	python3 -m venv .venv-resnet
	.venv-resnet/bin/pip install --upgrade pip
	.venv-resnet/bin/pip install tensorflow==2.14.0 wandb pyyaml Pillow

# Download ImageNet ILSVRC2012 and convert to TFRecord format (~150 GB).
# ImageNet requires manual download (academic license). See RUNBOOK for details.
prepare-mlperf-resnet-data:
	@echo "=== ImageNet ILSVRC2012 Data Preparation ==="
	@echo ""
	@echo "ImageNet requires manual download (academic license)."
	@echo ""
	@echo "1. Register at https://image-net.org"
	@echo "2. Download ILSVRC2012_img_train.tar (~138 GB) and ILSVRC2012_img_val.tar (~6.3 GB)"
	@echo "3. Extract:"
	@echo "     mkdir -p data/imagenet/raw/{train,val}"
	@echo "     tar -xf ILSVRC2012_img_train.tar -C data/imagenet/raw/train/"
	@echo "     tar -xf ILSVRC2012_img_val.tar -C data/imagenet/raw/val/"
	@echo "4. Convert to TFRecord format:"
	@echo "     source .venv-resnet/bin/activate"
	@echo "     python scripts/data/imagenet_to_tfrecord.py \\"
	@echo "       --raw-dir data/imagenet/raw --output-dir data/imagenet/tfrecord"
	@echo ""
	@echo "See scripts/launch/mlperf_resnet/RUNBOOK.md for full details."

# Run ResNet-50 v1.5 training benchmark (4 GPUs, ~6-12 hours)
run-mlperf-resnet:
	bash scripts/launch/mlperf_resnet/run_mlperf_resnet.sh

# Check top-1 accuracy >= 75.9% from result.txt
verify-mlperf-resnet:
	bash scripts/launch/mlperf_resnet/verify_run.sh

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
