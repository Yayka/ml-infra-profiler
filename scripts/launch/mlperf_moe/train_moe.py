"""
MoE pretraining with FSDP — pure PyTorch.

Random-init Mixtral-style MoE inspired by MLPerf small_llm_moe_pretraining
(MoE GPT, top-k expert routing). Runs on a 2-node x 2-GPU A100 cluster via
torchrun + FSDP FULL_SHARD; experts and dense layers are sharded the same way,
so every step generates all-gather (params) and reduce-scatter (grads) traffic
across the IB/Ethernet interconnect — which is what we want to profile.

Smoke test:   --fake-data --max-steps 5
Production:   --max-steps 2000  (2-3h on 2x2 A100)
"""

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoTokenizer,
    MixtralConfig,
    MixtralForCausalLM,
    get_cosine_schedule_with_warmup,
)
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

TIMEOUT_NCCL_MINUTES = int(os.environ.get("TIMEOUT_NCCL_MINUTES", "60"))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class FakeTokenizedDataset(IterableDataset):
    def __init__(self, seq_length: int, vocab_size: int, seed: int = 0):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + (dist.get_rank() if dist.is_initialized() else 0))
        while True:
            ids = torch.randint(0, self.vocab_size, (self.seq_length,), generator=g)
            yield {"input_ids": ids, "labels": ids.clone()}


class StreamingC4Dataset(IterableDataset):
    """C4 streamed from HF datasets, tokenized on the fly. Per-rank shard."""

    def __init__(self, tokenizer, seq_length: int, dataset_name: str = "allenai/c4"):
        from datasets import load_dataset
        from datasets.distributed import split_dataset_by_node

        self.tokenizer = tokenizer
        self.seq_length = seq_length
        ds = load_dataset(dataset_name, "en", split="train", streaming=True)
        if dist.is_initialized():
            ds = split_dataset_by_node(
                ds,
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
            )
        self.ds = ds.shuffle(seed=42, buffer_size=10_000)

    def __iter__(self):
        buffer: list[int] = []
        for sample in self.ds:
            ids = self.tokenizer(
                sample["text"],
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            buffer.extend(ids)
            buffer.append(self.tokenizer.eos_token_id or 2)
            while len(buffer) >= self.seq_length:
                chunk = buffer[: self.seq_length]
                buffer = buffer[self.seq_length :]
                t = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": t, "labels": t.clone()}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Scaled-down Mixtral-style MoE for 4xA100 80GB (random init, no checkpoint).
# Total params ~3.7B, active per token ~1.3B (top-2 of 8 experts).
MOE_DEFAULT_CONFIG = dict(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=12,
    num_attention_heads=16,
    num_key_value_heads=4,
    max_position_embeddings=4096,
    rms_norm_eps=1e-5,
    rope_theta=1000000.0,
    sliding_window=None,
    num_local_experts=8,
    num_experts_per_tok=2,
    output_router_logits=False,  # save memory; we don't use load-balancing aux loss in profiling run
    use_cache=False,
    tie_word_embeddings=False,
)

# Smaller config for fast smoke tests (~150M total).
MOE_TINY_CONFIG = dict(
    vocab_size=32000,
    hidden_size=512,
    intermediate_size=1024,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=2048,
    rms_norm_eps=1e-5,
    rope_theta=1000000.0,
    sliding_window=None,
    num_local_experts=8,
    num_experts_per_tok=2,
    output_router_logits=False,
    use_cache=False,
    tie_word_embeddings=False,
)

MODEL_CONFIGS = {
    "moe-3b": MOE_DEFAULT_CONFIG,
    "moe-tiny": MOE_TINY_CONFIG,
}


def build_model(args) -> MixtralForCausalLM:
    cfg = MixtralConfig(
        **MODEL_CONFIGS[args.model_size],
        attn_implementation=args.attn_implementation,
    )
    cfg.max_position_embeddings = max(cfg.max_position_embeddings, args.seq_length)
    return MixtralForCausalLM(cfg)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def setup_dist():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    import datetime as _dt

    dist.init_process_group(
        backend="nccl",
        timeout=_dt.timedelta(minutes=TIMEOUT_NCCL_MINUTES),
    )
    return rank, local_rank, world_size


def log(msg: str, rank: int):
    if rank == 0:
        print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="MoE pretraining with FSDP")
    p.add_argument(
        "--model-size",
        type=str,
        default="moe-3b",
        choices=list(MODEL_CONFIGS.keys()),
    )
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Only used when --fake-data is off",
    )
    p.add_argument("--fake-data", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--total-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    p.add_argument("--log-every-n-steps", type=int, default=5)
    p.add_argument("--metrics-csv", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_dist()
    device = torch.device(f"cuda:{local_rank}")

    log(f"=== MoE FSDP training ===", rank)
    log(f"  world_size:        {world_size}", rank)
    log(f"  model_size:        {args.model_size}", rank)
    log(f"  seq_length:        {args.seq_length}", rank)
    log(f"  per_device_batch:  {args.per_device_batch_size}", rank)
    log(f"  total_batch_size:  {args.total_batch_size}", rank)
    log(f"  fake_data:         {args.fake_data}", rank)
    log(f"  total_steps:       {args.total_steps}", rank)
    log(f"  max_steps:         {args.max_steps}", rank)

    # ---- Model ----
    log("Building model...", rank)
    t0 = time.perf_counter()
    model = build_model(args)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  total params: {n_params / 1e9:.2f}B  (built in {time.perf_counter() - t0:.1f}s)", rank)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # ---- FSDP ----
    auto_wrap_policy = lambda *a, **kw: transformer_auto_wrap_policy(
        *a, transformer_layer_cls={MixtralDecoderLayer}, **kw
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    log("Wrapping with FSDP FULL_SHARD...", rank)
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        use_orig_params=False,
        limit_all_gathers=True,
    )

    # ---- Data ----
    log("Building dataloader...", rank)
    if args.fake_data:
        # Use the model's own vocab size; safe regardless of tokenizer.
        ds = FakeTokenizedDataset(
            seq_length=args.seq_length,
            vocab_size=MODEL_CONFIGS[args.model_size]["vocab_size"],
            seed=42,
        )
    else:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        ds = StreamingC4Dataset(tokenizer=tok, seq_length=args.seq_length)

    loader = DataLoader(
        ds,
        batch_size=args.per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    data_iter = iter(loader)

    # ---- Optimizer ----
    grad_accum = args.total_batch_size // (args.per_device_batch_size * world_size)
    if grad_accum < 1:
        grad_accum = 1
    log(f"  grad_accum: {grad_accum}", rank)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )

    # ---- Train ----
    max_steps = args.max_steps or args.total_steps
    log(f"Starting training: {max_steps} steps", rank)
    csv_file = None
    if rank == 0 and args.metrics_csv:
        os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)
        csv_file = open(args.metrics_csv, "w", buffering=1)
        csv_file.write("step,loss,step_time_s,tokens_per_sec\n")

    model.train()
    step_start = time.perf_counter()
    accum_loss = 0.0
    micro = 0

    for step in range(1, max_steps + 1):
        # gradient accumulation loop
        accum_loss = 0.0
        for micro in range(grad_accum):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with model.no_sync() if micro < grad_accum - 1 else nullcontext():
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss / grad_accum
                loss.backward()
            accum_loss += loss.item()

        model.clip_grad_norm_(1.0)
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        step_time = time.perf_counter() - step_start
        step_start = time.perf_counter()

        tokens_this_step = (
            args.per_device_batch_size * args.seq_length * grad_accum * world_size
        )
        tps = tokens_this_step / step_time

        if step % args.log_every_n_steps == 0 or step == 1:
            log(
                f"step={step:5d}  loss={accum_loss:.4f}  step_time={step_time:.2f}s"
                f"  tokens/sec={tps:.0f}  lr={sched.get_last_lr()[0]:.2e}",
                rank,
            )
            if csv_file is not None:
                csv_file.write(f"{step},{accum_loss:.6f},{step_time:.4f},{tps:.1f}\n")

    if csv_file is not None:
        csv_file.close()

    log("Training complete.", rank)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
