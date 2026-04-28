"""
Llama 3.1 8B training with FSDP + optional DiLoCo optimization.

Single script, two modes:
  Baseline:  torchrun --nnodes=2 --nproc_per_node=2 ... train_llama8b.py
  DiLoCo:    torchrun --nnodes=2 --nproc_per_node=2 ... train_llama8b.py --diloco

Based on OpenDiloco's train_fsdp.py and train_diloco_torch.py, but uses
FSDP FULL_SHARD (fits 8B on 80GB A100) and pure PyTorch for DiLoCo
(no hivemind dependency).
"""

import datetime
import math
import os
import time
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

TIMEOUT_NCCL_MINUTES = int(os.environ.get("TIMEOUT_NCCL_MINUTES", "120"))


# ---------------------------------------------------------------------------
# Fake data for smoke tests
# ---------------------------------------------------------------------------
class FakeTokenizedDataset(IterableDataset):
    def __init__(self, seq_length: int, vocab_size: int):
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __iter__(self):
        while True:
            input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
            yield {"input_ids": input_ids, "labels": input_ids.clone()}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Llama 3.1 8B FSDP + DiLoCo training")

    # Model
    p.add_argument(
        "--path-model",
        type=str,
        default=None,
        help="HF model name or local path (if None, creates random-init 8B)",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name or path (defaults to --path-model)",
    )
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument("--attn-implementation", type=str, default="sdpa")
    p.add_argument(
        "--model-size",
        type=str,
        default="8b",
        choices=["8b", "150m"],
        help="Random-init model size when --path-model is not set. "
        "'8b' is the production Llama-3.1-8B config; '150m' is a small "
        "Llama for fast smoke tests (same tokenizer vocab).",
    )

    # Data
    p.add_argument("--dataset-name-or-path", type=str, default="allenai/c4")
    p.add_argument("--fake-data", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)

    # Optimization
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--total-batch-size", type=int, default=512)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--total-steps", type=int, default=88000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    p.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["bf16-mixed", "fp16-mixed", "32-true"],
    )
    p.add_argument("--torch-compile", action="store_true", default=False)

    # DiLoCo
    p.add_argument(
        "--diloco",
        action="store_true",
        help="Enable DiLoCo (per-node FSDP + periodic outer step)",
    )
    p.add_argument(
        "--diloco-local-steps",
        type=int,
        default=500,
        help="Inner optimizer steps between outer syncs",
    )
    p.add_argument("--diloco-outer-lr", type=float, default=0.7)

    # Logging
    p.add_argument("--project", type=str, default="llama8b_diloco")
    p.add_argument("--log-every-n-steps", type=int, default=10)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------
LLAMA_31_8B_CONFIG = dict(
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    vocab_size=128256,
    max_position_embeddings=131072,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    use_cache=False,
)

# Small config for fast smoke tests
LLAMA_150M_CONFIG = dict(
    hidden_size=768,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,
    vocab_size=128256,
    max_position_embeddings=2048,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    use_cache=False,
)

MODEL_CONFIGS = {
    "8b": LLAMA_31_8B_CONFIG,
    "150m": LLAMA_150M_CONFIG,
}


def get_model(args) -> LlamaForCausalLM:
    if args.path_model is not None:
        config = LlamaConfig.from_pretrained(
            args.path_model, attn_implementation=args.attn_implementation
        )
        return LlamaForCausalLM.from_pretrained(args.path_model, config=config)

    config = LlamaConfig(
        **MODEL_CONFIGS[args.model_size],
        attn_implementation=args.attn_implementation,
    )
    return LlamaForCausalLM(config)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def get_dataloader(args, tokenizer, world_size, rank):
    if args.fake_data:
        vocab_size = tokenizer.vocab_size if tokenizer else 1024
        dataset = FakeTokenizedDataset(args.seq_length, vocab_size)
    else:
        ds = load_dataset(args.dataset_name_or_path, "en", streaming=True)

        def tokenize_fn(data):
            return tokenizer(
                data["text"],
                truncation=True,
                max_length=args.seq_length,
                padding="max_length",
            )

        tokenized = ds.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text", "timestamp", "url"],
        )["train"]
        dataset = split_dataset_by_node(tokenized, world_size=world_size, rank=rank)

    if args.fake_data:
        # Fake data already returns input_ids + labels; just stack into batches
        def collate_fn(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
            }
    else:
        # Mask labels via attention_mask, not via pad_token_id. Llama tokenizers
        # have no pad token, so we set pad_token = eos_token below for the
        # tokenizer's padding step. If we used DataCollatorForLanguageModeling
        # it would mask every position where input_ids == pad_token_id, which
        # also silently masks real EOS tokens from the loss. Using
        # attention_mask masks only the actual padded positions.
        def collate_fn(batch):
            input_ids = torch.tensor(
                [b["input_ids"] for b in batch], dtype=torch.long
            )
            attention_mask = torch.tensor(
                [b["attention_mask"] for b in batch], dtype=torch.long
            )
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers if not args.fake_data else 0,
    )


# ---------------------------------------------------------------------------
# DiLoCo outer step
# ---------------------------------------------------------------------------
class DiLoCoState:
    """Holds CPU-resident outer optimizer state for DiLoCo.

    Every rank maintains its own copy of the outer optimizer.  Since all
    ranks see the same averaged pseudo-gradients, their optimizer states
    stay perfectly in sync, producing identical parameter updates.
    """

    def __init__(self, model: FSDP, outer_lr: float):
        # Gather full params to CPU on ALL ranks (rank0_only + writeback
        # is not supported in PyTorch <= 2.4)
        with FSDP.summon_full_params(model, offload_to_cpu=True, writeback=False):
            self.saved_params = [p.data.clone().float() for p in model.parameters()]
            self.outer_params = [
                nn.Parameter(p.clone(), requires_grad=True) for p in self.saved_params
            ]
            self.outer_optimizer = torch.optim.SGD(
                self.outer_params,
                lr=outer_lr,
                momentum=0.9,
                nesterov=True,
            )


def diloco_outer_step(
    model: FSDP,
    state: DiLoCoState,
    outer_gloo_pg: dist.ProcessGroup,
    world_size: int,
    nnodes: int,
):
    """Perform one DiLoCo outer step.

    Must be called by ALL ranks (summon_full_params is collective).
    All ranks compute the same pseudo-gradients, all-reduce them (getting
    identical averaged results), and apply the same outer optimizer step.
    This keeps all ranks in sync so that writeback produces consistent shards.

    Uses a gloo process group for the all-reduce since pseudo-gradients
    live on CPU (NCCL requires CUDA tensors).
    """
    with FSDP.summon_full_params(
        model,
        offload_to_cpu=True,
        writeback=True,
    ):
        # 1. Compute pseudo-gradients and all-reduce across all ranks.
        #    Within a node, ranks have identical params (FSDP keeps them
        #    in sync), so their pseudo-grads are the same.  The all-reduce
        #    sums across world_size ranks; dividing by world_size gives
        #    the correct cross-node average:
        #    SUM / world_size = (local_ws * PG0 + local_ws * PG1) / (nnodes * local_ws)
        #                     = (PG0 + PG1) / nnodes
        for saved, opt_p, model_p in zip(
            state.saved_params, state.outer_params, model.parameters()
        ):
            opt_p.grad = saved - model_p.data.float()
            dist.all_reduce(opt_p.grad, group=outer_gloo_pg)
            opt_p.grad.div_(world_size)

        # 2. Outer optimizer step (SGD + Nesterov)
        state.outer_optimizer.step()
        state.outer_optimizer.zero_grad()

        # 3. Write updated params back to model and refresh snapshot
        for saved, opt_p, model_p in zip(
            state.saved_params, state.outer_params, model.parameters()
        ):
            model_p.data.copy_(opt_p.data.to(model_p.dtype))
            saved.copy_(opt_p.data)

    # Context exit: writeback=True scatters each rank's full params back
    # to FSDP shards.  Since all ranks have identical full params, the
    # resulting shards are consistent.


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    nnodes = world_size // local_world_size

    torch.cuda.set_device(local_rank)

    # ---- Precision --------------------------------------------------------
    if args.precision == "bf16-mixed":
        amp_dtype = torch.bfloat16
    elif args.precision == "fp16-mixed":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    use_amp = amp_dtype is not None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == "fp16-mixed"))

    # ---- Process groups / device mesh -------------------------------------
    if args.diloco:
        mesh = init_device_mesh(
            "cuda",
            (nnodes, local_world_size),
            mesh_dim_names=("replicate", "shard"),
        )
        fsdp_mesh = mesh["shard"]
        # Gloo process group for CPU-based all-reduce of pseudo-gradients
        # (NCCL only works on CUDA tensors; pseudo-grads live on CPU)
        outer_gloo_pg = dist.new_group(backend="gloo")
        if rank == 0:
            print(
                f"DiLoCo: {nnodes} islands, {local_world_size} GPUs/island, "
                f"local_steps={args.diloco_local_steps}"
            )
    else:
        fsdp_mesh = None
        outer_gloo_pg = None
        if rank == 0:
            print(f"Baseline: FSDP across {world_size} GPUs")

    # ---- Batch size / gradient accumulation --------------------------------
    if args.diloco:
        # Each island sees total_batch_size independently
        island_size = local_world_size
    else:
        island_size = world_size

    assert args.total_batch_size % island_size == 0, (
        f"total_batch_size ({args.total_batch_size}) must be divisible by "
        f"island_size ({island_size})"
    )
    per_rank_batch = args.total_batch_size // island_size
    assert per_rank_batch % args.per_device_train_batch_size == 0
    grad_accum_steps = per_rank_batch // args.per_device_train_batch_size

    if rank == 0:
        print(
            f"Batch: total={args.total_batch_size}, per_rank={per_rank_batch}, "
            f"mbs={args.per_device_train_batch_size}, grad_accum={grad_accum_steps}"
        )

    # ---- Tokenizer ---------------------------------------------------------
    tokenizer_name = args.tokenizer or args.path_model
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if tokenizer.pad_token is None:
            # Required for tokenizer-side padding to work; the custom collator
            # below masks loss via attention_mask, not pad_token_id, so this
            # does NOT cause real EOS tokens to be masked from the loss.
            tokenizer.pad_token = tokenizer.eos_token
    elif args.fake_data:
        # Random-init smoke tests don't need a real tokenizer.
        tokenizer = None
    else:
        raise ValueError(
            "Real-data training requires a tokenizer. Pass --tokenizer (or "
            "--path-model). Use --fake-data for a tokenizer-free smoke test. "
            "Example: --tokenizer NousResearch/Meta-Llama-3.1-8B "
            "(meta-llama/Llama-3.1-8B is gated and needs HF_TOKEN in the "
            "container environment)."
        )

    # ---- Dataloader --------------------------------------------------------
    dataloader = get_dataloader(args, tokenizer, world_size, rank)

    # ---- Model -------------------------------------------------------------
    model = get_model(args)

    # Ensure identical initial weights across all nodes
    model = model.to(local_rank)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # Activation checkpointing (before FSDP wrapping)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # FSDP wrapping
    auto_wrap = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )
    mp = (
        MixedPrecision(
            param_dtype=amp_dtype,
            reduce_dtype=amp_dtype,
            buffer_dtype=amp_dtype,
        )
        if use_amp
        else None
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp,
        device_mesh=fsdp_mesh,
        device_id=local_rank,
        use_orig_params=True,
        limit_all_gathers=True,
    )

    if args.torch_compile:
        model = torch.compile(model)

    # ---- Optimizers / scheduler -------------------------------------------
    inner_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )

    # ---- DiLoCo state (CPU outer optimizer) --------------------------------
    diloco_state = None
    if args.diloco:
        diloco_state = DiLoCoState(model, args.diloco_outer_lr)
        if rank == 0:
            n_outer_params = sum(p.numel() for p in diloco_state.outer_params)
            cpu_gb = n_outer_params * 4 * 3 / 1e9  # fp32, 3 copies
            print(
                f"DiLoCo outer optimizer: {n_outer_params / 1e9:.2f}B params, "
                f"~{cpu_gb:.1f} GB CPU per rank (params+momentum+snapshot)"
            )

    # ---- Training loop -----------------------------------------------------
    model.train()
    loss_batch = 0.0
    step_time = time.time()

    if rank == 0:
        print(f"\nStarting training (max_steps={args.max_steps})")

    for step, batch in enumerate(dataloader):
        real_step = (step + 1) // grad_accum_steps
        is_accumulating = bool((step + 1) % grad_accum_steps)

        # Move batch to GPU
        batch = {k: v.to(local_rank) for k, v in batch.items()}

        # Forward + backward (skip gradient sync during accumulation)
        ctx = model.no_sync() if is_accumulating else nullcontext()
        with ctx:
            with torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext():
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps

            loss_batch += loss.detach()
            scaler.scale(loss).backward()

        if is_accumulating:
            continue

        # --- Optimizer step (non-accumulating) ---
        scaler.unscale_(inner_optimizer)
        model.clip_grad_norm_(1.0)
        scaler.step(inner_optimizer)
        scaler.update()
        scheduler.step()
        inner_optimizer.zero_grad()

        # --- DiLoCo outer step ---
        if args.diloco and real_step % args.diloco_local_steps == 0 and real_step > 0:
            t0 = time.time()
            diloco_outer_step(model, diloco_state, outer_gloo_pg, world_size, nnodes)
            if rank == 0:
                print(
                    f"  [DiLoCo] outer step at inner_step={real_step} "
                    f"({time.time() - t0:.1f}s)"
                )

        # --- Logging ---
        if real_step % args.log_every_n_steps == 0:
            if rank == 0:
                elapsed = time.time() - step_time
                tokens_per_sec = (
                    args.seq_length
                    * args.total_batch_size
                    * args.log_every_n_steps
                    / elapsed
                )
                lr_now = scheduler.get_last_lr()[0]
                avg_loss = loss_batch.item() / args.log_every_n_steps
                ppl = math.exp(avg_loss)

                print(
                    f"step {real_step:>6d} | loss {avg_loss:.4f} | "
                    f"ppl {ppl:.2f} | lr {lr_now:.2e} | "
                    f"tok/s {tokens_per_sec:.0f} | "
                    f"elapsed {elapsed:.1f}s"
                )

                step_time = time.time()
            loss_batch = 0.0

        if args.max_steps is not None and real_step >= args.max_steps:
            break

    # Ensure all ranks finish before teardown to avoid NCCL watchdog errors
    dist.barrier()

    if rank == 0:
        print("Training completed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=TIMEOUT_NCCL_MINUTES),
    )

    args = parse_args()

    torch.set_float32_matmul_precision("high")
    if args.torch_compile:
        torch._dynamo.config.suppress_errors = True

    train(args)

    dist.destroy_process_group()
