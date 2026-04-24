#!/usr/bin/env python3
"""
benchmark_runner.py — MLPerf Text-to-Image (Flux.1-schnell) training benchmark.

Runs Flux.1-schnell fine-tuning on CC12M with FSDP, evaluates validation loss
on COCO-2014, and logs metrics to W&B. Results are saved in MLPerf submission
format to results/mlperf_t2i/.

Usage:
    torchrun --nproc_per_node=2 --nnodes=2 --node_rank=$RANK \
        --master_addr=$MASTER --master_port=29500 \
        scripts/launch/mlperf_t2i/benchmark_runner.py \
        --config scripts/launch/mlperf_t2i/config/t2i_4gpu.yaml

Single-node dev (2 GPUs):
    torchrun --nproc_per_node=2 \
        scripts/launch/mlperf_t2i/benchmark_runner.py \
        --config scripts/launch/mlperf_t2i/config/t2i_4gpu.yaml
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def log_print(msg: str) -> None:
    if is_main_process():
        print(msg, flush=True)


def setup_distributed(cfg: dict) -> int:
    """Initialize torch.distributed and return global rank."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank
    else:
        # Single-GPU fallback
        torch.cuda.set_device(0)
        return 0


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def build_model(cfg: dict, device: torch.device):
    """Load Flux.1-schnell UNet (diffusion transformer) and wrap with FSDP."""
    from diffusers import FluxPipeline

    model_cfg = cfg["model"]
    pretrained_path = model_cfg["pretrained_path"]

    log_print(f"  Loading Flux.1-schnell from {pretrained_path} ...")
    pipe = FluxPipeline.from_pretrained(
        pretrained_path,
        torch_dtype=torch.bfloat16 if cfg["training"]["mixed_precision"] == "bf16" else torch.float32,
    )

    # Extract the transformer (trainable) and freeze encoders
    transformer = pipe.transformer
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    vae = pipe.vae
    scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2

    # Freeze encoders and VAE
    for param in text_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder_2.parameters():
        param.requires_grad = False
    for param in vae.parameters():
        param.requires_grad = False

    text_encoder.to(device).eval()
    text_encoder_2.to(device).eval()
    vae.to(device).eval()

    # Enable gradient checkpointing on transformer
    if cfg["training"].get("gradient_checkpointing", False):
        transformer.enable_gradient_checkpointing()

    transformer.to(device)
    transformer.train()

    # Wrap transformer with FSDP if distributed
    if dist.is_initialized() and dist.get_world_size() > 1:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        transformer = FSDP(transformer, mixed_precision=mp_policy, use_orig_params=True)

    log_print(f"  Transformer params: {sum(p.numel() for p in transformer.parameters()):,}")
    log_print(f"  Trainable params:   {sum(p.numel() for p in transformer.parameters() if p.requires_grad):,}")

    return transformer, text_encoder, text_encoder_2, vae, scheduler, tokenizer, tokenizer_2


def build_dataset(cfg: dict, split: str):
    """Build a simple image-caption dataset from a directory of images."""
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image
    import glob

    data_cfg = cfg["data"]
    if split == "train":
        data_dir = data_cfg["train_data_dir"]
    else:
        data_dir = data_cfg["val_data_dir"]

    resolution = cfg["model"]["image_resolution"]

    class ImageCaptionDataset(Dataset):
        def __init__(self, root_dir, resolution):
            self.root_dir = Path(root_dir)
            # Look for images with associated caption files
            self.image_files = sorted(
                glob.glob(str(self.root_dir / "**" / "*.jpg"), recursive=True)
                + glob.glob(str(self.root_dir / "**" / "*.png"), recursive=True)
            )
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)

            # Load caption from .txt sidecar file
            caption_path = Path(img_path).with_suffix(".txt")
            if caption_path.exists():
                caption = caption_path.read_text().strip()
            else:
                caption = ""

            return {"pixel_values": image, "caption": caption}

    dataset = ImageCaptionDataset(data_dir, resolution)
    log_print(f"  {split} dataset: {len(dataset)} samples from {data_dir}")
    return dataset


def build_dataloader(cfg: dict, dataset, split: str):
    from torch.utils.data import DataLoader, DistributedSampler

    batch_size = cfg["training"]["batch_size_per_gpu"]
    num_workers = cfg["data"].get("num_workers", 4)
    pin_memory = cfg["data"].get("pin_memory", True)

    sampler = None
    shuffle = split == "train"
    if dist.is_initialized() and dist.get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )


def encode_text(batch_captions, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device):
    """Encode captions using CLIP and T5 text encoders."""
    with torch.no_grad():
        # CLIP encoding
        clip_inputs = tokenizer(
            batch_captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        clip_embeds = text_encoder(**clip_inputs).pooler_output

        # T5 encoding
        t5_inputs = tokenizer_2(
            batch_captions,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        t5_embeds = text_encoder_2(**t5_inputs).last_hidden_state

    return clip_embeds, t5_embeds


def encode_images(pixel_values, vae):
    """Encode images to latent space using frozen VAE."""
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents


def compute_loss(transformer, latents, clip_embeds, t5_embeds, scheduler, timestep_idx=None):
    """Compute MSE loss over latents at a given (or random) timestep."""
    bsz = latents.shape[0]
    device = latents.device

    if timestep_idx is not None:
        # Fixed timestep for validation: t = timestep_idx / 8
        t = torch.full((bsz,), timestep_idx / 8.0, device=device, dtype=latents.dtype)
    else:
        # Random timestep for training
        t = torch.rand(bsz, device=device, dtype=latents.dtype)

    # Sample noise
    noise = torch.randn_like(latents)

    # Interpolate: noisy_latents = (1-t) * latents + t * noise  (rectified flow)
    t_expand = t.view(-1, 1, 1, 1)
    noisy_latents = (1.0 - t_expand) * latents + t_expand * noise

    # Pack latents for Flux transformer input
    img_ids = _build_img_ids(latents, device)
    txt_ids = torch.zeros(t5_embeds.shape[0], t5_embeds.shape[1], 3, device=device, dtype=latents.dtype)

    # Predict velocity
    model_pred = transformer(
        hidden_states=noisy_latents,
        timestep=t,
        encoder_hidden_states=t5_embeds,
        pooled_projections=clip_embeds,
        txt_ids=txt_ids,
        img_ids=img_ids,
        return_dict=False,
    )[0]

    # Target: noise - latents (velocity in rectified flow)
    target = noise - latents

    loss = torch.nn.functional.mse_loss(model_pred, target)
    return loss


def _build_img_ids(latents, device):
    """Build image position IDs for the Flux transformer."""
    bsz, ch, h, w = latents.shape
    img_ids = torch.zeros(h, w, 3, device=device, dtype=latents.dtype)
    img_ids[..., 1] = torch.arange(h, device=device, dtype=latents.dtype)[:, None]
    img_ids[..., 2] = torch.arange(w, device=device, dtype=latents.dtype)[None, :]
    img_ids = img_ids.reshape(1, h * w, 3).expand(bsz, -1, -1)
    return img_ids


@torch.no_grad()
def run_validation(
    transformer, text_encoder, text_encoder_2, vae, scheduler,
    tokenizer, tokenizer_2, val_loader, cfg, device
):
    """Run validation: compute MSE loss averaged over 8 equidistant timesteps."""
    transformer.eval()
    eval_cfg = cfg["evaluation"]
    n_timesteps = eval_cfg["val_timesteps"]

    total_loss = 0.0
    total_samples = 0

    for timestep_idx in range(n_timesteps):
        ts_loss = 0.0
        ts_count = 0

        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            captions = batch["caption"]

            clip_embeds, t5_embeds = encode_text(
                captions, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device
            )
            latents = encode_images(pixel_values, vae)

            loss = compute_loss(transformer, latents, clip_embeds, t5_embeds, scheduler, timestep_idx)
            ts_loss += loss.item() * pixel_values.shape[0]
            ts_count += pixel_values.shape[0]

        ts_loss /= max(ts_count, 1)
        total_loss += ts_loss
        total_samples = ts_count

    avg_loss = total_loss / n_timesteps

    # Reduce across ranks
    if dist.is_initialized() and dist.get_world_size() > 1:
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

    transformer.train()
    return avg_loss, total_samples


def write_result(results_dir: Path, val_loss: float, step: int, converged: bool, cfg: dict) -> None:
    """Write MLPerf submission result file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "benchmark": "text_to_image",
        "model": "flux1-schnell",
        "dataset": "cc12m_coco2014",
        "hardware": f"{cfg['hardware']['num_nodes']}x{cfg['hardware']['gpus_per_node']} A100 80GB",
        "precision": cfg["training"]["mixed_precision"],
        "parallelism": cfg["distributed"]["strategy"],
        "final_val_loss": val_loss,
        "convergence_target": cfg["evaluation"]["convergence_target"],
        "converged": converged,
        "final_step": step,
    }

    result_path = results_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    summary_path = results_dir / "result.txt"
    status = "VALID" if converged else "INVALID"
    summary_path.write_text(
        "================================================\n"
        "MLPerf Results Summary\n"
        "================================================\n"
        f"Benchmark       : text_to_image\n"
        f"Model           : Flux.1-schnell\n"
        f"Scenario        : Training\n"
        f"Val loss        : {val_loss:.6f}\n"
        f"Target          : {cfg['evaluation']['convergence_target']}\n"
        f"Steps completed : {step}\n"
        f"Result is       : {status}\n"
        "================================================\n"
    )
    log_print(f"  Results written to {results_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="MLPerf T2I (Flux.1-schnell) training benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rank = setup_distributed(cfg)
    device = torch.device("cuda")

    log_print("=== MLPerf Text-to-Image Benchmark (Flux.1-schnell) ===")
    log_print(f"  Config:     {args.config}")
    log_print(f"  Nodes:      {cfg['hardware']['num_nodes']}")
    log_print(f"  GPUs/node:  {cfg['hardware']['gpus_per_node']}")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    log_print(f"  World size: {world_size}")
    log_print(f"  Precision:  {cfg['training']['mixed_precision']}")
    log_print(f"  Strategy:   {cfg['distributed']['strategy']}")

    # Build model
    log_print("\nLoading model...")
    transformer, text_encoder, text_encoder_2, vae, scheduler, tokenizer, tokenizer_2 = build_model(cfg, device)

    # Build datasets and dataloaders
    log_print("\nLoading datasets...")
    train_dataset = build_dataset(cfg, "train")
    val_dataset = build_dataset(cfg, "val")
    train_loader = build_dataloader(cfg, train_dataset, "train")
    val_loader = build_dataloader(cfg, val_dataset, "val")

    # Optimizer
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["eps"],
    )

    # LR scheduler
    lr_cfg = cfg["lr_schedule"]
    warmup_steps = lr_cfg["warmup_steps"]
    max_steps = cfg["training"]["max_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return max(lr_cfg["min_lr"] / opt_cfg["lr"], 0.5 * (1.0 + math.cos(math.pi * progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # W&B init
    wandb_run = None
    wandb_cfg = cfg["logging"].get("wandb", {})
    if wandb_cfg.get("enabled", False) and is_main_process():
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_cfg["project"],
                name=wandb_cfg["name"],
                config=cfg,
            )
            log_print(f"  W&B run: {wandb_run.url}")
        except Exception as e:
            log_print(f"  W&B init failed ({e}); continuing without W&B")

    # Training loop
    log_print("\n=== Starting Training ===")
    grad_accum = cfg["training"]["gradient_accumulation_steps"]
    log_every = cfg["logging"]["log_every_n_steps"]
    eval_every_samples = cfg["evaluation"]["eval_every_n_samples"]
    convergence_target = cfg["evaluation"]["convergence_target"]
    batch_size = cfg["training"]["batch_size_per_gpu"]
    samples_per_step = batch_size * world_size

    results_dir = Path(cfg["output"]["results_dir"])
    best_val_loss = float("inf")
    converged = False
    global_step = 0
    samples_since_eval = 0
    epoch = 0

    while global_step < max_steps and not converged:
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            if global_step >= max_steps or converged:
                break

            step_start = time.perf_counter()

            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            captions = batch["caption"]

            # Encode text and images (frozen)
            clip_embeds, t5_embeds = encode_text(
                captions, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device
            )
            latents = encode_images(pixel_values, vae)

            # Forward pass
            loss = compute_loss(transformer, latents, clip_embeds, t5_embeds, scheduler)
            loss = loss / grad_accum

            # Backward pass
            loss.backward()

            samples_since_eval += pixel_values.shape[0] * world_size

            # Gradient accumulation step
            if (global_step + 1) % grad_accum == 0 or global_step == max_steps - 1:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            step_time = time.perf_counter() - step_start
            img_per_sec = pixel_values.shape[0] / step_time

            # Log
            if global_step % log_every == 0 and is_main_process():
                lr = optimizer.param_groups[0]["lr"]
                log_print(
                    f"Step {global_step}/{max_steps} | "
                    f"loss={loss.item() * grad_accum:.4f} | "
                    f"lr={lr:.2e} | "
                    f"img_per_sec={img_per_sec:.1f} | "
                    f"step_time={step_time:.2f}s"
                )
                if wandb_run:
                    wandb_run.log({
                        "train/loss": loss.item() * grad_accum,
                        "train/lr": lr,
                        "train/img_per_sec": img_per_sec,
                        "train/step_time": step_time,
                        "train/step": global_step,
                    }, step=global_step)

            # Evaluation
            if samples_since_eval >= eval_every_samples:
                samples_since_eval = 0
                log_print(f"\n--- Validation at step {global_step} ---")
                val_loss, val_count = run_validation(
                    transformer, text_encoder, text_encoder_2, vae, scheduler,
                    tokenizer, tokenizer_2, val_loader, cfg, device,
                )
                log_print(f"  val_loss={val_loss:.6f} (target={convergence_target})")

                if wandb_run:
                    wandb_run.log({
                        "eval/val_loss": val_loss,
                        "eval/step": global_step,
                    }, step=global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                if val_loss <= convergence_target:
                    log_print(f"  CONVERGED at step {global_step}: val_loss={val_loss:.6f} <= {convergence_target}")
                    converged = True

                # Write intermediate results
                if is_main_process():
                    write_result(results_dir, val_loss, global_step, converged, cfg)

        epoch += 1

    # Final evaluation if not already done
    if not converged and samples_since_eval > 0:
        log_print(f"\n--- Final Validation at step {global_step} ---")
        val_loss, _ = run_validation(
            transformer, text_encoder, text_encoder_2, vae, scheduler,
            tokenizer, tokenizer_2, val_loader, cfg, device,
        )
        log_print(f"  val_loss={val_loss:.6f} (target={convergence_target})")

        if val_loss <= convergence_target:
            converged = True

        if is_main_process():
            write_result(results_dir, val_loss, global_step, converged, cfg)

    # Summary
    log_print("\n=== Summary ===")
    log_print(f"  Steps completed : {global_step}")
    log_print(f"  Best val loss   : {best_val_loss:.6f}")
    log_print(f"  Target          : {convergence_target}")
    log_print(f"  Converged       : {converged}")

    if wandb_run:
        wandb_run.finish()

    cleanup_distributed()
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
