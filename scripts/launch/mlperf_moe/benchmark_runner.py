#!/usr/bin/env python3
"""
benchmark_runner.py — MLPerf Small LLM MoE Pretraining benchmark runner.

Launches NeMo MoE pretraining via subprocess inside the NeMo container,
logs metrics to W&B (step, loss, tokens/sec, step_time), and saves results
in MLPerf submission format.

Usage:
    python scripts/launch/mlperf_moe/benchmark_runner.py \
        --config scripts/launch/mlperf_moe/config/moe_4gpu.yaml \
        --log-dir logs/mlperf_moe_run \
        --results-dir results/mlperf_moe
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_nemo_command(config: dict, config_path: str) -> list[str]:
    """Build the NeMo pretraining command from config.

    Passes the full YAML via Hydra --config-path/--config-name so that
    all top-level keys (including exp_manager) are present in cfg.
    """
    config_file = Path(config_path).resolve()
    cmd = [
        "python3", "/opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py",
        f"--config-path={config_file.parent}",
        f"--config-name={config_file.name}",
    ]
    return cmd


def parse_nemo_log_line(line: str) -> dict | None:
    """Extract step metrics from NeMo training log lines.

    NeMo logs lines like:
      Epoch 0:  10%| 20/200 [16:40, reduced_train_loss=9.8, global_step=19, ...]
    """
    metrics = {}

    # Extract global_step
    step_match = re.search(r"global_step=(\d+)", line)
    if not step_match:
        return None
    metrics["step"] = int(step_match.group(1))

    # Extract train loss
    loss_match = re.search(r"reduced_train_loss=([\d.]+)", line)
    if loss_match:
        metrics["train/loss"] = float(loss_match.group(1))

    # Extract step timing
    timing_match = re.search(r"train_step_timing in s=([\d.]+)", line)
    if timing_match:
        metrics["train/step_time"] = float(timing_match.group(1))

    # Extract val_loss if present
    val_match = re.search(r"val_loss=([\d.]+)", line)
    if val_match:
        metrics["val/loss"] = float(val_match.group(1))

    return metrics


def run_training(config: dict, log_dir: Path, config_path: str) -> list[dict]:
    """Run NeMo pretraining and capture metrics from stdout."""
    cmd = build_nemo_command(config, config_path)
    all_metrics = []

    print(f"\nLaunching NeMo MoE pretraining...")
    print(f"  Command: {' '.join(cmd[:3])} ...")

    log_file = log_dir / "run.log"

    try:
        wandb_module = None
        try:
            import wandb
            wandb_module = wandb
        except ImportError:
            print("  WARNING: wandb not installed; metrics logged to CSV only.")

        # Initialize W&B if available
        wandb_run = None
        if wandb_module:
            exp = config.get("exp_manager", {})
            wandb_kwargs = exp.get("wandb_logger_kwargs", {})
            try:
                wandb_run = wandb_module.init(
                    project=wandb_kwargs.get("project", "ml-netprof"),
                    name=wandb_kwargs.get("name", config.get("run_name", "mlperf-moe")),
                    config=config,
                    tags=["mlperf", "moe", "pretraining"],
                )
            except Exception as e:
                print(f"  WARNING: W&B init failed ({e}); continuing without W&B.")

        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)

                metrics = parse_nemo_log_line(line)
                if metrics and "train/loss" in metrics:
                    # Compute tokens/sec if step_time available
                    if "train/step_time" in metrics:
                        tokens_per_step = (
                            config["model"]["global_batch_size"]
                            * config["model"]["seq_length"]
                        )
                        metrics["train/tokens_per_sec"] = (
                            tokens_per_step / metrics["train/step_time"]
                        )

                    all_metrics.append(metrics)

                    # Log to W&B
                    if wandb_run:
                        wandb_module.log(metrics, step=metrics["step"])

            proc.wait()

        if wandb_run:
            wandb_module.finish()

        if proc.returncode != 0:
            print(f"\nWARNING: NeMo exited with code {proc.returncode}")

    except FileNotFoundError:
        print("\nERROR: NeMo not found. Run inside the NeMo container:")
        print("  docker run --gpus all -v /data:/data -v $(pwd):/workspace \\")
        print("    nvcr.io/nvidia/nemo:24.12-rc0 \\")
        print("    python scripts/launch/mlperf_moe/benchmark_runner.py \\")
        print(f"      --config {config.get('_config_path', 'config/moe_4gpu.yaml')} \\")
        print(f"      --log-dir {log_dir}")
        return all_metrics

    return all_metrics


def save_results(metrics: list[dict], config: dict, results_dir: Path) -> None:
    """Save metrics to CSV and summary JSON in MLPerf submission format."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Write per-step CSV
    csv_path = results_dir / "metrics.csv"
    if metrics:
        fieldnames = ["step", "train/loss", "train/step_time", "train/tokens_per_sec", "val/loss"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in metrics:
                writer.writerow(row)
        print(f"  Metrics CSV: {csv_path}")

    # Write summary JSON
    summary = {
        "benchmark": "small_llm_moe_pretraining",
        "model": "GPT-OSS-20B",
        "num_experts": config["model"]["num_moe_experts"],
        "moe_router_topk": config["model"]["moe_router_topk"],
        "num_layers": config["model"]["num_layers"],
        "hidden_size": config["model"]["hidden_size"],
        "seq_length": config["model"]["seq_length"],
        "global_batch_size": config["model"]["global_batch_size"],
        "max_steps": config["trainer"]["max_steps"],
        "target_val_loss": config.get("convergence", {}).get("target_val_loss", 3.34),
        "total_steps_completed": len(metrics),
    }

    if metrics:
        summary["final_train_loss"] = metrics[-1].get("train/loss")
        # Find last val_loss
        val_losses = [m["val/loss"] for m in metrics if "val/loss" in m]
        if val_losses:
            summary["final_val_loss"] = val_losses[-1]
            summary["converged"] = val_losses[-1] <= summary["target_val_loss"]
        else:
            summary["final_val_loss"] = None
            summary["converged"] = False

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON: {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="MLPerf MoE pretraining benchmark runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log-dir", required=True, help="Directory for run logs")
    parser.add_argument("--results-dir", default="results/mlperf_moe", help="Directory for results output")
    args = parser.parse_args()

    config = load_config(args.config)
    config["_config_path"] = args.config
    log_dir = Path(args.log_dir)
    results_dir = Path(args.results_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=== MLPerf Small LLM MoE Pretraining Benchmark ===")
    print(f"  Config:      {args.config}")
    print(f"  Model:       GPT-OSS-20B ({config['model']['num_moe_experts']} experts, top-{config['model']['moe_router_topk']})")
    print(f"  Layers:      {config['model']['num_layers']}")
    print(f"  Hidden:      {config['model']['hidden_size']}")
    print(f"  Seq length:  {config['model']['seq_length']}")
    print(f"  GBS:         {config['model']['global_batch_size']}")
    print(f"  Max steps:   {config['trainer']['max_steps']}")
    print(f"  GPUs:        {config['trainer']['devices']}")
    print(f"  EP:          {config['model']['expert_model_parallel_size']}")
    print(f"  Target:      val_loss <= {config.get('convergence', {}).get('target_val_loss', 3.34)}")
    print(f"  Log dir:     {log_dir}")
    print(f"  Results dir: {results_dir}")

    # Run training
    metrics = run_training(config, log_dir, args.config)

    # Save results
    print("\nSaving results...")
    save_results(metrics, config, results_dir)

    # Print summary
    print("\n=== Summary ===")
    if metrics:
        print(f"  Steps completed : {len(metrics)}")
        print(f"  Final train loss: {metrics[-1].get('train/loss', 'N/A')}")
        val_losses = [m["val/loss"] for m in metrics if "val/loss" in m]
        if val_losses:
            target = config.get("convergence", {}).get("target_val_loss", 3.34)
            print(f"  Final val loss  : {val_losses[-1]}")
            converged = val_losses[-1] <= target
            print(f"  Converged       : {'YES' if converged else 'NO'} (target: {target})")
        else:
            print("  Final val loss  : N/A (no validation steps completed)")
    else:
        print("  No metrics captured. Check run.log for errors.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
