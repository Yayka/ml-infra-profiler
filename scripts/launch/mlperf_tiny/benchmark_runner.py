#!/usr/bin/env python3
"""
benchmark_runner.py — MLPerf Tiny IC benchmark runner (TFLite, Single Stream).

Runs performance and accuracy phases against the ResNet v1 int8 TFLite model
on CIFAR-10. Writes result files in MLPerf Tiny v1.1 submission format.

No W&B — submission files are the source of truth.

Usage:
    python scripts/launch/mlperf_tiny/benchmark_runner.py \
        --config scripts/launch/mlperf_tiny/config/ic_resnet_gpu.yaml \
        --log-dir logs/mlperf_tiny_run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


def load_config(path: str) -> Any:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return argparse.Namespace(**cfg)


def preprocess(image: np.ndarray, input_shape: tuple) -> np.ndarray:
    """Resize to model input shape, normalize to int8 [-128, 127]."""
    h, w = input_shape[1], input_shape[2]
    img = Image.fromarray(image.astype(np.uint8))
    img = img.resize((w, h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    # Int8 quantization: scale from [0, 255] to [-128, 127]
    arr = (arr - 128.0).astype(np.int8)
    return arr[np.newaxis, ...]  # add batch dim → (1, H, W, 3)


def build_interpreter(config: argparse.Namespace):
    """Build TFLite interpreter with the requested delegate."""
    import tensorflow as tf

    delegate_name = getattr(config, "delegate", "cpu").lower()
    num_threads = getattr(config, "num_threads", 1)

    if delegate_name == "gpu":
        try:
            gpu_delegate = tf.lite.experimental.load_delegate("libdelegate.so")
            interp = tf.lite.Interpreter(
                model_path=config.model_path,
                experimental_delegates=[gpu_delegate],
                num_threads=num_threads,
            )
            print(f"  Delegate: GPU")
        except Exception as e:
            print(f"  WARNING: GPU delegate failed ({e}); falling back to CPU", file=sys.stderr)
            interp = tf.lite.Interpreter(
                model_path=config.model_path,
                num_threads=num_threads,
            )
            print(f"  Delegate: CPU (fallback)")
    else:
        interp = tf.lite.Interpreter(
            model_path=config.model_path,
            num_threads=num_threads,
        )
        print(f"  Delegate: CPU")

    interp.allocate_tensors()
    return interp


def run_performance_phase(
    interp, images: np.ndarray, config: argparse.Namespace, log_dir: Path
) -> dict:
    """Single Stream performance phase — batch=1 enforced per MLPerf Tiny spec."""
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    input_id = input_details[0]["index"]
    output_id = output_details[0]["index"]
    input_shape = input_details[0]["shape"]  # (1, H, W, C)

    n_samples = min(config.performance_samples, len(images))
    print(f"\n[Performance] {n_samples} samples, Single Stream (batch=1 enforced)")

    latencies_ns = []
    log_lines = []

    for i in range(n_samples):
        tensor = preprocess(images[i], input_shape)
        interp.set_tensor(input_id, tensor)
        t0 = time.perf_counter()
        interp.invoke()
        elapsed_ns = int((time.perf_counter() - t0) * 1e9)
        latencies_ns.append(elapsed_ns)
        log_lines.append(json.dumps({"sample": i, "latency_ns": elapsed_ns}))

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_samples}")

    latencies_sorted = sorted(latencies_ns)
    p50 = latencies_sorted[int(0.50 * len(latencies_sorted))]
    p90 = latencies_sorted[int(0.90 * len(latencies_sorted))]
    p99 = latencies_sorted[int(0.99 * len(latencies_sorted))]
    mean = int(sum(latencies_ns) / len(latencies_ns))

    print(f"  p50={p50/1e6:.2f}ms  p90={p90/1e6:.2f}ms  p99={p99/1e6:.2f}ms  mean={mean/1e6:.2f}ms")

    # Write per-sample log
    (log_dir / "log_performance.txt").write_text("\n".join(log_lines) + "\n")

    return {"p50_ns": p50, "p90_ns": p90, "p99_ns": p99, "mean_ns": mean, "n_samples": n_samples}


def run_accuracy_phase(
    interp, images: np.ndarray, labels: np.ndarray, config: argparse.Namespace, log_dir: Path
) -> dict:
    """Accuracy phase — all 10,000 CIFAR-10 test images."""
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    input_id = input_details[0]["index"]
    output_id = output_details[0]["index"]
    input_shape = input_details[0]["shape"]

    total = len(images)
    print(f"\n[Accuracy] {total} samples")

    correct = 0
    log_lines = []

    for i in range(total):
        tensor = preprocess(images[i], input_shape)
        interp.set_tensor(input_id, tensor)
        interp.invoke()
        output = interp.get_tensor(output_id)
        pred = int(output.argmax())
        gt = int(labels[i])
        correct += pred == gt
        log_lines.append(json.dumps({"sample": i, "pred": pred, "label": gt}))

        if (i + 1) % 1000 == 0:
            running_acc = 100.0 * correct / (i + 1)
            print(f"  {i + 1}/{total}  running top-1: {running_acc:.2f}%")

    top1_pct = 100.0 * correct / total
    print(f"  Final top-1 accuracy: {top1_pct:.2f}%  (target: {config.accuracy_target}%)")

    (log_dir / "log_accuracy.txt").write_text("\n".join(log_lines) + "\n")

    return {"top1_pct": top1_pct, "correct": correct, "total": total}


def write_result_file(path: Path, mode: str, stats: dict, config: argparse.Namespace) -> None:
    """Write MLPerf Tiny v1.1 result summary file."""
    if mode == "Performance":
        validity = "VALID" if stats["p90_ns"] > 0 else "INVALID"
        body = (
            "================================================\n"
            "MLPerf Results Summary\n"
            "================================================\n"
            "SUT name : TFLite\n"
            f"Scenario : Single Stream\n"
            f"Mode     : Performance\n"
            f"90th percentile latency (ns) : {stats['p90_ns']}\n"
            f"Result is : {validity}\n"
            "================================================\n"
            "Additional Stats\n"
            "================================================\n"
            f"50th percentile latency (ns) : {stats['p50_ns']}\n"
            f"99th percentile latency (ns) : {stats['p99_ns']}\n"
            f"Mean latency (ns)            : {stats['mean_ns']}\n"
            f"Samples run                  : {stats['n_samples']}\n"
        )
    else:  # Accuracy
        validity = "VALID" if stats["top1_pct"] >= config.accuracy_target else "INVALID"
        body = (
            "================================================\n"
            "MLPerf Results Summary\n"
            "================================================\n"
            "SUT name : TFLite\n"
            f"Scenario : Single Stream\n"
            f"Mode     : Accuracy\n"
            f"Accuracy : {stats['top1_pct']:.2f}%\n"
            f"Result is : {validity}\n"
            "================================================\n"
            "Additional Stats\n"
            "================================================\n"
            f"Correct predictions : {stats['correct']}\n"
            f"Total samples       : {stats['total']}\n"
            f"Accuracy target     : {config.accuracy_target}%\n"
        )

    path.write_text(body)
    print(f"  Written: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="MLPerf Tiny IC benchmark runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log-dir", required=True, help="Directory for output logs and results")
    args = parser.parse_args()

    config = load_config(args.config)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=== MLPerf Tiny IC Benchmark ===")
    print(f"  Config:    {args.config}")
    print(f"  Model:     {config.model_path}")
    print(f"  Data:      {config.data_path}")
    print(f"  Log dir:   {log_dir}")

    # Load data
    print("\nLoading CIFAR-10 test set...")
    data = np.load(config.data_path)
    images = data["images"]   # (10000, 32, 32, 3) uint8
    labels = data["labels"]   # (10000,) int32
    print(f"  Images: {images.shape}  Labels: {labels.shape}")

    # Build interpreter
    print("\nBuilding TFLite interpreter...")
    interp = build_interpreter(config)

    # Performance phase
    perf_stats = run_performance_phase(interp, images, config, log_dir)
    write_result_file(log_dir / "result_performance.txt", "Performance", perf_stats, config)

    # Accuracy phase
    acc_stats = run_accuracy_phase(interp, images, labels, config, log_dir)
    write_result_file(log_dir / "result_accuracy.txt", "Accuracy", acc_stats, config)

    print("\n=== Summary ===")
    print(f"  p90 latency : {perf_stats['p90_ns'] / 1e6:.2f} ms")
    print(f"  Top-1 acc   : {acc_stats['top1_pct']:.2f}% (target: {config.accuracy_target}%)")

    passed = acc_stats["top1_pct"] >= config.accuracy_target
    print(f"  Result      : {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
