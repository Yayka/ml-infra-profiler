#!/usr/bin/env python3
"""
benchmark_runner.py — MLPerf Training ResNet-50 v1.5 benchmark runner (TF2, multi-GPU).

Trains ResNet-50 v1.5 on ImageNet ILSVRC2012 using tf.keras with
tf.distribute.MirroredStrategy. Logs per-epoch metrics and checks convergence
against the MLPerf target (top-1 accuracy >= 75.9%).

Reference: https://github.com/mlcommons/training/tree/master/retired_benchmarks/resnet-tf2

Usage:
    python scripts/launch/mlperf_resnet/benchmark_runner.py \
        --config scripts/launch/mlperf_resnet/config/resnet_4gpu.yaml \
        --log-dir logs/mlperf_resnet_run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# TF must be imported at module level for keras callback subclassing
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_imagenet_dataset(data_dir: str, split: str, batch_size: int,
                           crop_size: int, resize_size: int, is_training: bool):
    """Build a tf.data pipeline reading TFRecord files."""
    pattern = os.path.join(data_dir, f"{split}-*")
    files = tf.io.gfile.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No TFRecord files found matching {pattern}. "
            f"Run 'make prepare-mlperf-resnet-data' first."
        )

    feature_spec = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64),
    }

    def parse_fn(serialized):
        features = tf.io.parse_single_example(serialized, feature_spec)
        image = tf.io.decode_jpeg(features["image/encoded"], channels=3)
        label = tf.cast(features["image/class/label"], tf.int32)
        # ImageNet labels are 1-indexed; shift to 0-indexed
        label = label - 1
        return image, label

    def train_preprocess(image, label):
        image = tf.image.resize(image, [resize_size, resize_size])
        image = tf.image.random_crop(image, [crop_size, crop_size, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return image, label

    def val_preprocess(image, label):
        image = tf.image.resize(image, [resize_size, resize_size])
        image = tf.image.resize_with_crop_or_pad(image, crop_size, crop_size)
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return image, label

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.shuffle(10000)
        ds = ds.map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(val_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=is_training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_resnet50(num_classes: int, weight_decay: float):
    """Build ResNet-50 v1.5 using tf.keras.applications."""
    base = tf.keras.applications.ResNet50(
        weights=None,
        include_top=True,
        input_shape=(224, 224, 3),
        classes=num_classes,
    )

    # Apply weight decay via kernel_regularizer on all Conv2D and Dense layers
    if weight_decay > 0:
        for layer in base.layers:
            if hasattr(layer, "kernel_regularizer"):
                layer.kernel_regularizer = tf.keras.regularizers.l2(weight_decay)

    return base


def build_lr_schedule(cfg: dict, steps_per_epoch: int):
    """Piecewise constant LR schedule with linear warmup."""
    total_batch = cfg["batch_size_per_gpu"] * cfg["num_gpus"]
    scaled_lr = cfg["base_lr"] * (total_batch / 256.0)
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch

    decay_epochs = cfg.get("lr_decay_epochs", [30, 60, 80])
    decay_factor = cfg.get("lr_decay_factor", 0.1)

    # Build boundaries and values for piecewise schedule
    boundaries = [e * steps_per_epoch for e in decay_epochs]
    values = [scaled_lr]
    for _ in decay_epochs:
        values.append(values[-1] * decay_factor)

    piecewise = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries, values=values
    )

    if warmup_steps > 0:
        class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, base_schedule, warmup_steps, warmup_target):
                super().__init__()
                self.base_schedule = base_schedule
                self.warmup_steps = warmup_steps
                self.warmup_target = warmup_target

            def __call__(self, step):
                step = tf.cast(step, tf.float32)
                warmup = self.warmup_target * step / tf.cast(self.warmup_steps, tf.float32)
                return tf.where(step < self.warmup_steps, warmup, self.base_schedule(step))

            def get_config(self):
                return {
                    "warmup_steps": self.warmup_steps,
                    "warmup_target": self.warmup_target,
                }

        return WarmupSchedule(piecewise, warmup_steps, scaled_lr)

    return piecewise


class MetricsLogger(tf.keras.callbacks.Callback):
    """Logs per-epoch metrics to JSONL file and optionally to W&B."""

    def __init__(self, log_path: Path, wandb_run=None):
        super().__init__()
        self.log_path = log_path
        self.wandb_run = wandb_run
        self.epoch_start_time = None
        self.fh = open(log_path, "w")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0

        record = {
            "epoch": epoch + 1,
            "train/loss": float(logs.get("loss", 0)),
            "train/top1_accuracy": float(logs.get("accuracy", 0)) * 100,
            "eval/loss": float(logs.get("val_loss", 0)),
            "eval/top1_accuracy": float(logs.get("val_accuracy", 0)) * 100,
            "eval/top5_accuracy": float(logs.get("val_top5", 0)) * 100,
            "perf/epoch_time_s": round(elapsed, 1),
            "perf/images_per_sec": round(
                logs.get("num_train_images", 0) / max(elapsed, 1), 1
            ),
        }

        self.fh.write(json.dumps(record) + "\n")
        self.fh.flush()

        if self.wandb_run:
            self.wandb_run.log(record, step=epoch + 1)

        print(
            f"  Epoch {record['epoch']:3d} | "
            f"train_loss={record['train/loss']:.4f} | "
            f"val_top1={record['eval/top1_accuracy']:.2f}% | "
            f"val_top5={record['eval/top5_accuracy']:.2f}% | "
            f"{record['perf/epoch_time_s']}s"
        )

    def on_train_end(self, logs=None):
        self.fh.close()


def top5_accuracy(y_true, y_pred):
    """Top-5 accuracy metric for Keras."""
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)


def main() -> int:
    parser = argparse.ArgumentParser(description="MLPerf ResNet-50 v1.5 benchmark runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log-dir", required=True, help="Directory for output logs and results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=== MLPerf ResNet-50 v1.5 Training Benchmark ===")
    print(f"  Config:     {args.config}")
    print(f"  Data dir:   {cfg['data_dir']}")
    print(f"  GPUs:       {cfg['num_gpus']}")
    print(f"  Batch size: {cfg['batch_size_per_gpu']} x {cfg['num_gpus']} = "
          f"{cfg['batch_size_per_gpu'] * cfg['num_gpus']}")
    print(f"  Epochs:     {cfg['epochs']}")
    print(f"  Target:     top-1 >= {cfg['target_top1']}%")
    print(f"  Log dir:    {log_dir}")

    # --- Strategy ---
    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync
    print(f"\n  Replicas:   {num_replicas}")

    global_batch = cfg["batch_size_per_gpu"] * num_replicas
    steps_per_epoch = cfg["num_train_images"] // global_batch

    # --- Datasets ---
    print("\nBuilding data pipelines...")
    crop = cfg.get("crop_size", 224)
    resize = cfg.get("resize_size", 256)

    train_ds = build_imagenet_dataset(
        cfg["data_dir"], "train", global_batch, crop, resize, is_training=True
    )
    val_ds = build_imagenet_dataset(
        cfg["data_dir"], "validation", global_batch, crop, resize, is_training=False
    )

    # Distribute datasets
    train_dist = strategy.experimental_distribute_dataset(train_ds)
    val_dist = strategy.experimental_distribute_dataset(val_ds)

    # --- Model ---
    print("Building ResNet-50 v1.5...")
    with strategy.scope():
        model = build_resnet50(cfg["num_classes"], cfg.get("weight_decay", 1e-4))

        lr_schedule = build_lr_schedule(cfg, steps_per_epoch)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=cfg.get("momentum", 0.9),
        )

        # Mixed precision
        dtype = cfg.get("dtype", "float16")
        if dtype == "float16":
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        label_smoothing = cfg.get("label_smoothing", 0.0)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        )
        if label_smoothing > 0:
            loss_fn = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True,
                label_smoothing=label_smoothing,
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            )

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                "accuracy",
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
            ],
        )

    # --- W&B (optional) ---
    wandb_run = None
    if cfg.get("wandb_enabled", False):
        import wandb
        wandb_run = wandb.init(
            project=cfg.get("wandb_project", "mlperf-resnet"),
            entity=cfg.get("wandb_entity") or None,
            config=cfg,
            name=f"resnet50_{cfg['num_gpus']}gpu",
        )

    # --- Callbacks ---
    training_log_path = log_dir / "training_log.jsonl"
    metrics_logger = MetricsLogger(training_log_path, wandb_run=wandb_run)
    # Inject num_train_images so the callback can compute images/sec
    metrics_logger.num_train_images = cfg["num_train_images"]

    callbacks = [metrics_logger]

    # --- Train ---
    print(f"\nStarting training ({cfg['epochs']} epochs, {steps_per_epoch} steps/epoch)...")
    train_start = time.time()

    # If using label smoothing, we need to one-hot encode labels
    if label_smoothing > 0:
        def one_hot_map(image, label):
            return image, tf.one_hot(label, cfg["num_classes"])
        train_ds = train_ds.map(one_hot_map, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(one_hot_map, num_parallel_calls=tf.data.AUTOTUNE)
        train_dist = strategy.experimental_distribute_dataset(train_ds)
        val_dist = strategy.experimental_distribute_dataset(val_ds)

    model.fit(
        train_dist,
        epochs=cfg["epochs"],
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dist,
        validation_freq=cfg.get("eval_interval_epochs", 1),
        callbacks=callbacks,
        verbose=0,
    )

    total_time = time.time() - train_start

    # --- Final evaluation ---
    print("\nRunning final evaluation...")
    eval_results = model.evaluate(val_dist, verbose=0, return_dict=True)
    final_top1 = eval_results.get("accuracy", 0) * 100
    final_top5 = eval_results.get("top5", 0) * 100

    print(f"\n=== Final Results ===")
    print(f"  Top-1 accuracy : {final_top1:.2f}%")
    print(f"  Top-5 accuracy : {final_top5:.2f}%")
    print(f"  Training time  : {total_time / 3600:.1f}h")
    print(f"  Target         : {cfg['target_top1']}%")

    passed = final_top1 >= cfg["target_top1"]
    print(f"  Result         : {'PASS' if passed else 'FAIL'}")

    # --- Write result file ---
    results_dir = Path(cfg.get("results_dir", "results/mlperf_resnet"))
    results_dir.mkdir(parents=True, exist_ok=True)

    result_text = (
        "================================================\n"
        "MLPerf Training Results Summary\n"
        "================================================\n"
        f"Benchmark    : ResNet-50 v1.5 Image Classification\n"
        f"Dataset      : ImageNet ILSVRC2012\n"
        f"Framework    : TensorFlow {tf.__version__}\n"
        f"GPUs         : {num_replicas}\n"
        f"Batch size   : {global_batch} (global)\n"
        f"Epochs       : {cfg['epochs']}\n"
        f"Training time: {total_time / 3600:.2f} hours\n"
        "================================================\n"
        f"Top-1 Accuracy : {final_top1:.2f}%\n"
        f"Top-5 Accuracy : {final_top5:.2f}%\n"
        f"Target         : {cfg['target_top1']}%\n"
        f"Result         : {'PASS' if passed else 'FAIL'}\n"
        "================================================\n"
    )

    (results_dir / "result.txt").write_text(result_text)
    print(f"\n  Result written to: {results_dir / 'result.txt'}")

    # Copy training log to results dir
    import shutil
    shutil.copy2(training_log_path, results_dir / "training_log.jsonl")
    print(f"  Training log:     {results_dir / 'training_log.jsonl'}")

    # Also write to log dir
    (log_dir / "result.txt").write_text(result_text)

    if wandb_run:
        wandb_run.finish()

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
