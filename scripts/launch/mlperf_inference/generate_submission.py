#!/usr/bin/env python3
"""
generate_submission.py — Copy benchmark results into MLPerf Inference v5.0 submission tree.

Reads LoadGen output files from LOG_DIR and copies them into the standard
submission directory structure under OUTPUT_DIR. Also auto-generates the
system description JSON via platform.uname() + nvidia-smi.

Idempotent — overwrites existing files on re-run.

Usage:
    python scripts/launch/mlperf_inference/generate_submission.py \
        --log-dir logs/mlperf_inference_20240101_120000_offline \
        --scenario offline \
        --output-dir results/mlperf_inference_v5.0
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


DIVISION = "open"
SUBMITTER = "ml-infra-profiler"
SYSTEM_ID = "linux_a100_vllm"
BENCHMARK = "llama3.1-8b"


def get_nvidia_info() -> dict:
    """Query nvidia-smi for GPU model name. Returns defaults if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        ).strip().splitlines()[0]
        name, mem = [x.strip() for x in out.split(",")]
        return {"hw_model": name, "hw_memory_gb": int(int(mem) / 1024), "hw_count": 2}
    except Exception:
        return {"hw_model": "A100 80GB", "hw_memory_gb": 80, "hw_count": 2}


def generate_system_json(systems_dir: Path) -> Path:
    """Auto-generate the system description JSON."""
    uname = platform.uname()
    gpu_info = get_nvidia_info()

    system = {
        "system_name": SYSTEM_ID,
        "division": DIVISION,
        "submitter": SUBMITTER,
        "hw_vendor": "NVIDIA",
        "hw_model": gpu_info.get("hw_model", "A100 80GB"),
        "hw_memory_gb": gpu_info.get("hw_memory_gb", 80),
        "hw_count": gpu_info.get("hw_count", 2),
        "sw_framework": "vLLM",
        "sw_framework_version": "0.6.3",
        "sw_os": f"{uname.system} {uname.release}",
        "sw_python": platform.python_version(),
        "notes": (
            "vLLM inference engine; tensor_parallel_size=2; bfloat16; "
            "Open Division; Offline + Server scenarios"
        ),
    }

    systems_dir.mkdir(parents=True, exist_ok=True)
    out_path = systems_dir / f"{SYSTEM_ID}.json"
    out_path.write_text(json.dumps(system, indent=2) + "\n")
    print(f"  Written: {out_path}")
    return out_path


def copy_results(log_dir: Path, results_dir: Path) -> None:
    """Copy LoadGen result and log files into the submission tree."""
    # Map from files produced by run_mlperf_inference.sh → submission tree locations
    file_map = {
        "mlperf_log_summary.txt": "result.txt",
        "mlperf_log_detail.txt":  "log.txt",
    }

    for src_name, dst_name in file_map.items():
        src = log_dir / src_name
        dst = results_dir / dst_name
        results_dir.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            print(f"  WARNING: source file not found: {src}", file=sys.stderr)
            continue

        shutil.copy2(src, dst)
        print(f"  Written: {dst}")

    # Copy accuracy result if present
    accuracy_src = log_dir / "accuracy_result.txt"
    if accuracy_src.exists():
        accuracy_dst = results_dir.parent / "accuracy" / "result.txt"
        accuracy_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(accuracy_src, accuracy_dst)
        print(f"  Written: {accuracy_dst}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MLPerf Inference v5.0 submission tree")
    parser.add_argument("--log-dir", required=True, help="Directory containing LoadGen output files")
    parser.add_argument("--scenario", required=True, choices=["offline", "server"],
                        help="Benchmark scenario (offline or server)")
    parser.add_argument("--output-dir", required=True,
                        help="Root of submission tree (e.g. results/mlperf_inference_v5.0)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)

    if not log_dir.exists():
        print(f"ERROR: log-dir does not exist: {log_dir}", file=sys.stderr)
        return 1

    submitter_dir = output_dir / DIVISION / SUBMITTER
    systems_dir = submitter_dir / "systems"
    results_dir = submitter_dir / "results" / BENCHMARK / args.scenario

    print(f"=== Generating MLPerf Inference v5.0 submission tree ({args.scenario}) ===")
    print(f"  Source:      {log_dir}")
    print(f"  Destination: {submitter_dir}")
    print()

    print("System description:")
    generate_system_json(systems_dir)

    print("\nResults:")
    copy_results(log_dir, results_dir)

    print(f"\nSubmission tree: {submitter_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
