#!/usr/bin/env python3
"""Nanochat training wrapper — shared entry point for local and multi-node runs.

Sets up the environment (env vars, W&B metrics scraping via ml-netprof agent)
and hands off to nanochat's base_train.

Can be invoked directly or via torchrun:
    python scripts/launch/nanochat/train_wrapper.py [config.yaml]
    torchrun ... scripts/launch/nanochat/train_wrapper.py --config config.yaml [-- extra]
"""

import os
import runpy
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[2]

DEFAULT_CONFIG = HERE / "config/local_mac_tiny.yaml"


def config_to_flags(config_path: Path) -> list[str]:
    config = yaml.safe_load(config_path.read_text())
    return [f"--{k.replace('_', '-')}={v}" for k, v in config.items() if v is not None]


def run(config_path: Path, extra_flags: list[str] | None = None) -> None:
    """Set up environment and launch nanochat training."""
    load_dotenv(REPO_ROOT / ".env")
    os.environ["NANOCHAT_BASE_DIR"] = str(REPO_ROOT / "data")

    flags = config_to_flags(config_path) + (extra_flags or [])

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from netprof import setup_agent

    setup_agent()

    os.chdir(REPO_ROOT / "nanochat")
    sys.path.insert(0, str(REPO_ROOT / "nanochat"))
    sys.argv = ["base_train"] + flags

    runpy.run_module("scripts.base_train", run_name="__main__", alter_sys=True)


def main() -> None:
    """Parse CLI args and call run(). Supports two calling conventions:

    Direct:   train_wrapper.py [config.yaml]
    torchrun: train_wrapper.py --config config.yaml [-- extra nanochat flags]
    """
    args = sys.argv[1:]
    config_path = DEFAULT_CONFIG
    extra_flags: list[str] = []

    if "--config" in args:
        idx = args.index("--config")
        config_path = Path(args[idx + 1])
        args = args[:idx] + args[idx + 2:]
    elif args and not args[0].startswith("-"):
        # Positional config path (backwards-compat with run_local.py usage)
        config_path = Path(args[0])
        args = args[1:]

    if "--" in args:
        idx = args.index("--")
        extra_flags = args[idx + 1:]

    run(config_path, extra_flags)


if __name__ == "__main__":
    main()
