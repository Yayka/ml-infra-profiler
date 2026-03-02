#!/usr/bin/env python3
"""Launch a local nanochat training run, optionally scraping the ml-netprof agent.

nanochat: https://github.com/karpathy/nanochat

Usage:
    python scripts/launch/nanochat/run_local.py [config.yaml]

Defaults to configs/local_mac_tiny.yaml if no config is provided.
"""

import os
import runpy
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[2]


def config_to_flags(config_path: Path) -> list[str]:
    config = yaml.safe_load(config_path.read_text())
    return [f"--{k.replace('_', '-')}={v}" for k, v in config.items() if v is not None]


def main() -> None:
    default_config = HERE / "config/local_mac_tiny.yaml"
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_config

    load_dotenv(REPO_ROOT / ".env")
    os.environ["NANOCHAT_BASE_DIR"] = str(REPO_ROOT / "data")

    flags = config_to_flags(config_path)

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from netprof import setup_agent  # noqa: E402

    setup_agent()

    # Run from nanochat/ so that scripts.base_train is importable.
    os.chdir(REPO_ROOT / "nanochat")
    sys.path.insert(0, str(REPO_ROOT / "nanochat"))
    sys.argv = ["base_train"] + flags

    runpy.run_module("scripts.base_train", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
