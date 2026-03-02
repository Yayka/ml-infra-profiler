#!/usr/bin/env python3
"""Launch a local training run, optionally scraping the ml-netprof agent for network metrics.

Usage:
    python scripts/launch/run_local.py [config.yaml]

Defaults to configs/local_mac_tiny.yaml if no config is provided.
Start the agent first if you want network metrics:
    make agent-build-darwin && ./agent/bin/agent-darwin &
"""
import os
import runpy
import sys
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[2]
AGENT_HEALTHZ = "http://localhost:9100/healthz"
AGENT_METRICS = "http://localhost:9100/metrics"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


def agent_running() -> bool:
    try:
        urllib.request.urlopen(AGENT_HEALTHZ, timeout=1)
        return True
    except Exception:
        return False


def config_to_flags(config_path: Path) -> list[str]:
    config = yaml.safe_load(config_path.read_text())
    return [f"--{k.replace('_', '-')}={v}" for k, v in config.items() if v is not None]


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "configs/local_mac_tiny.yaml"

    load_dotenv(REPO_ROOT / ".env")
    os.environ["NANOCHAT_BASE_DIR"] = str(REPO_ROOT / "data")

    flags = config_to_flags(config_path)

    # Run from nanochat/ so that scripts.base_train is importable.
    os.chdir(REPO_ROOT / "nanochat")
    sys.path.insert(0, str(REPO_ROOT / "nanochat"))
    sys.argv = ["base_train"] + flags

    # wandb.setup() must be called before wandb.init() in the same process.
    if agent_running():
        import wandb
        wandb.setup(settings=wandb.Settings(
            x_stats_open_metrics_endpoints={"agent": AGENT_METRICS},
        ))

    runpy.run_module("scripts.base_train", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
