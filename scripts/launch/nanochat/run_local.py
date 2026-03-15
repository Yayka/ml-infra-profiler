#!/usr/bin/env python3
"""Launch a local nanochat training run, optionally scraping the ml-netprof agent.

nanochat: https://github.com/karpathy/nanochat

Usage:
    python scripts/launch/nanochat/run_local.py [config.yaml]

Defaults to configs/local_mac_tiny.yaml if no config is provided.
"""

import sys
from pathlib import Path

# train_wrapper lives in the same directory, but callers run this script from
# the repo root (e.g. `python scripts/launch/nanochat/run_local.py`), so the
# directory isn't on sys.path by default.
sys.path.insert(0, str(Path(__file__).parent))
from train_wrapper import DEFAULT_CONFIG, run  # noqa: E402

config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
run(config_path)
