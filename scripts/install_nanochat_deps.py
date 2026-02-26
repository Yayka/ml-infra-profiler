"""
Install nanochat's dependencies into the active venv without building the package.

nanochat uses a flat layout (dev/, runs/, nanochat/ at the same level) that
setuptools can't auto-discover, so `pip install -e nanochat/` fails. We parse
nanochat's pyproject.toml directly and install only its dependencies.

The nanochat package itself is importable at runtime because run_local.sh
cds into the nanochat/ directory before invoking python -m, which puts the
current directory on sys.path.
"""

import re
import subprocess
import sys

with open("nanochat/pyproject.toml") as f:
    content = f.read()

m = re.search(r"dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL)
assert m, "Could not find dependencies block in nanochat/pyproject.toml"
deps = re.findall(r'"([^"]+)"', m.group(1))

print(f"Installing {len(deps)} nanochat dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install"] + deps)
