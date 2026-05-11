"""Render the network/PCIe dashboard for the MoE pretraining run.

Produces:
    <data_root>/dashboard.png  (4-panel: Ethernet bytes, packets, NI total, PCIe)

Usage:
    python scripts/analysis/plot_moe_run.py            # uses results/moe/
    python scripts/analysis/plot_moe_run.py path/to/csvs

Folder must contain four Grafana CSV exports matching the patterns
expected by `plot_dashboard.plot_dataset` (Ethernet bytes/s, Ethernet
packets/s, network interface usage total, PCIe stats).
"""

import sys
from pathlib import Path

# Reuse the renderer from plot_dashboard.py (PR #14).
sys.path.insert(0, str(Path(__file__).parent))
from plot_dashboard import plot_dataset  # noqa: E402

DEFAULT_FOLDER = Path(__file__).parent.parent.parent / "results" / "moe"
TITLE = "MoE Pretraining (Mixtral-3.7B FSDP, 2x2 A100) — Network & PCIe Dashboard"


def main() -> None:
    folder = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_FOLDER
    if not folder.is_dir():
        raise SystemExit(f"data root not found: {folder}")
    plot_dataset(folder, TITLE)


if __name__ == "__main__":
    main()
