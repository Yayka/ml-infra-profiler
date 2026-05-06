"""Render plots for the baseline FSDP vs DiLoCo training experiment.

Generates three artifacts under `data_root` (default ~/data):
  baseline/dashboard.png    — 4-panel network/PCIe dashboard for the baseline run
  diloco/dashboard.png      — 4-panel dashboard for the DiLoCo run
  baseline_vs_diloco.png    — overlaid 4-panel comparison on shared axes

Usage:
    python scripts/analysis/plot_diloco_runs.py [data_root]

Both sub-directories must contain the four Grafana CSV exports that
`plot_dashboard.plot_dataset` expects (Ethernet bytes/s, Ethernet packets/s,
network interface usage total, PCIe stats).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Reuse the renderers and helpers from the existing plot_dashboard PR (#14).
sys.path.insert(0, str(Path(__file__).parent))
from plot_dashboard import (  # noqa: E402
    bytes_formatter,
    load_and_aggregate,
    packets_formatter,
    plot_dataset,
)


RUNS = [
    ("baseline", "Baseline FSDP — Network & PCIe Dashboard"),
    ("diloco", "DiLoCo — Network & PCIe Dashboard"),
]

# Visual style for the two-run comparison.
BASELINE_COLOR = "#d62728"  # red — sustained cross-node traffic
DILOCO_COLOR = "#1f77b4"    # blue — bursty cross-node traffic

# (csv_pattern, panel_title, unit_type, log_scale, differentiate, use_sum, force_unit)
COMPARISON_PANELS = [
    ("Ethernet*Bytes*",    "Internode Bytes Sent",       "bytes",   False, False, False, "GB/s"),
    ("Ethernet*Packets*",  "Internode Packets Sent",     "packets", False, False, False, None),
    ("PCIe*",              "Intranode Bytes Sent",       "bytes",   False, False, False, None),
    ("network interface*", "Internode Cumulative Bytes", "bytes",   True,  False, True,  "GB"),
]


def plot_baseline_vs_diloco(
    baseline_folder: Path,
    diloco_folder: Path,
    output_path: Path,
) -> None:
    """4-panel overlay of baseline vs DiLoCo on shared time axes.

    Same layout as plot_dashboard.plot_comparison but with labels/colors
    chosen for "FSDP baseline" vs "DiLoCo" rather than "Inference" vs
    "Training".
    """
    # Clip both runs to the shorter duration so x-axes align.
    any_pat = COMPARISON_PANELS[0][0]
    t_b_end = load_and_aggregate(baseline_folder, any_pat)[0][-1]
    t_d_end = load_and_aggregate(diloco_folder, any_pat)[0][-1]
    max_t = min(t_b_end, t_d_end)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.patch.set_facecolor("white")

    for ax, (pattern, panel_title, unit_type, log_scale, diff, use_sum, force_unit) in zip(axes.flat, COMPARISON_PANELS):
        t_b, v_b = load_and_aggregate(
            baseline_folder, pattern, max_minutes=max_t,
            differentiate=diff, use_sum=use_sum,
        )
        t_d, v_d = load_and_aggregate(
            diloco_folder, pattern, max_minutes=max_t,
            differentiate=diff, use_sum=use_sum,
        )

        ax.set_facecolor("white")
        ax.plot(t_b, v_b, color=BASELINE_COLOR, linewidth=1.6, label="Baseline FSDP")
        ax.plot(t_d, v_d, color=DILOCO_COLOR, linewidth=1.6, label="DiLoCo")

        ax.set_title(panel_title, fontsize=11, fontweight="bold",
                     color="#333333", pad=6)
        ax.set_xlabel("Elapsed time (min)", fontsize=8, color="#333333")
        ax.tick_params(axis="x", labelsize=8, colors="#333333")
        ax.tick_params(axis="y", labelsize=8, colors="#333333")

        if log_scale:
            ax.set_yscale("log")

        all_vals = np.concatenate([v_b, v_d])
        y_max = float(all_vals.max()) if len(all_vals) else 1.0
        if force_unit == "GB/s":
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{v / 1e9:.2f} GB/s"))
        elif force_unit == "GB":
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{v / 1e9:.2f} GB"))
        elif unit_type == "bytes":
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(bytes_formatter(y_max)))
        else:
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(packets_formatter))

        ax.legend(loc="upper left", fontsize=8,
                  framealpha=0.7, edgecolor="#cccccc")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")
        ax.grid(True, color="#eeeeee", linewidth=0.8)

    fig.suptitle(
        "Baseline FSDP vs DiLoCo — Cross-node Communication on Llama-3.1-8B",
        fontsize=14, fontweight="bold", color="#333333", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    root = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "data"
    if not root.is_dir():
        raise SystemExit(f"data root not found: {root}")

    baseline = root / "baseline"
    diloco = root / "diloco"
    for folder in (baseline, diloco):
        if not folder.is_dir():
            raise SystemExit(f"missing {folder} — expected baseline/ and diloco/ next to {root}")

    for sub, title in RUNS:
        plot_dataset(root / sub, title)

    plot_baseline_vs_diloco(baseline, diloco, root / "baseline_vs_diloco.png")


if __name__ == "__main__":
    main()
