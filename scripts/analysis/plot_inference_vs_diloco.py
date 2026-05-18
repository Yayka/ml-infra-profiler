"""Render a 4-panel dashboard comparing inference vs DiLoCo training.

Reads Grafana CSV exports from `results/inference/` and `results/diloco/` and
produces a single overlaid figure with shared time axes.

Usage:
    python scripts/analysis/plot_inference_vs_diloco.py

Output:
    results/inference_vs_diloco.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plot_dashboard import (  # noqa: E402
    _fmt_num,
    adaptive_bytes_formatter,
    apply_log_scale,
    bytes_formatter,
    load_and_aggregate,
    packets_formatter,
)

RESULTS = Path(__file__).parent.parent.parent / "results"

INF_COLOR = "#1f77b4"     # blue   — Inference
DILOCO_COLOR = "#d62728"  # red    — DiLoCo training
THRESH_COLOR = "#ff7f0e"  # orange — bandwidth limit reference

# (pattern, title, unit_type, threshold, log_scale, differentiate, use_sum, force_unit)
PANELS = [
    ("Ethernet*Bytes*",    "Internode Bytes Sent",            "bytes",   1e6,  True,  False, False, None),
    ("Ethernet*Packets*",  "Internode Packets Sent",          "packets", 100,  True,  False, False, None),
    ("PCIe*",              "Intranode Bytes Sent",            "bytes",   None, False, False, False, None),
    ("network interface*", "Internode Cumulative Bytes Sent", "bytes",   2e9,  True,  False, True,  "GB"),
]


def _threshold_label(thresh: float, unit_type: str, force_unit: str | None) -> str:
    if unit_type == "packets":
        if thresh >= 1e6:
            return f"{_fmt_num(thresh / 1e6)}M p/s"
        if thresh >= 1e3:
            return f"{_fmt_num(thresh / 1e3)}k p/s"
        return f"{_fmt_num(thresh)} p/s"
    if force_unit == "GB":
        return f"{thresh / 1e9:.0f} GB"
    if thresh >= 1e9:
        return f"{_fmt_num(thresh / 1e9)} GB/s"
    if thresh >= 1e6:
        return f"{_fmt_num(thresh / 1e6)} MB/s"
    if thresh >= 1e3:
        return f"{_fmt_num(thresh / 1e3)} kB/s"
    return f"{_fmt_num(thresh)} B/s"


def plot_inference_vs_diloco(
    inf_folder: Path,
    diloco_folder: Path,
    output_path: Path,
    title: str = "Inference vs DiLoCo Training — Network & PCIe",
) -> None:
    any_pat = PANELS[0][0]
    t_inf_end = load_and_aggregate(inf_folder, any_pat)[0][-1]
    t_dil_end = load_and_aggregate(diloco_folder, any_pat)[0][-1]
    max_t = min(t_inf_end, t_dil_end)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.patch.set_facecolor("white")

    for ax, (pattern, panel_title, unit_type, threshold, log_scale, diff, use_sum, force_unit) in zip(axes.flat, PANELS):
        t_inf, v_inf = load_and_aggregate(
            inf_folder, pattern, max_minutes=max_t,
            differentiate=diff, use_sum=use_sum,
        )
        t_dil, v_dil = load_and_aggregate(
            diloco_folder, pattern, max_minutes=max_t,
            differentiate=diff, use_sum=use_sum,
        )

        ax.set_facecolor("white")
        floor = 1e-3 if unit_type == "packets" else 1.0
        ax.plot(t_inf, np.maximum(v_inf, floor), color=INF_COLOR,
                linewidth=1.6, label="Inference")
        ax.plot(t_dil, np.maximum(v_dil, floor), color=DILOCO_COLOR,
                linewidth=1.6, label="DiLoCo Training")

        if threshold is not None:
            thresh = float(threshold)
            ax.axhline(thresh, color=THRESH_COLOR, linewidth=1.2,
                       linestyle="--", zorder=0)
            ax.text(
                1.0, thresh,
                f"bandwidth limit ({_threshold_label(thresh, unit_type, force_unit)})",
                color=THRESH_COLOR, fontsize=8, va="bottom", ha="right",
                transform=ax.get_yaxis_transform(),
            )

        ax.set_title(panel_title, fontsize=11, fontweight="bold",
                     color="#333333", pad=6)
        ax.set_xlabel("Elapsed time (min)", fontsize=8, color="#333333")
        ax.tick_params(axis="x", labelsize=8, colors="#333333")
        ax.tick_params(axis="y", labelsize=8, colors="#333333")

        if log_scale:
            apply_log_scale(ax)

        all_vals = np.concatenate([v_inf, v_dil])
        y_max = float(all_vals.max()) if len(all_vals) else 1.0
        if force_unit == "GB/s":
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{v / 1e9:.2f} GB/s"))
        elif force_unit == "GB":
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{v / 1e9:.2f} GB"))
        elif unit_type == "bytes" and log_scale:
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(adaptive_bytes_formatter))
        elif unit_type == "bytes":
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(bytes_formatter(y_max)))
        else:
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(packets_formatter))

        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")
        ax.grid(True, color="#eeeeee", linewidth=0.8)

    fig.suptitle(title, fontsize=14, fontweight="bold",
                 color="#333333", y=1.04)
    fig.tight_layout()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9,
               framealpha=0.7, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, 1.0))
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_inference_vs_diloco(
        RESULTS / "distributed inference" / "inference",
        RESULTS / "diloco",
        RESULTS / "inference_vs_diloco.png",
    )
