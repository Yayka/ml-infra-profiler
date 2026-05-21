"""
plot_cumulative_bytes.py — Cumulative internode bytes + bandwidth analysis.

Produces a two-panel figure:
  Top   : Cumulative internode bytes sent over elapsed time (TB scale).
  Bottom: Estimated training time vs available bandwidth (hyperbolic),
          with compute-bound / bandwidth-bound regions and measured BW annotated.

Input: the Grafana "network interface usage total" CSV export from a training run.
       Columns are per-instance cumulative byte counters; the script sums all
       ethernet columns to get total internode bytes.

Usage:
    python scripts/analysis/plot_cumulative_bytes.py
    python scripts/analysis/plot_cumulative_bytes.py --data-dir results/training \
        --output results/figures/cumulative_bytes_training.png \
        --title "Distributed Training — Total Internode Bytes Sent"
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── value parser ──────────────────────────────────────────────────────────────
_UNITS = {"b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12}


def parse_value(s: str) -> float:
    s = str(s).strip()
    if s in ("", "-", "N/A"):
        return 0.0
    m = re.match(r"([0-9.]+)\s*([A-Za-z]*)", s)
    if not m:
        return 0.0
    num, unit = float(m.group(1)), m.group(2).lower()
    return num * _UNITS.get(unit, 1)


def _fmt_bytes(val: float) -> str:
    for mult, unit in [(1e12, "TB"), (1e9, "GB"), (1e6, "MB"), (1e3, "kB")]:
        if val >= mult:
            return f"{val/mult:.1f} {unit}"
    return f"{val:.0f} B"


def _fmt_bw(bps: float) -> str:
    for mult, unit in [(1e9, "GB/s"), (1e6, "MB/s"), (1e3, "kB/s")]:
        if bps >= mult:
            v = bps / mult
            return f"{v:.0f} {unit}" if v >= 10 else f"{v:.1f} {unit}"
    return f"{bps:.0f} B/s"


# ── CSV loader ────────────────────────────────────────────────────────────────

def load_cumulative_bytes(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (elapsed_minutes, cumulative_bytes_total) arrays.
    Sums all ethernet columns (skips infiniband / unknown).
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    times = pd.to_datetime(df.iloc[:, 0], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    elapsed = (times - times.iloc[0]).dt.total_seconds() / 60.0

    total = np.zeros(len(df))
    for col in df.columns[1:]:
        col_lower = col.lower()
        if "infiniband" in col_lower or "unknown" in col_lower:
            continue
        total += df[col].apply(parse_value).to_numpy()

    mask = ~np.isnan(elapsed.to_numpy())
    return elapsed.to_numpy()[mask], total[mask]


def find_csv(folder: Path) -> Path:
    matches = list(folder.glob("network interface*"))
    if not matches:
        sys.exit(f"ERROR: no 'network interface*' CSV found in {folder}")
    return sorted(matches)[0]


# ── plotting ──────────────────────────────────────────────────────────────────

def plot(csv_path: Path, output_path: Path, title: str) -> None:
    elapsed, cum_bytes = load_cumulative_bytes(csv_path)

    total_bytes = cum_bytes[-1]
    actual_time_h = elapsed[-1] / 60.0
    avg_bw = total_bytes / (actual_time_h * 3600)   # bytes/s — also the crossover BW

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 10),
        gridspec_kw={"hspace": 0.45},
    )
    fig.patch.set_facecolor("white")

    # ── top panel: cumulative bytes ───────────────────────────────────────────
    ax_top.set_facecolor("white")
    ax_top.plot(elapsed, cum_bytes, color="black", linewidth=1.8, label="Total")

    ax_top.set_title(title, fontsize=16, fontweight="bold", color="#333333", pad=12)
    ax_top.set_xlabel("Elapsed time (min)", fontsize=11, color="#333333")

    def bytes_fmt(val, _):
        for mult, unit in [(1e12, "TB"), (1e9, "GB"), (1e6, "MB")]:
            if val >= mult:
                return f"{val/mult:.0f} {unit}"
        return f"{val:.0f} B"

    ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(bytes_fmt))
    ax_top.tick_params(labelsize=10, colors="#333333")
    ax_top.spines[["top", "right"]].set_visible(False)
    ax_top.spines[["left", "bottom"]].set_color("#cccccc")
    ax_top.grid(False)
    ax_top.legend(fontsize=10, framealpha=0.7, edgecolor="#cccccc")

    # ── bottom panel: training time vs bandwidth ──────────────────────────────
    ax_bot.set_facecolor("white")

    bw_start = total_bytes / (actual_time_h * 100 * 1.05 * 3600)  # BW that hits top of y-axis
    bw_range = np.logspace(np.log10(bw_start), 11, 500)
    time_h = total_bytes / bw_range / 3600

    ax_bot.plot(bw_range, time_h, color="black", linewidth=2.0)

    # Actual training time — horizontal dashed line (no legend entry)
    ax_bot.axhline(actual_time_h, color="#888888", linewidth=1.2, linestyle="--")

    # Compute-bound threshold — vertical dotted line at crossover BW
    ax_bot.axvline(avg_bw, color="#888888", linewidth=1.0, linestyle=":")

    # Measured avg BW — green dot (legend entry)
    ax_bot.scatter(
        [avg_bw], [actual_time_h],
        color="#2ca02c", s=80, zorder=5,
        label=f"Unconstrained run ({actual_time_h:.1f} h)",
    )

    # Low-BW what-if: bandwidth at which training takes 100× longer (legend entry)
    low_time_h = actual_time_h * 100
    low_bw = total_bytes / (low_time_h * 3600)
    ax_bot.scatter(
        [low_bw], [low_time_h], color="#d62728", s=80, zorder=5,
        label=f"BW limit for 100× slowdown ({_fmt_bw(low_bw)})",
    )
    ax_bot.annotate(
        f"{_fmt_bw(low_bw)} → {low_time_h:.0f} h (100×)",
        xy=(low_bw, low_time_h),
        xytext=(low_bw * 3.0, low_time_h * 0.82),
        color="#d62728", fontsize=10,
        arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.5),
    )

    ax_bot.set_xscale("log")
    ax_bot.set_title(
        f"Training Time vs Available Bandwidth\n"
        f"(total data transferred: {_fmt_bytes(total_bytes)})",
        fontsize=14, fontweight="bold", color="#333333", pad=12,
    )
    ax_bot.set_xlabel("Available bandwidth", fontsize=11, color="#333333")
    ax_bot.set_ylabel("Estimated training time (hours)", fontsize=11, color="#333333")

    def bw_fmt(val, _):
        for mult, unit in [(1e9, "GB/s"), (1e6, "MB/s")]:
            if val >= mult:
                v = val / mult
                return f"{v:.0f} {unit}" if v >= 10 else f"{v:.1f} {unit}"
        return f"{val/1e3:.0f} kB/s"

    ax_bot.xaxis.set_major_formatter(ticker.FuncFormatter(bw_fmt))
    ax_bot.set_ylim(bottom=0, top=low_time_h * 1.05)
    ax_bot.tick_params(labelsize=10, colors="#333333")
    ax_bot.spines[["top", "right"]].set_visible(False)
    ax_bot.spines[["left", "bottom"]].set_color("#cccccc")
    ax_bot.grid(False)
    ax_bot.legend(fontsize=10, framealpha=0.7, edgecolor="#cccccc", loc="upper right")

    # Region labels — placed after axes are scaled
    ymax = ax_bot.get_ylim()[1]
    label_y = ymax * 0.08
    ax_bot.text(avg_bw * 0.80, label_y, "← bandwidth-\nbound",
                ha="right", va="bottom", fontsize=9, color="#888888")
    ax_bot.text(avg_bw * 1.25, label_y, "compute-\nbound →",
                ha="left", va="bottom", fontsize=9, color="#888888")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"  Total bytes  : {_fmt_bytes(total_bytes)}")
    print(f"  Elapsed time : {actual_time_h:.1f} h")
    print(f"  Avg bandwidth: {_fmt_bw(avg_bw)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).resolve().parents[2]  # repo root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=here / "results" / "training",
        help="Folder containing the 'network interface usage total' CSV",
    )
    parser.add_argument(
        "--output", type=Path,
        default=here / "results" / "figures" / "cumulative_bytes_training.png",
    )
    parser.add_argument(
        "--title", type=str,
        default="Distributed Training — Total Internode Bytes Sent",
    )
    args = parser.parse_args()

    csv_path = find_csv(args.data_dir)
    print(f"Using: {csv_path}")
    plot(csv_path, args.output, args.title)


if __name__ == "__main__":
    main()
