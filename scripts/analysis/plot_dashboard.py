"""
Reproduce the 4-panel Grafana network/PCIe dashboard from exported CSVs.

Usage:
    python scripts/analysis/plot_dashboard.py

Outputs:
    results/inference/dashboard.png
    results/distributed inference/dashboard.png
"""

import fnmatch
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

RESULTS = Path(__file__).parent.parent.parent / "results"

UNIT_MULTIPLIERS = {
    "B": 1,
    "kB": 1e3,
    "MB": 1e6,
    "GB": 1e9,
}


def parse_value(s: str) -> float:
    """Parse a Grafana value string like '1.02 kB/s' or '5.02 p/s' to a float."""
    s = s.strip()
    # packets: no unit conversion needed
    if s.endswith("p/s"):
        return float(s[:-3].strip())
    for suffix, mult in UNIT_MULTIPLIERS.items():
        if s.endswith(f" {suffix}/s") or s == f"{suffix}/s":
            return float(s[: -len(suffix) - 2].strip()) * mult
    # fallback: try plain float
    return float(s)


def clean_label(raw: str) -> str:
    """Normalize a Grafana series label to a human-readable form."""
    # {instance="1.2.3.4:9100", transport="ethernet"}
    m = re.match(r'\{instance="([^"]+):9100",\s*transport="([^"]+)"\}', raw)
    if m:
        return f"{m.group(1)} ({m.group(2)})"

    # 1.2.3.4:9100direction: rx gpu_index: 0
    m = re.match(r"([\d.]+):9100direction:\s*(\w+)\s+gpu_index:\s*(\d+)", raw)
    if m:
        return f"{m.group(1)} {m.group(2)} GPU {m.group(3)}"

    # 1.2.3.4:9100 eth0 rx
    cleaned = re.sub(r":9100\b", "", raw)
    return cleaned.strip()


def bytes_formatter(ax_max):
    """Return a FuncFormatter that auto-scales bytes to the best prefix."""
    if ax_max >= 1e9:
        scale, unit = 1e9, "GB/s"
    elif ax_max >= 1e6:
        scale, unit = 1e6, "MB/s"
    elif ax_max >= 1e3:
        scale, unit = 1e3, "kB/s"
    else:
        scale, unit = 1, "B/s"

    def fmt(val, _pos):
        return f"{val / scale:.1f} {unit}"

    return fmt


def packets_formatter(val, _pos):
    return f"{val:.1f} p/s"


# Map filename-pattern keywords to panel metadata
PANEL_ORDER = [
    ("Ethernet*Bytes*", "Ethernet — Bytes/s", "bytes"),
    ("Ethernet*Packets*", "Ethernet — Packets/s", "packets"),
    ("network interface*", "Network Interface Stats", "bytes"),
    ("PCIe*", "PCIe Stats", "bytes"),
]


def find_csv(folder: Path, pattern: str) -> Path:
    """Find a CSV by glob pattern using fnmatch (avoids pathlib ** restriction)."""
    glob_pat = pattern + ".csv"
    all_csvs = list(folder.glob("*.csv"))
    matches = [f for f in all_csvs if fnmatch.fnmatch(f.name, glob_pat)]
    if not matches:
        matches = [f for f in all_csvs if fnmatch.fnmatch(f.name.lower(), glob_pat.lower())]
    if not matches:
        raise FileNotFoundError(f"No CSV matching '{glob_pat}' in {folder}")
    return sorted(matches)[0]


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def plot_dataset(folder: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor("white")

    for ax, (pattern, panel_title, unit_type) in zip(axes.flat, PANEL_ORDER):
        csv_path = find_csv(folder, pattern)
        df = pd.read_csv(csv_path, parse_dates=["Time"])

        # Parse value columns (all except Time)
        series_cols = [c for c in df.columns if c != "Time"]
        for col in series_cols:
            df[col] = df[col].apply(parse_value)

        ax.set_facecolor("white")
        for i, col in enumerate(series_cols):
            label = clean_label(col)
            ax.plot(
                df["Time"],
                df[col],
                label=label,
                color=COLORS[i % len(COLORS)],
                linewidth=1.4,
                marker="o",
                markersize=3,
                markevery=max(1, len(df) // 30),
            )

        ax.set_title(panel_title, fontsize=11, fontweight="bold", color="#333333", pad=6)
        ax.tick_params(axis="x", rotation=30, labelsize=8, colors="#333333")
        ax.tick_params(axis="y", labelsize=8, colors="#333333")
        ax.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%H:%M")
        )
        ax.xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())

        if unit_type == "bytes":
            y_max = max(df[series_cols].max()) if series_cols else 1
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(bytes_formatter(y_max)))
        else:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(packets_formatter))

        ax.legend(
            loc="lower left",
            fontsize=7,
            framealpha=0.7,
            edgecolor="#cccccc",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")
        ax.grid(True, color="#eeeeee", linewidth=0.8)

    fig.suptitle(title, fontsize=15, fontweight="bold", color="#333333", y=1.01)
    fig.tight_layout()

    out = folder / "dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


DATASETS = [
    (RESULTS / "inference", "Inference — Network & PCIe Dashboard"),
    (RESULTS / "distributed inference", "Distributed Inference — Network & PCIe Dashboard"),
]

if __name__ == "__main__":
    for folder, title in DATASETS:
        plot_dataset(folder, title)
