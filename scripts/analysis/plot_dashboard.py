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
    "TB": 1e12,
}

PACKET_MULTIPLIERS = {
    "kp/s": 1e3,
    "Mp/s": 1e6,
    "p/s": 1,
}


def parse_value(s: str) -> float:
    """Parse a Grafana value string like '1.02 kB/s', '5.02 p/s', or '1.07 kp/s' to a float."""
    s = s.strip()
    for suffix, mult in PACKET_MULTIPLIERS.items():
        if s.endswith(suffix):
            return float(s[: -len(suffix)].strip()) * mult
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
    if ax_max >= 1e12:
        scale, unit = 1e12, "TB/s"
    elif ax_max >= 1e9:
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

import numpy as np


def load_and_aggregate(folder: Path, pattern: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (elapsed_minutes, summed_values) arrays for all series in a CSV."""
    csv_path = find_csv(folder, pattern)
    df = pd.read_csv(csv_path, parse_dates=["Time"])
    series_cols = [c for c in df.columns if c != "Time"]
    for col in series_cols:
        df[col] = df[col].apply(parse_value)
    elapsed = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds() / 60
    total = df[series_cols].sum(axis=1)
    return elapsed.values, total.values


def plot_comparison(inf_folder: Path, train_folder: Path, output_path: Path) -> None:
    INF_COLOR = "#1f77b4"   # blue  — Distributed Inference
    TRAIN_COLOR = "#d62728"  # red   — Training
    THRESH_COLOR = "#888888"

    # Panels: (pattern, title, unit_type, add_threshold, log_scale)
    # Network Interface Stats uses log scale: both workloads export cumulative byte counters
    # from Prometheus, so inference (6-min run, ~88 MB total) and training (145-min run,
    # ~6.5 TB total) span orders of magnitude — log scale shows both clearly.
    panels = [
        ("Ethernet*Bytes*", "Ethernet — Bytes/s", "bytes", True, False),
        ("Ethernet*Packets*", "Ethernet — Packets/s", "packets", True, False),
        ("network interface*", "Network Interface Stats", "bytes", True, True),
        ("PCIe*", "PCIe Stats", "bytes", True, False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.patch.set_facecolor("white")

    for ax, (pattern, panel_title, unit_type, add_threshold, log_scale) in zip(axes.flat, panels):
        t_inf, v_inf = load_and_aggregate(inf_folder, pattern)
        t_train, v_train = load_and_aggregate(train_folder, pattern)

        ax.set_facecolor("white")
        if log_scale:
            ax.set_yscale("log")
        ax.plot(t_inf, v_inf, color=INF_COLOR, linewidth=1.6, label="Distributed Inference")
        ax.plot(t_train, v_train, color=TRAIN_COLOR, linewidth=1.6, label="Training")

        if add_threshold:
            med_inf = float(np.median(v_inf[v_inf > 0])) if (v_inf > 0).any() else 0
            med_train = float(np.median(v_train[v_train > 0])) if (v_train > 0).any() else 0
            if med_inf > 0 and med_train > 0:
                thresh = float(np.sqrt(med_inf * med_train))
                ax.axhline(thresh, color=THRESH_COLOR, linewidth=1.2, linestyle="--", zorder=0)
                x_text = 0.02 * max(t_inf[-1], t_train[-1])
                ax.text(
                    x_text, thresh,
                    "training threshold",
                    color=THRESH_COLOR, fontsize=8, va="bottom", ha="left",
                )

        ax.set_title(panel_title, fontsize=11, fontweight="bold", color="#333333", pad=6)
        ax.set_xlabel("Elapsed time (min)", fontsize=8, color="#333333")
        ax.tick_params(axis="x", labelsize=8, colors="#333333")
        ax.tick_params(axis="y", labelsize=8, colors="#333333")

        if not log_scale:
            all_vals = np.concatenate([v_inf, v_train])
            y_max = float(all_vals.max()) if len(all_vals) else 1.0
            if unit_type == "bytes":
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(bytes_formatter(y_max)))
            else:
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(packets_formatter))
        else:
            # log-scale: use bytes_formatter keyed to the axis max after rendering
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda val, _pos: bytes_formatter(val)(val, _pos)
            ))

        ax.legend(loc="upper left", fontsize=8, framealpha=0.7, edgecolor="#cccccc")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")
        ax.grid(True, color="#eeeeee", linewidth=0.8, which="both" if log_scale else "major")

    fig.suptitle(
        "Distributed Inference vs Training — Network & PCIe",
        fontsize=14,
        fontweight="bold",
        color="#333333",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    for folder, title in DATASETS:
        plot_dataset(folder, title)

    plot_comparison(
        RESULTS / "distributed inference",
        RESULTS / "training",
        RESULTS / "comparison.png",
    )
