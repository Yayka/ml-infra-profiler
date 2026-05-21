"""
plot_data.py — Plot Grafana CSV exports for distributed inference benchmarks.

Reads CSVs exported from the Grafana dashboards and produces publication-ready
figures comparing model load vs inference phases.

Thresholds (horizontal reference lines):
  - Ethernet bytes/s  : 1 MB/s
  - Ethernet packets/s: 100 p/s
  - Cumulative bytes  : 2 GB

Usage:
    python scripts/analysis/plot_data.py \
        --inference-dir "results/distributed inference/inference" \
        --model-load-dir "results/distributed inference/model load" \
        --output-dir results/figures/distributed_inference
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

COLORS_INFERENCE  = ["#1f77b4", "#6baed6"]   # blue shades (rx, tx)
COLORS_MODEL_LOAD = ["#d62728", "#fc8d59"]   # red-orange shades (rx, tx)

LABEL_INFERENCE  = "Distributed Inference"
LABEL_MODEL_LOAD = "Model Load"


# ── Y-axis engineering formatters ─────────────────────────────────────────────

def _fmt_bytes_s(val, _):
    if val == 0:
        return "0"
    for mult, unit in [(1e9, "GB/s"), (1e6, "MB/s"), (1e3, "kB/s")]:
        if val >= mult:
            return f"{val/mult:g} {unit}"
    return f"{val:g} B/s"


def _fmt_packets_s(val, _):
    if val == 0:
        return "0"
    for mult, unit in [(1e6, "Mp/s"), (1e3, "kp/s")]:
        if val >= mult:
            return f"{val/mult:g} {unit}"
    return f"{val:g} p/s"


def _fmt_bytes(val, _):
    if val == 0:
        return "0"
    for mult, unit in [(1e12, "TB"), (1e9, "GB"), (1e6, "MB"), (1e3, "kB")]:
        if val >= mult:
            return f"{val/mult:.2f} {unit}"
    return f"{val:g} B"


# ── unit parser ─────────────────────────────────────────────────────────────

_UNIT_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*"
    r"(T|G|M|k|m|µ|n|p)?"
    r"(B|b|p)?"
    r"(?:/s)?\s*$",
    re.IGNORECASE,
)

_PREFIX = {"T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "m": 1e-3,
           "µ": 1e-6, "n": 1e-9, "p": 1e-12}


def parse_value(s: str) -> float:
    """Convert strings like '1.17 kB/s', '200 MB/s', '3.33 p/s', '67.3 kB' to float."""
    if not isinstance(s, str):
        return float(s)
    s = s.strip()
    if s in ("", "-", "N/A"):
        return float("nan")
    m = _UNIT_RE.match(s)
    if not m:
        try:
            return float(s)
        except ValueError:
            return float("nan")
    val = float(m.group(1))
    prefix = _PREFIX.get(m.group(2) or "", 1.0)
    # p/s (packets) — no byte conversion needed; treat prefix as scaling only
    return val * prefix


def load_csv(path: Path) -> pd.DataFrame:
    """Load a Grafana CSV, parse the Time column and all value columns."""
    df = pd.read_csv(path, header=0)
    # Multi-line header: Grafana sometimes wraps the column name across rows
    # Drop rows where Time is NaN or not a timestamp
    df.columns = [c.strip().replace("\n", " ") for c in df.columns]
    df = df.rename(columns={df.columns[0]: "Time"})
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).reset_index(drop=True)

    for col in df.columns[1:]:
        df[col] = df[col].apply(parse_value)

    # Normalise time to seconds from start
    df["t"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds()
    return df


def find_csv(directory: Path, keyword: str) -> Path | None:
    for f in directory.glob("*.csv"):
        if keyword.lower() in f.name.lower():
            return f
    return None


# ── plotting helpers ─────────────────────────────────────────────────────────

_NODE_RE = re.compile(r"\d+\.\d+\.\d+\.\d+:\d+\s*")


def _short_label(col: str) -> str:
    """Shorten column names for legend readability (single-node view)."""
    # Strip IP:port prefix
    col = _NODE_RE.sub("", col)
    col = re.sub(r"eth0\s*", "", col)
    # Shorten PromQL query to just 'cumulative tx'
    if "ml_net_interface_bytes_total" in col:
        return "cumulative tx"
    # Shorten PCIe: "direction: rx gpu_index: 0" → "GPU0 rx"
    m = re.search(r"direction:\s*(\w+).*?gpu_index:\s*(\d+)", col, re.IGNORECASE)
    if m:
        return f"GPU{m.group(2)} {m.group(1)}"
    col = re.sub(r"\s+", " ", col).strip()
    return col


SERVER_NODE = "20.29.43.19"


def plot_metric(
    ax,
    df_inf: pd.DataFrame | None,
    df_load: pd.DataFrame | None,
    threshold: tuple | None,
    ylabel: str,
    scale: float = 1.0,
    cols_filter: str | None = None,
    node_filter: str | None = SERVER_NODE,
):
    """Plot inference + model-load data on ax with optional threshold line."""
    for df, colors, dataset_name in [
        (df_inf,  COLORS_INFERENCE,  LABEL_INFERENCE),
        (df_load, COLORS_MODEL_LOAD, LABEL_MODEL_LOAD),
    ]:
        if df is None:
            continue
        value_cols = [c for c in df.columns if c not in ("Time", "t")]
        if node_filter:
            value_cols = [c for c in value_cols if node_filter in c]
        if cols_filter:
            value_cols = [c for c in value_cols if cols_filter.lower() in c.lower()]
        for i, col in enumerate(value_cols):
            col_label = _short_label(col)
            label = dataset_name if len(value_cols) == 1 else f"{dataset_name} {col_label}"
            ax.plot(
                df["t"] / 60,
                df[col] * scale,
                color=colors[i % len(colors)],
                linewidth=1.4,
                label=label,
            )

    if threshold:
        val, label = threshold
        ax.axhline(val * scale, color="#ff7f0e", linestyle="--", linewidth=1.2,
                   label=label, zorder=5)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Elapsed time (min)")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)


# ── 3-way inference comparison ───────────────────────────────────────────────

def _server_tx(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_minutes, bytes_s) for the server TX column."""
    col = next(
        (c for c in df.columns if SERVER_NODE in c and " tx" in c.lower()),
        None,
    )
    if col is None:
        return np.array([]), np.array([])
    return df["t"].values / 60, df[col].values


def extrapolate_to(
    t: np.ndarray, v: np.ndarray, target_min: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tile the steady-state window (skipping warmup and tail drop) to target_min.
    The tail is detected as values dropping below 20 % of the median signal.
    """
    if len(t) < 4 or t[-1] >= target_min * 0.95:
        return np.array([]), np.array([])
    # Trim tail: find last index still above 20 % of median
    median_v = float(np.median(v))
    drop_thresh = median_v * 0.20
    steady_end = len(v)
    for i in range(len(v) - 1, -1, -1):
        if v[i] >= drop_thresh:
            steady_end = i + 1
            break
    steady_start = max(0, int(len(v) * 0.05))  # skip first 5 % warmup
    t_s, v_s = t[steady_start:steady_end], v[steady_start:steady_end]
    if len(t_s) < 2:
        return np.array([]), np.array([])
    step = np.mean(np.diff(t_s))
    t_parts, v_parts = [], []
    offset = t[steady_end - 1] + step   # continue from end of clean window
    while offset <= target_min:
        t_tile = t_s - t_s[0] + offset
        mask = t_tile <= target_min
        t_parts.append(t_tile[mask])
        v_parts.append(v_s[mask])
        offset = float(t_tile[-1]) + step
    if not t_parts:
        return np.array([]), np.array([])
    return np.concatenate(t_parts), np.concatenate(v_parts)


def plot_inference_comparison(
    df_text: pd.DataFrame | None,
    df_t2i:  pd.DataFrame | None,
    df_t2v:  pd.DataFrame | None,
    out_path: Path,
):
    """Single-panel comparison of all 3 workloads: text summarization / T2I / T2V."""
    datasets = [
        (df_text, "Text Summarization (Llama 3.1 8B)",  "#1f77b4"),
        (df_t2i,  "Text to Image (SDXL)",               "#2ca02c"),
        (df_t2v,  "Text to Video (Wan2.2-T2V-A14B)",    "#d62728"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(
        "Internode Bytes Sent — Inference Comparison",
        fontsize=13, fontweight="bold",
    )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_bytes_s))
    ax.set_ylabel("Throughput")
    ax.set_xlabel("Elapsed time (min)")
    ax.axhline(1e6, color="#ff7f0e", linestyle="--", linewidth=1.5,
               label="1 MB/s threshold")

    for df, label, color in datasets:
        if df is None:
            continue
        t, v = _server_tx(df)
        if len(t):
            ax.plot(t, v, color=color, linewidth=2.0, label=label)

    ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir",
                        default="results/distributed inference/inference")
    parser.add_argument("--model-load-dir",
                        default="results/distributed inference/model load")
    parser.add_argument("--text-dir",
                        default="results/distributed inference",
                        help="Directory with text inference (Llama) Bytes_s CSV")
    parser.add_argument("--t2i-dir", default="results/t2i/inference")
    parser.add_argument("--t2v-dir", default="results/t2v/inference")
    parser.add_argument("--output-dir",
                        default="results/figures/distributed_inference")
    args = parser.parse_args()

    inf_dir   = Path(args.inference_dir)
    load_dir  = Path(args.model_load_dir)
    text_dir  = Path(args.text_dir)
    t2i_dir   = Path(args.t2i_dir)
    t2v_dir   = Path(args.t2v_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load CSVs ──────────────────────────────────────────────────────────
    def _load(directory, keyword):
        f = find_csv(directory, keyword)
        return load_csv(f) if f else None

    inf_bytes   = _load(inf_dir,  "Bytes_s")
    inf_pkts    = _load(inf_dir,  "Packets_s")
    inf_cumul   = _load(inf_dir,  "network interface usage")
    inf_pcie    = _load(inf_dir,  "PCIe")

    load_bytes  = _load(load_dir, "Bytes_s")
    load_pkts   = _load(load_dir, "Packets_s")
    load_cumul  = _load(load_dir, "network interface usage")
    load_pcie   = _load(load_dir, "PCIe")

    _bytes_fmt   = ticker.FuncFormatter(_fmt_bytes_s)
    _pkts_fmt    = ticker.FuncFormatter(_fmt_packets_s)
    _cumul_fmt   = ticker.FuncFormatter(_fmt_bytes)
    _pcie_fmt    = ticker.FuncFormatter(_fmt_bytes_s)

    def _setup_bytes_log(ax):
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e2)
        ax.yaxis.set_major_formatter(_bytes_fmt)

    def _setup_pkts_log(ax):
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)
        ax.yaxis.set_major_formatter(_pkts_fmt)

    # ── figure 1: Internode Bytes Sent ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_metric(ax, inf_bytes, load_bytes,
                threshold=(1e6, "inference threshold"),
                ylabel="Throughput", cols_filter="tx")
    _setup_bytes_log(ax)
    ax.set_title("Internode Bytes Sent", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "ethernet_bytes_s.png")
    plt.close(fig)
    print(f"Saved: {out_dir}/ethernet_bytes_s.png")

    # ── figure 2: Internode Packets Sent ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_metric(ax, inf_pkts, load_pkts,
                threshold=(100, "inference threshold"),
                ylabel="Packet rate", cols_filter="tx")
    _setup_pkts_log(ax)
    ax.set_title("Internode Packets Sent", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "ethernet_packets_s.png")
    plt.close(fig)
    print(f"Saved: {out_dir}/ethernet_packets_s.png")

    # ── figure 3: Internode Cumulative Bytes Sent ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_metric(ax, inf_cumul, load_cumul,
                threshold=(2e9, "inference threshold"),
                ylabel="Cumulative bytes", node_filter=None)
    ax.yaxis.set_major_formatter(_cumul_fmt)
    ax.set_title("Internode Cumulative Bytes Sent", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_bytes.png")
    plt.close(fig)
    print(f"Saved: {out_dir}/cumulative_bytes.png")

    # ── figure 4: Intranode Bytes Sent (PCIe) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_metric(ax, inf_pcie, load_pcie,
                threshold=None,
                ylabel="Throughput", cols_filter="tx")
    ax.yaxis.set_major_formatter(_pcie_fmt)
    ax.set_title("Intranode Bytes Sent", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "pcie_stats.png")
    plt.close(fig)
    print(f"Saved: {out_dir}/pcie_stats.png")

    # ── figure 5: combined 2×2 overview ───────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Distributed Inference vs Model Load — Network & PCIe",
                 fontsize=13, fontweight="bold", y=1.01)

    plot_metric(axes[0, 0], inf_bytes, load_bytes,
                (1e6, "inference threshold"), "Throughput", cols_filter="tx")
    _setup_bytes_log(axes[0, 0])
    axes[0, 0].set_title("Internode Bytes Sent", fontweight="bold")

    plot_metric(axes[0, 1], inf_pkts, load_pkts,
                (100, "inference threshold"), "Packet rate", cols_filter="tx")
    _setup_pkts_log(axes[0, 1])
    axes[0, 1].set_title("Internode Packets Sent", fontweight="bold")

    plot_metric(axes[1, 0], inf_cumul, load_cumul,
                (2e9, "inference threshold"), "Cumulative bytes", node_filter=None)
    axes[1, 0].yaxis.set_major_formatter(_cumul_fmt)
    axes[1, 0].set_title("Internode Cumulative Bytes Sent", fontweight="bold")

    plot_metric(axes[1, 1], inf_pcie, load_pcie,
                None, "Throughput", cols_filter="tx")
    axes[1, 1].yaxis.set_major_formatter(_pcie_fmt)
    axes[1, 1].set_title("Intranode Bytes Sent", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/overview.png")

    # ── figure 6: 3-way inference comparison ──────────────────────────────
    text_bytes = _load(text_dir, "Bytes_s")
    t2i_bytes  = _load(t2i_dir, "Bytes_s") if t2i_dir.is_dir() else None
    t2v_bytes  = _load(t2v_dir, "Bytes_s") if t2v_dir.is_dir() else None
    plot_inference_comparison(
        text_bytes, t2i_bytes, t2v_bytes,
        out_dir / "inference_comparison.png",
    )


if __name__ == "__main__":
    main()
