"""
Reproduce the 4-panel Grafana network/PCIe dashboard from exported CSVs.

Usage:
    python scripts/analysis/plot_dashboard.py

Outputs:
    results/inference/dashboard.png
    results/distributed inference/dashboard.png
"""

import math
import numpy as np
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
    """Parse a Grafana value string to a float in base units (bytes or packets).

    Handles:
      '1.02 kB/s', '23.8 MB/s', '1.00 TB/s'  → bytes/s
      '5.02 p/s', '1.07 kp/s'                 → packets/s
      '30.8 MB', '1.12 TB'                    → bytes (bare, no /s — cumulative counters)
    """
    s = s.strip()
    for suffix, mult in PACKET_MULTIPLIERS.items():
        if s.endswith(suffix):
            return float(s[: -len(suffix)].strip()) * mult
    # rate units (e.g. "23.8 MB/s") — check before bare to avoid partial match
    for suffix, mult in UNIT_MULTIPLIERS.items():
        if s.endswith(f" {suffix}/s") or s == f"{suffix}/s":
            return float(s[: -len(suffix) - 2].strip()) * mult
    # bare units (e.g. "30.8 MB") — cumulative counter exports
    for suffix, mult in UNIT_MULTIPLIERS.items():
        if s.endswith(f" {suffix}") or s == suffix:
            return float(s[: -len(suffix)].strip()) * mult
    return float(s)


def clean_label(raw: str) -> str:
    """Normalize a Grafana series label to a generic, IP-free form.

    Drops `<host>:9100` instance identifiers (Prometheus scrape targets) so
    the rendered legend only shows the descriptive part of each series
    (transport / interface / GPU index).
    """
    # {instance="1.2.3.4:9100", transport="ethernet"} -> "ethernet"
    m = re.match(r'\{instance="[^"]+:9100",\s*transport="([^"]+)"\}', raw)
    if m:
        return m.group(1)

    # 1.2.3.4:9100direction: rx gpu_index: 0 -> "rx GPU 0"
    m = re.match(r"[\d.]+:9100direction:\s*(\w+)\s+gpu_index:\s*(\d+)", raw)
    if m:
        return f"{m.group(1)} GPU {m.group(2)}"

    # 1.2.3.4:9100 eth0 rx -> "eth0 rx"
    cleaned = re.sub(r"[\d.]+:9100\s*", "", raw)
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


def _fmt_num(val: float) -> str:
    """Format a scaled numeric value: round to nearest whole if <10, else ceil to nearest 10."""
    if val < 10:
        return str(round(val))
    else:
        return str(math.ceil(val / 10) * 10)


def packets_formatter(val, _pos):
    if val >= 1e6:
        return f"{val/1e6:.1f}M p/s"
    elif val >= 1e3:
        return f"{val/1e3:.1f}k p/s"
    return f"{int(val)} p/s"


def apply_log_scale(ax):
    """Set log y-scale with minor gridlines at 2×, 3×, 5× each decade."""
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs=[2, 3, 5], numticks=20))
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5, alpha=0.6)


def adaptive_bytes_formatter(val, _pos):
    """Per-tick adaptive formatter — works correctly on log-scale axes."""
    if val <= 0:
        return ""
    if val >= 1e9:
        return f"{val/1e9:.3g} GB/s"
    if val >= 1e6:
        return f"{val/1e6:.3g} MB/s"
    if val >= 1e3:
        return f"{val/1e3:.3g} kB/s"
    return f"{val:.3g} B/s"


# Map filename-pattern keywords to panel metadata
PANEL_ORDER = [
    ("Ethernet*Bytes*",    "Internode Bytes Sent",           "bytes"),
    ("Ethernet*Packets*",  "Internode Packets Sent",         "packets"),
    ("network interface*", "Internode Cumulative Bytes Sent","bytes"),
    ("PCIe*",              "Intranode Bytes Sent",           "bytes"),
]


def find_csv(folder: Path, pattern: str) -> Path:
    """Find a CSV by glob pattern using fnmatch (avoids pathlib ** restriction)."""
    glob_pat = pattern + ".csv"
    all_csvs = list(folder.glob("*.csv"))
    matches = [f for f in all_csvs if fnmatch.fnmatch(f.name, glob_pat)]
    if not matches:
        matches = [f for f in all_csvs if fnmatch.fnmatch(
            f.name.lower(), glob_pat.lower())]
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

        ax.set_title(panel_title, fontsize=11,
                     fontweight="bold", color="#333333", pad=6)
        ax.tick_params(axis="x", rotation=30, labelsize=8, colors="#333333")
        ax.tick_params(axis="y", labelsize=8, colors="#333333")
        ax.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%H:%M")
        )
        ax.xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())

        if unit_type == "bytes":
            y_max = max(df[series_cols].max()) if series_cols else 1
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(bytes_formatter(y_max)))
        else:
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(packets_formatter))

        ax.legend(
            loc="lower left",
            fontsize=7,
            framealpha=0.7,
            edgecolor="#cccccc",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")
        ax.grid(True, color="#eeeeee", linewidth=0.8)

    fig.suptitle(title, fontsize=15, fontweight="bold",
                 color="#333333", y=1.01)
    fig.tight_layout()

    out = folder / "dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


DATASETS = [
    (RESULTS / "inference", "Inference — Network & PCIe Dashboard"),
    (RESULTS / "distributed inference" / "inference",
     "Distributed Inference — Network & PCIe Dashboard"),
]


def load_and_aggregate(
    folder: Path | list[Path],
    pattern: str,
    max_minutes: float | None = None,
    differentiate: bool = False,
    use_sum: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (elapsed_minutes, aggregated_values) arrays for all series in a CSV.

    Args:
        folder: a single directory, or a list of directories whose CSVs are
            stitched end-to-end (e.g. model-load capture followed by inference
            capture from the same run, split across files).
        max_minutes: clip the time series to this many elapsed minutes.
        differentiate: convert a cumulative counter to an instantaneous rate
            (bytes/s) via diff/dt — for panels that export running totals.
        use_sum: sum all series instead of taking the max.  Use this for panels
            where series represent independent links (e.g. network interface,
            which mixes ethernet + infiniband transports that don't double-count).
            Default is max, which avoids tx/rx double-counting on Ethernet panels.
    """
    folders = list(folder) if isinstance(folder, (list, tuple)) else [folder]

    segments_t = []
    segments_v = []
    offset = 0.0
    for f in folders:
        csv_path = find_csv(f, pattern)
        df = pd.read_csv(csv_path, parse_dates=["Time"])
        series_cols = [c for c in df.columns if c != "Time"]
        for col in series_cols:
            df[col] = df[col].apply(parse_value)
        elapsed = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds() / 60
        total = df[series_cols].sum(
            axis=1) if use_sum else df[series_cols].max(axis=1)
        segments_t.append(elapsed.values + offset)
        segments_v.append(total.values)
        offset = segments_t[-1][-1]

    t = np.concatenate(segments_t)
    v = np.concatenate(segments_v)

    if max_minutes is not None:
        mask = t <= max_minutes
        t = t[mask]
        v = v[mask]

    if differentiate and len(t) > 1:
        dt_sec = np.diff(t) * 60  # minutes → seconds
        dv = np.diff(v)
        rate = np.where(dt_sec > 0, dv / dt_sec, 0.0)
        rate = np.maximum(rate, 0.0)          # drop counter resets
        rate = np.concatenate([[rate[0]], rate])  # restore original length
        v = rate

    return t, v


def plot_comparison(
    inference_segments: list[tuple[Path, str, str]],
    train_runs: list[tuple[Path, str, str]],
    output_path: Path,
    title: str = "Inference vs Training",
    xlim_minutes: float | None = None,
) -> None:
    """Overlay inference against one or more training runs on a 4-panel dashboard.

    `inference_segments` is a list of `(folder, label, linestyle)` tuples. When
    multiple entries are supplied, their captures are stitched end-to-end with
    cumulative time offsets — useful when a single run was exported as multiple
    CSVs (e.g. model load then inference). Each segment is plotted in the shared
    inference color with its own label and linestyle.

    `train_runs` is a list of `(folder, label, color)` tuples.
    """
    INF_COLOR = "#1f77b4"   # blue — Distributed Inference
    THRESH_COLOR = "#ff7f0e"  # orange

    # Panels: (pattern, title, unit_type, threshold_value, log_scale, differentiate, use_sum, force_unit)
    # threshold_value: None = no line, float = fixed value in base units
    panels = [
        ("Ethernet*Bytes*",    "Internode Bytes Sent",            "bytes",   1e6,   True,  False, False, None),
        ("Ethernet*Packets*",  "Internode Packets Sent",          "packets", 100,   True,  False, False, None),
        ("PCIe*",              "Intranode Bytes Sent",            "bytes",   None,  False, False, False, None),
        ("network interface*", "Internode Cumulative Bytes Sent", "bytes",   2e9,   True,  False, True,  "GB"),
    ]

    # X-axis spans the longest run; each curve ends naturally at its own duration.
    any_pat = panels[0][0]
    inf_folders = [f for f, _, _ in inference_segments]
    ends = [load_and_aggregate(inf_folders, any_pat)[0][-1]]
    for folder, _, _ in train_runs:
        ends.append(load_and_aggregate(folder, any_pat)[0][-1])
    max_t = max(ends)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.patch.set_facecolor("white")

    for ax, (pattern, panel_title, unit_type, threshold_value, log_scale, diff, use_sum, force_unit) in zip(axes.flat, panels):
        ax.set_facecolor("white")
        floor = 1e-3 if unit_type == "packets" else 1.0

        # For cumulative panels (use_sum is the current marker), rebase each
        # segment so its accumulation continues from the previous one's end —
        # otherwise the counter visually restarts at 0 between segments.
        all_vals = []
        offset = 0.0
        value_offset = 0.0
        for folder, label, linestyle in inference_segments:
            t_seg, v_seg = load_and_aggregate(
                folder, pattern, differentiate=diff, use_sum=use_sum)
            t_seg = t_seg + offset
            if use_sum:
                v_seg = v_seg + value_offset
            ax.plot(t_seg, np.maximum(v_seg, floor), color=INF_COLOR,
                    linewidth=1.6, linestyle=linestyle, label=label)
            offset = t_seg[-1] if len(t_seg) else offset
            if use_sum and len(v_seg):
                value_offset = v_seg[-1]
            all_vals.append(v_seg)

        for folder, label, color in train_runs:
            t_train, v_train = load_and_aggregate(
                folder, pattern, differentiate=diff, use_sum=use_sum)
            ax.plot(t_train, np.maximum(v_train, floor), color=color,
                    linewidth=1.6, label=label)
            all_vals.append(v_train)

        if threshold_value is not None and threshold_value is not False:
            thresh = float(threshold_value)
            if thresh > 0:
                if unit_type == "packets":
                    if thresh >= 1e6:
                        thresh_str = f"{_fmt_num(thresh/1e6)}M p/s"
                    elif thresh >= 1e3:
                        thresh_str = f"{_fmt_num(thresh/1e3)}k p/s"
                    else:
                        thresh_str = f"{_fmt_num(thresh)} p/s"
                elif force_unit == "GB":
                    thresh_str = f"{thresh/1e9:.0f} GB"
                elif thresh >= 1e9:
                    thresh_str = f"{_fmt_num(thresh/1e9)} GB/s"
                elif thresh >= 1e6:
                    thresh_str = f"{_fmt_num(thresh/1e6)} MB/s"
                elif thresh >= 1e3:
                    thresh_str = f"{_fmt_num(thresh/1e3)} kB/s"
                else:
                    thresh_str = f"{_fmt_num(thresh)} B/s"
                ax.axhline(thresh, color=THRESH_COLOR,
                           linewidth=1.2, linestyle="--", zorder=0)
                ax.text(
                    1.0, thresh,
                    f"bandwidth limit ({thresh_str})",
                    color=THRESH_COLOR, fontsize=8, va="bottom", ha="right",
                    transform=ax.get_yaxis_transform(),
                )

        ax.set_title(panel_title, fontsize=11,
                     fontweight="bold", color="#333333", pad=6)
        ax.set_xlabel("Elapsed time (min)", fontsize=8, color="#333333")
        ax.tick_params(axis="x", labelsize=8, colors="#333333")
        ax.tick_params(axis="y", labelsize=8, colors="#333333")

        if log_scale:
            apply_log_scale(ax)

        concat_vals = np.concatenate(all_vals)
        y_max = float(concat_vals.max()) if len(concat_vals) else 1.0
        if force_unit == "GB/s":
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{v/1e9:.2f} GB/s"))
        elif force_unit == "GB":
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{v/1e9:.2f} GB"))
        elif unit_type == "bytes" and log_scale:
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(adaptive_bytes_formatter))
        elif unit_type == "bytes":
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(bytes_formatter(y_max)))
        elif unit_type == "packets" and log_scale:
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(packets_formatter))
        else:
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(packets_formatter))

        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")
        ax.grid(True, color="#eeeeee", linewidth=0.8)

        if xlim_minutes is not None:
            ax.set_xlim(0, xlim_minutes)

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
        color="#333333",
        y=1.04,
    )
    fig.tight_layout()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               fontsize=9, framealpha=0.7, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, 1.0))
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_internode_bytes(
    inf_folder: Path,
    train_folders: list[tuple[Path, str]],
    output_path: Path,
    title: str = "Internode Bytes Sent",
    threshold: float = 1e6,
) -> None:
    THRESH_COLOR = "#ff7f0e"
    INF_COLOR = "#1f77b4"
    TRAIN_COLORS = ["#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    pattern = "Ethernet*Bytes*"

    ends = [load_and_aggregate(inf_folder, pattern)[0][-1]]
    for folder, _ in train_folders:
        ends.append(load_and_aggregate(folder, pattern)[0][-1])
    max_t = min(ends)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    t_inf, v_inf = load_and_aggregate(inf_folder, pattern, max_minutes=max_t)
    ax.plot(t_inf, np.maximum(v_inf, 1.0), color=INF_COLOR, linewidth=1.6, label="Inference")

    for (folder, label), color in zip(train_folders, TRAIN_COLORS):
        t, v = load_and_aggregate(folder, pattern, max_minutes=max_t)
        ax.plot(t, np.maximum(v, 1.0), color=color, linewidth=1.6, label=label)

    if threshold:
        thresh = float(threshold)
        if thresh >= 1e9:
            thresh_str = f"{_fmt_num(thresh/1e9)} GB/s"
        elif thresh >= 1e6:
            thresh_str = f"{_fmt_num(thresh/1e6)} MB/s"
        elif thresh >= 1e3:
            thresh_str = f"{_fmt_num(thresh/1e3)} kB/s"
        else:
            thresh_str = f"{_fmt_num(thresh)} B/s"
        ax.axhline(thresh, color=THRESH_COLOR, linewidth=1.2, linestyle="--", zorder=0)
        ax.text(1.0, thresh, f"bandwidth limit ({thresh_str})",
                color=THRESH_COLOR, fontsize=16, va="bottom", ha="right",
                transform=ax.get_yaxis_transform())

    apply_log_scale(ax)
    ax.set_title(title, fontsize=26, fontweight="bold", color="#333333", pad=16)
    ax.tick_params(axis="x", labelsize=16, colors="#333333")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)} min"))
    ax.tick_params(axis="y", labelsize=16, colors="#333333")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(adaptive_bytes_formatter))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.grid(True, color="#eeeeee", linewidth=0.8)

    fig.tight_layout()
    ax.legend(fontsize=16, framealpha=0.7, edgecolor="#cccccc",
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    for folder, title in DATASETS:
        plot_dataset(folder, title)

    inference_dir = RESULTS / "distributed inference" / "inference"
    model_load_dir = RESULTS / "distributed inference" / "model load"
    # Model load and inference came from a single run, split across two CSV
    # exports. Stitch them end-to-end and render model load as a dashed lead-in.
    inference_segments = [
        (model_load_dir, "Llama 2 70B Model Load", "--"),
        (inference_dir, "Llama 2 70B Inference", "-"),
    ]
    training_runs = [
        (RESULTS / "baseline", "Llama 3.1 8B Training", "#d62728"),  # red
        (RESULTS / "moe", "Mixtral 3.7B MoE Training", "#2ca02c"),  # green
    ]

    plot_comparison(
        inference_segments,
        training_runs,
        RESULTS / "comparison.png",
        title="Inference vs Training",
        xlim_minutes=130,
    )

    plot_internode_bytes(
        [f for f, _, _ in inference_segments],
        [(folder, label) for folder, label, _ in training_runs],
        RESULTS / "internode_bytes.png",
    )
