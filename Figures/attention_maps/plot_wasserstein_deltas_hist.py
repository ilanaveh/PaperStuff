"""Plot a histogram of paired Wasserstein deltas from a CSV file.

Run directly with a CSV path, or call plot_wasserstein_deltas_hist(csv_path, ax)
to embed the histogram in an existing figure.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_wasserstein_deltas_hist(csv_path, ax=None, *, bins=15, color="0.45",
                                 font_size=7, title="",
                                 title_weight="normal", xlim_quantiles=(0.005, 0.995),
                                 zero_linewidth=1.3, yticks=tuple(range(0, 101, 5))):
    """Plot negated finite delta_a_minus_b values as percentages and return the destination axes.

    Missing, nonnumeric, and infinite values are excluded. Raise ValueError if
    the required column is missing or no finite numeric observations remain.
    """
    frame = pd.read_csv(csv_path)
    column = "delta_a_minus_b"
    if column not in frame:
        raise ValueError(f"{csv_path}: missing required column {column!r}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError(f"{csv_path}: {column!r} contains no finite numeric values")
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 2.2), layout="constrained")
    values = -values  # Reverse A - B to D2 - D1.
    weights = np.full(values.size, 100.0 / values.size)
    ax.hist(values, bins=bins, weights=weights, color=color, edgecolor="white", linewidth=0.3)
    if xlim_quantiles is None:
        data_min, data_max = values.min(), values.max()
    else:
        low, high = xlim_quantiles
        if not 0 <= low < high <= 1:
            raise ValueError("xlim_quantiles must satisfy 0 <= low < high <= 1")
        data_min, data_max = np.quantile(values, [low, high])
    if data_min < data_max:
        ax.set_xlim(data_min, data_max)
    else:
        margin = max(abs(data_min) * 0.01, 0.001)
        ax.set_xlim(data_min - margin, data_max + margin)
    ax.axvline(0, color="black", linestyle="--", linewidth=zero_linewidth, zorder=5)
    upper = ax.get_ylim()[1]
    visible_ticks = [tick for tick in yticks if tick <= upper]
    ax.set_yticks(visible_ticks)
    ax.set_ylim(0, upper)
    ax.set_title(title, fontsize=font_size, weight=title_weight)
    ax.set_xlabel(r"$\Delta$ Distances (D2 - D1)", fontsize=font_size)
    ax.set_ylabel("% test images", fontsize=font_size, labelpad=1)
    ax.set_box_aspect(1)
    ax.set_anchor("SW")
    ax.tick_params(labelsize=font_size, pad=2)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(True)
    return ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    ax = plot_wasserstein_deltas_hist(args.csv_path, bins=args.bins)
    if args.output:
        ax.figure.savefig(args.output)
    if args.no_show:
        plt.close(ax.figure)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
