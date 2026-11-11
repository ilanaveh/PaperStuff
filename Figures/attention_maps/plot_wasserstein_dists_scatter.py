"""Scatter paired Wasserstein distances from the same CSV used for deltas.

X: wasserstein_distance_a (D1). Y: wasserstein_distance_b (D2).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_wasserstein_dists_scatter(csv_path, ax=None, *, color="0.45",
                                   marker_size=3, alpha=0.35, font_size=7,
                                   show_ticklabels=False):
    """Plot finite paired distances and y=x, returning the destination axes.

    Exclude a row if either distance is missing, nonnumeric, or infinite.
    """
    frame = pd.read_csv(csv_path)
    columns = ["wasserstein_distance_a", "wasserstein_distance_b"]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{csv_path}: missing required columns {missing}")
    pairs = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    pairs = pairs[np.isfinite(pairs).all(axis=1)]
    if not len(pairs):
        raise ValueError(f"{csv_path}: no finite numeric distance pairs")
    if ax is None:
        _, ax = plt.subplots(figsize=(2.5, 2.5), layout="constrained")
    ax.scatter(pairs[:, 0], pairs[:, 1], s=marker_size, alpha=alpha,
               color=color, linewidths=0, rasterized=True)
    low, high = pairs.min(), pairs.max()
    margin = (high - low) * 0.03 if high > low else max(abs(low) * 0.03, 0.001)
    limits = (max(0, low - margin) if low >= 0 else low - margin, high + margin)
    ax.plot(limits, limits, color="black", linestyle="--", linewidth=1, zorder=3)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_box_aspect(1)
    ax.set_anchor("SW")
    ax.set_xlabel("Distance D1", fontsize=font_size)
    ax.set_ylabel("Distance D2", fontsize=font_size, labelpad=1)
    ax.tick_params(labelsize=font_size, pad=2,
                   bottom=show_ticklabels, left=show_ticklabels, top=False, right=False,
                   labelbottom=show_ticklabels, labelleft=show_ticklabels)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(True)
    return ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--marker-size", type=float, default=3)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--show-ticklabels", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    ax = plot_wasserstein_dists_scatter(args.csv_path, marker_size=args.marker_size,
                                       alpha=args.alpha, show_ticklabels=args.show_ticklabels)
    if args.output:
        ax.figure.savefig(args.output)
    if args.no_show:
        plt.close(ax.figure)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
