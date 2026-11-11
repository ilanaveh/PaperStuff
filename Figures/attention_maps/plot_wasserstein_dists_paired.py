"""Plot paired D1/D2 Wasserstein distances, one connecting line per CSV image."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


def plot_wasserstein_dists_paired(csv_path, ax=None, *, color="0.45", alpha=0.08,
                                  linewidth=0.4, marker_size=2, font_size=7):
    """Plot every finite distance pair, preserving row correspondence."""
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
    segments = np.stack([np.column_stack([np.zeros(len(pairs)), pairs[:, 0]]),
                         np.column_stack([np.ones(len(pairs)), pairs[:, 1]])], axis=1)
    ax.add_collection(LineCollection(segments, colors=color, alpha=alpha,
                                     linewidths=linewidth, rasterized=True))
    ax.scatter(np.repeat([0, 1], len(pairs)), pairs.T.ravel(), color=color,
               alpha=alpha, s=marker_size, linewidths=0, rasterized=True)
    ax.autoscale_view()
    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks([0, 1], ["D1", "D2"])
    ax.set_ylabel("Wasserstein Distance", fontsize=font_size, labelpad=1)
    ax.tick_params(labelsize=font_size, pad=2)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(True)
    return ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--linewidth", type=float, default=0.4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    ax = plot_wasserstein_dists_paired(args.csv_path, alpha=args.alpha,
                                      linewidth=args.linewidth)
    if args.output:
        ax.figure.savefig(args.output)
    if args.no_show:
        plt.close(ax.figure)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
