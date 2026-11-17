"""Plot paired D1/D2 Wasserstein distances, one connecting line per CSV image."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


def plot_wasserstein_dists_paired(csv_path, ax=None, *, color="0.45", alpha=0.08,
                                  linewidth=0.4, marker_size=2, font_size=7,
                                  line_color_mode="gray", marker_color_mode="gray",
                                  positive_color="#0072B2", negative_color="#D55E00",
                                  d1_color="#D55E00", d2_color="#0072B2", random_seed=0,
                                  negative_on_top=False, line_alpha=None, transparent_slopes="all"):
    """Plot every finite distance pair, preserving row correspondence."""
    if line_color_mode not in {"gray", "random", "slope"}:
        raise ValueError("line_color_mode must be gray, random, or slope")
    if marker_color_mode not in {"gray", "distance", "slope"}:
        raise ValueError("marker_color_mode must be gray, distance, or slope")
    if transparent_slopes not in {"all", "positive", "negative", "none"}:
        raise ValueError("transparent_slopes must be all, positive, negative, or none")
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
    slopes = pairs[:, 1] - pairs[:, 0]
    slope_colors = [positive_color if slope > 0 else negative_color if slope < 0 else color
                    for slope in slopes]
    line_colors = color
    if line_color_mode == "random":
        line_colors = plt.get_cmap("hsv")(np.random.default_rng(random_seed).random(len(pairs)))
    elif line_color_mode == "slope":
        line_colors = slope_colors
    if marker_color_mode == "slope":
        marker_colors = slope_colors + slope_colors
    elif marker_color_mode == "distance":
        marker_colors = [d1_color] * len(pairs) + [d2_color] * len(pairs)
    else:
        marker_colors = color
    line_opacity = alpha if line_alpha is None else line_alpha
    marker_opacity = alpha
    if transparent_slopes == "none":
        line_opacity = marker_opacity = 1.0
    elif transparent_slopes != "all":
        selected = slopes > 0 if transparent_slopes == "positive" else slopes < 0
        line_opacity = np.where(selected, line_opacity, 1.0)
        marker_opacity = np.tile(np.where(selected, alpha, 1.0), 2)
    if negative_on_top:
        # Stable ordering preserves CSV order within each slope group, and keeps
        # each random/slope color attached to its original observation.
        order = np.argsort(pairs[:, 1] < pairs[:, 0], kind="stable")
        segments = segments[order]
        if np.ndim(line_opacity):
            line_opacity = line_opacity[order]
        if line_color_mode != "gray":
            line_colors = np.asarray(line_colors)[order]
    ax.add_collection(LineCollection(segments, colors=line_colors,
                                     alpha=line_opacity,
                                     linewidths=linewidth, rasterized=True))
    ax.scatter(np.repeat([0, 1], len(pairs)), pairs.T.ravel(), color=marker_colors,
               alpha=marker_opacity, s=marker_size, linewidths=0, rasterized=True)
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
    parser.add_argument("--line-alpha", type=float, help="Line opacity; defaults to --alpha")
    parser.add_argument("--transparent-slopes", choices=["all", "positive", "negative", "none"], default="all")
    parser.add_argument("--linewidth", type=float, default=0.4)
    parser.add_argument("--line-color-mode", choices=["gray", "random", "slope"], default="gray")
    parser.add_argument("--marker-color-mode", choices=["gray", "distance", "slope"], default="gray")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--negative-on-top", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    ax = plot_wasserstein_dists_paired(args.csv_path, alpha=args.alpha,
                                      linewidth=args.linewidth, line_color_mode=args.line_color_mode,
                                      marker_color_mode=args.marker_color_mode, random_seed=args.random_seed,
                                      negative_on_top=args.negative_on_top, line_alpha=args.line_alpha,
                                      transparent_slopes=args.transparent_slopes)
    if args.output:
        ax.figure.savefig(args.output)
    if args.no_show:
        plt.close(ax.figure)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
