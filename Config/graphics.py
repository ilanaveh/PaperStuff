"""Shared graphics settings for paper figures."""

# Pretraining-regime palette from plot_scatter_balanced.py (RGB, range 0-1).
model_colors = {
    "Low": (0.68, 0.56, 0.82),
    "Mixed-High": (0.38, 0.76, 0.38),
    "High": (1, 0.753, 0.525),
    "Mixed-Low": (0.651, 0.792, 0.925),
}

# Colors for wasserstein:
wasser_colors = {
    # "D2>D1": "#007C91",
    # "D2<D1": "#E69F00",
    "D2>D1": "#007C91",
    "D2<D1": "#D55E00",
}