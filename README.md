# Paper Figures and Visualization Code

> **Origin**: This repository is a mirror of the local folder `PaperStuff` from `W:\ilanaveh\code\PaperStuff`.
>
> **Description**: This repository contains code and assets for generating paper figures related to face attention visualization, model performance analysis, and behavioral experiments. It is not a fork of an existing public repository but rather a collection of custom visualization scripts developed for research papers.

## Local path setup

On each computer, copy `Config/path_config.example.py` to
`Config/path_config.py` and edit the external roots at the top for that machine.
The local `path_config.py` is ignored by Git. The example is tracked so new
checkouts have a setup template. Repository input and output paths automatically
follow the checkout location; individual paths can also be overridden in the
local file. Create configured output directories before running scripts.

Scripts can be run directly from any working directory, for example:
`python Figures/attention_maps/plot_attention_prototypes.py`.

## Attention-map figure options

Edit `Config/figure_configs.py` to configure attention-map figures. The named
options include pooling, prototype type, models, emotions, mean-face overlays,
interpolation, colors, alpha, labels, row order, figure size, and human panels.
Machine-specific input paths remain in `Config/path_config.py`.

- `current` (default) preserves the existing figure settings.
- `models_only_blurred_s8` omits human panels and `High-S0`, and blurs only
  the mean-face backgrounds of `Mixed-High-S8` and `Low-S8` with Gaussian sigma 8
  pixels. Attention maps remain unchanged.

```sh
python Figures/attention_maps/plot_attention_prototypes.py --config current
python Figures/attention_maps/plot_attention_prototypes.py --config models_only_blurred_s8
```

Set `input_column` in `Config/figure_configs.py` to `"blur"` for numeric blur
values or `"example_input"` for example faces in the table's second column.
S0 rows use `RAFDB_IMAGES_DIR/test_2934.jpg`; S8 rows use
`BLURRED_FACES_DIR/test_2934_blur8.jpg`. Filenames and the example-input header
are configurable. Only the pretraining column is colored.

Use `--output figure.png --no-show` to save without opening a window.
Use an empty `genders` list or `show_humans=False` to omit human panels;
`genders=["f", "m"]` adds one panel row per gender. Row and column counts follow
the selected models, human panels, and emotions.

## Contents

### Figures/attention_maps/
Code and outputs for generating attention map visualizations:
- `plot_attention_prototypes.py` - Generates prototype attention map figures
- `preprocess_behavior_fig.py` - Preprocesses behavioral data for figure generation
- `get_blurred_ims.py` - Creates blurred face images for attention analysis
- `bhvr_fig_split/` - Behavioral figure splits by emotion (anger, disgust, fear, happiness, neutral, sadness, surprise)
- `blurred_face_ims/` - Sample blurred face images at different blur levels

### Figures/models_performance_bars/
Bar plots for model performance comparison:
- `plot_bar_graph.py` - Main script for generating bar plots
- `config.py` - Configuration for different experimental setups
- Multiple config subdirectories for different experimental conditions (freeze backbone, test blur levels, etc.)

### Figures/models_performance_scatter/
Scatter plots for model performance analysis:
- `plot_scatter_graph.py` - Generates scatter plots comparing model performances

### tools/
Utility scripts:
- `split_rafdb.py` - Script for splitting RAF-DB dataset

## Purpose

These visualization tools were developed to support research on:
- **Attention mechanism interpretability** in vision transformers for face analysis
- **Model robustness** to input image blur variations
- **Behavioral analysis** of model predictions across emotional expressions
- **Performance comparison** across different model architectures and training configurations

The figures generated from this codebase have been used in academic publications and presentations.

### Wasserstein delta histogram

`Figures/attention_maps/plot_wasserstein_deltas_hist.py` plots the
`delta_a_minus_b` column of a supplied CSV. For example:

```sh
python Figures/attention_maps/plot_wasserstein_deltas_hist.py "path/to/paired_distances.csv" --bins 30 --output deltas.png --no-show
```

The attention-map figure includes this histogram to the right of the maps, aligned with their bottom edge.
In `Config/figure_configs.py`, set `wasserstein_histogram_csv` to the input CSV;
the default follows the local prototype root. Bin count, color, title, and panel
position/size are configurable. Set `show_wasserstein_histogram=False` to hide it.
The histogram negates `delta_a_minus_b` to show D2 - D1 and weights observations
so bin heights sum to 100% of the usable test images. All finite numeric deltas are plotted; missing, nonnumeric, and infinite values
are excluded. Missing columns or an input with no usable deltas raise an error.

The histogram defaults to 15 bins. `wasserstein_histogram_xlim_quantiles`
limits the visible x-range to the central 99% by default; set it to `None`
to show the full range. Tail observations still contribute to the histogram
and percentage normalization. The vertical offset, zero-line width, and y-ticks
are also configurable in `Config/figure_configs.py`.

### Paired Wasserstein distance scatter

Set `wasserstein_panel_type="scatter"` in `Config/figure_configs.py` to show
D1 (`wasserstein_distance_a`) on x and D2 (`wasserstein_distance_b`) on y,
with a dashed y=x line. Set it to `"histogram"` for the delta histogram (default).
Both use `wasserstein_histogram_csv`, the existing panel dimensions/position,
and `show_wasserstein_histogram` as the shared visibility switch. Scatter color,
marker size, and alpha have their own `wasserstein_scatter_*` settings.
Rows with either distance missing, nonnumeric, or infinite are excluded together.

To run the scatter independently:

```sh
python Figures/attention_maps/plot_wasserstein_dists_scatter.py "path/to/paired_distances.csv" --output distances.png --no-show
```

### Paired distances and stacked panels

`Figures/attention_maps/plot_wasserstein_dists_paired.py` accepts the paired CSV
as a positional argument (plus `--output` and `--no-show`). Each finite CSV pair
is drawn at D1 and D2 with a connecting line. No observations are subsampled.

In `Config/figure_configs.py`, choose one plot or two stacked plots:

```python
"wasserstein_panel_type": "paired",
# Or, top to bottom:
"wasserstein_panel_type": ["paired", "histogram"],
```

The stack is centered on the middle model row. Adjust
`wasserstein_panel_stack_gap_inches` for vertical spacing. Both panels use the
shared CSV and panel dimensions; `wasserstein_paired_*` controls paired styling.
The delta bracket is shown when the selection includes a histogram.
