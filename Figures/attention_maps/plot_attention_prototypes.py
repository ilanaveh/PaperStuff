"""
1/6/26
Based on plot_prototypes from Transformer_Explainability/compare_models.
Assumes:
 1. prototypes were created and saved (by Transformer_Explainability/compare_models/build_prototypes.py).
 2. behavior results from Wegrzyn et al 2017 were preprocessed and saved (by preprocess_behavior_fig.py).
"""
import sys
from pathlib import Path

# Support running this script directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Config import path_config as paths
from Config.graphics import model_colors
from Figures.attention_maps.plot_wasserstein_deltas_hist import plot_wasserstein_deltas_hist
from Figures.attention_maps.plot_wasserstein_dists_scatter import plot_wasserstein_dists_scatter
from Figures.attention_maps.plot_wasserstein_dists_paired import plot_wasserstein_dists_paired
from Config.figure_configs import (
    ATTENTION_MAP_CONFIGS, DEFAULT_ATTENTION_MAP_CONFIG, get_attention_map_config,
)

import numpy as np
import pandas as pd
import os
import ast
import cv2
import argparse
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import torch

def plot_attention_maps(config):
    """Build a figure using a preset from Config.figure_configs."""
    selection = config["wasserstein_panel_type"]
    panel_types = [selection] if isinstance(selection, str) else list(selection)
    if not 1 <= len(panel_types) <= 2 or any(
        kind not in {"histogram", "scatter", "paired"} for kind in panel_types
    ):
        raise ValueError("wasserstein_panel_type must select one or two of histogram, scatter, paired")
    wasserstein_axes = []
    title_weight = "bold" if config["titles_bold"] else "normal"
    POOLING = config["POOLING"]
    prototype_config = config["prototype_type"]
    models = config["models"]
    model_labels = config["model_labels"]
    input_column = config["input_column"]
    if input_column not in {"blur", "example_input"}:
        raise ValueError("input_column must be 'blur' or 'example_input'")
    example_inputs = {}
    if input_column == "example_input":
        for mdl in models:
            sigma = int(mdl.split("blur")[1].split("_")[0])
            if sigma not in example_inputs:
                directory = paths.RAFDB_IMAGES_DIR if sigma == 0 else paths.BLURRED_FACES_DIR
                image_path = Path(directory) / config["example_input_filenames"][sigma]
                example_inputs[sigma] = plt.imread(image_path)
    EMOTIONS = config["EMOTIONS"]
    emo_labels = config["emotion_labels"]
    show_on_mean_face = config["show_on_mean_face"]
    interpolation_mode = config["interpolation_mode"]
    cmap = config["cmap"]
    gamma = config["gamma"]
    min_alpha = config["min_alpha"]
    max_alpha = config["max_alpha"]
    genders = config["genders"] if config["show_humans"] else []
    show_ttl = config["show_ttl"]
    model_rows = {model_labels[m] for m in models}
    row_order = [r for r in config["row_order"]
                 if r in model_rows or (r == "humans" and genders)]
    if set(row_order) != model_rows | ({"humans"} if genders else set()):
        raise ValueError("row_order must include every selected model and human row")
    human_r = row_order.index("humans") if genders else None
    # Expand the human entry into one row per selected gender.
    model_row_indices = {
        label: i + (len(genders) - 1 if genders and i > human_r else 0)
        for i, label in enumerate(row_order) if label != "humans"
    }
    # ===========================================================================
    # 1. Load prototypes
    # ===========================================================================
    summary_path = paths.prototype_summary_path(prototype_config, POOLING)

    summary_df = pd.read_csv(summary_path)

    prototypes = {}

    for _, row in summary_df.iterrows():
        key = ast.literal_eval(row["key"])
        prototype_path = paths.resolve_prototype_path(row["prototype_path"])
        mean_face_path = paths.resolve_prototype_path(row["mean_face_path"])

        if not prototype_path.exists():
            raise FileNotFoundError(
                f"Could not find prototype file: {prototype_path}"
            )

        prototypes[key] = {
            "prototype": np.load(prototype_path),
            "mean_face": np.load(mean_face_path) if mean_face_path.exists() else None,
            "n": int(row["n"]),
            "map_paths": [],
        }

    # ===========================================================================
    # 2. Load behavior results from Wegrzyn et al 2017
    # ===========================================================================
    bhvr_figs_dir = paths.BEHAVIOR_PANELS_DIR

    # ===========================================================================
    # Create figure
    # ===========================================================================
    nrows = len(models) + len(genders)
    ncols = len(EMOTIONS)

    fig, axs = plt.subplots(nrows, ncols, squeeze=False)
    table_width = config["model_table_width_inches"]
    colorbar_space = config["colorbar_space_inches"] if config["show_colorbar"] else 0
    figure_width = (config["figsize"][0] + table_width) * config["figure_width_scale"]
    figure_height = config["figsize"][1] + colorbar_space
    fig.set_size_inches(figure_width, figure_height)
    table_fraction = table_width / figure_width

    for c, emo in enumerate(EMOTIONS):
        axs[0, c].set_title(emo, fontsize=config["model_table_header_font_size"], weight=title_weight,
                            pad=config["emotion_heading_pad_points"])

        # Model attention prototypes:
        for mdl in models:
            mdl_lbl = model_labels[mdl]
            sigma = int(mdl.split('blur')[1].split('_')[0])
            r = model_row_indices[mdl_lbl]
            ax = axs[r, c]
            key = (mdl, sigma, emo)

            if key in prototypes:
                proto = prototypes[key]["prototype"]
                proto_range = proto.max() - proto.min()
                proto_minmax = (proto - proto.min()) / proto_range if proto_range else np.zeros_like(proto)

                if show_on_mean_face:
                    face = prototypes[key]['mean_face']
                    if face is None:
                        raise ValueError(f"Missing mean face for {key}")
                    face = face.copy()
                    blur_sigma = config["mean_face_blur_sigma"].get(mdl, 0)
                    if blur_sigma > 0:
                        face = cv2.GaussianBlur(face, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
                    ax.imshow(np.clip(face, 0, 1))

                    if proto_minmax.shape != face.shape[:2]:
                        display_proto = torch.nn.functional.interpolate(
                            torch.from_numpy(proto_minmax).reshape(1, 1, *proto_minmax.shape),
                            size=face.shape[:2],
                            mode=interpolation_mode).reshape(face.shape[:2]).numpy()
                    else:
                        display_proto = proto_minmax

                    alpha = np.clip(display_proto, 0, 1)
                    alpha = alpha ** gamma
                    alpha_map = min_alpha + alpha * (max_alpha - min_alpha)

                    ax.imshow(display_proto, cmap=cmap, alpha=alpha_map, vmin=0, vmax=1)

                else:
                    ax.imshow(proto_minmax, cmap=cmap, vmin=0, vmax=1)

            ax.set_xticks([])
            ax.set_yticks([])

            if show_ttl and key in prototypes:
                ax.set_title(f"N={prototypes[key]['n']}", weight=title_weight, fontsize=config["font_size"])

        # Optional human behavior rows:
        for r, gen in enumerate(genders):
            ax = axs[r+human_r, c]
            fig_nm = f'{gen}_{emo_labels[emo]}.png'
            face = plt.imread(os.path.join(bhvr_figs_dir, fig_nm))
            ax.imshow(face, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout(rect=(table_fraction, colorbar_space / figure_height, 1, 0.93))
    for r in model_row_indices.values():
        for ax in axs[r]:
            pos = ax.get_position()
            scale = config["attention_map_scale"]
            width, height = pos.width * scale, pos.height * scale
            ax.set_position([pos.x0 + (pos.width - width) / 2,
                             pos.y0 + (pos.height - height) / 2, width, height])
    # Remove the automatic left padding between the table and image panels.
    table_right = table_fraction - 0.01
    desired_left = table_right + config["table_map_gap_inches"] / figure_width
    column_width = max(ax.get_position().width for ax in axs.flat)
    column_pitch = column_width + config["attention_map_column_gap_inches"] / figure_width
    row_height = max(ax.get_position().height for ax in axs.flat)
    row_pitch = row_height + config["attention_map_row_gap_inches"] / figure_height
    rows_top = max(ax.get_position().y1 for ax in axs[0]) + config["rows_top_shift_inches"] / figure_height
    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r, c]
            pos = ax.get_position()
            ax.set_position([desired_left + c * column_pitch,
                             rows_top - r * row_pitch - row_height / 2 - pos.height / 2,
                             pos.width, pos.height])

    maps_left = min(ax.get_position().x0 for ax in axs.flat)
    maps_right = max(ax.get_position().x1 for ax in axs.flat)
    attention_title = fig.text((maps_left + maps_right) / 2, config["attention_maps_title_y"],
             config["attention_maps_title"], ha="center", va="center",
             fontsize=config["model_table_header_font_size"], weight=title_weight)

    if config["show_colorbar"]:
        model_axes = [axs[r, c] for r in model_row_indices.values() for c in range(ncols)]
        bottom = min(ax.get_position().y0 for ax in model_axes)
        left = min(ax.get_position().x0 for ax in model_axes)
        right = max(ax.get_position().x1 for ax in model_axes)
        bar_height = config["colorbar_width_inches"] / figure_height
        bar_width = (right - left) * config["colorbar_length_fraction"]
        bar_left = left + (right - left) * config["colorbar_left_fraction"]
        bar_bottom = bottom - config["colorbar_gap_inches"] / figure_height - bar_height
        colorbar_ax = fig.add_axes([bar_left, bar_bottom, bar_width, bar_height])
        # Every prototype uses its own min-max normalization to the same 0-1 scale.
        colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                               cax=colorbar_ax, ticks=config["colorbar_ticks"],
                               orientation="horizontal")
        fig.text(bar_left - config["colorbar_label_gap_inches"] / figure_width,
                 bar_bottom + bar_height / 2 - config["colorbar_label_down_inches"] / figure_height,
                 config["colorbar_label"],
                 ha="right", va="center", multialignment="left",
                 fontsize=config["model_table_header_font_size"])
        colorbar.ax.tick_params(labelsize=config["model_table_font_size"])

    # Separate table cells share each model panel's vertical center. Human rows
    # stay outside the model table, including when their panels are resized.
    table_ax = fig.add_axes([0.01, 0, table_fraction - 0.02, 1])
    table_ax.set_axis_off()
    pretrain_fraction = config["pretrain_column_fraction"]
    column_widths = [pretrain_fraction, 1 - pretrain_fraction]
    headers = list(config["model_table_headers"])
    if input_column == "example_input":
        headers[1] = config["example_input_header"]
    title_y = config["attention_maps_title_y"] - config["left_titles_down_inches"] / figure_height
    title_style = dict(ha="center", va="center", weight=title_weight,
                       fontsize=config["model_table_header_font_size"])
    table_ax.text(pretrain_fraction / 2 + 0.03, title_y, headers[0], **title_style)
    title_offset = config["example_input_title_offset_inches"] / (
        table_ax.get_position().width * figure_width
    )
    example_title = table_ax.text((1 + pretrain_fraction) / 2 + title_offset,
                  title_y, headers[1], **title_style)
    for mdl in models:
        r = model_row_indices[model_labels[mdl]]
        pos = axs[r, 0].get_position()
        sigma = int(mdl.split("blur")[1].split("_")[0])
        row_table = table_ax.table(
            cellText=[[config["model_pretrain_regimes"][mdl],
                       str(sigma) if input_column == "blur" else ""]],
            cellLoc="center", colWidths=column_widths,
            bbox=[0, pos.y0, 1, pos.height],
        )
        row_table.auto_set_font_size(False)
        row_table.set_fontsize(config["model_table_font_size"])
        row_table[0, 0].get_text().set_rotation(config["pretrain_label_rotation"])
        color_group = model_labels[mdl].rsplit("-S", 1)[0]
        padding = config["row_background_padding_inches"]
        vertical_padding = config["row_vertical_padding_inches"]
        # Use table coordinates so backgrounds follow vertical figure trimming.
        background_right = maps_right + padding / figure_width
        background_width = (background_right - table_ax.get_position().x0) / table_ax.get_position().width
        fig.add_artist(Rectangle(
            (0, pos.y0 - vertical_padding / figure_height), background_width,
            pos.height + 2 * vertical_padding / figure_height,
            transform=table_ax.transAxes, facecolor="none",
            edgecolor="black", linewidth=config["row_frame_linewidth"],
            clip_on=False, zorder=10,
        ))
        fig.add_artist(Rectangle(
            (0, pos.y0 - vertical_padding / figure_height), pretrain_fraction,
            pos.height + 2 * vertical_padding / figure_height,
            transform=table_ax.transAxes, facecolor=model_colors[color_group],
            edgecolor="none", clip_on=False, zorder=-1,
        ))
        for cell in row_table.get_celld().values():
            cell.set_facecolor("none")
            cell.set_linewidth(0)
        if input_column == "example_input":
            input_gap = config["example_input_gap_inches"] / (
                table_ax.get_position().width * figure_width
            )
            image_down = (config["example_input_sigma_image_down_inches"] / figure_height
                          if config["example_input_show_sigma"] else 0)
            input_ax = table_ax.inset_axes([
                pretrain_fraction + input_gap, pos.y0 - image_down,
                1 - pretrain_fraction - input_gap, pos.height,
            ])
            input_ax.imshow(example_inputs[sigma])
            input_ax.set_axis_off()
            if config["example_input_show_sigma"]:
                input_ax.set_title(
                    rf"$\sigma = {sigma}$", fontsize=config["model_table_header_font_size"],
                    weight=title_weight, pad=config["example_input_sigma_title_pad_points"],
                )

    # Brackets use table coordinates to stay aligned when figure height is fitted.
    table_pos = table_ax.get_position()
    bracket_left = (maps_right + (config["row_background_padding_inches"]
                    + config["row_bracket_gap_inches"]) / figure_width - table_pos.x0) / table_pos.width
    bracket_width = config["row_bracket_width_inches"] / (figure_width * table_pos.width)

    def add_right_bracket(x, y1, y2, label):
        fig.add_artist(Line2D(
            [x, x + bracket_width, x + bracket_width, x], [y1, y1, y2, y2],
            transform=table_ax.transAxes, color="black",
            linewidth=config["row_bracket_linewidth"], clip_on=False, zorder=-0.5,
        ))
        # White label background masks the stem directly underneath the text.
        table_ax.text(x + bracket_width, (y1 + y2) / 2, label,
                      ha="center", va="center", fontsize=config["model_table_font_size"],
                      bbox=dict(facecolor="white", edgecolor="none",
                                pad=config["row_bracket_label_padding_points"]),
                      clip_on=False, zorder=20)

    bracket_centers = []
    for index, (first, second) in enumerate(config["row_bracket_pairs"]):
        if first not in model_row_indices or second not in model_row_indices:
            continue
        first_pos = axs[model_row_indices[first], 0].get_position()
        second_pos = axs[model_row_indices[second], 0].get_position()
        y1 = first_pos.y0 + first_pos.height / 2
        y2 = second_pos.y0 + second_pos.height / 2
        half_gap = config["row_bracket_vertical_gap_inches"] / (2 * figure_height)
        y_offset = config["row_bracket_y_offsets_inches"][index] / figure_height
        add_right_bracket(bracket_left, y1 - half_gap + y_offset,
                          y2 + half_gap + y_offset, config["row_bracket_labels"][index])
        bracket_centers.append((y1 + y2) / 2)
    if len(bracket_centers) == 2 and "histogram" in panel_types:
        outer_left = bracket_left + bracket_width + config["row_bracket_outer_gap_inches"] / (
            figure_width * table_pos.width
        )
        center = sum(bracket_centers) / 2
        half_height = abs(bracket_centers[0] - bracket_centers[1]) / 2
        half_height *= config["row_bracket_outer_height_scale"]
        add_right_bracket(outer_left, center + half_height, center - half_height,
                          config["row_bracket_outer_label"])

    if len(genders) == 1:
        axs[human_r, 0].set_ylabel(config["human_label"], fontsize=config["font_size"])
        axs[human_r, 0].yaxis.labelpad = config["human_label_pad"]

        for c in range(len(EMOTIONS)):
            ax = axs[human_r, c]
            pos = ax.get_position()

            new_w = pos.width * config["human_panel_scale"]
            new_h = pos.height * config["human_panel_scale"]

            # keep subplot centered around its original center
            new_x0 = pos.x0 + ((pos.width - new_w) / 2)
            new_y0 = pos.y0 + ((pos.height - new_h) / config["human_panel_vertical_divisor"])

            ax.set_position([new_x0, new_y0, new_w, new_h])
    elif genders:
        fig.supylabel(config["human_label"].replace("\n", " "),
                      x=config["multi_human_label_position"][0],
                      y=config["multi_human_label_position"][1],
                      fontsize=config["multi_human_font_size"])

    if config["show_wasserstein_histogram"]:
        # Add room on the right while preserving existing panel sizes/positions.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        right_edge = max(text.get_window_extent(renderer).x1 for text in table_ax.texts) / fig.dpi
        hist_left = right_edge + config["wasserstein_histogram_gap_inches"]
        hist_width = config["wasserstein_histogram_size_inches"]
        if len(panel_types) == 2:
            hist_width *= config["wasserstein_panel_stack_width_scale"]
        hist_height_inches = config["wasserstein_histogram_height_inches"]
        hist_height = hist_height_inches / figure_height
        row_centers = [axs[r, 0].get_position().y0 + axs[r, 0].get_position().height / 2
                       for r in model_row_indices.values()]
        stack_gap = config["wasserstein_panel_stack_gap_inches"] / figure_height
        stack_height = len(panel_types) * hist_height + (len(panel_types) - 1) * stack_gap
        stack_bottom = (float(np.median(row_centers)) - stack_height / 2
                        + config["wasserstein_histogram_y_offset_inches"] / figure_height)
        for index, panel_type in enumerate(panel_types):
            panel_bottom = stack_bottom + (len(panel_types) - 1 - index) * (hist_height + stack_gap)
            hist_ax = fig.add_axes([
                hist_left / figure_width, panel_bottom, hist_width / figure_width, hist_height,
            ])
            wasserstein_axes.append(hist_ax)
            if panel_type == "histogram":
                plot_wasserstein_deltas_hist(
                    config["wasserstein_histogram_csv"], hist_ax,
                    bins=config["wasserstein_histogram_bins"],
                    color=config["wasserstein_histogram_color"], font_size=config["font_size"],
                    title=config["wasserstein_histogram_title"], title_weight=title_weight,
                    xlim_quantiles=config["wasserstein_histogram_xlim_quantiles"],
                    zero_linewidth=config["wasserstein_histogram_zero_linewidth"],
                    yticks=config["wasserstein_histogram_yticks"],
                )
            elif panel_type == "scatter":
                plot_wasserstein_dists_scatter(
                    config["wasserstein_histogram_csv"], hist_ax,
                    color=config["wasserstein_scatter_color"],
                    marker_size=config["wasserstein_scatter_marker_size"],
                    alpha=config["wasserstein_scatter_alpha"], font_size=config["font_size"],
                    show_ticklabels=config["wasserstein_scatter_show_ticklabels"],
                )
            else:
                plot_wasserstein_dists_paired(
                    config["wasserstein_histogram_csv"], hist_ax,
                    color=config["wasserstein_paired_color"],
                    alpha=config["wasserstein_paired_alpha"],
                    linewidth=config["wasserstein_paired_linewidth"],
                    marker_size=config["wasserstein_paired_marker_size"], font_size=config["font_size"],
                )
            hist_ax.set_box_aspect(hist_height_inches / hist_width)
            hist_ax.set_anchor("SW")
        # Measure tick/axis labels so the histogram fits closely without overlap.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        decoration_width = max((ax.get_window_extent(renderer).x0
                                - ax.get_tightbbox(renderer).x0) / fig.dpi
                               for ax in wasserstein_axes)
        hist_left = right_edge + config["wasserstein_histogram_gap_inches"] + decoration_width
        new_width = hist_left + hist_width + 0.04
        old_width = figure_width
        positions = [(ax, ax.get_position().frozen()) for ax in fig.axes]
        text_positions = [(text, text.get_position()) for text in fig.texts]
        fig.set_size_inches(new_width, figure_height)
        for ax, pos in positions:
            ax.set_position([pos.x0 * old_width / new_width, pos.y0,
                             pos.width * old_width / new_width, pos.height])
        for text, (x, y) in text_positions:
            text.set_position((x * old_width / new_width, y))
        figure_width = new_width
        for ax in wasserstein_axes:
            pos = ax.get_position()
            ax.set_position([hist_left / new_width, pos.y0, hist_width / new_width, hist_height])

    # Align the rendered top edges, including the multiline example header.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_delta = (example_title.get_window_extent(renderer).y1
                   - attention_title.get_window_extent(renderer).y1)
    title_x, title_y = attention_title.get_position()
    attention_title.set_position((title_x, title_y + title_delta / fig.bbox.height))

    if len(wasserstein_axes) == 2:
        # Fit the stack including labels to the attention section's rendered span.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        main_artists = [ax for ax in fig.axes if ax is not table_ax and ax not in wasserstein_axes]
        main_artists += list(table_ax.texts) + list(table_ax.tables) + list(table_ax.child_axes)
        main_artists += list(fig.texts)
        main_bounds = [artist.get_tightbbox(renderer) for artist in main_artists]
        main_bounds = [box for box in main_bounds if box is not None]
        main_bottom = min(box.y0 for box in main_bounds) / fig.dpi
        main_top = max(box.y1 for box in main_bounds) / fig.dpi
        upper_ax, lower_ax = wasserstein_axes
        top_decoration = (upper_ax.get_tightbbox(renderer).y1
                          - upper_ax.get_window_extent(renderer).y1) / fig.dpi
        bottom_decoration = (lower_ax.get_window_extent(renderer).y0
                             - lower_ax.get_tightbbox(renderer).y0) / fig.dpi
        gap = config["wasserstein_panel_stack_gap_inches"]
        hist_height_inches = (main_top - main_bottom - top_decoration - bottom_decoration - gap) / 2
        if hist_height_inches <= 0:
            raise ValueError("Stack gap leaves no vertical room for Wasserstein panels")
        lower_bottom = main_bottom + bottom_decoration
        for index, ax in enumerate(wasserstein_axes):
            y_inches = lower_bottom + (1 - index) * (hist_height_inches + gap)
            pos = ax.get_position()
            ax.set_position([pos.x0, y_inches / figure_height,
                             hist_width / figure_width, hist_height_inches / figure_height])
            ax.set_box_aspect(hist_height_inches / hist_width)

    if config["fit_figure_height"]:
        # Keep physical panel sizes while trimming unused space above/below.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        artists = [ax for ax in fig.axes if ax is not table_ax]
        artists += list(table_ax.texts) + list(table_ax.tables) + list(table_ax.child_axes)
        artists += list(fig.texts)
        bounds = [artist.get_tightbbox(renderer) for artist in artists]
        bounds = [box for box in bounds if box is not None]
        padding = config["figure_vertical_padding_inches"]
        bottom = min(box.y0 for box in bounds) / fig.dpi - padding
        top = max(box.y1 for box in bounds) / fig.dpi + padding
        new_height = top - bottom
        positions = [(ax, ax.get_position().frozen()) for ax in fig.axes]
        text_positions = [(text, text.get_position()) for text in fig.texts]
        fig.set_size_inches(figure_width, new_height)
        for ax, pos in positions:
            ax.set_position([pos.x0, (pos.y0 * figure_height - bottom) / new_height,
                             pos.width, pos.height * figure_height / new_height])
        for text, (x, y) in text_positions:
            text.set_position((x, (y * figure_height - bottom) / new_height))

    for ax in wasserstein_axes:
        # Preserve physical panel dimensions after figure resizing/trimming.
        pos = ax.get_position()
        size = hist_width
        height = hist_height_inches
        ax.set_position([pos.x0, pos.y0, size / fig.get_figwidth(),
                         height / fig.get_figheight()])
        ax.set_box_aspect(height / size)
        ax.set_anchor("SW")
    fig.canvas.draw()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=ATTENTION_MAP_CONFIGS,
                        default='models_only_blurred_s8')
    parser.add_argument("--output", type=Path, help="Save the figure to this path",
                        default="attention_prototypes_with_wasser.png")
    parser.add_argument("--no-show", action="store_true", help="Skip the interactive window")
    args = parser.parse_args()
    fig = plot_attention_maps(get_attention_map_config(args.config))
    if args.output:
        fig.savefig(args.output)
    if not args.no_show:
        plt.show(block=True)
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
