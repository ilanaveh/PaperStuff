"""Named figure options. Select with --config in the corresponding plot script.

Paths are configured separately in path_config.py. Add future figure families
alongside ATTENTION_MAP_CONFIGS.
"""
from copy import deepcopy
from pathlib import Path
from . import path_config as paths


_ATTENTION_CURRENT = {
    "POOLING": True,
    "prototype_type": "pred_label_all",  # or "true_label_correct"
    "models": [
        "RAF_blur8_pretrained0-8_freeze_fix_projs",
        "RAF_blur8_freeze",
        "RAF_blur0_pretrained0-8_freeze",
        "RAF_blur0_freeze",
    ],
    "model_labels": {
        "RAF_blur0_freeze": "High-S0",
        "RAF_blur0_pretrained0-8_freeze": "Mixed-High-S0",
        "RAF_blur8_pretrained0-8_freeze_fix_projs": "Mixed-High-S8",
        "RAF_blur8_freeze": "Low-S8",
    },
    "EMOTIONS": ["Happy", "Angry", "Sad", "Surprise", "Fear", "Disgust", "Neutral"],
    "emotion_labels": {
        "Happy": "hap", "Sad": "sad", "Angry": "ang", "Fear": "fea",
        "Disgust": "dis", "Surprise": "sup", "Neutral": "ntr",
    },
    "show_on_mean_face": True,
    "interpolation_mode": "bilinear",  # or "nearest"
    "cmap": "jet",
    "gamma": 1.5,
    "min_alpha": 0.3,
    "max_alpha": 0.65,
    "show_humans": True,
    "genders": ["f"],  # choose from "f", "m"
    "show_ttl": False,  # show prototype sample counts
    "row_order": ["humans", "High-S0", "Mixed-High-S0", "Mixed-High-S8", "Low-S8"],
    # Per-model Gaussian sigma in image pixels; absent models stay unblurred.
    "mean_face_blur_sigma": {},
    "figsize": (6.2, 3.4),
    "figure_width_scale": 0.8,
    "font_size": 7,
    "titles_bold": False,  # False uses normal weight for all titles/headings.
    "attention_maps_title": "",  # "Mean Relevance-Maps Across all Test Images"
    "attention_maps_title_y": 0.94,
    "input_column": "example_input",  # "blur" or "example_input"
    "example_input_show_sigma": True,  # Add a sigma title above each example face.
    "example_input_sigma_title_pad_points": 3,
    "example_input_sigma_image_down_inches": 0.04,
    "example_input_header": "Example\nTest\nInput",
    "example_input_filenames": {
        0: "test_2788.jpg",       # in RAFDB_IMAGES_DIR
        8: "test_2788_blur8.jpg", # in BLURRED_FACES_DIR
    },
    "model_table_headers": ["Pretrain\nRegime", "Image\nBlur\n(pixels)"],
    "model_pretrain_regimes": {
        "RAF_blur0_freeze": "High\nResolution",
        "RAF_blur0_pretrained0-8_freeze": "Mixed\nResolution",
        "RAF_blur8_pretrained0-8_freeze_fix_projs": "Mixed\nResolution",
        "RAF_blur8_freeze": "Low\nResolution",
    },
    "model_table_width_inches": 1.05,
    "model_table_font_size": 7,
    "model_table_header_font_size": 7,
    "pretrain_column_fraction": 0.3,
    "pretrain_label_rotation": 90,
    "example_input_gap_inches": 0.04,
    "table_map_gap_inches": 0.04,
    "attention_map_scale": 0.85,
    "attention_map_column_gap_inches": 0.07,
    "attention_map_row_gap_inches": 0.31,
    "left_titles_down_inches": 0.12,
    "rows_top_shift_inches": 0.12,
    "row_background_padding_inches": 0.07,
    "row_vertical_padding_inches": 0.13,
    "row_bracket_pairs": [
        ("Mixed-High-S0", "Mixed-High-S8"),
        ("Mixed-High-S8", "Low-S8"),
    ],
    "row_bracket_gap_inches": 0.07,
    "row_bracket_width_inches": 0.055,
    "row_bracket_outer_gap_inches": 0.09,
    "row_bracket_linewidth": 1.2,
    "row_bracket_vertical_gap_inches": 0.06,
    "row_bracket_outer_height_scale": 0.7,
    "row_bracket_labels": ["D1", "D2"],
    "row_bracket_y_offsets_inches": [0.06, -0.06],
    "row_bracket_outer_label": "\u0394",
    "row_bracket_label_gap_inches": 0.04,
    "row_bracket_label_padding_points": 2.6,
    "row_frame_linewidth": 1,
    "emotion_heading_pad_points": 17,
    "show_colorbar": True,
    "colorbar_label": "Normalized\nAttention",
    "colorbar_ticks": [0, 0.5, 1],
    "colorbar_space_inches": 0.65,
    "colorbar_width_inches": 0.05,
    "colorbar_gap_inches": 0.19,
    "colorbar_length_fraction": 0.35,
    "colorbar_label_gap_inches": 0.12,
    "colorbar_label_down_inches": 0.07,
    "colorbar_left_fraction": 0.2,
    "example_input_title_offset_inches": 0.06,
    "show_wasserstein_histogram": True,  # Show/hide the distance panel in either mode.
    "wasserstein_panel_type": ["paired", "histogram"],  # "histogram", "scatter", or "paired"
    # A list selects two panels in top-to-bottom order, e.g. ["paired", "histogram"].
    "wasserstein_panel_stack_gap_inches": 0.4,
    "wasserstein_panel_stack_width_scale": 0.9,
    "wasserstein_paired_color": "0.45",
    "wasserstein_paired_alpha": 0.08,
    "wasserstein_paired_linewidth": 0.4,
    "wasserstein_paired_marker_size": 2,
    "wasserstein_scatter_color": "0.45",
    "wasserstein_scatter_marker_size": 3,
    "wasserstein_scatter_alpha": 0.35,
    "wasserstein_scatter_show_ticklabels": False,
    "wasserstein_histogram_csv": str(
        Path(paths.PROTOTYPE_DIR) / "attention_comparisons"
        / "paired_resolution_vs_model" / "paired_distances.csv"
    ),
    "wasserstein_histogram_bins": 15,
    "wasserstein_histogram_color": "0.45",
    "wasserstein_histogram_title": "",
    "wasserstein_histogram_xlim_quantiles": (0.005, 0.995),  # None shows full range.
    "wasserstein_histogram_y_offset_inches": 0.0,  # Relative to middle model-row center.
    "wasserstein_histogram_zero_linewidth": 1.3,
    "wasserstein_histogram_yticks": list(range(0, 101, 5)),
    # Shared panel dimensions (histogram or scatter), centered on the middle model row.
    "wasserstein_histogram_size_inches": 1.35,  # Plot width.
    "wasserstein_histogram_height_inches": 1.3,
    "wasserstein_histogram_gap_inches": 0.20,
    "fit_figure_height": True,
    "figure_vertical_padding_inches": 0.06,
    "human_label": "Human\nObservers",
    "human_label_pad": 15,
    "human_panel_scale": 0.75,
    "human_panel_vertical_divisor": 1.5,
    "multi_human_label_position": (0.05, 0.26),
    "multi_human_font_size": 7,
}

ATTENTION_MAP_CONFIGS = {"current": deepcopy(_ATTENTION_CURRENT)}
ATTENTION_MAP_CONFIGS["models_only_blurred_s8"] = {
    **deepcopy(_ATTENTION_CURRENT),
    "models": [m for m in _ATTENTION_CURRENT["models"] if m != "RAF_blur0_freeze"],
    "show_humans": False,
    "row_order": ["Mixed-High-S0", "Mixed-High-S8", "Low-S8"],
    "mean_face_blur_sigma": {
        "RAF_blur8_pretrained0-8_freeze_fix_projs": 8,
        "RAF_blur8_freeze": 8,
    },
}
DEFAULT_ATTENTION_MAP_CONFIG = "current"


def get_attention_map_config(name=DEFAULT_ATTENTION_MAP_CONFIG):
    """Return an independent copy so edits do not mutate another preset."""
    return deepcopy(ATTENTION_MAP_CONFIGS[name])
