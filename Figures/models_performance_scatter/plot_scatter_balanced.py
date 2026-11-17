"""
29/10/26
Based on plot_scatter_graph.py, but:
 - use results_balnced, generated from APViT.eval.get_eval_results_balanced.py.
 - show log-scale x axis.
 - add accuracies for blur 32 & 64 (to show convergence to chance).
 - currently show only blur 8 (remove 4).
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Config import path_config as paths
from Config.graphics import model_colors as colors

import numpy as np
import matplotlib.pyplot as plt
import json
import math
# from plot_scatter_graph import clean_model_names


save_fig = True

# style
marker_size = 1
yerr_capsize = 2
plot_linewidth = 2.5
pt_linewidth = 4
pt_row_spacing = 2
pt_vertical_padding = 4
chance_color = 'k'  # Matplotlib color shared by the chance lines and label
plt.rcParams.update({'font.size': 12})
linebreak = '\n'
labl_spc = {'Low': 8, 'Mixed-High': 1, 'High': 7, 'Mixed-Low': 1}

# defenitions
results = {}
n_test_samples = [162, 160, 74, 478, 1185, 329, 680]  # [Angry, Disgust, Fear, Sad, Happy, Surprise, Neutral]
low_blur = 8
lower_blurs = {4: 8, 8: 16}  # for 'Mixed-Low'.
fr_sufs = ['_freeze']  # ['_freeze', '']

add_HR_model = True
add_mixed_low_model = False
add_low_blur_grid = True
add_pt_background = True  # shade the pretraining-range axis
add_chance_line = True
add_intermediate_test_blurs = False  # Include test blurs 12 and 24.

y_pt_top = 8
# Positive blurs are spaced in log2 steps; zero blur precedes them.
x_for_bl = {0: -2,
            1: 0,
            2: 2,
            4: 4,
            8: 6,
            12: 2 * np.log2(12),
            16: 8,
            24: 2 * np.log2(24),
            32: 10,
            64: 12,
            128: 14}

pt_ranges = {
    '0': [0],  # [-.1, .1],
    '4': [4],  # [3.9, 4.1],
    '0-4': [0, 4],
    '4-8': [4, 8],
    '8': [8],  # [7.9, 8.1],
    '0-8': [0, 8],
    '8-16': [8, 16],
}
pt_ys = {
    '0': y_pt_top,
    '4': y_pt_top,
    '0-4': y_pt_top-2*pt_row_spacing,
    '4-8': y_pt_top-3*pt_row_spacing,
    '8': y_pt_top,
    '0-8': y_pt_top-2*pt_row_spacing,
    '8-16': y_pt_top-3*pt_row_spacing,
}

y_lim = [.1, .85]
x_lim = [min(x_for_bl.values()) - .475, max(x_for_bl.values()) + .475]

grid_color = plt.rcParams['grid.color']
grid_style = plt.rcParams['grid.linestyle']
grid_width = plt.rcParams['grid.linewidth']
grid_alpha = plt.rcParams['grid.alpha']



ALIASES = {
    'RAF_blur8_pretrained0-8_freeze': 'RAF_blur8_pretrained0-8_freeze_fix_projs',
    'RAF_blur8_pretrained8_freeze': 'RAF_blur8_freeze',
    'RAF_blur8_pretrained8': 'RAF_blur8',
    'RAF_blur0_pretrained0_freeze': 'RAF_blur0_freeze',
}


def collect_results(results, models, balanced=False, n=3068):
    """Validate all requested results before returning canonical model names."""
    means, errors, problems = {}, {}, []
    for canonical, blur in models.items():
        test = f'test_{blur}'
        means.setdefault(test, {})
        errors.setdefault(test, {})
        actual = canonical if canonical in results else ALIASES.get(canonical, canonical)
        try:
            entry = results[actual]
            mean = entry[f'{test}_acc'] if balanced else entry['rep0'][test]
            error = entry[f'{test}_sem'] if balanced else None
            if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean) or not 0 <= mean <= 100:
                raise ValueError('accuracy must be a finite percentage in [0, 100]')
            if balanced:
                if isinstance(error, bool) or not isinstance(error, (int, float)) or not math.isfinite(error) or error < 0:
                    raise ValueError('SEM must be finite and nonnegative')
            else:
                p = mean / 100
                error = 100 * math.sqrt(p * (1 - p) / n)
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f'{canonical} ({test}; lookup {actual}): {exc}')
            continue
        means[test][canonical] = mean
        errors[test][canonical] = error
    if problems:
        raise ValueError('Missing or invalid evaluation results:\n  ' + '\n  '.join(problems))
    return means, errors


fig = plt.figure(figsize=[3.8, 3.4])
grid = fig.add_gridspec(
    2, 1, height_ratios=[5, .65],
    left=.18, right=.96, bottom=.16, top=.97, wspace=.08, hspace=.035
)
ax = fig.add_subplot(grid[0, 0])
ax_pt = fig.add_subplot(grid[1, 0], sharex=ax)

with open(paths.APVIT_BALANCED_RESULTS, "r") as f:
    results = json.load(f)

# test_blurs = [0, 2, 4, 8, 16, 32, 64]
test_blurs = [0, 1, 2, 4, 8, 16, 32, 64, 128]
if add_intermediate_test_blurs:
    test_blurs = sorted(test_blurs + [12, 24])
pt_blurs = [f'{low_blur}', f'0-{low_blur}']
pt_blurs = [0] + pt_blurs if add_HR_model else pt_blurs
pt_blurs = pt_blurs + [f'{low_blur}-{lower_blurs[low_blur]}'] if add_mixed_low_model else pt_blurs

if add_low_blur_grid:
    ax.axvline(x=x_for_bl[low_blur], color=grid_color, linestyle=grid_style, linewidth=grid_width, alpha=grid_alpha, zorder=1)

if add_chance_line:
    ax.axhline(y=1/7, linestyle='--', color=chance_color)
    ax.annotate('Chance', xy=(.02, 1/7), xycoords=ax.get_yaxis_transform(),
                xytext=(0, 2), textcoords='offset points',
                ha='left', va='bottom', fontsize=10, color=chance_color)

models = {f'RAF_blur{tb}_pretrained{pt}{fr_suf}': tb for tb in test_blurs for fr_suf in fr_sufs
          for pt in pt_blurs}

means, errs = collect_results(results, models, balanced=True)

test_keys = sorted(means.keys(), key=lambda k: int(k.split("_")[1]))

# Collect grouped results
grouped_means = {}
grouped_errors = {}

for test_key in test_keys:
    blur = int(test_key.split("_")[1])

    for model_name, value in means[test_key].items():
        group = model_name.split("pretrained", 1)[1]

        if group not in grouped_means:
            grouped_means[group] = {}
            grouped_errors[group] = {}

        grouped_means[group][blur] = value / 100
        grouped_errors[group][blur] = errs[test_key][model_name] / 100

for group in sorted(grouped_means.keys()):
    x = sorted(grouped_means[group])
    y = [grouped_means[group][blur] for blur in x]
    yerr = [grouped_errors[group][blur] for blur in x]
    x = [x_for_bl[v] for v in x]

    blur_lbl = 'Mixed-High' if (('-' in group) & ('0' in group)) else 'High' if '0' in group else 'Mixed-Low' if '-' in group else 'Low'
    grp_no_freeze = group.replace("_freeze", "")
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker=".",
        linestyle='solid',
        linewidth=plot_linewidth,
        markersize=marker_size,
        capsize=yerr_capsize,
        label=f'{blur_lbl}{" " * labl_spc[blur_lbl]}({grp_no_freeze.replace("-", " - ")} px)',
        color=colors[blur_lbl]
    )
    pt_blurs_for_group = pt_ranges[grp_no_freeze]
    pt_y = pt_ys[grp_no_freeze]
    if len(pt_blurs_for_group) > 1:
        ax_pt.plot([x_for_bl[value] for value in pt_blurs_for_group],
                   [pt_y, pt_y], color=colors[blur_lbl], linewidth=pt_linewidth)
    else:
        ax_pt.scatter(x_for_bl[pt_blurs_for_group[0]], pt_y,
                      marker='s', color=colors[blur_lbl], s=pt_linewidth**2)

ax.set_ylim(y_lim)
ax.set_yticks([.2, .4, .6, .8])
ax.set_xlim(x_lim)
ax.set_xticks([x_for_bl[blur] for blur in test_blurs])
ax.set_xticklabels(test_blurs)
ax_pt.set_xlabel('Image Blur (pixels)', labelpad=3)
ax_pt.set_ylabel('Pretrain\nRegime', rotation=0, ha='right',
                    va='center', multialignment='left', labelpad=5, fontsize=10)
ax_pt.yaxis.set_label_coords(-.035, .5)
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.tick_params(axis='y', pad=2)
ax.set_ylabel('Proportion Correct', labelpad=3, fontsize=12)
ax_pt.set_ylim(min(pt_ys[str(pt)] for pt in pt_blurs) - pt_vertical_padding,
                  y_pt_top + pt_vertical_padding)
ax_pt.set_yticks([])
ax_pt.tick_params(axis='x', bottom=True, labelbottom=True, pad=2, labelsize=9, labelrotation=45)
if add_pt_background:
    ax_pt.set_facecolor(grid_color)
    ax_pt.patch.set_alpha(.2)

# Mark the special interval, leaving a gap in the axis border between slashes.
break_x = (x_for_bl[0] + x_for_bl[1]) / 2
ax_pt.plot([break_x], [0], marker='_', markersize=10,
           markeredgewidth=2, color=fig.get_facecolor(),
           transform=ax_pt.get_xaxis_transform(), clip_on=False, zorder=4)
ax_pt.annotate('//', xy=(break_x, 0),
               xycoords=ax_pt.get_xaxis_transform(), xytext=(0, -1),
               textcoords='offset points', ha='center', va='center',
               fontsize=12, annotation_clip=False, zorder=5)

# Align the visible left edges of the two y-axis labels.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
accuracy_label = ax.yaxis.label.get_window_extent(renderer)
pretrain_label = ax_pt.yaxis.label.get_window_extent(renderer)
label_x = ax.yaxis.label.get_transform().transform(ax.yaxis.label.get_position())[0]
label_x += pretrain_label.x0 - accuracy_label.x0
ax.yaxis.set_label_coords(ax.transAxes.inverted().transform((label_x, 0))[0], .5)

if save_fig:
    output_path = Path(paths.SCATTER_FIGURE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('.png', '.pdf'):
        fig.savefig(output_path.with_suffix(extension), dpi=300)
plt.show(block=True)
