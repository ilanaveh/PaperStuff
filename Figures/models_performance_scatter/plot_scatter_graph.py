"""
4/8/26
Create scatter plots of apvit models for different test-blurs.
Based on plot_bar_graph and on Transformers/APViT/eval/plot_scatter.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os


save_fig = False

# style
marker_size = 1
yerr_capsize = 2
single_fig = False
plt.rcParams.update({'font.size': 14})
linebreak = '\n'
colors = {
            'Low': (0.68, 0.56, 0.82),#(.745, .68, .83),  # Purple (low)
            'Mixed-High': (0.38, 0.76, 0.38),#(.5, .788, 0.5),  # Green (mixed)
            'High': (1, .753, 0.525),  # Orange (High)
            'Mixed-Low': (.651, .792, .925),  # Blue
    }
labl_spc = {'Low': 8, 'Mixed-High': 1, 'High': 7, 'Mixed-Low': 1}

# defenitions
not_finished_yet = ['RAF_blur11_pretrained0_freeze', 'RAF_blur14_pretrained0_freeze', 'RAF_blur9_pretrained4-8_freeze', 'RAF_blur13_pretrained8-16_freeze']
results = {}
n_test_samples = {}
home_dir = '/home/projects/bagon/ilanaveh/code/Transformers'
low_blurs = [4, 8]
lower_blurs = {4: 8, 8: 16}  # for 'Mixed-Low'.
fr_sufs = ['_freeze']  # ['_freeze', ']

add_HR_model = True
add_mixed_low_model = True
add_16_pt = True
add_16_acc = False
add_low_blur_grid = True
add_pt_background = True

y_pt_top = 9
x_for_16 = 9.5

pt_ranges = {
    '0': [0],  # [-.1, .1],
    '4': [4],  # [3.9, 4.1],
    '0-4': [0, 4],
    '4-8': [4, 8],
    '8': [8],  # [7.9, 8.1],
    '0-8': [0, 8],
    '8-16': [8, x_for_16],
}
pt_ys = {
    '0': y_pt_top,
    '4': y_pt_top,
    '0-4': y_pt_top-2,
    '4-8': y_pt_top-4,
    '8': y_pt_top,
    '0-8': y_pt_top-2,
    '8-16': y_pt_top-4,
}

n_test_samples['apvit'] = 3068

y_lim = [30, 90] if add_HR_model else [40, 80]
x_lim = [-.475, x_for_16+.475] if add_16_pt else [-.475, np.max(list(lower_blurs.keys()))+.475]

grid_color = plt.rcParams['grid.color']
grid_style = plt.rcParams['grid.linestyle']
grid_width = plt.rcParams['grid.linewidth']
grid_alpha = plt.rcParams['grid.alpha']


def clean_model_names(models, means=None, errs=None, clean_models=False):
    if clean_models:
        if 'RAF_blur8_pretrained0-8_freeze' in models:
            models['RAF_blur8_pretrained0-8_freeze_fix_projs'] = models.pop('RAF_blur8_pretrained0-8_freeze')
        if 'RAF_blur8_pretrained8_freeze' in models:
            models['RAF_blur8_freeze'] = models.pop('RAF_blur8_pretrained8_freeze')
        if 'RAF_blur8_pretrained8' in models:
            models['RAF_blur8'] = models.pop('RAF_blur8_pretrained8')
        if 'RAF_blur0_pretrained0_freeze' in models:
            models['RAF_blur0_freeze'] = models.pop('RAF_blur0_pretrained0_freeze')
        models = {mdl:val for mdl,val in models.items() if mdl not in not_finished_yet}
        return models

    if 'RAF_blur8_pretrained0-8_freeze_fix_projs' in models:
        means['test_8']['RAF_blur8_pretrained0-8_freeze'] = means['test_8'].pop(
            'RAF_blur8_pretrained0-8_freeze_fix_projs')
        errs['test_8']['RAF_blur8_pretrained0-8_freeze'] = errs['test_8'].pop(
            'RAF_blur8_pretrained0-8_freeze_fix_projs')
    if 'RAF_blur8_freeze' in models:
        means['test_8']['RAF_blur8_pretrained8_freeze'] = means['test_8'].pop('RAF_blur8_freeze')
        errs['test_8']['RAF_blur8_pretrained8_freeze'] = errs['test_8'].pop('RAF_blur8_freeze')
    if 'RAF_blur8' in models:
        means['test_8']['RAF_blur8_pretrained8'] = means['test_8'].pop('RAF_blur8')
        errs['test_8']['RAF_blur8_pretrained8'] = errs['test_8'].pop('RAF_blur8')
    if 'RAF_blur0_freeze' in models:
        means['test_0']['RAF_blur0_pretrained0_freeze'] = means['test_0'].pop('RAF_blur0_freeze')
        errs['test_0']['RAF_blur0_pretrained0_freeze'] = errs['test_0'].pop('RAF_blur0_freeze')

    return means, errs


fig, axs = plt.subplots(2, 1, figsize=[3.4, 4.8])

with open(os.path.join(home_dir, 'APViT/eval/results.json'), "r") as f:
    results['apvit'] = json.load(f)

for i, lb in enumerate(low_blurs):
    test_blurs = np.arange(9)
    test_blurs_for_means = np.append(test_blurs, 16) if add_16_acc else test_blurs
    pt_blurs = [f'{lb}', f'0-{lb}']
    pt_blurs = [0] + pt_blurs if add_HR_model else pt_blurs
    pt_blurs = pt_blurs + [f'{lb}-{lower_blurs[lb]}'] if add_mixed_low_model else pt_blurs

    ax = axs[i]

    if add_low_blur_grid:
        ax.axvline(x=lb, color=grid_color, linestyle=grid_style, linewidth=grid_width, alpha=grid_alpha, zorder=1)

    models = {f'RAF_blur{tb}_pretrained{pt}{fr_suf}': tb for tb in test_blurs_for_means for fr_suf in fr_sufs
              for pt in pt_blurs}

    models = clean_model_names(models, clean_models=True)

    means = {f'test_{tb}': {} for tb in test_blurs_for_means}
    errs = {f'test_{tb}': {} for tb in test_blurs_for_means}

    for mdl_nm, tb in models.items():
        for k in results['apvit'].keys():
            if (mdl_nm == k) and (f'RAF_blur{tb}' in k):
                means[f'test_{tb}'][mdl_nm] = results['apvit'][k]['rep0'][f'test_{tb}']
                # Compute err [sqrt( (p * (1-p)) / n)]:
                p = results['apvit'][k]['rep0'][f'test_{tb}'] / 100
                errs[f'test_{tb}'][mdl_nm] = 100 * np.sqrt((p * (1 - p)) / n_test_samples['apvit'])

    means, errs = clean_model_names(models, means, errs, clean_models=False)

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

            grouped_means[group][blur] = value
            grouped_errors[group][blur] = errs[test_key][model_name]

    for group in sorted(grouped_means.keys()):
        x = sorted(grouped_means[group].keys())
        y = [grouped_means[group][blur] for blur in x]
        yerr = [grouped_errors[group][blur] for blur in x]
        x = [x_for_16 if (v == 16) else v for v in x]

        blur_lbl = 'Mixed-High' if (('-' in group) & ('0' in group)) else 'High' if '0' in group else 'Mixed-Low' if '-' in group else 'Low'
        grp_no_freeze = group.replace("_freeze", "")
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=".",
            linestyle='none' if add_16_acc else 'solid',
            markersize=marker_size,
            capsize=yerr_capsize,
            label=f'{blur_lbl}{" " * labl_spc[blur_lbl]}({grp_no_freeze.replace("-", " - ")} px)',
            color=colors[blur_lbl]
        )

        if add_16_acc:
            ax.plot(x[:-1], y[:-1], color=colors[blur_lbl])

        pt_rng = pt_ranges[grp_no_freeze]
        pt_y = y_lim[0]+pt_ys[grp_no_freeze]
        if len(pt_rng) > 1:
            ax.plot(pt_rng, [pt_y, pt_y], color=colors[blur_lbl], linewidth=3)
        else:
            ax.scatter(pt_rng[0], pt_y, marker='s', color=colors[blur_lbl], s=8)

    ax.set_ylim(y_lim)
    ax.set_yticks(np.arange(y_lim[0]+20, y_lim[1]+1, 10))
    ax.set_xlim(x_lim)
    ax.set_xticks(np.append(x, x_for_16) if (add_16_pt and not add_16_acc) else x)
    ax.set_xticklabels(np.append(test_blurs_for_means, 16) if (add_16_pt and not add_16_acc) else x)
    if i == 1:
        ax.set_xlabel('Image Blur (pixels)', labelpad=12)
    ax.grid(axis='y', alpha=0.3)
    if add_pt_background:
        rect = patches.Rectangle((x_lim[0], y_lim[0]), np.diff(x_lim)[0], y_pt_top + 5,
                                 facecolor=grid_color, alpha=.2, zorder=1)
        ax.add_patch(rect)

    # leg = ax.legend(fontsize=10, loc='lower right')
    # leg = ax.legend(title="Pretraining Blur", fontsize=10, title_fontsize=10, loc='lower right')
    # leg.get_title().set_weight("bold")
    # ax.legend(fontsize=10)
fig.tight_layout()
fig.subplots_adjust(left=.23)
fig.supylabel("Test Accuracy", y=.58)

if save_fig:
    fig.savefig('scatter.png')
plt.show(block=True)