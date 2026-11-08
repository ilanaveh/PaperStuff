"""
4/5/26
Create bar plots of different models (deit, APViT).
Based on plot_bar_graph files from each project.
"""

import sys
from pathlib import Path

# Support running this script directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Config import path_config as paths

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from config import fig_config

cfg = fig_config[5]

# style
yerr_capsize = 2
single_fig = False
plt.rcParams.update({'font.size': 14})
linebreak = '\n'

# defenitions
results = {}
n_test_samples = {}
# x_lbls = ['High', 'Low', 'Mixed']
ttls = {'deit': 'Object Classification', 'apvit': 'Facial Expression Recognition',
        **{tb: f"Blur {tb} Test" for tb in cfg['test_blurs']}}
# chance = {'deit': 100/1000, 'apvit': 100/7}

# deit
n_test_samples['deit'] = 50000
with open(paths.DEIT_RESULTS, "r") as f:
    results['deit'] = json.load(f)

# apvit
n_test_samples['apvit'] = 3068
with open(paths.APVIT_RESULTS, "r") as f:
    results['apvit'] = json.load(f)

# plot
print(f"Separating figures by: {cfg['sep_fig_by']}")

# if cfg['sep_fig_by'] == 'test_blurs':
#     iterby = cfg['test_blurs']
# elif cfg['sep_fig_by'] == 'models':
#     iterby = cfg['models'].keys()

if single_fig:
    f, axs = plt.subplots(2, 1)
    f.set_size_inches([3.4, 4.8])

for i, k in enumerate(cfg[cfg['sep_fig_by']].keys()):
    # k is either model / test blur
    if cfg['sep_fig_by'] == 'test_blurs':
        tb = k
        mdl = cfg['models'][0]
    elif cfg['sep_fig_by'] == 'models':
        tb = cfg['test_blurs'][0]
        mdl = k
    if single_fig:
        ax = axs[i]
        ax.set_title(ttls[k])
    else:
        f, ax = plt.subplots()
        f.set_size_inches([3.5, 2.8])
        # Shrink axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.set_ylabel(f'{linebreak.join(ttls[k].rsplit(" ", 1))} Accuracy')
    if mdl == 'deit':
        means = {mdl_nm: results[mdl][mdl_nm][f"test_{tb}"] for mdl_nm in cfg['models'][mdl]}
    elif mdl == 'apvit':
        if cfg['sep_fig_by'] == 'test_blurs':
            means = {mdl_nm: results[mdl][mdl_nm]['rep0'][f"test_{tb}"] for mdl_nm in cfg['test_blurs'][tb]}
        elif cfg['sep_fig_by'] == 'models':
            means = {mdl_nm: results[mdl][mdl_nm]['rep0'][f"test_{tb}"] for mdl_nm in cfg['models'][mdl]}
    # Compute err [sqrt( (p * (1-p)) / n)]:
    if cfg['sep_fig_by'] == 'test_blurs':
        ps = {mdl_nm: (means[mdl_nm] / 100) for mdl_nm in cfg['test_blurs'][tb]}
        errs = {mdl_nm: 100 * np.sqrt((ps[mdl_nm] * (1 - ps[mdl_nm])) / n_test_samples[mdl]) for mdl_nm in cfg['test_blurs'][tb]}
    elif cfg['sep_fig_by'] == 'models':
        ps = {mdl_nm: (means[mdl_nm] / 100) for mdl_nm in cfg['models'][mdl]}
        errs = {mdl_nm: 100 * np.sqrt((ps[mdl_nm] * (1 - ps[mdl_nm])) / n_test_samples[mdl]) for mdl_nm in cfg['models'][mdl]}

    if 'widths' in cfg:
        ax.bar(cfg['x_lbls'][:len(means)], [means[m] for m in cfg['test_blurs'][tb]], yerr=[errs[m] for m in cfg['test_blurs'][tb]],
               align='center', color=cfg['colors'][:len(means)], ecolor='black', capsize=yerr_capsize, width=cfg['widths'][:len(means)])
    else:
        ax.bar(cfg['x_lbls'][:len(means)], [means[m] for m in cfg['test_blurs'][tb]], yerr=[errs[m] for m in cfg['test_blurs'][tb]],
               align='center', color=cfg['colors'][:len(means)], ecolor='black', capsize=yerr_capsize)
    ax.set_ylim(cfg['y_lim'][mdl])
    ax.set_yticks(np.arange(cfg['y_lim'][mdl][0], cfg['y_lim'][mdl][1]+1, 10))
    ax.set_xlabel('Model')
    plt.tight_layout()

if single_fig:
    f.supylabel('Task Accuracy')
plt.show(block=True)
