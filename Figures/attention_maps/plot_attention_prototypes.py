"""
1/6/26
Based on plot_prototypes from Transformer_Explainability/compare_models.
Assumes:
 1. prototypes were created and saved (by Transformer_Explainability/compare_models/build_prototypes.py).
 2. behavior results from Wegrzyn et al 2017 were preprocessed and saved (by preprocess_behavior_fig.py).
"""
import numpy as np
import pandas as pd
import os
import ast
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch

POOLING = True
prototype_config = 'pred_label_all'  # 'true_label_correct' / 'pred_label_all'

home_dir = Path("/home/projects/bagon/ilanaveh")
# models = ['RAF_blur8_pretrained0-8_freeze_fix_projs', 'RAF_blur8_freeze']
models = ['RAF_blur8_pretrained0-8_freeze_fix_projs', 'RAF_blur8_freeze', 'RAF_blur0_pretrained0-8_freeze', 'RAF_blur0_freeze']
mdl_lables = {
    'RAF_blur0_freeze': 'High-S0',
    'RAF_blur0_pretrained0-8_freeze': 'Mixed-High-S0',
    'RAF_blur8_pretrained0-8_freeze_fix_projs': 'Mixed-High-S8',
    'RAF_blur8_freeze': 'Low-S8',
}
# EMOTIONS = ['Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise', 'Neutral']
# EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTIONS = ['Happy', 'Angry', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Neutral']
emo_labels = {'Happy': 'hap',
              'Sad': 'sad',
              'Angry': 'ang',
              'Fear': 'fea',
              'Disgust': 'dis',
              'Surprise': 'sup',
              'Neutral': 'ntr'}

# Visualization params:
show_on_mean_face = True
interpolation_mode = 'bilinear'  # 'bilinear' for smoothing, 'nearest' for keeping patch-resolution (relevant for native prototypes)
cmap = 'jet'
gamma = 1.5
min_alpha = .3
max_alpha = .65
genders = ['f']  # choose from ['f', 'm'].
show_ttl = False
row_order = ['humans', 'High-S0', 'Mixed-High-S0', 'Mixed-High-S8', 'Low-S8']
human_r = row_order.index('humans')
# ===========================================================================
# 1. Load prototypes
# ===========================================================================
prototype_dir = home_dir / Path("code/Transformer-Explainability/compare_models")
summary_path = prototype_dir / Path(f'attention_prototypes/NATIVE_RESOLUTION{"/WITH_POOLING" if POOLING else ""}/{prototype_config}/prototype_summary.csv')

summary_df = pd.read_csv(summary_path)

prototypes = {}

for _, row in summary_df.iterrows():
    key = ast.literal_eval(row["key"])
    prototype_path = Path(row["prototype_path"])
    mean_face_path = Path(row["mean_face_path"])

    # Allow paths saved as either absolute or relative.
    if not prototype_path.exists():
        prototype_path = prototype_dir / prototype_path.parent / prototype_path.name
        mean_face_path = prototype_dir / mean_face_path.parent / mean_face_path.name

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
bhvr_figs_dir = 'bhvr_fig_split'

# ===========================================================================
# Create figure
# ===========================================================================
nrows = len(models) + len(genders)  # 2 for model prototypes and 1/2 for behavior
ncols = 7  # 7 emotions

fig, axs = plt.subplots(nrows, ncols)
# fig.set_size_inches([10.4, 5.6])
fig.set_size_inches([6.2, 3.4])

for c, emo in enumerate(EMOTIONS):
    axs[0, c].set_title(emo, fontsize=10)

    # First two rows: model attention-prototypes:
    for mdl in models:
        mdl_lbl = mdl_lables[mdl]
        sigma = int(mdl.split('blur')[1].split('_')[0])
        r = row_order.index(mdl_lbl)
        ax = axs[r, c]
        key = (mdl, sigma, emo)

        if key in prototypes:
            proto = prototypes[key]["prototype"]
            proto_minmax = (proto - proto.min()) / (proto.max() - proto.min())

            if show_on_mean_face:
                face = prototypes[key]['mean_face'].copy()
                ax.imshow(np.clip(face, 0, 1))

                if proto_minmax.shape != face.shape[:2]:
                    display_proto = torch.nn.functional.interpolate(
                        torch.from_numpy(proto_minmax).reshape(1, 1, 14, 14),
                        scale_factor=face.shape[0] / proto_minmax.shape[0],
                        mode=interpolation_mode).reshape(face.shape[:2]).numpy()
                else:
                    display_proto = proto_minmax

                alpha = np.clip(display_proto, 0, 1)
                alpha = alpha ** gamma
                alpha_map = min_alpha + alpha * (max_alpha - min_alpha)

                ax.imshow(display_proto, cmap=cmap, alpha=alpha_map)

            else:
                ax.imshow(proto_minmax)

        ax.set_xticks([])
        ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(mdl_lbl.replace(' ', '\n'), fontsize=10)

        if show_ttl:
            ax.set_title(f"N={prototypes[key]['n']}")

    # Last one/two rows: behavior results:
    for r, gen in enumerate(genders):
        ax = axs[r+human_r, c]
        fig_nm = f'{gen}_{emo_labels[emo]}.png'
        face = plt.imread(os.path.join(bhvr_figs_dir, fig_nm))
        ax.imshow(face, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # ax.set_aspect("equal", adjustable="box")
        # ax.set_anchor("C")
fig.supylabel("Model", x=.04, y=.32, fontsize=10)
fig.tight_layout()

if len(genders) == 1:
    axs[human_r, 0].set_ylabel('Human\nObservers', fontsize=10)
    axs[human_r, 0].yaxis.labelpad = 15

    for c in range(len(EMOTIONS)):
        ax = axs[human_r, c]
        pos = ax.get_position()

        new_w = pos.width * .75
        new_h = pos.height * .75

        # keep subplot centered around its original center
        new_x0 = pos.x0 + ((pos.width - new_w) / 2)
        new_y0 = pos.y0 + ((pos.height - new_h) / 1.5)

        ax.set_position([new_x0, new_y0, new_w, new_h])
else:
    fig.supylabel("Human Observers", x=.05, y=.26, fontsize=12)

plt.show(block=True)
print()
