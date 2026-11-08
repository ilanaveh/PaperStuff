"""Copy to path_config.py and edit the external roots for this computer.

Repository paths follow this checkout automatically. All paths are strings so
they also work with OpenCV versions that do not accept pathlib.Path objects.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = Path('/home/projects/bagon/ilanaveh')
TRANSFORMERS_ROOT = RESEARCH_ROOT / 'code/Transformers'
EXPLAINABILITY_ROOT = RESEARCH_ROOT / 'code/Transformer-Explainability'
RAFDB_ROOT = RESEARCH_ROOT / 'data/RAF-DB'

ATTENTION_MAPS_DIR = str(PROJECT_ROOT / 'Figures/attention_maps')
BEHAVIOR_PANELS_DIR = str(Path(ATTENTION_MAPS_DIR) / 'bhvr_fig_split')
BEHAVIOR_FIGURE = str(Path(ATTENTION_MAPS_DIR) / 'figs_from_bhvr_paper/fig4_weights_with_colorbar.png')
BEHAVIOR_ORIGINAL_FIGURE = str(RESEARCH_ROOT / 'data/S10Code/mappingTheEmotionalFace-master/figures/mainWeightPlot.png')
BLURRED_FACES_DIR = str(Path(ATTENTION_MAPS_DIR) / 'blurred_face_ims')
NIMSTIM_IMAGES_DIR = str(RESEARCH_ROOT / 'data/NimStim/NimStim')
NIMSTIM_CROPS_DIR = str(EXPLAINABILITY_ROOT / 'samples/test_images_nimstim')
PROTOTYPE_DIR = str(EXPLAINABILITY_ROOT / 'compare_models')
RAFDB_IMAGES_DIR = str(RAFDB_ROOT / 'basic/Image/aligned_224')
RAFDB_TEST_LABELS = str(RAFDB_ROOT / 'basic/EmoLabel/test.txt')
RAFDB_SPLIT_TEST_DIR = str(RAFDB_ROOT / 'split_test')
DEIT_RESULTS = str(TRANSFORMERS_ROOT / 'deit/eval/results.json')
APVIT_RESULTS = str(TRANSFORMERS_ROOT / 'APViT/eval/results.json')
APVIT_BALANCED_RESULTS = str(TRANSFORMERS_ROOT / 'APViT/eval/results_balanced.json')
SCATTER_FIGURE = str(PROJECT_ROOT / 'Figures/models_performance_scatter/scatter.png')


def prototype_summary_path(config, pooling):
    directory = Path(PROTOTYPE_DIR) / 'attention_prototypes/NATIVE_RESOLUTION'
    if pooling:
        directory /= 'WITH_POOLING'
    return directory / config / 'prototype_summary.csv'


def resolve_prototype_path(saved_path):
    """Resolve CSV paths under the local prototype root, including old roots."""
    # Normalize separators so CSVs can be shared between Windows and Linux.
    path = Path(str(saved_path).replace('\\', '/'))
    if path.is_absolute() and path.exists():
        return path
    parts = path.parts
    if 'attention_prototypes' in parts:
        return Path(PROTOTYPE_DIR).joinpath(*parts[parts.index('attention_prototypes'):])
    return Path(PROTOTYPE_DIR) / path
