# Paper Figures and Visualization Code

> **Origin**: This repository is a mirror of the local folder `PaperStuff` from `W:\ilanaveh\code\PaperStuff`.
>
> **Description**: This repository contains code and assets for generating paper figures related to face attention visualization, model performance analysis, and behavioral experiments. It is not a fork of an existing public repository but rather a collection of custom visualization scripts developed for research papers.

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