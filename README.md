# Sheet Metal Forming Measurement Comparison

Deep learning analysis of sheet metal forming signals for tool wear classification with model interpretability via attribution analysis.

## Overview

This project trains a 1D CNN to predict forming parameters from force sensor signals and provides interpretable attribution analysis using Captum's Integrated Gradients to identify which signals and time steps are most important for predictions.

## Contents

- `data/` - Put your JSON measurement files here
- `prepare_data.ipynb` - Data loading and preprocessing to HDF5 format
- `eval_models.ipynb` - Model training (using PyTorch & Lightning), evaluation, and attribution analysis for a single set of signals and target(s)
- `run_all_experiments.sh` - Batch runner for all signal/target combinations
- `visualize_results.ipynb` - Visualization of results across multiple experiments from `run_all_experiments.sh`

## Quick Start

TODO: Explain how to obtain the data.

```shell
pip install -r requirements.txt
jupyter lab eval_models.ipynb

CUDA_VISIBLE_DEVICES=05 python hp_search/multilabeling_exact_match.py
optuna-dashboard "sqlite:///hp_search/output/optuna.db"
```

### Installing Times New Roman Font (for plots)
The notebooks use Times New Roman for publication-quality plots. This font is not installed by default in most Linux environments. To install it:

```bash
# Install cabextract (needed to extract Microsoft fonts)
sudo apt-get update
sudo apt-get install -y cabextract

# Download and install Times New Roman from Microsoft's corefonts
cd /tmp
curl -sL "http://downloads.sourceforge.net/corefonts/times32.exe" -o times32.exe
cabextract -q times32.exe
mkdir -p ~/.local/share/fonts
mv *.TTF ~/.local/share/fonts/
rm -f times32.exe
fc-cache -fv ~/.local/share/fonts

# Clear matplotlib's font cache so it picks up the new font
rm -rf ~/.cache/matplotlib/font*.json
```

After installing, **restart your Jupyter kernel** for matplotlib to recognize the new font.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citing

If you use this code in your research, please cite:

```bibtex
TODO
```
