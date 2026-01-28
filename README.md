# Sheet Metal Forming Measurement Comparison

Deep learning analysis of sheet metal forming signals with model interpretability via attribution analysis.

## Overview

This project trains a 1D CNN to predict forming parameters from force sensor signals and provides interpretable attribution analysis using Captum's Integrated Gradients to identify which signals and time steps are most important for predictions.

## Contents

- `data/` - Input JSON measurement files and processed HDF5 archive
- `prepare-data.ipynb` - Data loading and preprocessing from HDF5 with train/val/test splitting
- `eval-models.ipynb` - Model training (PyTorch Lightning), evaluation, and attribution analysis with visualizations

## Quick Start

```shell
pip install -r requirements.txt
jupyter lab eval-models.ipynb
```

## Outputs

- Time-domain and frequency-domain attribution plots with confidence bands
- Per-signal importance distributions (box plots)
- Model checkpoints in `checkpoints/simplecnn/`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citing

If you use this code in your research, please cite:

```bibtex
TODO
```
