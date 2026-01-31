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
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citing

If you use this code in your research, please cite:

```bibtex
TODO
```
