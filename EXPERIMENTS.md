# Batch Experiment Guide

This guide explains how to run systematic experiments testing all signal combinations.

## Quick Start

```bash
# Run all experiments in parallel (default: 12 parallel jobs)
./run_all_experiments.sh
```

You'll be prompted for an optional WandB group name to organize your experiments.

## What Gets Tested

The batch script runs **22 total experiments**:

### Individual Signals (16 experiments)
Each of these 8 signals tested for both Ironing and Deep Drawing:
- Channel 1: Strip Connection Cut (Direct)
- Channel 2: Deep Drawing (Direct)
- Channel 3: Ironing (Direct)
- Channel 4: Deep Drawing - Top (Indirect)
- Channel 5: Deep Drawing - Bottom (Indirect)
- Channel 7: Ironing - Top (Indirect)
- Channel 8: Ironing - Stemp Holder (Indirect)
- Channel 9: Ironing - Bottom (Indirect)

### Predefined Groups (6 experiments)
- **DIRECT** [1, 2, 3]: All direct force sensors
- **INDIRECT** [4, 5, 7, 8, 9]: All indirect measurements
- **ALL** [1, 2, 3, 4, 5, 7, 8, 9]: All sensors combined

Each group tested for both Ironing and Deep Drawing classification.

## Configuration

### WandB Group Name
When you run the script, you'll be prompted:
```
Enter WandB group name for this experiment batch (optional, press Enter to skip):
```

This groups all experiments together in Weights & Biases for easier comparison.

### Adjusting Parallelism

Edit `run_all_experiments.sh` and change:
```bash
MAX_PARALLEL_JOBS=12  # Adjust this value
```

**Recommendations:**
- **1 GPU available**: Set to `1` (run sequentially to avoid GPU contention)
- **Multiple GPUs**: Set to number of GPUs (e.g., 4 GPUs → `MAX_PARALLEL_JOBS=4`)
- **CPU-only**: Set to `num_cores / 2` for reasonable performance
- **Limited memory**: Reduce to avoid OOM errors

Current default is **12** for fast parallel execution.

### Execution Time Estimates

With default settings (15 epochs per experiment):
- **Sequential (1 job)**: ~5-10 hours for all 22 experiments
- **12 parallel jobs**: ~30-60 minutes (depends on GPU/CPU)
- **Per experiment**: ~15-30 minutes average

## Output Structure

```
experiment_results/
├── experiment_log.txt              # Detailed execution log with timestamps
├── summary.txt                     # Auto-generated summary with metrics
├── all_results.csv                 # All results combined (after running visualize_results.ipynb)
├── f1_scores_heatmap.pdf/png       # Combined F1 score heatmaps
├── deep_drawing_f1_heatmap.pdf/png # Deep Drawing specific heatmap
├── ironing_f1_heatmap.pdf/png      # Ironing specific heatmap
├── accuracy_vs_f1.pdf/png          # Accuracy vs F1 diagnostic plots
├── signal_1_ironing/               # Individual experiment directories
│   ├── test_results.txt            # Performance metrics
│   ├── output.ipynb                # Executed notebook
│   ├── execution.log               # Detailed execution log
│   ├── sample_signals.pdf          # Visualizations
│   ├── attribution_*.pdf           # Attribution analysis plots
│   └── ...
└── ...
```

## Analyzing Results

After experiments complete, run the visualization notebook to generate comprehensive analysis:

```bash
jupyter lab visualize_results.ipynb
# Or execute all cells programmatically:
jupyter nbconvert --to notebook --execute visualize_results.ipynb
```

This generates:
- **all_results.csv**: Combined table of all experiment metrics
- **Heatmaps**: Visual comparison of F1 scores across all configurations
- **Performance rankings**: Best/worst configurations per target
- **Channel importance**: Single-channel performance analysis

The heatmaps use descriptive channel names and group DIRECT, INDIRECT, and ALL configurations for easy comparison.

## Troubleshooting

### Out of Memory (OOM) Errors

If experiments fail with OOM:
1. Reduce `MAX_PARALLEL_JOBS` to 1
2. Check GPU memory: `nvidia-smi`
3. Reduce batch size in `eval-models.ipynb` (default is 32)

### Experiment Failures

Check specific experiment logs:
```bash
cat experiment_results/signal_1_ironing/execution.log
```

Common issues:
- **CUDA out of memory**: Reduce parallelism or batch size
- **Kernel died**: Likely memory issue, reduce parallelism
- **Import errors**: Check environment: `pip install -r requirements.txt`
- **Papermill not found**: Install with `pip install papermill`

### Resuming Interrupted Runs

The script doesn't automatically resume. To re-run failed experiments:

1. Check which failed:
   ```bash
   grep "FAILED" experiment_results/experiment_log.txt
   ```

2. Manually re-run specific experiments using papermill:
   ```bash
   # Single channel example
   papermill eval-models.ipynb experiment_results/signal_1_ironing/output.ipynb \
     -p data_path all_data.hdf5 \
     -p plot_path experiment_results/signal_1_ironing \
     -p classification_target Ironing \
     -r signal_only_channels [1] \
     --kernel python3 \
     --execution-timeout 3600

   # Multi-channel example (DIRECT)
   papermill eval-models.ipynb experiment_results/direct_ironing/output.ipynb \
     -p data_path all_data.hdf5 \
     -p plot_path experiment_results/direct_ironing \
     -p classification_target Ironing \
     -r signal_only_channels [1,2,3] \
     --kernel python3 \
     --execution-timeout 3600
   ```

## Customizing Experiments

To add custom signal combinations, edit `run_all_experiments.sh`:

```bash
# Add after the predefined groups section, before "Count total experiments":
EXPERIMENTS+=("Single|2,4,7|Ironing|custom_combo_ironing")
EXPERIMENTS+=("Single|2,4,7|DeepDrawing|custom_combo_deepdrawing")
```

Note: Channels in custom combos are passed as comma-separated values.

## Performance Tips

1. **Use fast_dev_run for testing**:
   - Uncomment `fast_dev_run=True` in `eval-models.ipynb` to test pipeline
   - Comment it out for actual experiments

2. **WandB offline mode**:
   - Set `WANDB_MODE=offline` before running if no internet
   - Or skip the WandB group prompt (just press Enter)

3. **Use SSD for output**:
   - Experiments write many files
   - SSD recommended over NFS/network storage

4. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

5. **Check progress**:
   ```bash
   # See which experiments are running/completed
   tail -f experiment_results/experiment_log.txt

   # Count completed experiments
   ls -d experiment_results/signal_* experiment_results/direct_* experiment_results/indirect_* experiment_results/all_* 2>/dev/null | wc -l
   ```

## Technical Details

### Implementation
- Uses **papermill** for parameterized notebook execution
- Experiments run with isolated parameters via cell injection
- Each experiment gets its own output directory and logs

### Channel Indexing
Channels follow the HDF5 data structure:
- Index 0 = time_s (not used for training)
- Indices 1-9 = actual sensor channels
- Index 6 = "Not Connected" (skipped in ALL/INDIRECT)

### Visualization
The `visualize_results.ipynb` notebook automatically detects and labels:
- Individual channels with full descriptive names
- DIRECT/INDIRECT/ALL groups with clear labels
- Custom multi-channel combinations
