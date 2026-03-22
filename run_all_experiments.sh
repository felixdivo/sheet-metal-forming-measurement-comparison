#!/bin/bash

# Script to run all signal combination experiments for sheet metal forming classification
# This script tests all individual signals and predefined groups for both Ironing and Deep Drawing
# Experiments run in parallel for faster execution
# Supports multiple random seeds for statistical robustness

# Configuration
NOTEBOOK="eval_models.ipynb"
OUTPUT_DIR="experiment_results"
LOG_FILE="${OUTPUT_DIR}/experiment_log.txt"
MAX_PARALLEL_JOBS=8  # Adjust based on available CPU/GPU resources
CONFIG_FILE="channel_config.json"

# Channel index to readable name mapping (from channel_config.json)
# Names have spaces and parentheses removed for filesystem compatibility
declare -A CHANNEL_NAMES
CHANNEL_NAMES[1]="StripConnectionCut-Direct"
CHANNEL_NAMES[2]="DeepDrawing-Direct"
CHANNEL_NAMES[3]="Ironing-Direct"
CHANNEL_NAMES[4]="DeepDrawing-Top-Indirect"
CHANNEL_NAMES[5]="DeepDrawing-Bottom-Indirect"
CHANNEL_NAMES[7]="Ironing-Top-Indirect"
CHANNEL_NAMES[8]="Ironing-PunchHolder-Indirect"
CHANNEL_NAMES[9]="Ironing-Bottom-Indirect"

# Prompt for WandB group name (optional)
read -p "Enter WandB group name for this experiment batch (optional, press Enter to skip): " WANDB_GROUP
if [ -z "$WANDB_GROUP" ]; then
    echo "No WandB group specified - experiments will not be grouped"
    WANDB_GROUP_PARAM=""
else
    echo "Using WandB group: $WANDB_GROUP"
    WANDB_GROUP_PARAM="-p wandb_group \"$WANDB_GROUP\""
fi

# Prompt for cropping initial time steps (optional)
read -p "Skip first N time steps? Enter number to crop, or press Enter for no cropping: " SKIP_TIMESTEPS
if [ -z "$SKIP_TIMESTEPS" ]; then
    echo "No cropping - using full time series"
    SKIP_TIMESTEPS=0
else
    echo "Will skip first ${SKIP_TIMESTEPS} time steps"
fi

# Prompt for seed range
read -p "Enter seed range (e.g., '1-5' for seeds 1-5, or press Enter for single seed 42): " SEED_RANGE
if [ -z "$SEED_RANGE" ]; then
    echo "Using single seed: 42"
    SEEDS=(42)
else
    # Parse seed range (e.g., "1-5" -> 1 2 3 4 5)
    SEED_START=$(echo "$SEED_RANGE" | cut -d'-' -f1)
    SEED_END=$(echo "$SEED_RANGE" | cut -d'-' -f2)
    SEEDS=($(seq $SEED_START $SEED_END))
    echo "Will run with ${#SEEDS[@]} seeds: ${SEEDS[*]}"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Initialize log file
{
    echo "========================================"
    echo "Experiment Run Started: $(date)"
    echo "Max parallel jobs: ${MAX_PARALLEL_JOBS}"
    echo "Seeds: ${SEEDS[*]}"
    if [ -n "$WANDB_GROUP" ]; then
        echo "WandB Group: ${WANDB_GROUP}"
    else
        echo "WandB Group: (none)"
    fi
    if [ "$SKIP_TIMESTEPS" -gt 0 ]; then
        echo "Skip first N timesteps: ${SKIP_TIMESTEPS}"
    else
        echo "Skip first N timesteps: (none)"
    fi
    echo "========================================"
    echo ""
} > "${LOG_FILE}"

# Function to safely append to log file (thread-safe with flock)
log_message() {
    local message="$1"
    (
        flock -x 200
        echo "$message" >> "${LOG_FILE}"
    ) 200>"${LOG_FILE}.lock"
}

# Function to run a single experiment (designed to run in background)
run_experiment() {
    local signal_portion=$1
    local signal_channels=$2
    local target=$3
    local experiment_name=$4
    local experiment_seed=$5

    # Create experiment directory
    local exp_dir="${OUTPUT_DIR}/${experiment_name}"
    mkdir -p "${exp_dir}"

    # Log experiment start
    log_message "---"
    log_message "Experiment: ${experiment_name}"
    log_message "Started: $(date)"
    log_message "  Signal Portion: ${signal_portion}"
    log_message "  Channels: ${signal_channels}"
    log_message "  Target: ${target}"
    log_message "  Seed: ${experiment_seed}"
    log_message "  Skip Timesteps: ${SKIP_TIMESTEPS}"

    # Construct papermill command with parameters
    local papermill_params=""
    papermill_params+=" -p data_path all_data.hdf5"
    papermill_params+=" -p plot_path ${exp_dir}"
    papermill_params+=" -p classification_target ${target}"
    papermill_params+=" -p seed ${experiment_seed}"

    # Pass channel configuration based on signal portion type
    if [ "${signal_portion}" == "Single" ]; then
        # For single channels, pass as YAML list
        papermill_params+=" -r signal_only_channels [${signal_channels}]"
    elif [ "${signal_portion}" == "DIRECT" ]; then
        # DIRECT channels: 1, 2, 3
        papermill_params+=" -r signal_only_channels [1,2,3]"
    elif [ "${signal_portion}" == "INDIRECT" ]; then
        # INDIRECT channels: 4, 5, 7, 8, 9 (skipping 6 which is "Not Connected")
        papermill_params+=" -r signal_only_channels [4,5,7,8,9]"
    elif [ "${signal_portion}" == "ALL" ]; then
        # ALL channels: DIRECT + INDIRECT
        papermill_params+=" -r signal_only_channels [1,2,3,4,5,7,8,9]"
    fi

    # Add WandB group if specified
    if [ -n "$WANDB_GROUP" ]; then
        papermill_params+=" -p wandb_group \"${WANDB_GROUP}\""
    fi

    # Always pass skip_first_n_timesteps so the notebook doesn't fall back to its default
    papermill_params+=" -p skip_first_n_timesteps ${SKIP_TIMESTEPS}"

    # Execute the notebook with papermill
    echo "[$(date +%H:%M:%S)] Starting: ${experiment_name}"

    if papermill ${NOTEBOOK} "${exp_dir}/output.ipynb" \
        ${papermill_params} \
        --kernel python3 \
        --execution-timeout 3600 \
        --request-save-on-cell-execute \
        > "${exp_dir}/execution.log" 2>&1; then
        echo "[$(date +%H:%M:%S)] ✓ Success: ${experiment_name}"
        log_message "Status: SUCCESS"
    else
        echo "[$(date +%H:%M:%S)] ✗ Failed: ${experiment_name} (see ${exp_dir}/execution.log)"
        log_message "Status: FAILED"
        log_message "Error: Check ${exp_dir}/execution.log for details"
    fi

    log_message "Completed: $(date)"
    log_message ""
}

# Function to wait for background jobs with limit
wait_for_slots() {
    local max_jobs=$1
    while [ $(jobs -r | wc -l) -ge ${max_jobs} ]; do
        sleep 1
    done
}

# Array to store all experiment configurations
# Format: signal_portion|channels|target|experiment_dir|seed
declare -a EXPERIMENTS

# Individual signal experiments and predefined groups, for each seed
INDIVIDUAL_SIGNALS=(1 2 3 4 5 7 8 9)

for current_seed in "${SEEDS[@]}"; do
    for signal in "${INDIVIDUAL_SIGNALS[@]}"; do
        readable_name="${CHANNEL_NAMES[$signal]}"
        EXPERIMENTS+=("Single|${signal}|Ironing|target_ironing-${readable_name}/seed_${current_seed}|${current_seed}")
        EXPERIMENTS+=("Single|${signal}|DeepDrawing|target_deepdrawing-${readable_name}/seed_${current_seed}|${current_seed}")
    done

    # Predefined group experiments
    for group in DIRECT INDIRECT ALL; do
        for target in Ironing DeepDrawing; do
            target_lower=$(echo "$target" | tr '[:upper:]' '[:lower:]')
            EXPERIMENTS+=("${group}||${target}|target_${target_lower}-${group}/seed_${current_seed}|${current_seed}")
        done
    done
done

# Count total experiments
TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}

echo "========================================"
echo "Running ${TOTAL_EXPERIMENTS} Experiments in Parallel"
echo "  (22 configurations × ${#SEEDS[@]} seeds)"
echo "Max parallel jobs: ${MAX_PARALLEL_JOBS}"
echo "========================================"
echo ""

# Launch all experiments
EXPERIMENT_COUNT=0
for exp_config in "${EXPERIMENTS[@]}"; do
    EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))

    # Parse configuration (format: signal_portion|channels|target|name|seed)
    IFS='|' read -r signal_portion signal_channels target experiment_name experiment_seed <<< "$exp_config"

    # Wait if we've hit the parallel job limit
    wait_for_slots ${MAX_PARALLEL_JOBS}

    # Launch experiment in background
    echo "[${EXPERIMENT_COUNT}/${TOTAL_EXPERIMENTS}] Queueing: ${experiment_name}"
    run_experiment "${signal_portion}" "${signal_channels}" "${target}" "${experiment_name}" "${experiment_seed}" &

    # Small delay to avoid race conditions
    sleep 0.1
done

# Wait for all remaining jobs to complete
echo ""
echo "========================================"
echo "Waiting for all experiments to complete..."
echo "========================================"
wait

echo ""
echo "========================================"
echo "All Experiments Completed"
echo "========================================"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo ""

# Generate summary report
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"
echo "Generating summary report..."

{
    echo "========================================="
    echo "Experiment Summary"
    echo "========================================="
    echo "Generated: $(date)"
    echo "Seeds: ${SEEDS[*]}"
    echo ""
    echo "Total Experiments Run: ${TOTAL_EXPERIMENTS}"

    # Count successes and failures
    SUCCESS_COUNT=$(grep -c "Status: SUCCESS" "${LOG_FILE}" 2>/dev/null || echo "0")
    FAILED_COUNT=$(grep -c "Status: FAILED" "${LOG_FILE}" 2>/dev/null || echo "0")

    echo "Successful: ${SUCCESS_COUNT}"
    echo "Failed: ${FAILED_COUNT}"
    echo ""
    echo "Results by Configuration:"
    echo "-----------------------------------------"
    echo ""

    # Process each experiment (find test_results.txt recursively for nested seed dirs)
    while IFS= read -r result_file; do
        if [ -f "${result_file}" ]; then
            exp_path=$(dirname "${result_file}")
            exp_name=${exp_path#${OUTPUT_DIR}/}
            echo "Experiment: ${exp_name}"

            # Extract key metrics from test_results.txt
            grep -E "(test_acc|test_f1|test_loss)" "${result_file}" 2>/dev/null | sed 's/^/  /' || echo "  No results found"
            echo ""
        fi
    done < <(find "${OUTPUT_DIR}" -name "test_results.txt" | sort)

    echo "========================================="
    echo "Full log available at: ${LOG_FILE}"
    echo "========================================="
} > "${SUMMARY_FILE}"

cat "${SUMMARY_FILE}"

echo ""
echo "Summary saved to: ${SUMMARY_FILE}"
echo ""
echo "To visualize results, run the visualize_results.ipynb notebook"
