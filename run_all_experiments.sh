#!/bin/bash

# Script to run all signal combination experiments for sheet metal forming classification
# This script tests all individual signals and predefined groups for both Ironing and Deep Drawing
# Experiments run in parallel for faster execution

# Configuration
NOTEBOOK="eval-models.ipynb"
OUTPUT_DIR="experiment_results"
LOG_FILE="${OUTPUT_DIR}/experiment_log.txt"
MAX_PARALLEL_JOBS=8  # Adjust based on available CPU/GPU resources

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Initialize log file
{
    echo "========================================"
    echo "Experiment Run Started: $(date)"
    echo "Max parallel jobs: ${MAX_PARALLEL_JOBS}"
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

    # Construct notebook execution command
    local nb_cmd="jupyter nbconvert --to notebook --execute ${NOTEBOOK}"
    nb_cmd+=" --output ${exp_dir}/output.ipynb"
    nb_cmd+=" --ExecutePreprocessor.timeout=3600"
    nb_cmd+=" --ExecutePreprocessor.kernel_name=python3"

    # Build argument list for the notebook
    local nb_args="--classification_target ${target} --plot_path ${exp_dir}"

    if [ "${signal_portion}" == "Single" ]; then
        nb_args+=" --signal_portion Single --signal_only_channels ${signal_channels}"
    else
        nb_args+=" --signal_portion ${signal_portion}"
    fi

    # Execute the notebook with arguments
    echo "[$(date +%H:%M:%S)] Starting: ${experiment_name}"

    if ${nb_cmd} -- ${nb_args} > "${exp_dir}/execution.log" 2>&1; then
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
declare -a EXPERIMENTS

# Individual signal experiments (all 8 individual signals × 2 targets = 16 experiments)
INDIVIDUAL_SIGNALS=(1 2 3 4 5 7 8 9)

for signal in "${INDIVIDUAL_SIGNALS[@]}"; do
    EXPERIMENTS+=("Single|${signal}|Ironing|signal_${signal}_ironing")
    EXPERIMENTS+=("Single|${signal}|DeepDrawing|signal_${signal}_deepdrawing")
done

# Predefined group experiments (3 groups × 2 targets = 6 experiments)
EXPERIMENTS+=("DIRECT||Ironing|direct_ironing")
EXPERIMENTS+=("DIRECT||DeepDrawing|direct_deepdrawing")
EXPERIMENTS+=("INDIRECT||Ironing|indirect_ironing")
EXPERIMENTS+=("INDIRECT||DeepDrawing|indirect_deepdrawing")
EXPERIMENTS+=("ALL||Ironing|all_ironing")
EXPERIMENTS+=("ALL||DeepDrawing|all_deepdrawing")

# Count total experiments
TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}

echo "========================================"
echo "Running ${TOTAL_EXPERIMENTS} Experiments in Parallel"
echo "Max parallel jobs: ${MAX_PARALLEL_JOBS}"
echo "========================================"
echo ""

# Launch all experiments
EXPERIMENT_COUNT=0
for exp_config in "${EXPERIMENTS[@]}"; do
    EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))

    # Parse configuration (format: signal_portion|channels|target|name)
    IFS='|' read -r signal_portion signal_channels target experiment_name <<< "$exp_config"

    # Wait if we've hit the parallel job limit
    wait_for_slots ${MAX_PARALLEL_JOBS}

    # Launch experiment in background
    echo "[${EXPERIMENT_COUNT}/${TOTAL_EXPERIMENTS}] Queueing: ${experiment_name}"
    run_experiment "${signal_portion}" "${signal_channels}" "${target}" "${experiment_name}" &

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

    # Process each experiment
    for exp_dir in ${OUTPUT_DIR}/*/; do
        if [ -d "${exp_dir}" ] && [ -f "${exp_dir}/test_results.txt" ]; then
            exp_name=$(basename "${exp_dir}")
            echo "Experiment: ${exp_name}"

            # Extract key metrics from test_results.txt
            grep -E "(test_acc|test_f1|test_loss)" "${exp_dir}/test_results.txt" 2>/dev/null | sed 's/^/  /' || echo "  No results found"
            echo ""
        fi
    done

    echo "========================================="
    echo "Full log available at: ${LOG_FILE}"
    echo "========================================="
} > "${SUMMARY_FILE}"

cat "${SUMMARY_FILE}"

echo ""
echo "Summary saved to: ${SUMMARY_FILE}"
echo ""
echo "To analyze results, run:"
echo "  python analyze_experiments.py --results_dir ${OUTPUT_DIR}"
