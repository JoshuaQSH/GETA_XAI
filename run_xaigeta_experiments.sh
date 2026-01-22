#!/bin/bash
#
# XAI-GETA Experiment Runner
# ==========================
# This script runs qvgg7_xaigeta_demo.py with different attribution methods
# and weights, saving logs to the logs/ directory.
#
# Usage:
#   ./run_xaigeta_experiments.sh                    # Run all default experiments
#   ./run_xaigeta_experiments.sh -m saliency -w 0.3 -e 50   # Run single experiment
#   ./run_xaigeta_experiments.sh --help             # Show help
#
# Log naming convention: <method>_<weight>_<epochs>epoch.log
# Example: saliency_0.3_50epoch.log
#

set -e  # Exit on error

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PYTHON_SCRIPT="${SCRIPT_DIR}/qvgg7_xaigeta_demo.py"

# Default experiment parameters
DEFAULT_EPOCHS=50
DEFAULT_DEVICE="cuda:0"

# Attribution methods to test (all compatible methods)
ALL_METHODS=(
    # "saliency"
    "input_x_gradient"
    "guided_backprop"
    "deconvolution"
    "layer_conductance"
    "layer_gradient_x_activation"
    "layer_integrated_gradients"
    "deep_lift"
    "integrated_gradients"
    "lrp"
    "layer_lrp"
)

# Fast methods (for quick testing)
FAST_METHODS=(
    "saliency"
    "input_x_gradient"
    "deconvolution"
)

# Default weights to test
DEFAULT_WEIGHTS=(0.3 0.5)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run XAI-GETA experiments with different attribution methods and weights."
    echo ""
    echo "Options:"
    echo "  -m, --method METHOD     Attribution method (default: run all methods)"
    echo "  -w, --weight WEIGHT     Attribution weight 0.0-1.0 (default: 0.3)"
    echo "  -e, --epochs EPOCHS     Number of epochs (default: $DEFAULT_EPOCHS)"
    echo "  -d, --device DEVICE     Device to use (default: $DEFAULT_DEVICE)"
    echo "  -a, --all               Run all methods with all default weights"
    echo "  -f, --fast              Run only fast methods (saliency, input_x_gradient, deconvolution)"
    echo "  -l, --list              List all available attribution methods"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -m saliency -w 0.3 -e 50    # Single experiment"
    echo "  $0 --all                        # All methods with weights 0.1, 0.3, 0.5"
    echo "  $0 --fast -e 10                 # Fast methods, 10 epochs each"
    echo ""
    echo "Available methods:"
    for method in "${ALL_METHODS[@]}"; do
        echo "  - $method"
    done
}

# List all methods
list_methods() {
    echo "Available attribution methods:"
    echo ""
    echo "Gradient-Based (Fast):"
    echo "  - saliency"
    echo "  - input_x_gradient"
    echo "  - guided_backprop"
    echo "  - deconvolution"
    echo ""
    echo "Layer-Specific:"
    echo "  - layer_conductance"
    echo "  - layer_gradient_x_activation"
    echo "  - layer_integrated_gradients"
    echo ""
    echo "Path-Based:"
    echo "  - deep_lift"
    echo "  - integrated_gradients"
    echo ""
    echo "Decomposition (LRP):"
    echo "  - lrp"
    echo "  - layer_lrp"
}

# Run a single experiment
run_experiment() {
    local method=$1
    local weight=$2
    local epochs=$3
    local device=$4
    
    # Create log filename
    local log_file="${LOG_DIR}/${method}_${weight}_${epochs}epoch.log"
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${YELLOW}Running experiment:${NC}"
    echo -e "  Method: ${GREEN}${method}${NC}"
    echo -e "  Weight: ${GREEN}${weight}${NC}"
    echo -e "  Epochs: ${GREEN}${epochs}${NC}"
    echo -e "  Device: ${GREEN}${device}${NC}"
    echo -e "  Log:    ${GREEN}${log_file}${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Create logs directory if it doesn't exist
    mkdir -p "${LOG_DIR}"
    
    # Run the experiment
    python "${PYTHON_SCRIPT}" \
        --method "${method}" \
        --weight "${weight}" \
        --epochs "${epochs}" \
        --device "${device}" \
        > "${log_file}" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Experiment completed successfully!${NC}"
        echo -e "  Log saved to: ${log_file}"
        
        # Extract final accuracy from log
        local final_acc=$(grep -o "Final Top-1 Accuracy: [0-9.]*%" "${log_file}" | tail -1)
        if [ -n "$final_acc" ]; then
            echo -e "  ${final_acc}"
        fi
    else
        echo -e "${RED}✗ Experiment failed with exit code ${exit_code}${NC}"
        echo -e "  Check log for details: ${log_file}"
    fi
    
    echo ""
    return $exit_code
}

# Run all experiments
run_all_experiments() {
    local methods=("${!1}")
    local weights=("${!2}")
    local epochs=$3
    local device=$4
    
    local total=$((${#methods[@]} * ${#weights[@]}))
    local current=0
    local passed=0
    local failed=0
    
    echo -e "${BLUE}============================================${NC}"
    echo -e "${YELLOW}Running ${total} experiments${NC}"
    echo -e "  Methods: ${#methods[@]}"
    echo -e "  Weights: ${weights[*]}"
    echo -e "  Epochs:  ${epochs}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    
    for method in "${methods[@]}"; do
        for weight in "${weights[@]}"; do
            current=$((current + 1))
            echo -e "${YELLOW}[${current}/${total}]${NC} Running ${method} with weight ${weight}..."
            
            if run_experiment "${method}" "${weight}" "${epochs}" "${device}"; then
                passed=$((passed + 1))
            else
                failed=$((failed + 1))
            fi
        done
    done
    
    # Summary
    echo -e "${BLUE}============================================${NC}"
    echo -e "${YELLOW}Experiment Summary${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo -e "  Total:  ${total}"
    echo -e "  ${GREEN}Passed: ${passed}${NC}"
    echo -e "  ${RED}Failed: ${failed}${NC}"
    echo -e "  Logs:   ${LOG_DIR}/"
    echo -e "${BLUE}============================================${NC}"
}

# Parse command line arguments
METHOD=""
WEIGHT=""
EPOCHS="${DEFAULT_EPOCHS}"
DEVICE="${DEFAULT_DEVICE}"
RUN_ALL=false
RUN_FAST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--method)
            METHOD="$2"
            shift 2
            ;;
        -w|--weight)
            WEIGHT="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -a|--all)
            RUN_ALL=true
            shift
            ;;
        -f|--fast)
            RUN_FAST=true
            shift
            ;;
        -l|--list)
            list_methods
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Check if Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo -e "${RED}Error: Python script not found: ${PYTHON_SCRIPT}${NC}"
    exit 1
fi

# Run experiments based on options
if [ "$RUN_ALL" = true ]; then
    # Run all methods with all default weights
    run_all_experiments ALL_METHODS[@] DEFAULT_WEIGHTS[@] "${EPOCHS}" "${DEVICE}"
elif [ "$RUN_FAST" = true ]; then
    # Run fast methods only
    if [ -n "$WEIGHT" ]; then
        WEIGHTS=("$WEIGHT")
    else
        WEIGHTS=("${DEFAULT_WEIGHTS[@]}")
    fi
    run_all_experiments FAST_METHODS[@] WEIGHTS[@] "${EPOCHS}" "${DEVICE}"
elif [ -n "$METHOD" ]; then
    # Run single experiment
    if [ -z "$WEIGHT" ]; then
        WEIGHT="0.3"
    fi
    run_experiment "${METHOD}" "${WEIGHT}" "${EPOCHS}" "${DEVICE}"
else
    # Default: show usage
    echo -e "${YELLOW}No experiment specified. Use --help for usage.${NC}"
    echo ""
    echo "Quick examples:"
    echo "  $0 -m saliency -w 0.3 -e 50   # Single experiment"
    echo "  $0 --fast -e 10               # Fast methods, quick test"
    echo "  $0 --all                      # All methods (takes a long time)"
    exit 0
fi
