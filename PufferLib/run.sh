#!/bin/bash

# PufferLib Drone Training/Evaluation Script
# Usage:
#   ./run.sh train [--wandb] [other args...]
#   ./run.sh eval [--model-path /path/to/model.pt] [other args...]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to find the latest model
find_latest_model() {
    local experiments_dir="experiments"
    local env_name="puffer_drone_pp"

    if [ ! -d "$experiments_dir" ]; then
        print_error "Experiments directory not found: $experiments_dir"
        return 1
    fi

    # Find the most recent experiment directory matching our environment
    local latest_exp_dir=$(find "$experiments_dir" -maxdepth 1 -type d -name "${env_name}_*" | \
                          xargs ls -dt 2>/dev/null | head -1)

    if [ -z "$latest_exp_dir" ]; then
        print_error "No experiment directories found matching pattern: ${env_name}_*"
        return 1
    fi

    print_status "Found latest experiment: $latest_exp_dir" >&2

    # Find the latest model file in that directory
    local latest_model=$(find "$latest_exp_dir" -name "model_${env_name}_*.pt" | \
                        xargs ls -t 2>/dev/null | head -1)

    if [ -z "$latest_model" ]; then
        print_error "No model files found in $latest_exp_dir"
        return 1
    fi

    # Return just the model path, don't echo extra info
    echo "$latest_model"
    return 0
}

# Function to compile the environment
compile_env() {
    print_status "Compiling PufferLib environment..."

    if python setup.py build_ext --inplace --force; then
        print_success "Environment compiled successfully"
    else
        print_error "Failed to compile environment"
        exit 1
    fi
}

# Function to run training
run_training() {
    local args=("$@")
    print_status "Starting training with args: ${args[*]}"

    # Build the command
    local cmd="puffer train puffer_drone_pp"

    # Add any additional arguments
    for arg in "${args[@]}"; do
        cmd="$cmd $arg"
    done

    print_status "Executing: $cmd"
    exec $cmd
}

# Function to run evaluation
run_evaluation() {
    local args=()
    local model_path=""
    local custom_model=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model-path)
                model_path="$2"
                custom_model=true
                shift 2
                ;;
            --load-model-path)
                model_path="$2"
                custom_model=true
                shift 2
                ;;
            *)
                args+=("$1")
                shift
                ;;
        esac
    done

    # If no custom model specified, find the latest one
    if [ "$custom_model" = false ]; then
        print_status "No model path specified, searching for latest model..."
        if model_path=$(find_latest_model); then
            print_success "Found latest model: $model_path"
        else
            print_error "Could not find a model to evaluate"
            exit 1
        fi
    fi

    # Verify model file exists
    if [ ! -f "$model_path" ]; then
        print_error "Model file not found: $model_path"
        exit 1
    fi

    # Build the command
    local cmd="puffer eval puffer_drone_pp --load-model-path \"$model_path\""

    # Add any additional arguments
    for arg in "${args[@]}"; do
        cmd="$cmd $arg"
    done

    print_status "Executing: $cmd"
    eval $cmd
}

# Function to show usage
show_usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  train [--wandb] [args...]     Train a new model"
    echo "  eval [--model-path PATH]      Evaluate a model"
    echo ""
    echo "Training Examples:"
    echo "  $0 train                      Basic training"
    echo "  $0 train --wandb             Training with Weights & Biases logging"
    echo ""
    echo "Evaluation Examples:"
    echo "  $0 eval                       Evaluate latest model (auto-detected)"
    echo "  $0 eval --model-path /path/to/model.pt"
    echo "  $0 eval --load-model-path /path/to/model.pt"
    echo ""
    echo "The script automatically:"
    echo "  - Compiles the environment before running"
    echo "  - Finds the latest model for evaluation if none specified"
    echo "  - Passes through additional arguments to puffer command"
}

# Main script logic
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi

    local command="$1"
    shift

    case "$command" in
        train)
            print_status "=== PufferLib Drone Training ==="
            compile_env
            run_training "$@"
            ;;
        eval|evaluate)
            print_status "=== PufferLib Drone Evaluation ==="
            compile_env
            run_evaluation "$@"
            ;;
        compile)
            print_status "=== PufferLib Environment Compilation ==="
            compile_env
            print_success "Compilation complete"
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Check if running from correct directory
if [ ! -f "setup.py" ]; then
    print_error "setup.py not found. Please run this script from the PufferLib root directory."
    exit 1
fi

# Run main function with all arguments
main "$@"