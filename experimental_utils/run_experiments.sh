#!/bin/bash
# W&B Experiment Matrix Runner - Shell Wrapper
# Convenience CLI for common experiment workflows

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="${PROJECT_NAME:-qaic-training-opt}"
CONFIG_PATH="${CONFIG_PATH:-./base_config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./wandb_experiments}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print helpers
print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Commands
show_help() {
    cat <<EOF
W&B Experiment Matrix Runner

USAGE:
    $0 <command> [options]

COMMANDS:
    quick-test      Run experiments 1, 3, 4 for quick validation (1 epoch)
    full-run        Run all 6 experiments for full validation (5 epochs)
    baseline        Run only baseline experiment (exp 1)
    optimized       Run only optimized experiment (exp 4)
    dry-run         Preview configs without running
 
    single EXP_ID   Run single experiment (e.g., single 2)
    subset IDS      Run subset of experiments (e.g., subset 1 2 4)
 
    analyze         Analyze results from last runs
    report          Generate summary report
    csv             Export results to CSV
    dataload EXP_ID Show per-epoch dataloading times (e.g., dataload 4)

OPTIONS:
    --config PATH       Base config file (default: ${CONFIG_PATH})
    --project NAME      W&B project (default: ${PROJECT_NAME})
    --epochs N          Training epochs (default: 1)
    --output DIR        Output directory (default: ${OUTPUT_DIR})
    --entity NAME       W&B entity name (optional) 
    --help              Show this help message

EXAMPLES:
    # Run quick test with 1 epoch
    $0 quick-test
    
    # Run full matrix with 5 epochs
    $0 full-run --epochs 5
    
    # Run experiments 1, 3, 4 with custom config
    $0 quick-test --config my_config.yaml
    
    # Analyze previous results
    $0 analyze --project qaic-training-opt
    
    # View per-epoch dataloading for optimized config (exp 4)
    $0 dataload 4
    
    # Generate report and export CSV
    $0 report --project qaic-training-opt
    
EOF
}

# Validation
check_prereqs() {
    print_header "Checking Prerequisites"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi
    print_success "Python 3: $(python3 --version)"
    
    # Check wandb
    if ! python3 -c "import wandb" 2>/dev/null; then
        print_error "wandb not installed. Run: pip install wandb"
        exit 1
    fi
    print_success "wandb installed"
    
    # Check config file
    if [[ "$CMD" != "analyze" && "$CMD" != "report" && "$CMD" != "csv" ]]; then
        if [[ ! -f "$CONFIG_PATH" ]]; then
            print_error "Config file not found: $CONFIG_PATH"
            echo "Create one with: cp QEfficient/finetune/experimental/configs/sft_ddp_config_optimized.yaml ./base_config.yaml"
            exit 1
        fi
        print_success "Config file: $CONFIG_PATH"
    fi
    
    echo ""
}

# Commands
cmd_quick_test() {
    print_header "Running Quick Test (Exp 19 20 21 22 23)"
    print_info "Using QAIC devices: 16,17,18,19,20"
    QAIC_VISIBLE_DEVICES=38,39,40,41,42 python3 "$SCRIPT_DIR/run_wandb_experiment_matrix.py" \
        --config "$CONFIG_PATH" \
        --project "$PROJECT_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --epochs 6 \
        --exp-ids 19 20 21 22 23 \
        ${ENTITY:+--entity "$ENTITY"}
}

cmd_full_run() {
    print_header "Running Full Matrix (All 6 Experiments)"
    print_info "Using QAIC devices: 32,33,34,35,36,37,38"
    QAIC_VISIBLE_DEVICES=32,33,34,35,36,37,38 python3 "$SCRIPT_DIR/run_wandb_experiment_matrix.py" \
        --config "$CONFIG_PATH" \
        --project "$PROJECT_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --epochs "${EPOCHS:-5}" \
        ${ENTITY:+--entity "$ENTITY"}
}

cmd_baseline() {
    print_header "Running Baseline (Exp 1 Only)"
    print_info "Using QAIC devices: 16,17,18,19"
    QAIC_VISIBLE_DEVICES=16,17,18,19 python3 "$SCRIPT_DIR/run_wandb_experiment_matrix.py" \
        --config "$CONFIG_PATH" \
        --project "$PROJECT_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --epochs "${EPOCHS:-1}" \
        --exp-ids 1 \
        ${ENTITY:+--entity "$ENTITY"}
}

cmd_optimized() {
    print_header "Running Optimized Config (Exp 4 Only)"
    print_info "Using QAIC devices: 16,17,18,19"
    QAIC_VISIBLE_DEVICES=16,17,18,19 python3 "$SCRIPT_DIR/run_wandb_experiment_matrix.py" \
        --config "$CONFIG_PATH" \
        --project "$PROJECT_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --epochs "${EPOCHS:-5}" \
        --exp-ids 4 \
        ${ENTITY:+--entity "$ENTITY"}
}

cmd_dry_run() {
    print_header "Dry Run (Preview Configs)"
    print_info "Using QAIC devices: 16,17,18,19"
    QAIC_VISIBLE_DEVICES=16,17,18,19 python3 "$SCRIPT_DIR/run_wandb_experiment_matrix.py" \
        --config "$CONFIG_PATH" \
        --project "$PROJECT_NAME" \
        --dry-run \
        --epochs "${EPOCHS:-1}"
}

cmd_single() {
    if [[ -z "$1" ]]; then
        print_error "Experiment ID required. Usage: single <exp_id>"
        exit 1
    fi
    print_header "Running Single Experiment (Exp $1)"
    print_info "Using QAIC devices: 16,17,18,19"
    QAIC_VISIBLE_DEVICES=16,17,18,19 python3 "$SCRIPT_DIR/run_wandb_experiment_matrix.py" \
        --config "$CONFIG_PATH" \
        --project "$PROJECT_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --epochs "${EPOCHS:-1}" \
        --exp-ids "$1" \
        ${ENTITY:+--entity "$ENTITY"}
}

cmd_subset() {
    if [[ -z "$1" ]]; then
        print_error "Experiment IDs required. Usage: subset <id1> <id2> ..."
        exit 1
    fi
    print_header "Running Subset: $@"
    print_info "Using QAIC devices: 38,39,40,41,42"
    QAIC_VISIBLE_DEVICES=38,39,40,41,42 python3 "$SCRIPT_DIR/run_wandb_experiment_matrix.py" \
        --config "$CONFIG_PATH" \
        --project "$PROJECT_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --epochs "${EPOCHS:-1}" \
        --exp-ids "$@" \
        ${ENTITY:+--entity "$ENTITY"}
}

cmd_analyze() {
    print_header "Analyzing Results"
    python3 "$SCRIPT_DIR/analyze_wandb_results.py" \
        --project "$PROJECT_NAME" \
        --show-table \
        ${ENTITY:+--entity "$ENTITY"}
}

cmd_report() {
    print_header "Generating Summary Report"
    REPORT_FILE="${OUTPUT_DIR}/experiment_summary_report.txt"
    mkdir -p "$OUTPUT_DIR"
    python3 "$SCRIPT_DIR/analyze_wandb_results.py" \
        --project "$PROJECT_NAME" \
        --report "$REPORT_FILE" \
        ${ENTITY:+--entity "$ENTITY"}
    print_success "Report saved to: $REPORT_FILE"
    echo ""
    cat "$REPORT_FILE"
}

cmd_csv() {
    print_header "Exporting Results to CSV"
    CSV_FILE="${OUTPUT_DIR}/experiment_results.csv"
    mkdir -p "$OUTPUT_DIR"
    python3 "$SCRIPT_DIR/analyze_wandb_results.py" \
        --project "$PROJECT_NAME" \
        --csv "$CSV_FILE" \
        ${ENTITY:+--entity "$ENTITY"}
    print_success "CSV saved to: $CSV_FILE"
}

cmd_dataload() {
    print_header "Per-Epoch Dataloading Analysis"
    EXP_ID="${1:-1}"
    python3 "$SCRIPT_DIR/analyze_wandb_results.py" \
        --project "$PROJECT_NAME" \
        --dataload-exp "$EXP_ID" \
        ${ENTITY:+--entity "$ENTITY"}
}

# Parse arguments
CMD="${1:---help}"
shift || true

# Parse global options while processing remaining args
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --project)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --entity)
            ENTITY="$2"
            shift 2
            ;;
        --help | -h)
            show_help
            exit 0
            ;;
        *)
            # Remaining args for commands
            break
            ;;
    esac
done

# Execute command
case $CMD in
    quick-test)
        check_prereqs
        cmd_quick_test
        ;;
    full-run)
        check_prereqs
        cmd_full_run
        ;;
    baseline)
        check_prereqs
        cmd_baseline
        ;;
    optimized)
        check_prereqs
        cmd_optimized
        ;;
    dry-run)
        check_prereqs
        cmd_dry_run
        ;;
    single)
        check_prereqs
        cmd_single "$@"
        ;;
    subset)
        check_prereqs
        cmd_subset "$@"
        ;;
    analyze)
        cmd_analyze
        ;;
    report)
        cmd_report
        ;;
    csv)
        cmd_csv
        ;;
    dataload)
        cmd_dataload "$@"
        ;;
    --help | -h)
        show_help
        ;;
    *)
        print_error "Unknown command: $CMD"
        echo ""
        show_help
        exit 1
        ;;
esac
