#!/bin/bash
# =============================================================================
# Isolated Track Benchmark Script
# =============================================================================
# Runs fair benchmarks by loading only ONE track at a time with equalized
# instance counts. Prevents resource contention between tracks.
#
# Usage:
#   ./isolated_benchmark.sh [OPTIONS]
#
# Options:
#   --track TRACK    Run single track (A, B, C, D_streaming, D_balanced, D_batch, E, E_full)
#   --all            Run all tracks sequentially (default)
#   --duration SEC   Benchmark duration per track (default: 60)
#   --clients NUM    Number of concurrent clients (default: 128)
#   --skip-warmup    Skip model warmup
#   --no-restore     Don't restore production configs after benchmark
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"
BENCHMARK_CONFIGS="$SCRIPT_DIR/configs"
OUTPUT_DIR="$SCRIPT_DIR/isolated"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Defaults
DURATION=60
CLIENTS=128
WARMUP=20
RUN_ALL=true
SINGLE_TRACK=""
SKIP_WARMUP=false
NO_RESTORE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Track Definitions
# =============================================================================

declare -A TRACK_MODELS
declare -A TRACK_IDS
declare -A TRACK_CONFIGS

# Track A: PyTorch (no Triton models needed)
TRACK_IDS[A]="A"
TRACK_MODELS[A]=""
TRACK_CONFIGS[A]=""

# Track B: TensorRT + CPU NMS
TRACK_IDS[B]="B"
TRACK_MODELS[B]="yolov11_small_trt"
TRACK_CONFIGS[B]="track_b"

# Track C: TensorRT + GPU NMS
TRACK_IDS[C]="C"
TRACK_MODELS[C]="yolov11_small_trt_end2end"
TRACK_CONFIGS[C]="track_c"

# Track D Streaming: DALI + TRT (low latency)
TRACK_IDS[D_streaming]="D_streaming"
TRACK_MODELS[D_streaming]="yolo_preprocess_dali_streaming,yolov11_small_trt_end2end_streaming,yolov11_small_gpu_e2e_streaming"
TRACK_CONFIGS[D_streaming]="track_d_streaming"

# Track D Balanced: DALI + TRT (general purpose)
TRACK_IDS[D_balanced]="D_balanced"
TRACK_MODELS[D_balanced]="yolo_preprocess_dali,yolov11_small_trt_end2end,yolov11_small_gpu_e2e"
TRACK_CONFIGS[D_balanced]="track_d_balanced"

# Track D Batch: DALI + TRT (max throughput)
TRACK_IDS[D_batch]="D_batch"
TRACK_MODELS[D_batch]="yolo_preprocess_dali_batch,yolov11_small_trt_end2end_batch,yolov11_small_gpu_e2e_batch"
TRACK_CONFIGS[D_batch]="track_d_batch"

# Track E Simple: YOLO + MobileCLIP (fast)
TRACK_IDS[E]="E"
TRACK_MODELS[E]="yolo_clip_preprocess_dali,yolov11_small_trt_end2end,mobileclip2_s2_image_encoder,yolo_clip_ensemble"
TRACK_CONFIGS[E]="track_e_simple"

# Track E Full: YOLO + MobileCLIP + per-box embeddings
TRACK_IDS[E_full]="E_full"
TRACK_MODELS[E_full]="dual_preprocess_dali,yolov11_small_trt_end2end,mobileclip2_s2_image_encoder,box_embedding_extractor,yolo_mobileclip_ensemble"
TRACK_CONFIGS[E_full]="track_e_full"

ALL_TRACKS=("A" "B" "C" "D_streaming" "D_balanced" "D_batch" "E" "E_full")

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --track TRACK    Run single track (A, B, C, D_streaming, D_balanced, D_batch, E, E_full)"
    echo "  --all            Run all tracks sequentially (default)"
    echo "  --duration SEC   Benchmark duration per track (default: 60)"
    echo "  --clients NUM    Number of concurrent clients (default: 128)"
    echo "  --skip-warmup    Skip model warmup"
    echo "  --no-restore     Don't restore production configs after benchmark"
    echo "  --help           Show this help message"
}

wait_for_triton() {
    local max_attempts=60
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -sf http://localhost:4600/v2/health/ready > /dev/null 2>&1; then
            return 0
        fi
        echo -n "."
        sleep 1
        ((attempt++))
    done

    return 1
}

wait_for_model() {
    local model=$1
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -sf "http://localhost:4600/v2/models/$model/ready" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
        ((attempt++))
    done

    return 1
}

unload_all_models() {
    log_info "Unloading all models from Triton..."

    # Get list of loaded models
    local models=$(curl -sf http://localhost:4600/v2/repository/index 2>/dev/null | jq -r '.[].name' 2>/dev/null || echo "")

    local count=0
    for model in $models; do
        curl -sf -X POST "http://localhost:4600/v2/repository/models/$model/unload" > /dev/null 2>&1 || true
        ((count++))
    done

    sleep 2
    log_info "Unloaded $count models"
}

load_models_for_track() {
    local track=$1
    local models="${TRACK_MODELS[$track]}"

    if [ -z "$models" ]; then
        log_info "Track $track uses no Triton models (PyTorch)"
        return 0
    fi

    IFS=',' read -ra MODEL_ARRAY <<< "$models"

    log_info "Loading ${#MODEL_ARRAY[@]} models for Track $track..."

    for model in "${MODEL_ARRAY[@]}"; do
        # Load model
        curl -sf -X POST "http://localhost:4600/v2/repository/models/$model/load" > /dev/null 2>&1 || true

        if ! wait_for_model "$model"; then
            log_error "Failed to load model: $model"
            return 1
        fi

        log_info "  Loaded: $model"
    done

    return 0
}

apply_benchmark_config() {
    local track=$1
    local config_name="${TRACK_CONFIGS[$track]}"

    if [ -z "$config_name" ]; then
        log_info "No benchmark config needed for Track $track"
        return 0
    fi

    local config_dir="$BENCHMARK_CONFIGS/$config_name"

    if [ ! -d "$config_dir" ]; then
        log_warn "Benchmark config dir not found: $config_dir"
        log_warn "Run 'make bench-create-configs' first"
        return 0
    fi

    log_info "Applying benchmark configs from $config_name..."

    for config_file in "$config_dir"/*.config.pbtxt; do
        if [ -f "$config_file" ]; then
            local model_name=$(basename "$config_file" .config.pbtxt)
            local target="$MODELS_DIR/$model_name/config.pbtxt"
            local backup="$MODELS_DIR/$model_name/config.pbtxt.production"

            if [ ! -f "$target" ]; then
                log_warn "  Target config not found: $target"
                continue
            fi

            # Backup production config (only if not already backed up)
            if [ ! -f "$backup" ]; then
                cp "$target" "$backup"
            fi

            # Apply benchmark config
            cp "$config_file" "$target"
            log_info "  Applied: $model_name"
        fi
    done
}

restore_production_configs() {
    log_info "Restoring production configurations..."

    local restored=0
    for model_dir in "$MODELS_DIR"/*/; do
        local backup="$model_dir/config.pbtxt.production"
        local target="$model_dir/config.pbtxt"

        if [ -f "$backup" ]; then
            cp "$backup" "$target"
            rm "$backup"
            ((restored++))
        fi
    done

    if [ $restored -gt 0 ]; then
        log_info "Restored $restored production configs"
    else
        log_info "No configs to restore"
    fi
}

run_benchmark() {
    local track=$1
    local track_id="${TRACK_IDS[$track]}"
    local output_file="$OUTPUT_DIR/benchmark_${track}_${TIMESTAMP}.json"

    log_info "Running benchmark for Track $track..."
    log_info "  Track ID: $track_id"
    log_info "  Duration: ${DURATION}s"
    log_info "  Clients: $CLIENTS"

    # Build warmup arg
    local warmup_arg="--warmup $WARMUP"
    if [ "$SKIP_WARMUP" = true ]; then
        warmup_arg="--warmup 0"
    fi

    # Run Go benchmark tool
    cd "$SCRIPT_DIR"

    if ./triton_bench --mode full --track "$track_id" --clients $CLIENTS --duration $DURATION $warmup_arg --output "$output_file" 2>&1; then
        log_success "Results saved to: $output_file"
        return 0
    else
        log_error "Benchmark failed for Track $track"
        return 1
    fi
}

benchmark_track() {
    local track=$1

    echo ""
    echo "=============================================================================="
    echo " BENCHMARKING TRACK: $track"
    echo " Time: $(date)"
    echo "=============================================================================="
    echo ""

    # Step 1: Unload all models
    unload_all_models

    # Step 2: Apply benchmark configs (equalized instances)
    apply_benchmark_config "$track"

    # Step 3: Load only this track's models
    if ! load_models_for_track "$track"; then
        log_error "Failed to load models for Track $track"
        return 1
    fi

    # Step 4: Wait for stabilization
    log_info "Waiting for models to stabilize..."
    sleep 3

    # Step 5: Run benchmark
    if ! run_benchmark "$track"; then
        log_error "Benchmark failed for Track $track"
        return 1
    fi

    log_success "Track $track benchmark complete"
    return 0
}

# =============================================================================
# Main Script
# =============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --track)
                SINGLE_TRACK="$2"
                RUN_ALL=false
                shift 2
                ;;
            --all)
                RUN_ALL=true
                shift
                ;;
            --duration)
                DURATION="$2"
                shift 2
                ;;
            --clients)
                CLIENTS="$2"
                shift 2
                ;;
            --skip-warmup)
                SKIP_WARMUP=true
                shift
                ;;
            --no-restore)
                NO_RESTORE=true
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

main() {
    parse_args "$@"

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Check prerequisites
    if ! command -v curl &> /dev/null; then
        log_error "curl is required"
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        log_error "jq is required"
        exit 1
    fi

    if [ ! -f "$SCRIPT_DIR/triton_bench" ]; then
        log_error "Benchmark tool not found. Run: make bench-build"
        exit 1
    fi

    # Check Triton is running
    log_info "Checking Triton server..."
    if ! wait_for_triton; then
        log_error "Triton server not responding. Run: make up"
        exit 1
    fi
    log_success "Triton server ready"

    # Validate single track if specified
    if [ -n "$SINGLE_TRACK" ]; then
        local valid=false
        for t in "${ALL_TRACKS[@]}"; do
            if [ "$t" = "$SINGLE_TRACK" ]; then
                valid=true
                break
            fi
        done
        if [ "$valid" = false ]; then
            log_error "Invalid track: $SINGLE_TRACK"
            log_error "Valid tracks: ${ALL_TRACKS[*]}"
            exit 1
        fi
    fi

    # Determine which tracks to run
    local tracks_to_run=()

    if [ "$RUN_ALL" = true ]; then
        tracks_to_run=("${ALL_TRACKS[@]}")
    else
        tracks_to_run=("$SINGLE_TRACK")
    fi

    local start_time=$(date +%s)
    local failed_tracks=()
    local completed_tracks=()

    echo ""
    echo "=============================================================================="
    echo " ISOLATED TRACK BENCHMARK SUITE"
    echo " Started: $(date)"
    echo " Tracks: ${tracks_to_run[*]}"
    echo " Duration per track: ${DURATION}s"
    echo " Concurrent clients: $CLIENTS"
    echo " Output directory: $OUTPUT_DIR"
    echo "=============================================================================="

    for track in "${tracks_to_run[@]}"; do
        if benchmark_track "$track"; then
            completed_tracks+=("$track")
        else
            failed_tracks+=("$track")
        fi
    done

    # Restore production configs (unless --no-restore)
    if [ "$NO_RESTORE" = false ]; then
        restore_production_configs
    fi

    # Reload all models for normal operation
    log_info "Reloading all models for normal operation..."
    curl -sf -X POST "http://localhost:4600/v2/repository/index" > /dev/null 2>&1 || true

    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))

    echo ""
    echo "=============================================================================="
    echo " BENCHMARK SUITE COMPLETE"
    echo " Total time: ${total_time}s"
    echo " Completed: ${#completed_tracks[@]} tracks"
    echo " Results saved to: $OUTPUT_DIR"
    echo "=============================================================================="

    if [ ${#failed_tracks[@]} -gt 0 ]; then
        log_error "Failed tracks: ${failed_tracks[*]}"
        echo ""
        echo "To view results, run: make bench-isolated-results"
        exit 1
    fi

    log_success "All benchmarks completed successfully!"
    echo ""
    echo "To view results, run: make bench-isolated-results"
    echo "Or: python3 benchmarks/aggregate_results.py --input benchmarks/isolated"
}

main "$@"
