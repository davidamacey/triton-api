#!/bin/bash
# =============================================================================
# PP-OCRv5 Model Export Script for Triton Inference Server
# =============================================================================
#
# This script automates the full export pipeline for PP-OCRv5 OCR models:
# 1. Download ONNX models (detection + recognition)
# 2. Convert to TensorRT engines with optimal settings
# 3. Setup dictionary files
# 4. Restart Triton to load new models
#
# Usage:
#   ./scripts/export_paddleocr.sh [command]
#
# Commands:
#   all        - Run complete export pipeline (default)
#   download   - Download ONNX models only
#   trt        - Convert ONNX to TensorRT only
#   det        - Export detection model only
#   rec        - Export recognition model only
#   restart    - Restart Triton server
#   test       - Run OCR test
#   clean      - Remove all OCR model files
#   status     - Check model status
#
# Requirements:
#   - Docker with triton-api and yolo-api containers running
#   - NVIDIA GPU with sufficient memory (4GB+ for TRT build)
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
PYTORCH_MODELS_DIR="$PROJECT_DIR/pytorch_models/paddleocr"
EXPORTS_DIR="$PROJECT_DIR/models/exports/ocr"

# Model paths
DET_ONNX_NAME="ppocr_det_v5_mobile.onnx"
REC_ONNX_NAME="en_ppocrv5_mobile_rec.onnx"
DET_PLAN="$MODELS_DIR/paddleocr_det_trt/1/model.plan"
REC_PLAN="$MODELS_DIR/paddleocr_rec_trt/1/model.plan"
DICT_FILE="$MODELS_DIR/paddleocr_rec_trt/en_ppocrv5_dict.txt"

# TensorRT configuration
# CRITICAL: Use --memPoolSize=workspace:4G syntax (not --workspace)
TRT_WORKSPACE="4G"  # 4GB workspace for TRT optimization
TRT_FP16="--fp16"   # Use FP16 precision

# Detection model shapes (H,W are multiples of 32, max 960)
DET_MIN_SHAPES="x:1x3x32x32"
DET_OPT_SHAPES="x:1x3x736x736"
DET_MAX_SHAPES="x:4x3x960x960"

# Recognition model shapes (height=48, width=8-2048 dynamic)
REC_MIN_SHAPES="x:1x3x48x8"
REC_OPT_SHAPES="x:32x3x48x320"
REC_MAX_SHAPES="x:64x3x48x2048"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_container() {
    local container=$1
    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        return 0
    else
        return 1
    fi
}

wait_for_triton() {
    log_info "Waiting for Triton to be ready..."
    for i in {1..30}; do
        if curl -s localhost:4600/v2/health/ready > /dev/null 2>&1; then
            log_success "Triton is ready"
            return 0
        fi
        sleep 2
    done
    log_error "Triton did not become ready in time"
    return 1
}

unload_models_for_memory() {
    log_info "Unloading models to free GPU memory..."

    local models_to_unload=(
        "ocr_pipeline"
        "paddleocr_det_trt"
        "paddleocr_rec_trt"
        "yolov11_small_trt"
        "yolov11_small_trt_end2end"
        "mobileclip_image_trt"
        "mobileclip_text_trt"
    )

    for model in "${models_to_unload[@]}"; do
        curl -s -X POST "localhost:4600/v2/repository/models/${model}/unload" > /dev/null 2>&1 || true
    done

    sleep 3
    log_success "Models unloaded"
}

# =============================================================================
# Download Functions
# =============================================================================

download_models() {
    log_info "Downloading PP-OCRv5 models..."

    if ! check_container "yolo-api"; then
        log_error "yolo-api container not running. Start with: docker compose up -d yolo-api"
        return 1
    fi

    docker compose exec yolo-api python /app/export/download_paddleocr.py

    if [ $? -eq 0 ]; then
        log_success "Models downloaded successfully"
    else
        log_error "Model download failed"
        return 1
    fi
}

# =============================================================================
# TensorRT Export Functions
# =============================================================================

export_detection() {
    log_info "Exporting detection model to TensorRT..."

    local onnx_path="/models/$DET_ONNX_NAME"
    local plan_path="/models/paddleocr_det_trt/1/model.plan"

    # Check ONNX exists
    if [ ! -f "$PYTORCH_MODELS_DIR/$DET_ONNX_NAME" ]; then
        log_error "Detection ONNX not found: $PYTORCH_MODELS_DIR/$DET_ONNX_NAME"
        log_info "Run: ./scripts/export_paddleocr.sh download"
        return 1
    fi

    # Copy ONNX to models dir for container access
    cp "$PYTORCH_MODELS_DIR/$DET_ONNX_NAME" "$MODELS_DIR/$DET_ONNX_NAME"

    # Create output directory
    mkdir -p "$MODELS_DIR/paddleocr_det_trt/1"

    # Remove old plan
    rm -f "$DET_PLAN"

    log_info "Running trtexec for detection model..."
    log_info "  Min shapes: $DET_MIN_SHAPES"
    log_info "  Opt shapes: $DET_OPT_SHAPES"
    log_info "  Max shapes: $DET_MAX_SHAPES"
    log_info "  Workspace: $TRT_WORKSPACE"

    docker compose exec triton-api /usr/src/tensorrt/bin/trtexec \
        --onnx="$onnx_path" \
        --saveEngine="$plan_path" \
        --minShapes="$DET_MIN_SHAPES" \
        --optShapes="$DET_OPT_SHAPES" \
        --maxShapes="$DET_MAX_SHAPES" \
        --memPoolSize=workspace:$TRT_WORKSPACE \
        $TRT_FP16 \
        2>&1 | tee /tmp/trtexec_det.log

    if [ -f "$DET_PLAN" ] && [ -s "$DET_PLAN" ]; then
        local size=$(du -h "$DET_PLAN" | cut -f1)
        log_success "Detection TensorRT engine created: $size"
    else
        log_error "Detection TensorRT conversion failed"
        log_info "Check log: /tmp/trtexec_det.log"
        return 1
    fi
}

export_recognition() {
    log_info "Exporting recognition model to TensorRT..."

    local onnx_path="/models/$REC_ONNX_NAME"
    local plan_path="/models/paddleocr_rec_trt/1/model.plan"

    # Check ONNX exists (may be in exports dir or pytorch_models)
    local onnx_source=""
    if [ -f "$EXPORTS_DIR/$REC_ONNX_NAME" ]; then
        onnx_source="$EXPORTS_DIR/$REC_ONNX_NAME"
    elif [ -f "$PYTORCH_MODELS_DIR/$REC_ONNX_NAME" ]; then
        onnx_source="$PYTORCH_MODELS_DIR/$REC_ONNX_NAME"
    else
        log_error "Recognition ONNX not found"
        log_info "Expected locations:"
        log_info "  - $EXPORTS_DIR/$REC_ONNX_NAME"
        log_info "  - $PYTORCH_MODELS_DIR/$REC_ONNX_NAME"
        log_info "Run: docker compose exec yolo-api python /app/export/export_paddleocr_rec.py --skip-tensorrt"
        return 1
    fi

    # Copy ONNX to models dir for container access
    cp "$onnx_source" "$MODELS_DIR/$REC_ONNX_NAME"

    # Create output directory
    mkdir -p "$MODELS_DIR/paddleocr_rec_trt/1"

    # Remove old plan
    rm -f "$REC_PLAN"

    log_info "Running trtexec for recognition model..."
    log_info "  Min shapes: $REC_MIN_SHAPES"
    log_info "  Opt shapes: $REC_OPT_SHAPES"
    log_info "  Max shapes: $REC_MAX_SHAPES"
    log_info "  Workspace: $TRT_WORKSPACE"
    log_warn "This may take 10-20 minutes for dynamic width optimization..."

    docker compose exec triton-api /usr/src/tensorrt/bin/trtexec \
        --onnx="$onnx_path" \
        --saveEngine="$plan_path" \
        --minShapes="$REC_MIN_SHAPES" \
        --optShapes="$REC_OPT_SHAPES" \
        --maxShapes="$REC_MAX_SHAPES" \
        --memPoolSize=workspace:$TRT_WORKSPACE \
        $TRT_FP16 \
        2>&1 | tee /tmp/trtexec_rec.log

    if [ -f "$REC_PLAN" ] && [ -s "$REC_PLAN" ]; then
        local size=$(du -h "$REC_PLAN" | cut -f1)
        log_success "Recognition TensorRT engine created: $size"
    else
        log_error "Recognition TensorRT conversion failed"
        log_info "Check log: /tmp/trtexec_rec.log"
        return 1
    fi
}

export_all_trt() {
    log_info "Converting all models to TensorRT..."

    if ! check_container "triton-api"; then
        log_error "triton-api container not running. Start with: docker compose up -d triton-api"
        return 1
    fi

    # Unload models to free GPU memory
    unload_models_for_memory

    # Export detection
    export_detection || return 1

    # Export recognition
    export_recognition || return 1

    log_success "All TensorRT exports complete"
}

# =============================================================================
# Config and Dictionary Functions
# =============================================================================

setup_dictionary() {
    log_info "Setting up dictionary file..."

    mkdir -p "$(dirname "$DICT_FILE")"

    # Check if dictionary already exists
    if [ -f "$DICT_FILE" ]; then
        local char_count=$(wc -l < "$DICT_FILE")
        log_info "Dictionary already exists: $char_count characters"
        return 0
    fi

    # Try to extract from PaddleX model config
    docker compose exec yolo-api python -c "
import yaml
from pathlib import Path

paddlex_dir = Path.home() / '.paddlex/official_models/en_PP-OCRv5_mobile_rec'
config_path = paddlex_dir / 'inference.yml'

if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    char_dict = config.get('PostProcess', {}).get('character_dict', [])
    if char_dict:
        output_path = Path('/models/paddleocr_rec_trt/en_ppocrv5_dict.txt')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for char in char_dict:
                f.write(f'{char}\n')

        print(f'Wrote {len(char_dict)} characters to dictionary')
    else:
        print('No character_dict found in config')
else:
    print(f'Config not found: {config_path}')
" 2>/dev/null || true

    if [ -f "$DICT_FILE" ]; then
        local char_count=$(wc -l < "$DICT_FILE")
        log_success "Dictionary created: $char_count characters"
    else
        log_warn "Could not extract dictionary automatically"
        log_info "Please ensure en_ppocrv5_dict.txt is placed in $MODELS_DIR/paddleocr_rec_trt/"
    fi
}

create_configs() {
    log_info "Creating Triton configs..."

    # Detection config
    cat > "$MODELS_DIR/paddleocr_det_trt/config.pbtxt" << 'EOF'
# PP-OCRv5 Text Detection Model (TensorRT)
#
# DB++ architecture for text region detection
#
# Input:
#   - x: [B, 3, H, W] FP32, preprocessed (x / 127.5 - 1), BGR
#        H,W must be multiples of 32, max 960
#
# Output:
#   - fetch_name_0: [B, 1, H, W] FP32, probability map [0, 1]

name: "paddleocr_det_trt"
platform: "tensorrt_plan"
max_batch_size: 4

input [
  {
    name: "x"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]  # Dynamic H, W (multiples of 32)
  }
]

output [
  {
    name: "fetch_name_0"
    data_type: TYPE_FP32
    dims: [ 1, -1, -1 ]  # Same H, W as input
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

dynamic_batching {
  preferred_batch_size: [ 1, 2, 4 ]
  max_queue_delay_microseconds: 5000
}
EOF

    # Recognition config
    cat > "$MODELS_DIR/paddleocr_rec_trt/config.pbtxt" << 'EOF'
# PP-OCRv5 Text Recognition Model (TensorRT)
#
# SVTR-LCNet architecture for text sequence recognition
# Supports dynamic width for variable-length text recognition
#
# Input:
#   - x: [B, 3, 48, W] FP32, preprocessed (x / 127.5 - 1), BGR
#        Text crops with height=48, width=8-2048
#
# Output:
#   - fetch_name_0: [B, T, 438] FP32, character probabilities
#        T timesteps (dynamic), 438 character classes (English dict)

name: "paddleocr_rec_trt"
platform: "tensorrt_plan"
max_batch_size: 1

input [
  {
    name: "x"
    data_type: TYPE_FP32
    dims: [ 3, 48, -1 ]
  }
]

output [
  {
    name: "fetch_name_0"
    data_type: TYPE_FP32
    dims: [ -1, 438 ]
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

# No dynamic batching - each request has different width
# Process individually for best accuracy
EOF

    log_success "Triton configs created"
}

# =============================================================================
# Control Functions
# =============================================================================

restart_triton() {
    log_info "Restarting Triton server..."
    docker compose restart triton-api
    wait_for_triton
}

reload_models() {
    log_info "Reloading OCR models..."

    local models=("paddleocr_det_trt" "paddleocr_rec_trt" "ocr_pipeline")

    for model in "${models[@]}"; do
        log_info "Loading $model..."
        curl -s -X POST "localhost:4600/v2/repository/models/${model}/load" > /dev/null 2>&1 || true
    done

    sleep 2
    log_success "Models reloaded"
}

run_test() {
    log_info "Running OCR test..."

    if [ -f "$PROJECT_DIR/scripts/test_ocr_pipeline.py" ]; then
        python "$PROJECT_DIR/scripts/test_ocr_pipeline.py"
    else
        # Quick curl test
        local test_image="$PROJECT_DIR/test_images/ocr-synthetic/hello_world.jpg"
        if [ -f "$test_image" ]; then
            log_info "Testing with: $test_image"
            curl -s -X POST http://localhost:4603/track_e/ocr/predict \
                -F "image=@$test_image" | python -m json.tool
        else
            log_warn "No test image found at: $test_image"
            log_info "Testing model health..."
            curl -s localhost:4600/v2/models/paddleocr_det_trt | python -m json.tool
        fi
    fi
}

show_status() {
    echo ""
    echo "=========================================="
    echo "PP-OCRv5 Model Status"
    echo "=========================================="
    echo ""

    # Check files
    echo "Model Files:"
    echo "-----------"

    if [ -f "$DET_PLAN" ]; then
        local det_size=$(du -h "$DET_PLAN" | cut -f1)
        echo -e "  Detection TRT:   ${GREEN}OK${NC} ($det_size)"
    else
        echo -e "  Detection TRT:   ${RED}MISSING${NC}"
    fi

    if [ -f "$REC_PLAN" ]; then
        local rec_size=$(du -h "$REC_PLAN" | cut -f1)
        echo -e "  Recognition TRT: ${GREEN}OK${NC} ($rec_size)"
    else
        echo -e "  Recognition TRT: ${RED}MISSING${NC}"
    fi

    if [ -f "$DICT_FILE" ]; then
        local char_count=$(wc -l < "$DICT_FILE")
        echo -e "  Dictionary:      ${GREEN}OK${NC} ($char_count chars)"
    else
        echo -e "  Dictionary:      ${RED}MISSING${NC}"
    fi

    echo ""
    echo "Triton Models:"
    echo "-------------"

    for model in paddleocr_det_trt paddleocr_rec_trt ocr_pipeline; do
        local status=$(curl -s "localhost:4600/v2/models/$model" 2>/dev/null | grep -o '"state":"[^"]*"' | cut -d'"' -f4)
        if [ "$status" = "READY" ]; then
            echo -e "  $model: ${GREEN}READY${NC}"
        elif [ -n "$status" ]; then
            echo -e "  $model: ${YELLOW}$status${NC}"
        else
            echo -e "  $model: ${RED}NOT LOADED${NC}"
        fi
    done

    echo ""
}

clean_all() {
    log_warn "This will remove all OCR model files. Continue? [y/N]"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf "$MODELS_DIR/paddleocr_det_trt/1/model.plan"
        rm -rf "$MODELS_DIR/paddleocr_rec_trt/1/model.plan"
        rm -f "$MODELS_DIR/$DET_ONNX_NAME"
        rm -f "$MODELS_DIR/$REC_ONNX_NAME"
        log_success "OCR model files removed"
    else
        log_info "Cancelled"
    fi
}

# =============================================================================
# Full Pipeline
# =============================================================================

run_full_pipeline() {
    echo ""
    echo "=========================================="
    echo "PP-OCRv5 Full Export Pipeline"
    echo "=========================================="
    echo ""

    # 1. Download models
    log_info "Step 1/5: Downloading models..."
    download_models || return 1

    # 2. Setup dictionary
    log_info "Step 2/5: Setting up dictionary..."
    setup_dictionary

    # 3. Export to TensorRT
    log_info "Step 3/5: Exporting to TensorRT..."
    export_all_trt || return 1

    # 4. Create configs
    log_info "Step 4/5: Creating Triton configs..."
    create_configs

    # 5. Restart Triton
    log_info "Step 5/5: Restarting Triton..."
    restart_triton

    echo ""
    log_success "Export pipeline complete!"
    show_status
}

# =============================================================================
# Main
# =============================================================================

show_help() {
    echo "PP-OCRv5 Model Export Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  all        Run complete export pipeline (default)"
    echo "  download   Download ONNX models only"
    echo "  trt        Convert ONNX to TensorRT only"
    echo "  det        Export detection model only"
    echo "  rec        Export recognition model only"
    echo "  config     Create Triton configs only"
    echo "  dict       Setup dictionary file only"
    echo "  restart    Restart Triton server"
    echo "  reload     Reload OCR models without restart"
    echo "  test       Run OCR test"
    echo "  status     Check model status"
    echo "  clean      Remove all OCR model files"
    echo "  help       Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 all           # Full pipeline: download + TRT + configs + restart"
    echo "  $0 trt           # Convert existing ONNX to TensorRT"
    echo "  $0 det           # Export detection model only"
    echo "  $0 status        # Check current status"
    echo ""
}

main() {
    local command="${1:-all}"

    case "$command" in
        all)
            run_full_pipeline
            ;;
        download)
            download_models
            ;;
        trt)
            export_all_trt
            ;;
        det)
            unload_models_for_memory
            export_detection
            ;;
        rec)
            unload_models_for_memory
            export_recognition
            ;;
        config)
            create_configs
            ;;
        dict)
            setup_dictionary
            ;;
        restart)
            restart_triton
            ;;
        reload)
            reload_models
            ;;
        test)
            run_test
            ;;
        status)
            show_status
            ;;
        clean)
            clean_all
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
