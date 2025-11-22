#!/bin/bash
# Download PyTorch YOLO Models Script
# Downloads all required YOLO .pt files to local storage for faster container startup

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/pytorch_models"

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}PyTorch YOLO Model Downloader${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo "Target directory: $MODELS_DIR"
echo ""

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# Model list
# SIMPLIFIED: Only download small model for clean benchmarking
declare -A MODELS=(
    ["yolo11s.pt"]="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
)

# Download each model
for model_file in "${!MODELS[@]}"; do
    model_url="${MODELS[$model_file]}"
    target_path="$MODELS_DIR/$model_file"

    if [ -f "$target_path" ]; then
        echo -e "${GREEN}✓${NC} $model_file already exists"
    else
        echo -e "${YELLOW}→${NC} Downloading $model_file..."
        curl -L -o "$target_path" "$model_url"

        if [ -f "$target_path" ]; then
            file_size=$(ls -lh "$target_path" | awk '{print $5}')
            echo -e "${GREEN}✓${NC} $model_file downloaded ($file_size)"
        else
            echo -e "${RED}✗${NC} Failed to download $model_file"
            exit 1
        fi
    fi
done

echo ""
echo -e "${BLUE}==========================================${NC}"
echo -e "${GREEN}✓ All models downloaded successfully${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo "Models location: $MODELS_DIR"
echo ""
ls -lh "$MODELS_DIR"
