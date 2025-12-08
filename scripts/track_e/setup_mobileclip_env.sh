#!/bin/bash
# =============================================================================
# Track E: MobileCLIP2-S2 Environment Setup (Host-Side)
# =============================================================================
#
# This script sets up the MobileCLIP2 environment on the HOST machine by:
# 1. Cloning Apple's ml-mobileclip repository to ./reference_repos/
# 2. Cloning and patching OpenCLIP with MobileCLIP2 support
# 3. Downloading MobileCLIP2-S2 checkpoint to ./pytorch_models/
#
# These directories are then mounted into the Docker container.
#
# Run from: Host machine (repo root)
#   bash scripts/track_e/setup_mobileclip_env.sh
#
# After running this script:
#   docker compose up -d --build
#   docker compose exec yolo-api bash /app/scripts/track_e/install_mobileclip_deps.sh
#
# The script is idempotent - safe to run multiple times.
#
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "Track E: MobileCLIP2-S2 Host Setup"
echo "=============================================="

# Get the repo root (script is in scripts/track_e/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Repo root: $REPO_ROOT"

# Paths on host
REFERENCE_REPOS="$REPO_ROOT/reference_repos"
MOBILECLIP_REPO="$REFERENCE_REPOS/ml-mobileclip"
OPENCLIP_DIR="$REFERENCE_REPOS/open_clip"
PYTORCH_MODELS="$REPO_ROOT/pytorch_models"
CHECKPOINT_DIR="$PYTORCH_MODELS/mobileclip2_s2"

# Create directories
mkdir -p "$REFERENCE_REPOS"
mkdir -p "$PYTORCH_MODELS"
mkdir -p "$CHECKPOINT_DIR"

# =============================================================================
# Step 1: Clone Apple's ml-mobileclip repository
# =============================================================================
echo ""
echo "Step 1: Cloning Apple's ml-mobileclip repository..."

if [ -d "$MOBILECLIP_REPO" ]; then
    echo "  ✓ ml-mobileclip already exists at $MOBILECLIP_REPO"
    cd "$MOBILECLIP_REPO"
    git pull --quiet 2>/dev/null || echo "  (using cached version)"
else
    echo "  Cloning from GitHub..."
    cd "$REFERENCE_REPOS"
    git clone --depth 1 https://github.com/apple/ml-mobileclip.git
    echo "  ✓ ml-mobileclip cloned successfully"
fi

# =============================================================================
# Step 2: Clone and patch OpenCLIP
# =============================================================================
echo ""
echo "Step 2: Setting up OpenCLIP with MobileCLIP2 support..."

if [ -d "$OPENCLIP_DIR" ]; then
    echo "  ✓ OpenCLIP already exists at $OPENCLIP_DIR"
else
    echo "  Cloning OpenCLIP repository..."
    cd "$REFERENCE_REPOS"
    git clone --depth 1 https://github.com/mlfoundations/open_clip.git

    echo "  Applying MobileCLIP2 patch..."
    cd "$OPENCLIP_DIR"

    # Apply the patch (handles if already applied)
    if git apply --check "$MOBILECLIP_REPO/mobileclip2/open_clip_inference_only.patch" 2>/dev/null; then
        git apply "$MOBILECLIP_REPO/mobileclip2/open_clip_inference_only.patch"
        echo "  ✓ Patch applied"
    else
        echo "  ✓ Patch already applied or not needed"
    fi

    echo "  Copying MobileCLIP2 modules..."
    cp -r "$MOBILECLIP_REPO/mobileclip2/"* ./src/open_clip/

    echo "  ✓ OpenCLIP patched successfully"
fi

# =============================================================================
# Step 3: Download MobileCLIP2-S2 checkpoint
# =============================================================================
echo ""
echo "Step 3: Downloading MobileCLIP2-S2 checkpoint..."

CHECKPOINT_FILE="$CHECKPOINT_DIR/mobileclip2_s2.pt"

if [ -f "$CHECKPOINT_FILE" ]; then
    SIZE_MB=$(du -m "$CHECKPOINT_FILE" | cut -f1)
    echo "  ✓ Checkpoint already exists at $CHECKPOINT_FILE ($SIZE_MB MB)"
else
    echo "  Downloading from HuggingFace (~398 MB)..."

    # Try using huggingface-cli if available, otherwise use curl
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download apple/MobileCLIP2-S2 mobileclip2_s2.pt \
            --local-dir "$CHECKPOINT_DIR" \
            --local-dir-use-symlinks False
    elif command -v python3 &> /dev/null && python3 -c "import huggingface_hub" 2>/dev/null; then
        python3 << DOWNLOAD_SCRIPT
from huggingface_hub import hf_hub_download
import os

checkpoint_dir = '$CHECKPOINT_DIR'

print('  Downloading MobileCLIP2-S2 checkpoint...')
try:
    model_path = hf_hub_download(
        repo_id='apple/MobileCLIP2-S2',
        filename='mobileclip2_s2.pt',
        local_dir=checkpoint_dir
    )
    print(f'  ✓ Downloaded to: {model_path}')
except Exception as e:
    print(f'  ✗ Download failed: {e}')
    exit(1)
DOWNLOAD_SCRIPT
    else
        # Fallback to curl
        echo "  Using curl to download..."
        curl -L --progress-bar -o "$CHECKPOINT_FILE" \
            "https://huggingface.co/apple/MobileCLIP2-S2/resolve/main/mobileclip2_s2.pt"
    fi

    if [ -f "$CHECKPOINT_FILE" ]; then
        SIZE_MB=$(du -m "$CHECKPOINT_FILE" | cut -f1)
        echo "  ✓ Checkpoint downloaded successfully ($SIZE_MB MB)"
    else
        echo "  ERROR: Failed to download checkpoint"
        exit 1
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "✅ Host Setup Complete!"
echo "=============================================="
echo ""
echo "Directory structure created:"
echo "  $REFERENCE_REPOS/"
echo "    ├── ml-mobileclip/     (Apple's MobileCLIP repo)"
echo "    └── open_clip/         (Patched OpenCLIP)"
echo "  $PYTORCH_MODELS/"
echo "    └── mobileclip2_s2/"
echo "        └── mobileclip2_s2.pt (~398 MB)"
echo ""
echo "Next steps:"
echo "  1. Build and start containers:"
echo "     docker compose up -d --build"
echo ""
echo "  2. Install MobileCLIP Python deps in container:"
echo "     docker compose exec yolo-api bash /app/scripts/track_e/install_mobileclip_deps.sh"
echo ""
echo "  3. Export models:"
echo "     docker compose exec yolo-api python /app/scripts/track_e/export_mobileclip_image_encoder.py"
echo "     docker compose exec yolo-api python /app/scripts/track_e/export_mobileclip_text_encoder.py"
echo ""
