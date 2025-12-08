#!/bin/bash
# =============================================================================
# Track E: Install MobileCLIP Python Dependencies (Container-Side)
# =============================================================================
#
# This script installs the MobileCLIP Python packages from the mounted
# reference_repos volume. Run this INSIDE the container after host setup.
#
# Prerequisites:
#   1. Run setup_mobileclip_env.sh on the host first
#   2. Container must be running with volumes mounted
#
# Run from: Inside yolo-api container
#   docker compose exec yolo-api bash /app/scripts/track_e/install_mobileclip_deps.sh
#
# The script is idempotent - safe to run multiple times.
#
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "Track E: Installing MobileCLIP Dependencies"
echo "=============================================="

# Paths in container (mounted from host)
REFERENCE_REPOS="/app/reference_repos"
MOBILECLIP_REPO="$REFERENCE_REPOS/ml-mobileclip"
OPENCLIP_DIR="$REFERENCE_REPOS/open_clip"
CHECKPOINT_PATH="/app/pytorch_models/mobileclip2_s2/mobileclip2_s2.pt"

# =============================================================================
# Verify mounted volumes
# =============================================================================
echo ""
echo "Step 1: Verifying mounted volumes..."

if [ ! -d "$MOBILECLIP_REPO" ]; then
    echo "  ✗ ERROR: ml-mobileclip not found at $MOBILECLIP_REPO"
    echo "    Run setup_mobileclip_env.sh on the HOST first!"
    exit 1
fi
echo "  ✓ ml-mobileclip found"

if [ ! -d "$OPENCLIP_DIR" ]; then
    echo "  ✗ ERROR: OpenCLIP not found at $OPENCLIP_DIR"
    echo "    Run setup_mobileclip_env.sh on the HOST first!"
    exit 1
fi
echo "  ✓ OpenCLIP found"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "  ✗ ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo "    Run setup_mobileclip_env.sh on the HOST first!"
    exit 1
fi
echo "  ✓ Checkpoint found"

# =============================================================================
# Install Python packages
# =============================================================================
echo ""
echo "Step 2: Installing OpenCLIP (editable mode)..."
cd "$OPENCLIP_DIR"
pip install -e . --quiet 2>/dev/null || pip install -e .
echo "  ✓ OpenCLIP installed"

echo ""
echo "Step 3: Installing mobileclip module (editable mode)..."
cd "$MOBILECLIP_REPO"
pip install -e . --quiet 2>/dev/null || pip install -e .
echo "  ✓ mobileclip installed"

# =============================================================================
# Verify installation
# =============================================================================
echo ""
echo "Step 4: Verifying installation..."

python3 << 'VERIFY_SCRIPT'
import sys
import os

errors = []

# Check OpenCLIP
try:
    import open_clip
    print(f"  ✓ OpenCLIP version: {open_clip.__version__}")
except ImportError as e:
    errors.append(f"OpenCLIP import failed: {e}")
    print(f"  ✗ OpenCLIP import failed: {e}")

# Check if MobileCLIP2-S2 is available
try:
    available_models = open_clip.list_models()
    if 'MobileCLIP2-S2' in available_models:
        print("  ✓ MobileCLIP2-S2 model registered in OpenCLIP")
    else:
        print("  ⚠ MobileCLIP2-S2 not in list_models() - will use pretrained path")
except Exception as e:
    print(f"  ⚠ Could not check models: {e}")

# Check mobileclip module
try:
    from mobileclip.modules.common.mobileone import reparameterize_model
    print("  ✓ mobileclip module available (reparameterize_model)")
except ImportError as e:
    errors.append(f"mobileclip module failed: {e}")
    print(f"  ✗ mobileclip module failed: {e}")

# Check timm
try:
    import timm
    print(f"  ✓ timm version: {timm.__version__}")
except ImportError as e:
    errors.append(f"timm import failed: {e}")
    print(f"  ✗ timm import failed: {e}")

# Check checkpoint exists
checkpoint_path = "/app/pytorch_models/mobileclip2_s2/mobileclip2_s2.pt"
if os.path.exists(checkpoint_path):
    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"  ✓ Checkpoint exists: {size_mb:.1f} MB")
else:
    errors.append("Checkpoint not found")
    print(f"  ✗ Checkpoint not found at {checkpoint_path}")

# Final result
if errors:
    print(f"\n❌ Setup incomplete - {len(errors)} error(s)")
    sys.exit(1)
else:
    print("\n✅ All verifications passed!")
VERIFY_SCRIPT

echo ""
echo "=============================================="
echo "✅ MobileCLIP2-S2 Ready!"
echo "=============================================="
echo ""
echo "Next steps - Export models:"
echo "  python /app/scripts/track_e/export_mobileclip_image_encoder.py"
echo "  python /app/scripts/track_e/export_mobileclip_text_encoder.py"
echo ""
