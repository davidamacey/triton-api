#!/bin/bash
# =============================================================================
# Face Recognition Test Data Setup
# =============================================================================
# Downloads and organizes face datasets for testing Track E face recognition.
#
# Datasets:
# - LFW Deep Funneled: 13,233 images, 5,749 people (111MB) - Primary
# - Test Subset: 100 images, 20 people (~2MB) - Quick validation
#
# Usage:
#   bash scripts/setup_face_test_data.sh [--full|--subset|--all]
#
# Options:
#   --full    Download full LFW dataset only (default)
#   --subset  Create test subset only (requires full dataset)
#   --all     Download full dataset and create subset
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FACES_DIR="$PROJECT_ROOT/test_images/faces"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Download LFW Deep Funneled Dataset
# =============================================================================
download_lfw() {
    local LFW_URL="http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
    local LFW_FILE="$FACES_DIR/lfw-deepfunneled.tgz"
    local LFW_DIR="$FACES_DIR/lfw-deepfunneled"
    local EXPECTED_MD5="68331da3eb755a505a502b5aacb3c201"

    log_info "Setting up LFW Deep Funneled dataset..."

    # Create directory
    mkdir -p "$FACES_DIR"

    # Check if already extracted
    if [ -d "$LFW_DIR" ] && [ "$(ls -A "$LFW_DIR" 2>/dev/null | wc -l)" -gt 100 ]; then
        log_info "LFW dataset already exists at $LFW_DIR"
        log_info "  People: $(ls -d "$LFW_DIR"/*/ 2>/dev/null | wc -l)"
        log_info "  Images: $(find "$LFW_DIR" -name "*.jpg" 2>/dev/null | wc -l)"
        return 0
    fi

    # Download if needed
    if [ ! -f "$LFW_FILE" ]; then
        log_info "Downloading LFW Deep Funneled (111MB)..."
        log_info "  Source: $LFW_URL"

        if command -v wget &> /dev/null; then
            wget -q --show-progress -O "$LFW_FILE" "$LFW_URL"
        elif command -v curl &> /dev/null; then
            curl -L --progress-bar -o "$LFW_FILE" "$LFW_URL"
        else
            log_error "Neither wget nor curl found. Please install one."
            exit 1
        fi
    else
        log_info "Archive already downloaded: $LFW_FILE"
    fi

    # Verify checksum
    log_info "Verifying checksum..."
    if command -v md5sum &> /dev/null; then
        ACTUAL_MD5=$(md5sum "$LFW_FILE" | cut -d' ' -f1)
        if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
            log_warn "Checksum mismatch (expected: $EXPECTED_MD5, got: $ACTUAL_MD5)"
            log_warn "File may be corrupted. Re-downloading..."
            rm -f "$LFW_FILE"
            download_lfw
            return $?
        fi
        log_info "Checksum verified: $ACTUAL_MD5"
    else
        log_warn "md5sum not available, skipping checksum verification"
    fi

    # Extract
    log_info "Extracting dataset..."
    cd "$FACES_DIR"
    tar -xzf "$(basename "$LFW_FILE")"

    # Cleanup archive
    rm -f "$LFW_FILE"

    # Verify extraction
    local num_people=$(ls -d "$LFW_DIR"/*/ 2>/dev/null | wc -l)
    local num_images=$(find "$LFW_DIR" -name "*.jpg" 2>/dev/null | wc -l)

    log_info "LFW dataset extracted successfully!"
    log_info "  Location: $LFW_DIR"
    log_info "  People: $num_people"
    log_info "  Images: $num_images"
}

# =============================================================================
# Create Test Subset (100 images from 20 people)
# =============================================================================
create_test_subset() {
    local LFW_DIR="$FACES_DIR/lfw-deepfunneled"
    local SUBSET_DIR="$FACES_DIR/test_subset"
    local NUM_PEOPLE=20
    local IMAGES_PER_PERSON=5

    log_info "Creating test subset..."

    # Check LFW exists
    if [ ! -d "$LFW_DIR" ]; then
        log_error "LFW dataset not found. Run with --full first."
        exit 1
    fi

    # Remove existing subset
    rm -rf "$SUBSET_DIR"
    mkdir -p "$SUBSET_DIR"

    # Find people with at least 5 images (for meaningful testing)
    log_info "Finding people with $IMAGES_PER_PERSON+ images..."
    local count=0

    for person_dir in "$LFW_DIR"/*/; do
        if [ $count -ge $NUM_PEOPLE ]; then
            break
        fi

        local person_name=$(basename "$person_dir")
        local num_images=$(ls "$person_dir"/*.jpg 2>/dev/null | wc -l)

        if [ "$num_images" -ge "$IMAGES_PER_PERSON" ]; then
            # Copy first N images
            mkdir -p "$SUBSET_DIR/$person_name"
            local img_count=0
            for img in "$person_dir"/*.jpg; do
                if [ $img_count -ge $IMAGES_PER_PERSON ]; then
                    break
                fi
                cp "$img" "$SUBSET_DIR/$person_name/"
                ((img_count++))
            done
            ((count++))
            log_info "  Added: $person_name ($img_count images)"
        fi
    done

    local total_images=$(find "$SUBSET_DIR" -name "*.jpg" | wc -l)
    log_info "Test subset created successfully!"
    log_info "  Location: $SUBSET_DIR"
    log_info "  People: $count"
    log_info "  Images: $total_images"
}

# =============================================================================
# Create pairs.txt for verification testing
# =============================================================================
create_test_pairs() {
    local SUBSET_DIR="$FACES_DIR/test_subset"
    local PAIRS_FILE="$FACES_DIR/test_pairs.txt"

    log_info "Creating test pairs file..."

    if [ ! -d "$SUBSET_DIR" ]; then
        log_error "Test subset not found. Run with --subset first."
        exit 1
    fi

    # Create pairs: positive (same person) and negative (different people)
    echo "# Face verification test pairs" > "$PAIRS_FILE"
    echo "# Format: person1 img1 person2 img2 label (1=same, 0=different)" >> "$PAIRS_FILE"
    echo "# Generated: $(date)" >> "$PAIRS_FILE"
    echo "" >> "$PAIRS_FILE"

    local people=()
    for person_dir in "$SUBSET_DIR"/*/; do
        people+=("$(basename "$person_dir")")
    done

    local pair_count=0

    # Positive pairs (same person, different images)
    for person in "${people[@]}"; do
        local images=("$SUBSET_DIR/$person"/*.jpg)
        if [ ${#images[@]} -ge 2 ]; then
            local img1=$(basename "${images[0]}")
            local img2=$(basename "${images[1]}")
            echo "$person $img1 $person $img2 1" >> "$PAIRS_FILE"
            ((pair_count++))
        fi
    done

    # Negative pairs (different people)
    local num_people=${#people[@]}
    for ((i=0; i<num_people-1; i++)); do
        local person1="${people[$i]}"
        local person2="${people[$((i+1))]}"
        local img1=$(ls "$SUBSET_DIR/$person1"/*.jpg | head -1 | xargs basename)
        local img2=$(ls "$SUBSET_DIR/$person2"/*.jpg | head -1 | xargs basename)
        echo "$person1 $img1 $person2 $img2 0" >> "$PAIRS_FILE"
        ((pair_count++))
    done

    log_info "Test pairs created: $PAIRS_FILE"
    log_info "  Total pairs: $pair_count"
}

# =============================================================================
# Print dataset summary
# =============================================================================
print_summary() {
    log_info ""
    log_info "=========================================="
    log_info "Face Test Data Summary"
    log_info "=========================================="

    if [ -d "$FACES_DIR/lfw-deepfunneled" ]; then
        local lfw_people=$(ls -d "$FACES_DIR/lfw-deepfunneled"/*/ 2>/dev/null | wc -l)
        local lfw_images=$(find "$FACES_DIR/lfw-deepfunneled" -name "*.jpg" 2>/dev/null | wc -l)
        log_info "LFW Deep Funneled:"
        log_info "  Path: test_images/faces/lfw-deepfunneled/"
        log_info "  People: $lfw_people"
        log_info "  Images: $lfw_images"
    else
        log_warn "LFW dataset not installed"
    fi

    if [ -d "$FACES_DIR/test_subset" ]; then
        local subset_people=$(ls -d "$FACES_DIR/test_subset"/*/ 2>/dev/null | wc -l)
        local subset_images=$(find "$FACES_DIR/test_subset" -name "*.jpg" 2>/dev/null | wc -l)
        log_info ""
        log_info "Test Subset:"
        log_info "  Path: test_images/faces/test_subset/"
        log_info "  People: $subset_people"
        log_info "  Images: $subset_images"
    fi

    if [ -f "$FACES_DIR/test_pairs.txt" ]; then
        local pair_count=$(grep -v "^#" "$FACES_DIR/test_pairs.txt" | grep -v "^$" | wc -l)
        log_info ""
        log_info "Test Pairs:"
        log_info "  Path: test_images/faces/test_pairs.txt"
        log_info "  Pairs: $pair_count"
    fi

    log_info "=========================================="
}

# =============================================================================
# Main
# =============================================================================
main() {
    local mode="${1:---full}"

    case "$mode" in
        --full)
            download_lfw
            ;;
        --subset)
            create_test_subset
            create_test_pairs
            ;;
        --all)
            download_lfw
            create_test_subset
            create_test_pairs
            ;;
        --summary)
            print_summary
            ;;
        --help|-h)
            echo "Usage: $0 [--full|--subset|--all|--summary]"
            echo ""
            echo "Options:"
            echo "  --full    Download full LFW dataset only (default)"
            echo "  --subset  Create test subset from LFW (requires --full first)"
            echo "  --all     Download full dataset and create subset"
            echo "  --summary Show current dataset status"
            echo "  --help    Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $mode"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac

    print_summary
}

main "$@"
