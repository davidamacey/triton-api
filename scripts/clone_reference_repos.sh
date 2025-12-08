#!/bin/bash
# Clone reference repositories for development and attribution
# These repos are gitignored but provide important reference implementations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
REF_DIR="$REPO_ROOT/reference_repos"

# Reference repositories with descriptions
declare -A REPOS=(
    ["ultralytics-end2end"]="https://github.com/levipereira/ultralytics.git"
    ["ultralytics-official"]="https://github.com/ultralytics/ultralytics.git"
    ["ml-mobileclip"]="https://github.com/apple/ml-mobileclip.git"
    ["open_clip"]="https://github.com/mlfoundations/open_clip.git"
    ["triton-server-yolo"]="https://github.com/levipereira/triton-server-yolo.git"
    ["DeepStream-Yolo"]="https://github.com/marcoslucianops/DeepStream-Yolo.git"
    ["yolov8-triton"]="https://github.com/omarabid59/yolov8-triton.git"
)

# Descriptions for each repo
declare -A DESCRIPTIONS=(
    ["ultralytics-end2end"]="End2End NMS export patch (REQUIRED for Track C/D attribution)"
    ["ultralytics-official"]="Official Ultralytics repo for comparison"
    ["ml-mobileclip"]="Apple MobileCLIP for Track E embeddings"
    ["open_clip"]="OpenCLIP reference implementation"
    ["triton-server-yolo"]="Triton YOLO server reference by levipereira"
    ["DeepStream-Yolo"]="DeepStream YOLO implementation reference"
    ["yolov8-triton"]="YOLOv8 Triton deployment reference"
)

# Categories
ESSENTIAL=("ultralytics-end2end" "ml-mobileclip")
RECOMMENDED=("open_clip" "ultralytics-official")
OPTIONAL=("triton-server-yolo" "DeepStream-Yolo" "yolov8-triton")

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Clone reference repositories for development and attribution."
    echo ""
    echo "Options:"
    echo "  --essential    Clone essential repos (ultralytics-end2end, ml-mobileclip)"
    echo "  --recommended  Clone essential + recommended repos"
    echo "  --all          Clone all reference repos"
    echo "  --list         List available repos and their status"
    echo "  --repo NAME    Clone a specific repo by name"
    echo "  -h, --help     Show this help message"
    echo ""
    echo "Essential repos (required for attribution):"
    for repo in "${ESSENTIAL[@]}"; do
        echo "  - $repo: ${DESCRIPTIONS[$repo]}"
    done
    echo ""
    echo "Recommended repos:"
    for repo in "${RECOMMENDED[@]}"; do
        echo "  - $repo: ${DESCRIPTIONS[$repo]}"
    done
    echo ""
    echo "Optional repos:"
    for repo in "${OPTIONAL[@]}"; do
        echo "  - $repo: ${DESCRIPTIONS[$repo]}"
    done
}

list_repos() {
    echo "Reference Repositories Status:"
    echo "=============================="
    echo ""

    for repo in "${!REPOS[@]}"; do
        local status="NOT CLONED"
        local category="optional"

        if [[ -d "$REF_DIR/$repo/.git" ]]; then
            status="CLONED"
        fi

        if [[ " ${ESSENTIAL[*]} " =~ " $repo " ]]; then
            category="essential"
        elif [[ " ${RECOMMENDED[*]} " =~ " $repo " ]]; then
            category="recommended"
        fi

        printf "%-25s [%-11s] %-10s\n" "$repo" "$category" "$status"
        echo "  ${DESCRIPTIONS[$repo]}"
        echo "  ${REPOS[$repo]}"
        echo ""
    done
}

clone_repo() {
    local name=$1
    local url=${REPOS[$name]}

    if [[ -z "$url" ]]; then
        echo "Error: Unknown repo '$name'"
        return 1
    fi

    local target="$REF_DIR/$name"

    if [[ -d "$target/.git" ]]; then
        echo "✓ $name already cloned, updating..."
        git -C "$target" fetch --all
        git -C "$target" pull --ff-only 2>/dev/null || echo "  (not on tracking branch, skipping pull)"
    else
        echo "→ Cloning $name..."
        mkdir -p "$REF_DIR"
        git clone "$url" "$target"
        echo "✓ $name cloned successfully"
    fi
}

clone_repos() {
    local repos=("$@")

    echo "Cloning ${#repos[@]} repository(ies)..."
    echo ""

    for repo in "${repos[@]}"; do
        clone_repo "$repo"
        echo ""
    done

    echo "Done!"
}

# Parse arguments
if [[ $# -eq 0 ]]; then
    print_usage
    exit 0
fi

case "$1" in
    --essential)
        clone_repos "${ESSENTIAL[@]}"
        ;;
    --recommended)
        clone_repos "${ESSENTIAL[@]}" "${RECOMMENDED[@]}"
        ;;
    --all)
        clone_repos "${!REPOS[@]}"
        ;;
    --list)
        list_repos
        ;;
    --repo)
        if [[ -z "$2" ]]; then
            echo "Error: --repo requires a repository name"
            exit 1
        fi
        clone_repo "$2"
        ;;
    -h|--help)
        print_usage
        ;;
    *)
        echo "Unknown option: $1"
        print_usage
        exit 1
        ;;
esac
