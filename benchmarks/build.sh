#!/bin/bash
#
# Triton Benchmark Tool - Build Script
# Builds the triton_bench binary for easy distribution
#

set -e

echo "========================================="
echo "Building Triton Benchmark Tool"
echo "========================================="

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "Error: Go is not installed"
    echo ""
    echo "Install Go with:"
    echo "  cd /tmp"
    echo "  wget https://go.dev/dl/go1.23.4.linux-amd64.tar.gz"
    echo "  sudo tar -C /usr/local -xzf go1.23.4.linux-amd64.tar.gz"
    echo "  export PATH=\$PATH:/usr/local/go/bin"
    echo ""
    exit 1
fi

echo "✓ Go version: $(go version)"
echo ""

# Build the binary
echo "Building triton_bench..."
go build -o triton_bench -ldflags="-s -w" triton_bench.go

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo ""
    echo "Binary location: $(pwd)/triton_bench"
    echo "Size: $(du -h triton_bench | cut -f1)"
    echo ""
    echo "Quick start:"
    echo "  ./triton_bench --mode quick"
    echo ""
    echo "See all modes:"
    echo "  ./triton_bench --list-modes"
    echo ""
    echo "Full documentation:"
    echo "  cat README.md"
    echo ""
else
    echo "✗ Build failed"
    exit 1
fi
