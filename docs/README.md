# Documentation Index

Welcome to the Triton YOLO Inference Server documentation.

---

## Quick Links

- **[Main README](../README.md)** - Project overview, quick start, benchmarking (START HERE)
- **[Benchmarks Guide](../benchmarks/README.md)** - How to use the triton_bench tool
- **[Attribution](../ATTRIBUTION.md)** - Third-party code attribution and licensing

---

## Documentation Structure

### Getting Started
- **[MODEL_EXPORT_GUIDE.md](MODEL_EXPORT_GUIDE.md)** - Model building and export for all 5 tracks
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Step-by-step deployment instructions
- **[TESTING.md](TESTING.md)** - Testing strategy and validation procedures

### Attribution & Analysis
- **[Attribution/END2END_ANALYSIS.md](Attribution/END2END_ANALYSIS.md)** - End2end ONNX export analysis and fork details
- **[Attribution/FORK_COMPARISON.md](Attribution/FORK_COMPARISON.md)** - Comparison of levipereira fork vs official ultralytics

### Technical Reference
- **[Technical/TRITON_BEST_PRACTICES.md](Technical/TRITON_BEST_PRACTICES.md)** - Triton configuration and optimization
- **[Technical/MODEL_FOLDER_STRUCTURE.md](Technical/MODEL_FOLDER_STRUCTURE.md)** - Model repository structure and config.pbtxt patterns
- **[Technical/DALI_LETTERBOX_IMPLEMENTATION.md](Technical/DALI_LETTERBOX_IMPLEMENTATION.md)** - DALI GPU preprocessing implementation
- **[Technical/STREAMING_OPTIMIZATION.md](Technical/STREAMING_OPTIMIZATION.md)** - Streaming inference and async processing

### Performance Tracks
- **[Tracks/BENCHMARKING_GUIDE.md](Tracks/BENCHMARKING_GUIDE.md)** - Comprehensive benchmarking methodology and results
- **[Tracks/TRACK_D_COMPLETE.md](Tracks/TRACK_D_COMPLETE.md)** - Complete Track D guide (DALI + TRT + GPU NMS)

### Track E Documentation
- **[TRACK_E_GUIDE.md](TRACK_E_GUIDE.md)** - Track E (Visual Search) setup and usage guide
- **[TRACK_E_IMPLEMENTATION_STATUS.md](TRACK_E_IMPLEMENTATION_STATUS.md)** - Track E implementation status
- **[TRACK_E_SUMMARY.md](TRACK_E_SUMMARY.md)** - Track E architecture summary
- **[TRACK_E_DEPLOYMENT_CHECKLIST.md](TRACK_E_DEPLOYMENT_CHECKLIST.md)** - Track E deployment checklist

### Implementation Notes
- **[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)** - Design decisions and technical notes

---

## Five Performance Tracks

| Track | Technology | Speedup | Documentation |
|-------|-----------|---------|---------------|
| **A** | PyTorch + CPU NMS | 1x (baseline) | See [main README](../README.md) |
| **B** | TensorRT + CPU NMS | 2x | See [main README](../README.md) |
| **C** | TensorRT + GPU NMS | 4x | See [Attribution/END2END_ANALYSIS.md](Attribution/END2END_ANALYSIS.md) |
| **D** | DALI + TRT + GPU NMS | **10-15x** | See [Tracks/TRACK_D_COMPLETE.md](Tracks/TRACK_D_COMPLETE.md) |
| **E** | MobileCLIP + OpenSearch | Visual Search | See [TRACK_E_GUIDE.md](TRACK_E_GUIDE.md) |

---

## Project Architecture

```
triton-api/
├── README.md                       # Main project documentation
├── ATTRIBUTION.md                  # Third-party code attribution
├── CLAUDE.md                       # Project instructions for Claude Code
│
├── benchmarks/
│   ├── README.md                   # Benchmark tool documentation
│   ├── triton_bench.go             # Master benchmark tool
│   └── results/                    # Auto-generated results
│
├── docs/                           # THIS DIRECTORY
│   ├── README.md                   # This file (documentation index)
│   ├── MODEL_EXPORT_GUIDE.md       # Model building and export (all tracks)
│   ├── DEPLOYMENT_GUIDE.md         # Deployment instructions
│   ├── TESTING.md                  # Testing procedures
│   ├── IMPLEMENTATION_NOTES.md     # Design decisions
│   │
│   ├── Attribution/                # Fork attribution & analysis
│   │   ├── END2END_ANALYSIS.md
│   │   └── FORK_COMPARISON.md
│   │
│   ├── Technical/                  # Technical reference docs
│   │   ├── TRITON_BEST_PRACTICES.md
│   │   ├── MODEL_FOLDER_STRUCTURE.md
│   │   ├── DALI_LETTERBOX_IMPLEMENTATION.md
│   │   └── STREAMING_OPTIMIZATION.md
│   │
│   ├── Tracks/                     # Performance track guides
│   │   ├── BENCHMARKING_GUIDE.md
│   │   └── TRACK_D_COMPLETE.md
│   │
│   └── future_work/                # Track E planning
│       ├── TRACK_E_PROJECT_PLAN.md
│       ├── TRACK_E_IMPLEMENTATION_PHASES.md
│       └── TRACK_E_FINAL_PHASES.md
│
├── models/                         # Triton model repository
│   ├── yolov11_small_trt/          # Track B
│   ├── yolov11_small_trt_end2end/  # Track C
│   ├── yolo_preprocess_dali/       # Track D preprocessing
│   └── yolov11_small_gpu_e2e*/     # Track D variants
│
├── monitoring/                     # Prometheus & Grafana configs
└── src/                            # FastAPI services
```

---

## Common Tasks

### Export Models
```bash
# See MODEL_EXPORT_GUIDE.md for complete instructions
cat docs/MODEL_EXPORT_GUIDE.md
```

### Deploy the System
```bash
# See main README.md Quick Start section
docker compose up -d
```

### Run Benchmarks
```bash
# See benchmarks/README.md
cd benchmarks
./build.sh
./triton_bench --mode quick
```

### Understand Track C (End2End)
```bash
# Read the fork attribution analysis
cat docs/Attribution/END2END_ANALYSIS.md
```

### Optimize Track D
```bash
# Read the complete Track D guide
cat docs/Tracks/TRACK_D_COMPLETE.md
```

### Learn Triton Best Practices
```bash
# Read Triton optimization guide
cat docs/Technical/TRITON_BEST_PRACTICES.md
```

---

## External Resources

- **NVIDIA Triton Docs**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Ultralytics Docs**: https://docs.ultralytics.com/
- **NVIDIA DALI Docs**: https://docs.nvidia.com/deeplearning/dali/
- **levipereira/ultralytics Fork**: https://github.com/levipereira/ultralytics
- **TensorRT Plugin Docs**: https://docs.nvidia.com/deeplearning/tensorrt/

---

## Contributing

When adding new documentation:
1. Place in appropriate subfolder (Attribution/, Technical/, Tracks/, future_work/)
2. Update this index with link and description
3. Use relative links for cross-references
4. Keep main [README.md](../README.md) concise - detailed docs go here

---

**Last Updated:** December 2025
