# ==================================================================================
# Makefile for Triton Inference Server YOLO Deployment
# ==================================================================================
# This Makefile provides convenient shortcuts for common development tasks
# across five performance tracks (A/B/C/D/E) with unified service management.
# ==================================================================================

# Default shell
SHELL := /bin/bash

# Variables
COMPOSE := docker compose
API_SERVICE := yolo-api
TRITON_SERVICE := triton-api
OPENSEARCH_SERVICE := opensearch
BENCHMARK_DIR := benchmarks
SCRIPTS_DIR := scripts
TRACK_E_DIR := scripts/track_e

# Port configurations
API_PORT := 4603
TRITON_HTTP_PORT := 4600
TRITON_GRPC_PORT := 4601
TRITON_METRICS_PORT := 4602
PROMETHEUS_PORT := 4604
GRAFANA_PORT := 4605
LOKI_PORT := 4606
OPENSEARCH_PORT := 4607
OPENSEARCH_DASH_PORT := 4608

# Default target
.DEFAULT_GOAL := help

# ==================================================================================
# Help
# ==================================================================================

.PHONY: help
help: ## Show this help message
	@echo "==================================================================================="
	@echo "Triton Inference Server - YOLO Deployment Makefile"
	@echo "==================================================================================="
	@echo ""
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick Start:"
	@echo "  make up          # Start all services"
	@echo "  make status      # Check service status"
	@echo "  make bench-quick # Run quick benchmark"
	@echo "  make logs        # View all logs"
	@echo ""

# ==================================================================================
# Service Management
# ==================================================================================

.PHONY: up
up: ## Start all services (Triton + API + Monitoring + OpenSearch)
	@echo "Starting all services..."
	$(COMPOSE) up -d
	@echo ""
	@echo "Services starting. Check status with: make status"
	@echo "API available at: http://localhost:$(API_PORT)"
	@echo "Grafana dashboard: http://localhost:$(GRAFANA_PORT) (admin/admin)"

.PHONY: down
down: ## Stop all services
	@echo "Stopping all services..."
	$(COMPOSE) down

.PHONY: restart
restart: ## Restart all services
	@echo "Restarting all services..."
	$(COMPOSE) restart
	@echo "Services restarted. Check status with: make status"

.PHONY: restart-triton
restart-triton: ## Restart only Triton server (after model changes)
	@echo "Restarting Triton server..."
	$(COMPOSE) restart $(TRITON_SERVICE)
	@sleep 5
	@echo "Triton restarted. Checking model status..."
	@$(MAKE) status

.PHONY: restart-api
restart-api: ## Restart only API service
	@echo "Restarting API service..."
	$(COMPOSE) restart $(API_SERVICE)

.PHONY: build
build: ## Build all containers
	@echo "Building containers..."
	$(COMPOSE) build

.PHONY: rebuild
rebuild: ## Rebuild containers without cache
	@echo "Rebuilding containers (no cache)..."
	$(COMPOSE) build --no-cache

# ==================================================================================
# Logs and Monitoring
# ==================================================================================

.PHONY: logs
logs: ## Follow logs from all services
	$(COMPOSE) logs -f

.PHONY: logs-triton
logs-triton: ## Follow Triton server logs
	$(COMPOSE) logs -f $(TRITON_SERVICE)

.PHONY: logs-api
logs-api: ## Follow API service logs
	$(COMPOSE) logs -f $(API_SERVICE)

.PHONY: logs-opensearch
logs-opensearch: ## Follow OpenSearch logs (Track E)
	$(COMPOSE) logs -f $(OPENSEARCH_SERVICE)

.PHONY: status
status: ## Check health of all services
	@bash $(SCRIPTS_DIR)/check_services.sh

.PHONY: health
health: status ## Alias for status

.PHONY: ps
ps: ## Show running containers
	$(COMPOSE) ps

# ==================================================================================
# Consolidated Deployment
# ==================================================================================
#
# All tracks (A, B, C, D, E) run on a single unified deployment:
# - GPU 0: Track A/B/C/E models
# - GPU 1 (host GPU 2): Track D DALI pipeline models
# - Single API on port 4603, single Triton on ports 4600-4602
#
# ==================================================================================

# ==================================================================================
# Benchmarking
# ==================================================================================

.PHONY: bench-build
bench-build: ## Build benchmark tool
	@echo "Building benchmark tool..."
	cd $(BENCHMARK_DIR) && go build -o triton_bench triton_bench.go
	@echo "✓ Benchmark tool built successfully at $(BENCHMARK_DIR)/triton_bench"

.PHONY: bench-quick
bench-quick: ## Quick benchmark (30 seconds, 16 clients)
	@echo "Running quick benchmark (30 seconds, 16 clients)..."
	cd $(BENCHMARK_DIR) && ./triton_bench --mode quick

.PHONY: bench-full
bench-full: ## Full benchmark (60 seconds, 128 clients)
	@echo "Running full benchmark (60 seconds, 128 clients)..."
	cd $(BENCHMARK_DIR) && ./triton_bench --mode full --clients 128 --duration 60

.PHONY: bench-matrix
bench-matrix: ## Matrix test with multiple concurrency levels (32,64,128,256,512,1024)
	@echo "Running matrix benchmark across multiple concurrency levels..."
	cd $(BENCHMARK_DIR) && ./triton_bench --mode matrix --matrix-clients 32,64,128,256,512,1024 --duration 30

.PHONY: bench-track-a
bench-track-a: ## Benchmark Track A (PyTorch baseline)
	@echo "Benchmarking Track A (PyTorch)..."
	cd $(BENCHMARK_DIR) && ./triton_bench --mode full --track A --clients 64 --duration 60

.PHONY: bench-track-b
bench-track-b: ## Benchmark Track B (TensorRT + CPU NMS)
	@echo "Benchmarking Track B (TensorRT + CPU NMS)..."
	cd $(BENCHMARK_DIR) && ./triton_bench --mode full --track B --clients 128 --duration 60

.PHONY: bench-track-c
bench-track-c: ## Benchmark Track C (TensorRT + GPU NMS)
	@echo "Benchmarking Track C (TensorRT + GPU NMS)..."
	cd $(BENCHMARK_DIR) && ./triton_bench --mode full --track C --clients 128 --duration 60

.PHONY: bench-track-d
bench-track-d: ## Benchmark Track D batch variant (max throughput)
	@echo "Benchmarking Track D batch (max throughput)..."
	cd $(BENCHMARK_DIR) && ./triton_bench --mode full --track D_batch --clients 256 --duration 60

.PHONY: bench-track-d-streaming
bench-track-d-streaming: ## Benchmark Track D streaming variant (low latency)
	@echo "Benchmarking Track D streaming (low latency)..."
	cd $(BENCHMARK_DIR) && ./triton_bench --mode full --track D_streaming --clients 128 --duration 60

.PHONY: bench-track-d-balanced
bench-track-d-balanced: ## Benchmark Track D balanced variant
	@echo "Benchmarking Track D balanced..."
	cd $(BENCHMARK_DIR) && ./triton_bench --mode full --track D_balanced --clients 128 --duration 60

.PHONY: bench-track-d-batch
bench-track-d-batch: bench-track-d ## Alias for bench-track-d

.PHONY: bench-track-e
bench-track-e: ## Benchmark Track E (Visual Search - YOLO+CLIP)
	@echo "Benchmarking Track E (Visual Search)..."
	cd $(BENCHMARK_DIR) && ./triton_bench --mode full --track E --clients 128 --duration 60

.PHONY: bench-track-e-full
bench-track-e-full: ## Benchmark Track E full variant (with per-box embeddings)
	@echo "Benchmarking Track E full (per-box embeddings)..."
	cd $(BENCHMARK_DIR) && ./triton_bench --mode full --track E_full --clients 128 --duration 60

.PHONY: bench-stress
bench-stress: ## Stress test with 512 clients (high load)
	@echo "Running stress test (512 concurrent clients)..."
	cd $(BENCHMARK_DIR) && ./triton_bench --mode full --clients 512 --duration 120

.PHONY: bench-results
bench-results: ## Show recent benchmark results
	@echo "Recent benchmark results:"
	@ls -lt $(BENCHMARK_DIR)/results/ | head -n 10

.PHONY: lint-go
lint-go: ## Run golangci-lint on benchmark code
	@echo "Running golangci-lint on benchmark code..."
	cd $(BENCHMARK_DIR) && $(HOME)/go/bin/golangci-lint run triton_bench.go
	@echo "✓ Linting complete"

.PHONY: fmt-go
fmt-go: ## Format Go code with gofmt
	@echo "Formatting Go code..."
	cd $(BENCHMARK_DIR) && gofmt -w -s triton_bench.go
	@echo "✓ Code formatted"

# ==================================================================================
# Development and Testing
# ==================================================================================

.PHONY: shell-api
shell-api: ## Open shell in API container
	$(COMPOSE) exec $(API_SERVICE) /bin/bash

.PHONY: shell-triton
shell-triton: ## Open shell in Triton container
	$(COMPOSE) exec $(TRITON_SERVICE) /bin/bash

.PHONY: shell-opensearch
shell-opensearch: ## Open shell in OpenSearch container
	$(COMPOSE) exec $(OPENSEARCH_SERVICE) /bin/bash

.PHONY: test-inference
test-inference: ## Test inference on all tracks (shell script)
	@echo "Testing inference on all tracks..."
	@bash tests/test_inference.sh

.PHONY: test-track-e-suite
test-track-e-suite: ## Run full Track E test suite with multiple images (inside container)
	@echo "Running full Track E test suite..."
	$(COMPOSE) exec $(API_SERVICE) python /app/$(TRACK_E_DIR)/test_track_e_images.py

.PHONY: test-integration
test-integration: ## Run Track E integration tests
	@echo "Running Track E integration tests..."
	$(COMPOSE) exec $(API_SERVICE) python /app/$(TRACK_E_DIR)/test_integration.py

.PHONY: test-patch
test-patch: ## Verify End2End TRT NMS patch is applied
	@echo "Verifying End2End TensorRT NMS patch..."
	$(COMPOSE) exec $(API_SERVICE) python /app/tests/test_end2end_patch.py

.PHONY: test-onnx
test-onnx: ## Test ONNX End2End model locally (bypasses Triton)
	@echo "Testing ONNX End2End model locally..."
	$(COMPOSE) exec $(API_SERVICE) python /app/tests/test_onnx_end2end.py

.PHONY: test-validate-models
test-validate-models: compare-tracks ## Compare detections across tracks (alias for compare-tracks)

.PHONY: test-shared-client
test-shared-client: ## Test shared vs per-request client performance
	@echo "Testing shared vs per-request client..."
	@bash tests/test_shared_vs_per_request.sh

.PHONY: test-compare-padding
test-compare-padding: ## Compare DALI padding methods (center vs simple)
	@echo "Comparing DALI padding methods..."
	$(COMPOSE) exec $(API_SERVICE) python /app/tests/compare_padding_methods.py

.PHONY: profile-api
profile-api: ## Profile API with py-spy (DURATION=30, OUTPUT=profile.svg)
	@echo "======================================"
	@echo "FastAPI Performance Profiling"
	@echo "======================================"
	@DURATION=$${DURATION:-30}; OUTPUT=$${OUTPUT:-profile.svg}; \
	echo "Duration: $${DURATION} seconds"; \
	echo "Output: $${OUTPUT}"; \
	CONTAINER_PID=$$($(COMPOSE) exec $(API_SERVICE) pgrep -f "uvicorn src.main:app" | head -1 | tr -d '[:space:]'); \
	if [ -z "$$CONTAINER_PID" ]; then \
		echo "ERROR: Could not find uvicorn process"; \
		exit 1; \
	fi; \
	echo "Found process: PID $$CONTAINER_PID"; \
	FORMAT="flamegraph"; \
	case "$$OUTPUT" in *.speedscope) FORMAT="speedscope";; esac; \
	echo "Generating $$FORMAT visualization..."; \
	$(COMPOSE) exec $(API_SERVICE) py-spy record --pid $$CONTAINER_PID --duration $$DURATION --rate 100 --format $$FORMAT --output /tmp/$$OUTPUT --subprocesses; \
	docker cp $$(docker compose ps -q $(API_SERVICE)):/tmp/$$OUTPUT ./$$OUTPUT; \
	echo "Profile saved to: $$OUTPUT"

.PHONY: resize-images
resize-images: ## Resize images for testing (SOURCE_DIR, OUTPUT_DIR, SIZE)
	@echo "Resizing images..."
	@. .venv/bin/activate && python scripts/resize_images.py \
		--source $${SOURCE_DIR:-test_images} \
		--output $${OUTPUT_DIR:-test_images_resized} \
		--size $${SIZE:-640}

.PHONY: test-create-images
test-create-images: ## Generate test images in various sizes (SOURCE required)
	@echo "Creating test images..."
	@if [ -z "$(SOURCE)" ]; then \
		echo "Error: SOURCE parameter required"; \
		echo "Example: make test-create-images SOURCE=/path/to/image.jpg"; \
		exit 1; \
	fi
	python tests/create_test_images.py --source "$(SOURCE)"

.PHONY: test-all
test-all: ## Run all tests (comprehensive)
	@echo "==================================================================================="
	@echo "Running All Tests"
	@echo "==================================================================================="
	@echo ""
	@echo "--- 1. Verify patch is applied ---"
	@$(MAKE) test-patch || echo "WARNING: Patch test failed"
	@echo ""
	@echo "--- 2. Test all tracks via API ---"
	@$(MAKE) test-inference
	@echo ""
	@echo "--- 3. Test Track E Full pipeline ---"
	@$(MAKE) test-track-e-full || echo "WARNING: Track E Full test failed"
	@echo ""
	@echo "--- 4. Validate models ---"
	@$(MAKE) test-validate-models || echo "WARNING: Model validation had issues"
	@echo ""
	@echo "==================================================================================="
	@echo "All tests completed!"
	@echo "==================================================================================="

# ==================================================================================
# Model Management API (Dynamic Upload & Export)
# ==================================================================================
#
# Upload .pt files via API, auto-export to TRT, and load into Triton dynamically.
# No restart required! Uses background tasks for long-running exports.
#
# Endpoints:
#   POST /models/upload       Upload .pt file and start export
#   GET  /models/export/{id}  Check export status
#   GET  /models              List all models in repository
#   POST /models/{name}/load  Load model into Triton
#   POST /models/{name}/unload Unload model from Triton
#   DELETE /models/{name}     Delete model from repository
#
# ==================================================================================

.PHONY: api-upload-model
api-upload-model: ## Upload a model via API (usage: make api-upload-model MODEL=/path/to/model.pt [NAME=custom_name])
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL parameter required"; \
		echo "Usage: make api-upload-model MODEL=/path/to/model.pt [NAME=custom_name]"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make api-upload-model MODEL=./my_model.pt"; \
		echo "  make api-upload-model MODEL=./my_model.pt NAME=vehicle_detector"; \
		exit 1; \
	fi
	@NAME_ARG=""; \
	if [ -n "$(NAME)" ]; then NAME_ARG="-F triton_name=$(NAME)"; fi; \
	echo "Uploading model $(MODEL) via API..."; \
	curl -s -X POST http://localhost:$(API_PORT)/models/upload \
		-F "file=@$(MODEL)" \
		$$NAME_ARG | jq '.'

.PHONY: api-export-status
api-export-status: ## Check export task status (usage: make api-export-status ID=task_id)
	@if [ -z "$(ID)" ]; then \
		echo "Error: ID parameter required"; \
		echo "Usage: make api-export-status ID=task_id"; \
		exit 1; \
	fi
	@curl -s http://localhost:$(API_PORT)/models/export/$(ID) | jq '.'

.PHONY: api-exports
api-exports: ## List all export tasks
	@echo "Export tasks:"
	@curl -s http://localhost:$(API_PORT)/models/exports | jq '.'

.PHONY: api-models
api-models: ## List all models in Triton repository
	@echo "Models in Triton repository:"
	@curl -s http://localhost:$(API_PORT)/models/ | jq '.'

.PHONY: api-load-model
api-load-model: ## Load a model into Triton (usage: make api-load-model NAME=model_name)
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter required"; \
		echo "Usage: make api-load-model NAME=model_name"; \
		exit 1; \
	fi
	@echo "Loading model $(NAME) into Triton..."
	@curl -s -X POST http://localhost:$(API_PORT)/models/$(NAME)/load | jq '.'

.PHONY: api-unload-model
api-unload-model: ## Unload a model from Triton (usage: make api-unload-model NAME=model_name)
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter required"; \
		echo "Usage: make api-unload-model NAME=model_name"; \
		exit 1; \
	fi
	@echo "Unloading model $(NAME) from Triton..."
	@curl -s -X POST http://localhost:$(API_PORT)/models/$(NAME)/unload | jq '.'

.PHONY: api-delete-model
api-delete-model: ## Delete a model from repository (usage: make api-delete-model NAME=model_name)
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter required"; \
		echo "Usage: make api-delete-model NAME=model_name"; \
		exit 1; \
	fi
	@echo "Deleting model $(NAME)..."
	@curl -s -X DELETE http://localhost:$(API_PORT)/models/$(NAME) | jq '.'

.PHONY: api-health
api-health: ## Check if API is healthy and ready
	@echo "Checking API health..."
	@curl -sf http://localhost:$(API_PORT)/health > /dev/null && echo "API is healthy" || (echo "API not ready"; exit 1)

.PHONY: api-wait-ready
api-wait-ready: ## Wait for API to be ready (up to 60 seconds)
	@echo "Waiting for API to be ready..."
	@for i in $$(seq 1 12); do \
		if curl -sf http://localhost:$(API_PORT)/health > /dev/null 2>&1; then \
			echo "API is ready!"; \
			exit 0; \
		fi; \
		echo "  Attempt $$i/12 - waiting 5 seconds..."; \
		sleep 5; \
	done; \
	echo "ERROR: API not ready after 60 seconds"; \
	exit 1

.PHONY: api-test-e2e
api-test-e2e: ## Run full E2E test of Model Management API with YOLO11 nano
	@echo "==================================================================================="
	@echo "Model Management API - End-to-End Test (YOLO11 Nano)"
	@echo "==================================================================================="
	@echo ""
	@echo "--- Step 1: Restart API to load new code ---"
	@$(MAKE) restart-api
	@sleep 3
	@$(MAKE) api-wait-ready
	@echo ""
	@echo "--- Step 2: Download YOLO11 nano model if needed ---"
	@$(MAKE) api-download-nano
	@echo ""
	@echo "--- Step 3: List current models ---"
	@$(MAKE) api-models
	@echo ""
	@echo "--- Step 4: Upload nano model for export ---"
	@$(MAKE) api-upload-nano
	@echo ""
	@echo "--- Step 5: Monitor export (this may take 3-5 minutes) ---"
	@$(MAKE) api-wait-export
	@echo ""
	@echo "--- Step 6: Verify model in repository ---"
	@$(MAKE) api-models
	@echo ""
	@echo "--- Step 7: Test unload/load cycle ---"
	@$(MAKE) api-unload-model NAME=test_nano_trt_end2end || true
	@sleep 2
	@$(MAKE) api-load-model NAME=test_nano_trt_end2end
	@echo ""
	@echo "--- Step 8: Run inference test on uploaded model ---"
	@$(MAKE) api-test-inference MODEL=test_nano_trt_end2end
	@echo ""
	@echo "--- Step 9: Cleanup - delete test model ---"
	@$(MAKE) api-delete-model NAME=test_nano_trt_end2end || true
	@$(MAKE) api-delete-model NAME=test_nano_end2end || true
	@echo ""
	@echo "==================================================================================="
	@echo "E2E Test Complete!"
	@echo "==================================================================================="

.PHONY: api-test-inference
api-test-inference: ## Run inference test on uploaded model (usage: make api-test-inference MODEL=model_name)
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL parameter required"; \
		echo "Usage: make api-test-inference MODEL=model_name"; \
		exit 1; \
	fi
	@echo "Running inference test on $(MODEL)..."
	@echo ""
	@echo "Test 1: Single image inference"
	@RESPONSE=$$(curl -s -X POST http://localhost:$(API_PORT)/predict/$(MODEL) \
		-F "image=@test_images/bus.jpg"); \
	echo "$$RESPONSE" | jq '.'; \
	DETECTIONS=$$(echo "$$RESPONSE" | jq '.detections | length'); \
	if [ "$$DETECTIONS" = "null" ] || [ "$$DETECTIONS" = "0" ]; then \
		echo ""; \
		echo "ERROR: No detections returned!"; \
		echo "Response: $$RESPONSE"; \
		exit 1; \
	fi; \
	echo ""; \
	echo "SUCCESS: Got $$DETECTIONS detections"; \
	echo ""
	@echo "Test 2: Verify detection format (normalized boxes)"
	@RESPONSE=$$(curl -s -X POST http://localhost:$(API_PORT)/predict/$(MODEL) \
		-F "image=@test_images/bus.jpg"); \
	X1=$$(echo "$$RESPONSE" | jq '.detections[0].x1 // 0'); \
	Y1=$$(echo "$$RESPONSE" | jq '.detections[0].y1 // 0'); \
	X2=$$(echo "$$RESPONSE" | jq '.detections[0].x2 // 0'); \
	Y2=$$(echo "$$RESPONSE" | jq '.detections[0].y2 // 0'); \
	echo "First detection: x1=$$X1, y1=$$Y1, x2=$$X2, y2=$$Y2"; \
	if echo "$$X1 $$Y1 $$X2 $$Y2" | awk '{exit !($$1 >= 0 && $$1 <= 1 && $$2 >= 0 && $$2 <= 1 && $$3 >= 0 && $$3 <= 1 && $$4 >= 0 && $$4 <= 1)}'; then \
		echo "SUCCESS: Boxes are normalized [0,1]"; \
	else \
		echo "WARNING: Boxes may not be normalized (expected [0,1] range)"; \
	fi
	@echo ""
	@echo "Inference test PASSED for $(MODEL)!"

.PHONY: api-download-nano
api-download-nano: ## Download YOLO11 nano model for testing
	@if [ -f "pytorch_models/yolo11n.pt" ]; then \
		echo "YOLO11 nano model already exists"; \
	else \
		echo "Downloading YOLO11 nano model..."; \
		$(COMPOSE) exec $(API_SERVICE) python -c "from ultralytics import YOLO; m = YOLO('yolo11n.pt'); import shutil; shutil.copy(str(m.ckpt_path), '/app/pytorch_models/yolo11n.pt')"; \
		echo "Downloaded to pytorch_models/yolo11n.pt"; \
	fi

.PHONY: api-upload-nano
api-upload-nano: ## Upload YOLO11 nano model via API (for testing)
	@echo "Uploading YOLO11 nano model..."
	@RESPONSE=$$(curl -s -X POST http://localhost:$(API_PORT)/models/upload \
		-F "file=@pytorch_models/yolo11n.pt" \
		-F "triton_name=test_nano" \
		-F "max_batch=16" \
		-F "formats=trt_end2end"); \
	echo "$$RESPONSE" | jq '.'; \
	TASK_ID=$$(echo "$$RESPONSE" | jq -r '.task_id'); \
	echo "$$TASK_ID" > /tmp/export_task_id.txt; \
	echo "Task ID saved: $$TASK_ID"

.PHONY: api-wait-export
api-wait-export: ## Wait for current export task to complete (reads task ID from /tmp/export_task_id.txt)
	@if [ ! -f /tmp/export_task_id.txt ]; then \
		echo "Error: No task ID found. Run api-upload-nano first."; \
		exit 1; \
	fi; \
	TASK_ID=$$(cat /tmp/export_task_id.txt); \
	echo "Monitoring export task: $$TASK_ID"; \
	echo "This typically takes 3-5 minutes for nano model..."; \
	for i in $$(seq 1 60); do \
		RESPONSE=$$(curl -s http://localhost:$(API_PORT)/models/export/$$TASK_ID); \
		STATUS=$$(echo "$$RESPONSE" | jq -r '.status'); \
		PROGRESS=$$(echo "$$RESPONSE" | jq -r '.progress'); \
		STEP=$$(echo "$$RESPONSE" | jq -r '.current_step'); \
		echo "  [$$i/60] Status: $$STATUS | Progress: $$PROGRESS% | Step: $$STEP"; \
		if [ "$$STATUS" = "completed" ]; then \
			echo ""; \
			echo "Export completed successfully!"; \
			echo "$$RESPONSE" | jq '.'; \
			rm -f /tmp/export_task_id.txt; \
			exit 0; \
		elif [ "$$STATUS" = "failed" ]; then \
			echo ""; \
			echo "Export FAILED!"; \
			echo "$$RESPONSE" | jq '.'; \
			rm -f /tmp/export_task_id.txt; \
			exit 1; \
		fi; \
		sleep 10; \
	done; \
	echo "ERROR: Export timed out after 10 minutes"; \
	rm -f /tmp/export_task_id.txt; \
	exit 1

.PHONY: api-test-quick
api-test-quick: ## Quick API test (no export, just endpoint verification)
	@echo "==================================================================================="
	@echo "Model Management API - Quick Endpoint Test"
	@echo "==================================================================================="
	@echo ""
	@echo "--- Testing GET /models/ ---"
	@curl -s http://localhost:$(API_PORT)/models/ | jq '.triton_status, .total'
	@echo ""
	@echo "--- Testing GET /models/exports ---"
	@curl -s http://localhost:$(API_PORT)/models/exports | jq 'length'
	@echo ""
	@echo "--- Testing API Health ---"
	@curl -sf http://localhost:$(API_PORT)/health | jq '.status'
	@echo ""
	@echo "All quick tests passed!"

# ==================================================================================
# Model Export (CLI-based)
# ==================================================================================

.PHONY: export-models
export-models: ## Export YOLO models (TRT + End2End with normalized boxes)
	@echo "Exporting YOLO models to TensorRT formats (normalized boxes)..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py --models small --formats trt trt_end2end --normalize-boxes --save-labels --generate-config

.PHONY: export-all
export-all: ## Export all models (nano through xlarge) in all formats
	@echo "Exporting all YOLO models in all formats (normalized boxes)..."
	@echo "WARNING: This will take 60-120 minutes depending on GPU"
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py --models nano small medium large xlarge --formats all --normalize-boxes --save-labels --generate-config

.PHONY: export-small
export-small: ## Quick export for small model (TRT + End2End with normalized boxes)
	@echo "Exporting small model (TRT + End2End, normalized boxes)..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py --models small --formats trt trt_end2end --normalize-boxes --save-labels --generate-config

.PHONY: export-onnx
export-onnx: ## Export ONNX-only format (with normalized boxes)
	@echo "Exporting ONNX models only (normalized boxes)..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py --models small --formats onnx onnx_end2end --normalize-boxes --save-labels

.PHONY: export-custom
export-custom: ## Export custom model (usage: make export-custom MODEL=/path/to/model.pt [NAME=custom_name] [BATCH=32])
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL parameter required"; \
		echo "Usage: make export-custom MODEL=/path/to/model.pt [NAME=custom_name] [BATCH=32]"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make export-custom MODEL=/app/pytorch_models/my_model.pt"; \
		echo "  make export-custom MODEL=/app/pytorch_models/my_model.pt NAME=my_detector BATCH=64"; \
		exit 1; \
	fi
	@CUSTOM_ARG="$(MODEL)"; \
	if [ -n "$(NAME)" ]; then CUSTOM_ARG="$$CUSTOM_ARG:$(NAME)"; elif [ -n "$(BATCH)" ]; then CUSTOM_ARG="$$CUSTOM_ARG:"; fi; \
	if [ -n "$(BATCH)" ]; then CUSTOM_ARG="$$CUSTOM_ARG:$(BATCH)"; fi; \
	echo "Exporting custom model: $$CUSTOM_ARG"; \
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py \
		--custom-model "$$CUSTOM_ARG" \
		--formats trt trt_end2end \
		--normalize-boxes \
		--save-labels \
		--generate-config

.PHONY: export-config
export-config: ## Export models from YAML config file (usage: make export-config CONFIG=/path/to/config.yaml)
	@if [ -z "$(CONFIG)" ]; then \
		echo "Error: CONFIG parameter required"; \
		echo "Usage: make export-config CONFIG=/path/to/config.yaml"; \
		echo ""; \
		echo "Example YAML format:"; \
		echo "  models:"; \
		echo "    my_model:"; \
		echo "      pt_file: /app/pytorch_models/my_model.pt"; \
		echo "      triton_name: my_custom_detector  # optional"; \
		echo "      max_batch: 32                    # optional"; \
		exit 1; \
	fi
	@echo "Exporting models from config: $(CONFIG)"
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py \
		--config-file "$(CONFIG)" \
		--formats trt trt_end2end \
		--normalize-boxes \
		--save-labels \
		--generate-config

.PHONY: export-list
export-list: ## List available built-in models
	@$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py --list-models

.PHONY: export-end2end
export-end2end: ## Export ONNX + TRT end2end with normalized boxes (Track E)
	@echo "Exporting end2end models with normalized boxes for Track E..."
	@echo "This exports ONNX and TensorRT end2end formats with box normalization."
	@echo "Required for Track E visual search pipeline (YOLO + CLIP)."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py \
		--models small \
		--formats onnx_end2end trt_end2end \
		--normalize-boxes
	@echo ""
	@echo "Export complete! Next steps:"
	@echo "  1. make rebuild-dali          # Rebuild DALI pipeline"
	@echo "  2. make restart-triton         # Restart Triton to load new models"

.PHONY: export-end2end-standard
export-end2end-standard: ## Export ONNX + TRT end2end with normalized boxes (Track D)
	@echo "Exporting end2end models with normalized boxes for Track D..."
	@echo "This exports ONNX and TensorRT end2end formats with normalization."
	@echo "Required for Track D DALI pipeline."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py \
		--models small \
		--formats onnx_end2end trt_end2end \
		--normalize-boxes
	@echo ""
	@echo "Export complete! Next steps:"
	@echo "  1. make create-dali            # Create DALI pipeline"
	@echo "  2. make restart-triton         # Restart Triton"

.PHONY: download-pytorch
download-pytorch: ## Download PyTorch models (usage: make download-pytorch MODELS="small medium")
	@echo "Downloading PyTorch models using Ultralytics..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/download_pytorch_models.py --models $(or $(MODELS),small)

.PHONY: download-pytorch-all
download-pytorch-all: ## Download all PyTorch models (nano, small, medium)
	@echo "Downloading all PyTorch models..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/download_pytorch_models.py --models all

.PHONY: download-pytorch-list
download-pytorch-list: ## List available PyTorch models
	@$(COMPOSE) exec $(API_SERVICE) python /app/export/download_pytorch_models.py --list

.PHONY: export-status
export-status: ## Show status of all exported models
	@echo "==================================================================================="
	@echo "Model Export Status"
	@echo "==================================================================================="
	@echo ""
	@echo "PyTorch Models (pytorch_models/):"
	@ls -lh pytorch_models/*.pt 2>/dev/null || echo "  No PyTorch models found"
	@echo ""
	@echo "Triton Models (models/):"
	@for dir in models/yolov11*; do \
		if [ -d "$$dir" ]; then \
			name=$$(basename $$dir); \
			model=""; config=""; \
			[ -f "$$dir/1/model.onnx" ] && model="ONNX"; \
			[ -f "$$dir/1/model.plan" ] && model="TRT"; \
			[ -f "$$dir/config.pbtxt" ] && config="OK" || config="MISSING"; \
			printf "  %-35s model: %-5s config: %s\n" "$$name" "$${model:-NONE}" "$$config"; \
		fi \
	done
	@echo ""

.PHONY: validate-exports
validate-exports: ## Validate that Triton can load exported models
	@echo "Validating exported models with Triton..."
	@echo "Checking Triton model repository status..."
	@curl -s http://localhost:$(TRITON_HTTP_PORT)/v2/models | jq -r '.models[]? | "\(.name): \(.state)"' 2>/dev/null || echo "Triton not running. Start with: make up"
	@echo ""
	@echo "Model details:"
	@curl -s http://localhost:$(TRITON_HTTP_PORT)/v2/models/yolov11_small_trt/config 2>/dev/null | jq '.name, .max_batch_size' || echo "  yolov11_small_trt: Not loaded"
	@curl -s http://localhost:$(TRITON_HTTP_PORT)/v2/models/yolov11_small_trt_end2end/config 2>/dev/null | jq '.name, .max_batch_size' || echo "  yolov11_small_trt_end2end: Not loaded"

.PHONY: create-dali
create-dali: ## Create DALI preprocessing pipeline for Track D
	@echo "Creating DALI preprocessing pipeline (Track D)..."
	$(COMPOSE) exec $(API_SERVICE) python /app/dali/create_dali_letterbox_pipeline.py
	@echo "Creating ensemble models..."
	$(COMPOSE) exec $(API_SERVICE) python /app/dali/create_ensembles.py --models small
	@echo "Restarting Triton to load new models..."
	@$(MAKE) restart-triton

.PHONY: create-ensembles
create-ensembles: ## Create Track D ensemble models only (without rebuilding DALI)
	@echo "Creating Track D ensemble models..."
	$(COMPOSE) exec $(API_SERVICE) python /app/dali/create_ensembles.py --models small

.PHONY: rebuild-dali
rebuild-dali: ## Rebuild Track E DALI pipeline (triple-branch: YOLO + CLIP + HD)
	@echo "Rebuilding Track E DALI preprocessing pipeline..."
	@echo "This creates triple-branch pipeline: YOLO + MobileCLIP + HD cropping"
	$(COMPOSE) exec $(API_SERVICE) python /app/dali/create_dual_dali_pipeline.py
	@echo "DALI pipeline rebuilt. Restarting Triton..."
	@$(MAKE) restart-triton

.PHONY: create-dali-dual
create-dali-dual: ## Create Track E triple-branch DALI pipeline (YOLO + CLIP + HD)
	@echo "Creating Track E triple-branch DALI pipeline..."
	@echo "Outputs: yolo_images (640x640), clip_images (256x256), original_images (max 1920px)"
	$(COMPOSE) exec $(API_SERVICE) python /app/dali/create_dual_dali_pipeline.py
	@$(MAKE) restart-triton

.PHONY: create-dali-simple
create-dali-simple: ## Create Track E simple DALI pipeline (YOLO + CLIP only, faster)
	@echo "Creating Track E simple dual-branch DALI pipeline..."
	@echo "Outputs: yolo_images (640x640), clip_images (256x256) - faster, no HD cropping"
	$(COMPOSE) exec $(API_SERVICE) python /app/dali/create_yolo_clip_dali_pipeline.py
	@$(MAKE) restart-triton

.PHONY: validate-dali
validate-dali: ## Validate Track D DALI pipeline
	@echo "Validating Track D DALI pipeline..."
	$(COMPOSE) exec $(API_SERVICE) python /app/dali/validate_dali_letterbox.py

.PHONY: validate-dali-dual
validate-dali-dual: ## Validate Track E DALI pipeline against PyTorch reference
	@echo "Validating Track E dual DALI preprocessing..."
	$(COMPOSE) exec $(API_SERVICE) python /app/dali/validate_dual_dali_preprocessing.py

.PHONY: export-mobileclip
export-mobileclip: ## Export MobileCLIP models for Track E
	@echo "Exporting MobileCLIP image encoder..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_mobileclip_image_encoder.py
	@echo "Exporting MobileCLIP text encoder..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_mobileclip_text_encoder.py
	@$(MAKE) restart-triton

.PHONY: setup-track-e
setup-track-e: ## Complete Track E setup (models + DALI + configs)
	@echo "Setting up Track E (Visual Search)..."
	@bash $(TRACK_E_DIR)/setup_mobileclip_env.sh
	@$(MAKE) export-mobileclip
	@$(MAKE) create-dali-dual
	@echo "Track E setup complete!"

# ==================================================================================
# Monitoring and Metrics
# ==================================================================================

.PHONY: open-grafana
open-grafana: ## Open Grafana dashboard in browser
	@echo "Opening Grafana dashboard..."
	@echo "URL: http://localhost:$(GRAFANA_PORT)"
	@echo "Login: admin/admin"
	@xdg-open http://localhost:$(GRAFANA_PORT) 2>/dev/null || open http://localhost:$(GRAFANA_PORT) 2>/dev/null || echo "Please open http://localhost:$(GRAFANA_PORT) in your browser"

.PHONY: open-prometheus
open-prometheus: ## Open Prometheus UI in browser
	@echo "Opening Prometheus..."
	@xdg-open http://localhost:$(PROMETHEUS_PORT) 2>/dev/null || open http://localhost:$(PROMETHEUS_PORT) 2>/dev/null || echo "Please open http://localhost:$(PROMETHEUS_PORT) in your browser"

.PHONY: open-opensearch
open-opensearch: ## Open OpenSearch Dashboards in browser
	@echo "Opening OpenSearch Dashboards..."
	@xdg-open http://localhost:$(OPENSEARCH_DASH_PORT) 2>/dev/null || open http://localhost:$(OPENSEARCH_DASH_PORT) 2>/dev/null || echo "Please open http://localhost:$(OPENSEARCH_DASH_PORT) in your browser"

.PHONY: metrics
metrics: ## Show Triton metrics
	@curl -s http://localhost:$(TRITON_METRICS_PORT)/metrics | grep -E "nv_inference_|nv_gpu_" | head -n 20

.PHONY: gpu
gpu: ## Show GPU status
	@nvidia-smi

.PHONY: gpu-watch
gpu-watch: ## Watch GPU status (updates every second)
	@watch -n 1 nvidia-smi

# ==================================================================================
# Cleanup
# ==================================================================================

.PHONY: clean
clean: ## Stop services and remove containers
	@echo "Stopping and removing containers..."
	$(COMPOSE) down

.PHONY: clean-all
clean-all: ## Stop services, remove containers and volumes (WARNING: deletes all data)
	@echo "WARNING: This will remove all containers, volumes, and data!"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read
	$(COMPOSE) down -v

.PHONY: clean-logs
clean-logs: ## Clear Docker logs
	@echo "Clearing Docker logs..."
	$(COMPOSE) down
	@docker system prune -f

.PHONY: clean-bench
clean-bench: ## Remove benchmark results
	@echo "Removing benchmark results..."
	@rm -rf $(BENCHMARK_DIR)/results/*
	@echo "Benchmark results cleared."

.PHONY: clean-exports
clean-exports: ## Clean old model exports (keeps configs, prepares for re-export)
	@echo "==================================================================================="
	@echo "Cleaning old model exports..."
	@echo "==================================================================================="
	@echo ""
	@echo "Backing up config.pbtxt files..."
	@mkdir -p models/backup_configs_$$(date +%Y%m%d_%H%M%S)
	@for config in models/*/config.pbtxt; do \
		if [ -f "$$config" ]; then \
			model_name=$$(dirname "$$config" | xargs basename); \
			cp "$$config" "models/backup_configs_$$(date +%Y%m%d_%H%M%S)/$${model_name}_config.pbtxt" 2>/dev/null || true; \
		fi; \
	done
	@echo ""
	@echo "Removing old ONNX and TRT files..."
	@for dir in models/yolov11_*/1; do \
		if [ -d "$$dir" ]; then \
			rm -f "$$dir/model.onnx" "$$dir/model.onnx.old" "$$dir/model.plan" "$$dir/model.plan.old" 2>/dev/null || true; \
		fi; \
	done
	@echo ""
	@echo "Clearing TRT cache..."
	@rm -rf trt_cache/* 2>/dev/null || true
	@echo ""
	@echo "Done! Run 'make export-status' to see current state."
	@echo "Then run 'make export-models' or 'make export-all' to re-export."

# ==================================================================================
# OpenSearch / Track E Data Management
# ==================================================================================

.PHONY: opensearch-reset
opensearch-reset: ## Reset OpenSearch indices (WARNING: deletes all visual search data)
	@echo "WARNING: This will delete all OpenSearch indices and data!"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read
	@curl -X DELETE "http://localhost:$(OPENSEARCH_PORT)/_all" || echo "Failed to delete indices"
	@echo "OpenSearch indices cleared."

.PHONY: opensearch-status
opensearch-status: ## Show OpenSearch cluster status
	@echo "OpenSearch Cluster Status:"
	@curl -s http://localhost:$(OPENSEARCH_PORT)/_cluster/health?pretty

.PHONY: opensearch-indices
opensearch-indices: ## List OpenSearch indices
	@echo "OpenSearch Indices:"
	@curl -s http://localhost:$(OPENSEARCH_PORT)/_cat/indices?v

# ==================================================================================
# API Testing
# ==================================================================================
#
# All tracks run on the unified API at port 4603:
# - Track A: /pytorch/predict/small
# - Track B: /predict/small (TensorRT)
# - Track C: /predict/small_end2end (TensorRT + GPU NMS)
# - Track D: /predict/small_gpu_e2e_batch (DALI + GPU pipeline)
# - Track E: /track_e/detect (Visual Search)
#
# ==================================================================================

.PHONY: test-api-health
test-api-health: ## Test API health
	@echo "Testing API health (port $(API_PORT))..."
	@curl -sf http://localhost:$(API_PORT)/health && echo " OK" || echo " FAILED"

.PHONY: test-track-a
test-track-a: ## Quick test Track A (PyTorch)
	@echo "Testing Track A (PyTorch) on port $(API_PORT)..."
	@curl -s -X POST http://localhost:$(API_PORT)/pytorch/predict/small \
		-F "image=@test_images/bus.jpg" | jq '.' || echo "Test failed - is the API running? (make up)"

.PHONY: test-track-b
test-track-b: ## Quick test Track B (TensorRT)
	@echo "Testing Track B (TensorRT) on port $(API_PORT)..."
	@curl -s -X POST http://localhost:$(API_PORT)/predict/small \
		-F "image=@test_images/bus.jpg" | jq '.' || echo "Test failed - is the API running? (make up)"

.PHONY: test-track-c
test-track-c: ## Quick test Track C (TensorRT + GPU NMS)
	@echo "Testing Track C (TensorRT + GPU NMS) on port $(API_PORT)..."
	@curl -s -X POST http://localhost:$(API_PORT)/predict/small_end2end \
		-F "image=@test_images/bus.jpg" | jq '.' || echo "Test failed - is the API running? (make up)"

.PHONY: test-track-d
test-track-d: ## Quick test Track D (DALI + GPU pipeline)
	@echo "Testing Track D (DALI + GPU pipeline) on port $(API_PORT)..."
	@curl -s -X POST http://localhost:$(API_PORT)/predict/small_gpu_e2e_batch \
		-F "image=@test_images/bus.jpg" | jq '.' || echo "Test failed - is the API running? (make up)"

.PHONY: test-track-e
test-track-e: ## Quick test Track E (Visual Search - simple detection + embedding)
	@echo "Testing Track E (Visual Search - YOLO+CLIP ensemble) on port $(API_PORT)..."
	@curl -s -X POST http://localhost:$(API_PORT)/track_e/predict \
		-F "image=@test_images/bus.jpg" | jq '.' || echo "Test failed - is the API running? (make up)"

.PHONY: test-track-e-full
test-track-e-full: ## Quick test Track E Full (Visual Search - detection + per-box embeddings)
	@echo "Testing Track E Full (Visual Search - per-box embeddings) on port $(API_PORT)..."
	@curl -s -X POST http://localhost:$(API_PORT)/track_e/predict_full \
		-F "image=@test_images/bus.jpg" | jq '{status, track, num_detections, box_embeddings: (.box_embeddings | length), normalized_boxes: (.normalized_boxes | length), embedding_norm, total_time_ms, detections}' || echo "Test failed - is the API running? (make up)"

.PHONY: test-track-e-full-pipeline
test-track-e-full-pipeline: ## Test Track E full pipeline with timing (detections + box embeddings)
	@echo "==================================================================================="
	@echo "Track E Full Pipeline Test"
	@echo "==================================================================================="
	@echo ""
	@echo "--- Track E Simple (detection + global embedding) ---"
	@time curl -s -X POST http://localhost:$(API_PORT)/track_e/predict \
		-F "image=@test_images/bus.jpg" | jq '{status, track, num_detections, embedding_norm, total_time_ms}'
	@echo ""
	@echo "--- Track E Full (detection + global + per-box embeddings) ---"
	@time curl -s -X POST http://localhost:$(API_PORT)/track_e/predict_full \
		-F "image=@test_images/bus.jpg" | jq '{status, track, num_detections, box_embeddings: (.box_embeddings | length), normalized_boxes: (.normalized_boxes | length), embedding_norm, total_time_ms, detections}'
	@echo ""
	@echo "==================================================================================="
	@echo "Track E Full Pipeline Test Complete"
	@echo "==================================================================================="

.PHONY: test-all-tracks
test-all-tracks: ## Test all tracks (A, B, C, D, E, E_full)
	@echo "==================================================================================="
	@echo "Testing All Tracks on Unified API (port $(API_PORT))"
	@echo "==================================================================================="
	@echo ""
	@echo "--- Track A (PyTorch) ---"
	@$(MAKE) test-track-a
	@echo ""
	@echo "--- Track B (TensorRT + CPU NMS) ---"
	@$(MAKE) test-track-b
	@echo ""
	@echo "--- Track C (TensorRT + GPU NMS) ---"
	@$(MAKE) test-track-c
	@echo ""
	@echo "--- Track D (DALI + GPU pipeline) ---"
	@$(MAKE) test-track-d
	@echo ""
	@echo "--- Track E (Visual Search - detection + global embedding) ---"
	@$(MAKE) test-track-e
	@echo ""
	@echo "--- Track E Full (Visual Search - detection + global + per-box embeddings) ---"
	@$(MAKE) test-track-e-full

.PHONY: compare-tracks
compare-tracks: ## Compare detection outputs across tracks A, B, C, D, E
	@echo "==================================================================================="
	@echo "Comparing Detection Outputs Across All Tracks"
	@echo "==================================================================================="
	@. .venv/bin/activate && python tests/compare_tracks.py --host localhost --port $(API_PORT) --tracks A,B,C,D,E --images test_images

# ==================================================================================
# Documentation
# ==================================================================================

.PHONY: info
info: ## Show service URLs and ports
	@echo "==================================================================================="
	@echo "Triton Inference Server - Unified Deployment"
	@echo "==================================================================================="
	@echo ""
	@echo "All tracks run on a single unified API (port $(API_PORT))."
	@echo ""
	@echo "Quick Start:"
	@echo "  make up                    Start all services"
	@echo "  make status                Check service health"
	@echo "  make test-all-tracks       Test all tracks"
	@echo ""
	@echo "Services:"
	@echo "  YOLO API:                  http://localhost:$(API_PORT)"
	@echo "  Triton HTTP:               http://localhost:$(TRITON_HTTP_PORT)"
	@echo "  Triton gRPC:               http://localhost:$(TRITON_GRPC_PORT)"
	@echo "  OpenSearch:                http://localhost:$(OPENSEARCH_PORT)"
	@echo ""
	@echo "Monitoring:"
	@echo "  Grafana:                   http://localhost:$(GRAFANA_PORT) (admin/admin)"
	@echo "  Prometheus:                http://localhost:$(PROMETHEUS_PORT)"
	@echo ""
	@echo "Track Endpoints (all on port $(API_PORT)):"
	@echo "  Track A (PyTorch):         POST /pytorch/predict/small"
	@echo "  Track B (TensorRT):        POST /predict/small"
	@echo "  Track C (TRT + GPU NMS):   POST /predict/small_end2end"
	@echo "  Track D (DALI + GPU):      POST /predict/small_gpu_e2e_batch"
	@echo "  Track E (Visual Search):   POST /track_e/detect"
	@echo ""
	@echo "GPU Assignment:"
	@echo "  GPU 0: Tracks A, B, C, E models"
	@echo "  GPU 1 (host GPU 2): Track D DALI pipeline"
	@echo ""

.PHONY: docs
docs: info ## Alias for info

# ==================================================================================
# Reference Repositories (for attribution and development)
# ==================================================================================
#
# These repos are gitignored but provide important reference implementations.
# Clone them for development, attribution verification, or to understand the
# source of key features like the End2End NMS export patch.
#
# ==================================================================================

.PHONY: clone-refs-essential
clone-refs-essential: ## Clone essential reference repos (ultralytics-end2end, ml-mobileclip)
	@echo "Cloning essential reference repositories..."
	@bash $(SCRIPTS_DIR)/clone_reference_repos.sh --essential

.PHONY: clone-refs-recommended
clone-refs-recommended: ## Clone essential + recommended reference repos
	@echo "Cloning recommended reference repositories..."
	@bash $(SCRIPTS_DIR)/clone_reference_repos.sh --recommended

.PHONY: clone-refs-all
clone-refs-all: ## Clone all reference repositories
	@echo "Cloning all reference repositories..."
	@bash $(SCRIPTS_DIR)/clone_reference_repos.sh --all

.PHONY: clone-refs-list
clone-refs-list: ## List available reference repos and their status
	@bash $(SCRIPTS_DIR)/clone_reference_repos.sh --list

.PHONY: clone-ref
clone-ref: ## Clone a specific reference repo (usage: make clone-ref REPO=ultralytics-end2end)
	@if [ -z "$(REPO)" ]; then \
		echo "Error: REPO parameter required"; \
		echo "Usage: make clone-ref REPO=repo_name"; \
		echo ""; \
		echo "Available repos:"; \
		bash $(SCRIPTS_DIR)/clone_reference_repos.sh --list; \
		exit 1; \
	fi
	@bash $(SCRIPTS_DIR)/clone_reference_repos.sh --repo $(REPO)

# ==================================================================================
# Phony targets (targets that don't create files)
# ==================================================================================

.PHONY: help up down restart restart-triton restart-api build rebuild \
        logs logs-triton logs-api logs-opensearch status health ps \
        bench-build bench-quick bench-full bench-matrix \
        bench-track-a bench-track-b bench-track-c bench-track-d \
        bench-track-d-streaming bench-track-d-balanced bench-track-d-batch \
        bench-track-e bench-track-e-full bench-stress bench-results \
        lint-go fmt-go \
        shell-api shell-triton shell-opensearch \
        test-inference test-track-e-suite test-integration \
        test-patch test-onnx test-validate-models test-shared-client \
        test-compare-padding test-create-images test-all \
        profile-api resize-images \
        api-upload-model api-export-status api-exports api-models \
        api-load-model api-unload-model api-delete-model \
        api-health api-wait-ready api-test-e2e api-download-nano \
        api-upload-nano api-wait-export api-test-quick api-test-inference \
        export-models export-all export-small export-onnx export-end2end export-end2end-standard \
        export-custom export-config export-list \
        download-pytorch download-pytorch-all download-pytorch-list export-status validate-exports \
        create-dali create-ensembles rebuild-dali create-dali-dual create-dali-simple \
        validate-dali validate-dali-dual \
        export-mobileclip setup-track-e \
        open-grafana open-prometheus open-opensearch metrics gpu gpu-watch \
        clean clean-all clean-logs clean-bench clean-exports \
        opensearch-reset opensearch-status opensearch-indices \
        test-api-health test-track-a test-track-b test-track-c test-track-d test-track-e \
        test-track-e-full test-track-e-full-pipeline test-all-tracks compare-tracks info docs \
        clone-refs-essential clone-refs-recommended clone-refs-all clone-refs-list clone-ref
