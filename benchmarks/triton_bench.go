// Package main provides a high-performance benchmark tool for Triton Inference Server.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	version  = "1.0.0"
	trackAll = "all"
)

// Track represents an inference track
type Track struct {
	ID          string
	Name        string
	URL         string
	Description string
}

// TestMode represents different benchmark modes
type TestMode int

const (
	ModeSingleImage TestMode = iota
	ModeImageSet
	ModeQuickConcurrency
	ModeFullConcurrency
	ModeAllImages
	ModeThroughputSustained
	ModeVariableLoad
	ModeMatrix // NEW: Run tests across multiple concurrency levels
)

// BenchmarkConfig holds all configuration
type BenchmarkConfig struct {
	Mode          TestMode
	ImageDir      string
	ImageLimit    int
	Clients       int
	Duration      int
	Warmup        int
	TrackFilter   string
	OutputFile    string
	Quiet         bool
	JSONOutput    bool
	LoadPattern   string // "constant", "burst", "ramp"
	BurstInterval int    // seconds between bursts
	RampStep      int    // client increment for ramp
	EnableResize  bool   // enable pre-resize in FastAPI
	MaxSize       int    // max dimension for resize
	Port          int    // API port (default 4603, use 4613 for Track D benchmark)
	MatrixClients string // comma-separated list of client counts for matrix mode
}

// BenchmarkResults holds results for a single track
type BenchmarkResults struct {
	TrackID         string  `json:"track_id"`
	TrackName       string  `json:"track_name"`
	TestMode        string  `json:"test_mode"`
	TotalRequests   int64   `json:"total_requests"`
	SuccessRequests int64   `json:"success_requests"`
	FailedRequests  int64   `json:"failed_requests"`
	TotalDuration   float64 `json:"total_duration_sec"`
	Throughput      float64 `json:"throughput_rps"`
	MeanLatency     float64 `json:"mean_latency_ms"`
	MedianLatency   float64 `json:"median_latency_ms"`
	P95Latency      float64 `json:"p95_latency_ms"`
	P99Latency      float64 `json:"p99_latency_ms"`
	MinLatency      float64 `json:"min_latency_ms"`
	MaxLatency      float64 `json:"max_latency_ms"`
}

// DetectionResponse from API
type DetectionResponse struct {
	Detections []map[string]interface{} `json:"detections"`
	Status     string                   `json:"status"`
}

// Track represents an inference track
// IsBatch indicates if this track uses batch requests (multiple images per request)
type TrackConfig struct {
	IsBatch   bool
	BatchSize int // Default batch size for this track
}

// All available tracks
var allTracks = []Track{
	{ID: "A", Name: "PyTorch Baseline", URL: "http://localhost:4603/pytorch/predict/small", Description: "Native PyTorch + CPU NMS"},
	{ID: "B", Name: "Triton Standard TRT", URL: "http://localhost:4603/predict/small", Description: "TensorRT + CPU NMS"},
	{ID: "C", Name: "Triton End2End TRT", URL: "http://localhost:4603/predict/small_end2end", Description: "TensorRT + GPU NMS"},
	{ID: "D_streaming", Name: "DALI+TRT Streaming", URL: "http://localhost:4603/predict/small_gpu_e2e_streaming", Description: "Full GPU (low latency)"},
	{ID: "D_balanced", Name: "DALI+TRT Balanced", URL: "http://localhost:4603/predict/small_gpu_e2e", Description: "Full GPU (balanced)"},
	{ID: "D_batch", Name: "DALI+TRT Batch", URL: "http://localhost:4603/predict/small_gpu_e2e_batch", Description: "Full GPU (max throughput)"},
	{ID: "E", Name: "DALI+YOLO+CLIP (simple)", URL: "http://localhost:4603/track_e/predict", Description: "YOLO + global embedding (single image)"},
	{ID: "E_full", Name: "DALI+YOLO+CLIP (full)", URL: "http://localhost:4603/track_e/predict_full", Description: "YOLO + global + per-box embeddings"},
	{ID: "E_batch16", Name: "DALI+YOLO+CLIP Batch-16", URL: "http://localhost:4603/track_e/predict_batch", Description: "Batch 16 images per request"},
	{ID: "E_batch32", Name: "DALI+YOLO+CLIP Batch-32", URL: "http://localhost:4603/track_e/predict_batch", Description: "Batch 32 images per request"},
	{ID: "E_batch64", Name: "DALI+YOLO+CLIP Batch-64", URL: "http://localhost:4603/track_e/predict_batch", Description: "Batch 64 images per request"},
	{ID: "E_faces", Name: "SCRFD+ArcFace (faces)", URL: "http://localhost:4603/track_e/faces/recognize", Description: "Face detection + ArcFace embeddings"},
	{ID: "E_faces_detect", Name: "SCRFD Only (detect)", URL: "http://localhost:4603/track_e/faces/detect", Description: "Face detection only (no embeddings)"},
	{ID: "E_quad", Name: "Quad Pipeline (full)", URL: "http://localhost:4603/track_e/faces/full", Description: "YOLO + CLIP + SCRFD + ArcFace (unified)"},
	{ID: "E_unified", Name: "Unified (person-only)", URL: "http://localhost:4603/track_e/unified", Description: "YOLO + CLIP + face detection on person crops"},
	{ID: "F", Name: "CPU+TRT Direct", URL: "http://localhost:4603/track_f/predict", Description: "CPU preprocessing + direct TRT (no DALI)"},
}

// Track configs for batch tracks
var trackConfigs = map[string]TrackConfig{
	"E_batch16": {IsBatch: true, BatchSize: 16},
	"E_batch32": {IsBatch: true, BatchSize: 32},
	"E_batch64": {IsBatch: true, BatchSize: 64},
}

func main() {
	// Define flags
	var (
		mode          = flag.String("mode", "quick", "Test mode: single, set, quick, full, all, sustained, variable")
		imageDir      = flag.String("images", "/mnt/nvm/KILLBOY_SAMPLE_PICTURES", "Directory containing test images")
		imageLimit    = flag.Int("limit", 100, "Maximum number of images to load")
		clients       = flag.Int("clients", 64, "Number of concurrent clients")
		duration      = flag.Int("duration", 60, "Test duration in seconds")
		warmup        = flag.Int("warmup", 10, "Number of warmup requests per track")
		track         = flag.String("track", "all", "Track filter (A, B, C, D_*, E, E_full, E_batch*, E_faces, E_faces_detect, E_quad, F, or all)")
		output        = flag.String("output", "results/benchmark_results.json", "Output JSON file")
		quiet         = flag.Bool("quiet", false, "Quiet mode (minimal output)")
		jsonOut       = flag.Bool("json", false, "JSON output only")
		loadPattern   = flag.String("load-pattern", "constant", "Load pattern: constant, burst, ramp")
		burstInterval = flag.Int("burst-interval", 10, "Seconds between bursts (for burst mode)")
		rampStep      = flag.Int("ramp-step", 16, "Client increment for ramp mode")
		enableResize  = flag.Bool("resize", false, "Enable pre-resize for large images (recommended for 5MP+ images)")
		maxSize       = flag.Int("max-size", 1024, "Maximum dimension for resize (640-4096)")
		port          = flag.Int("port", 4603, "API port (4603=Track E, 4613=Track D benchmark)")
		matrixClients = flag.String("matrix-clients", "32,64,128,256,512,1024", "Comma-separated client counts for matrix mode")
		showVersion   = flag.Bool("version", false, "Show version and exit")
		listModes     = flag.Bool("list-modes", false, "List all available test modes")
	)

	flag.Parse()

	// Handle version
	if *showVersion {
		fmt.Printf("triton_bench version %s\n", version)
		os.Exit(0)
	}

	// Handle list modes
	if *listModes {
		printModes()
		os.Exit(0)
	}

	// Parse mode
	testMode, err := parseMode(*mode)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		fmt.Fprintf(os.Stderr, "Run with --list-modes to see available modes\n")
		os.Exit(1)
	}

	// Create config
	config := &BenchmarkConfig{
		Mode:          testMode,
		ImageDir:      *imageDir,
		ImageLimit:    *imageLimit,
		Clients:       *clients,
		Duration:      *duration,
		Warmup:        *warmup,
		TrackFilter:   *track,
		OutputFile:    *output,
		Quiet:         *quiet,
		JSONOutput:    *jsonOut,
		LoadPattern:   *loadPattern,
		BurstInterval: *burstInterval,
		RampStep:      *rampStep,
		EnableResize:  *enableResize,
		MaxSize:       *maxSize,
		Port:          *port,
		MatrixClients: *matrixClients,
	}

	// Run benchmark
	if err := runBenchmark(config); err != nil {
		fmt.Fprintf(os.Stderr, "Benchmark failed: %v\n", err)
		os.Exit(1)
	}
}

func printModes() {
	fmt.Println("Available Test Modes:")
	fmt.Println()
	fmt.Println("  single       Test a single image through all tracks")
	fmt.Println("  set          Test a set of images on all tracks (use --limit)")
	fmt.Println("  quick        Quick concurrency check (16 clients, 30s)")
	fmt.Println("  full         Full concurrency benchmark (use --clients and --duration)")
	fmt.Println("  all          Process all images in directory with concurrency")
	fmt.Println("  sustained    Max throughput sustainment test")
	fmt.Println("  variable     Variable load pattern test (use --load-pattern)")
	fmt.Println("  matrix       Run tests across multiple concurrency levels (use --matrix-clients)")
	fmt.Println()
	fmt.Println("Load Patterns (for variable mode):")
	fmt.Println("  constant     Steady load throughout test")
	fmt.Println("  burst        Periodic bursts (use --burst-interval)")
	fmt.Println("  ramp         Gradually increase load (use --ramp-step)")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("  triton_bench --mode single")
	fmt.Println("  triton_bench --mode quick --track D_batch")
	fmt.Println("  triton_bench --mode full --clients 256 --duration 120")
	fmt.Println("  triton_bench --mode variable --load-pattern ramp --ramp-step 32")
	fmt.Println("  triton_bench --mode matrix --track D_batch --matrix-clients 32,64,128,256,512 --duration 30")
}

func parseMode(mode string) (TestMode, error) {
	switch mode {
	case "single":
		return ModeSingleImage, nil
	case "set":
		return ModeImageSet, nil
	case "quick":
		return ModeQuickConcurrency, nil
	case "full":
		return ModeFullConcurrency, nil
	case trackAll:
		return ModeAllImages, nil
	case "sustained":
		return ModeThroughputSustained, nil
	case "variable":
		return ModeVariableLoad, nil
	case "matrix":
		return ModeMatrix, nil
	default:
		return 0, fmt.Errorf("invalid mode: %s", mode)
	}
}

func runBenchmark(config *BenchmarkConfig) error {
	if !config.Quiet && !config.JSONOutput {
		printHeader(config)
	}

	// Load images
	images, err := loadImages(config.ImageDir, config.ImageLimit)
	if err != nil {
		return fmt.Errorf("failed to load images: %v", err)
	}

	if !config.Quiet && !config.JSONOutput {
		fmt.Printf("✓ Loaded %d images from %s\n\n", len(images), config.ImageDir)
	}

	// Filter tracks (applies custom port if specified)
	tracks := filterTracks(config.TrackFilter, config.Port)

	// Check availability
	available := checkAvailability(tracks, config.Quiet)
	if len(available) == 0 {
		return fmt.Errorf("no tracks available")
	}

	// Run tests based on mode
	results, err := runTestMode(available, images, config)
	if err != nil {
		return err
	}

	// Output results
	return outputResults(results, config)
}

func runTestMode(available []Track, images [][]byte, config *BenchmarkConfig) ([]*BenchmarkResults, error) {
	switch config.Mode {
	case ModeSingleImage:
		return testSingleImage(available, images, config)
	case ModeImageSet:
		return testImageSet(available, images, config)
	case ModeQuickConcurrency:
		return testQuickConcurrency(available, images, config)
	case ModeFullConcurrency:
		return testFullConcurrency(available, images, config)
	case ModeAllImages:
		return testAllImages(available, images, config)
	case ModeThroughputSustained:
		return testSustained(available, images, config)
	case ModeVariableLoad:
		return testVariableLoad(available, images, config)
	case ModeMatrix:
		return testMatrix(available, images, config)
	default:
		return nil, fmt.Errorf("unknown test mode: %v", config.Mode)
	}
}

func testSingleImage(tracks []Track, images [][]byte, config *BenchmarkConfig) ([]*BenchmarkResults, error) {
	if !config.Quiet {
		fmt.Println("Test Mode: Single Image")
		fmt.Println("Testing one image through all tracks with 10 iterations for accuracy")
		fmt.Println(strings.Repeat("=", 80))
	}

	results := make([]*BenchmarkResults, 0, len(tracks))
	client := createHTTPClient(1)

	for _, track := range tracks {
		if !config.Quiet {
			fmt.Printf("\nTrack %s: %s\n", track.ID, track.Name)
		}

		var latencies []float64
		var success int64
		var failed int64

		start := time.Now()
		for i := 0; i < 10; i++ {
			latency, ok := sendSingleRequest(client, track.URL, images[0], config)
			if ok {
				latencies = append(latencies, latency)
				success++
			} else {
				failed++
			}
		}
		duration := time.Since(start).Seconds()

		results = append(results, calculateResults(track, "single_image", success, failed, latencies, duration))

		if !config.Quiet {
			fmt.Printf("  Mean latency: %.2fms\n", results[len(results)-1].MeanLatency)
			fmt.Printf("  Success: %d/10\n", success)
		}
	}

	return results, nil
}

func testImageSet(tracks []Track, images [][]byte, config *BenchmarkConfig) ([]*BenchmarkResults, error) {
	if !config.Quiet {
		fmt.Printf("Test Mode: Image Set (%d images)\n", len(images))
		fmt.Printf("Processing %d images sequentially through each track\n", len(images))
		fmt.Println(strings.Repeat("=", 80))
	}

	results := make([]*BenchmarkResults, 0, len(tracks))
	client := createHTTPClient(1)

	for _, track := range tracks {
		if !config.Quiet {
			fmt.Printf("\nTrack %s: %s\n", track.ID, track.Name)
		}

		var latencies []float64
		var success int64
		var failed int64

		start := time.Now()
		for i, img := range images {
			if !config.Quiet && (i+1)%10 == 0 {
				fmt.Printf("  Processed %d/%d images...\r", i+1, len(images))
			}

			latency, ok := sendSingleRequest(client, track.URL, img, config)
			if ok {
				latencies = append(latencies, latency)
				success++
			} else {
				failed++
			}
		}
		duration := time.Since(start).Seconds()

		if !config.Quiet {
			fmt.Printf("  Processed %d/%d images - Complete\n", len(images), len(images))
		}

		results = append(results, calculateResults(track, "image_set", success, failed, latencies, duration))

		if !config.Quiet {
			fmt.Printf("  Throughput: %.2f images/sec\n", float64(success)/duration)
			fmt.Printf("  Mean latency: %.2fms\n", results[len(results)-1].MeanLatency)
		}
	}

	return results, nil
}

func testQuickConcurrency(tracks []Track, images [][]byte, config *BenchmarkConfig) ([]*BenchmarkResults, error) {
	// Override config for quick test
	quickClients := 16
	quickDuration := 30

	if !config.Quiet {
		fmt.Printf("Test Mode: Quick Concurrency Check\n")
		fmt.Printf("Testing with %d concurrent clients for %d seconds\n", quickClients, quickDuration)
		fmt.Println(strings.Repeat("=", 80))
	}

	results := make([]*BenchmarkResults, 0, len(tracks))

	for _, track := range tracks {
		if !config.Quiet {
			fmt.Printf("\nTrack %s: %s\n", track.ID, track.Name)
		}

		// Warmup
		if config.Warmup > 0 {
			runWarmup(track, images, config.Warmup, config.Quiet, config)
		}

		// Run concurrent test
		result := runConcurrentTest(track, images, quickClients, quickDuration, "quick_concurrency", config)

		results = append(results, result)

		if !config.Quiet {
			fmt.Printf("  Throughput: %.2f req/sec\n", result.Throughput)
			fmt.Printf("  P50 latency: %.2fms\n", result.MedianLatency)
			fmt.Printf("  Success rate: %.1f%%\n", float64(result.SuccessRequests)/float64(result.TotalRequests)*100)
		}
	}

	return results, nil
}

func testFullConcurrency(tracks []Track, images [][]byte, config *BenchmarkConfig) ([]*BenchmarkResults, error) {
	if !config.Quiet {
		fmt.Printf("Test Mode: Full Concurrency Benchmark\n")
		fmt.Printf("Testing with %d concurrent clients for %d seconds\n", config.Clients, config.Duration)
		fmt.Println(strings.Repeat("=", 80))
	}

	results := make([]*BenchmarkResults, 0, len(tracks))

	for _, track := range tracks {
		if !config.Quiet {
			fmt.Printf("\nTrack %s: %s\n", track.ID, track.Name)
		}

		// Warmup
		if config.Warmup > 0 {
			runWarmup(track, images, config.Warmup, config.Quiet, config)
		}

		// Run concurrent test
		result := runConcurrentTest(track, images, config.Clients, config.Duration, "full_concurrency", config)

		results = append(results, result)

		if !config.Quiet {
			fmt.Printf("  Total requests: %d\n", result.TotalRequests)
			fmt.Printf("  Throughput: %.2f req/sec\n", result.Throughput)
			fmt.Printf("  Mean latency: %.2fms\n", result.MeanLatency)
			fmt.Printf("  P95 latency: %.2fms\n", result.P95Latency)
			fmt.Printf("  P99 latency: %.2fms\n", result.P99Latency)
			fmt.Printf("  Success rate: %.1f%%\n", float64(result.SuccessRequests)/float64(result.TotalRequests)*100)
		}
	}

	return results, nil
}

func testAllImages(tracks []Track, images [][]byte, config *BenchmarkConfig) ([]*BenchmarkResults, error) {
	if !config.Quiet {
		fmt.Printf("Test Mode: All Images with Concurrency\n")
		fmt.Printf("Processing all %d images with %d concurrent clients\n", len(images), config.Clients)
		fmt.Println(strings.Repeat("=", 80))
	}

	results := make([]*BenchmarkResults, 0, len(tracks))

	for _, track := range tracks {
		if !config.Quiet {
			fmt.Printf("\nTrack %s: %s\n", track.ID, track.Name)
		}

		// Process all images with concurrency
		result := processAllImagesConcurrent(track, images, config.Clients, config)

		results = append(results, result)

		if !config.Quiet {
			fmt.Printf("  Processed: %d images\n", result.SuccessRequests)
			fmt.Printf("  Failed: %d images\n", result.FailedRequests)
			fmt.Printf("  Total time: %.2f seconds\n", result.TotalDuration)
			fmt.Printf("  Throughput: %.2f images/sec\n", result.Throughput)
			fmt.Printf("  Mean latency: %.2fms\n", result.MeanLatency)
		}
	}

	return results, nil
}

func testSustained(tracks []Track, images [][]byte, config *BenchmarkConfig) ([]*BenchmarkResults, error) {
	sustainedDuration := 300 // 5 minutes

	if !config.Quiet {
		fmt.Printf("Test Mode: Sustained Throughput\n")
		fmt.Printf("Testing maximum sustained throughput for %d seconds\n", sustainedDuration)
		fmt.Printf("Will automatically adjust client count to maximize throughput\n")
		fmt.Println(strings.Repeat("=", 80))
	}

	results := make([]*BenchmarkResults, 0, len(tracks))

	for _, track := range tracks {
		if !config.Quiet {
			fmt.Printf("\nTrack %s: %s\n", track.ID, track.Name)
			fmt.Println("Finding optimal client count...")
		}

		// Find optimal client count
		optimalClients := findOptimalClients(track, images, config.Quiet, config)

		if !config.Quiet {
			fmt.Printf("  Optimal clients: %d\n", optimalClients)
			fmt.Printf("  Running sustained test for %d seconds...\n", sustainedDuration)
		}

		// Run sustained test
		result := runConcurrentTest(track, images, optimalClients, sustainedDuration, "sustained_throughput", config)

		results = append(results, result)

		if !config.Quiet {
			fmt.Printf("  Total requests: %d\n", result.TotalRequests)
			fmt.Printf("  Sustained throughput: %.2f req/sec\n", result.Throughput)
			fmt.Printf("  Mean latency: %.2fms\n", result.MeanLatency)
			fmt.Printf("  P99 latency: %.2fms\n", result.P99Latency)
		}
	}

	return results, nil
}

func testVariableLoad(tracks []Track, images [][]byte, config *BenchmarkConfig) ([]*BenchmarkResults, error) {
	if !config.Quiet {
		fmt.Printf("Test Mode: Variable Load Pattern (%s)\n", config.LoadPattern)
		fmt.Println(strings.Repeat("=", 80))
	}

	results := make([]*BenchmarkResults, 0, len(tracks))

	for _, track := range tracks {
		if !config.Quiet {
			fmt.Printf("\nTrack %s: %s\n", track.ID, track.Name)
		}

		var result *BenchmarkResults

		switch config.LoadPattern {
		case "burst":
			result = runBurstPattern(track, images, config)
		case "ramp":
			result = runRampPattern(track, images, config)
		default: // constant
			result = runConcurrentTest(track, images, config.Clients, config.Duration, "variable_constant", config)
		}

		results = append(results, result)

		if !config.Quiet {
			fmt.Printf("  Total requests: %d\n", result.TotalRequests)
			fmt.Printf("  Throughput: %.2f req/sec\n", result.Throughput)
			fmt.Printf("  Mean latency: %.2fms\n", result.MeanLatency)
			fmt.Printf("  P95 latency: %.2fms\n", result.P95Latency)
		}
	}

	return results, nil
}

// MatrixResult holds results for a single concurrency level in matrix mode
type MatrixResult struct {
	Clients     int
	Throughput  float64
	MeanLatency float64
	P95Latency  float64
	P99Latency  float64
	Success     float64
}

func testMatrix(tracks []Track, images [][]byte, config *BenchmarkConfig) ([]*BenchmarkResults, error) {
	// Parse client counts from comma-separated string
	clientStrs := strings.Split(config.MatrixClients, ",")
	var clientCounts []int
	for _, s := range clientStrs {
		s = strings.TrimSpace(s)
		if count, err := strconv.Atoi(s); err == nil && count > 0 {
			clientCounts = append(clientCounts, count)
		}
	}

	if len(clientCounts) == 0 {
		return nil, fmt.Errorf("no valid client counts in matrix-clients: %s", config.MatrixClients)
	}

	if !config.Quiet {
		fmt.Printf("Test Mode: Matrix Benchmark\n")
		fmt.Printf("Testing across %d concurrency levels: %v\n", len(clientCounts), clientCounts)
		fmt.Printf("Duration per level: %d seconds\n", config.Duration)
		fmt.Println(strings.Repeat("=", 80))
	}

	var results []*BenchmarkResults

	for _, track := range tracks {
		if !config.Quiet {
			fmt.Printf("\nTrack %s: %s\n", track.ID, track.Name)
			fmt.Println(strings.Repeat("-", 80))
			fmt.Printf("%-10s | %-12s | %-12s | %-12s | %-12s | %-10s\n",
				"Clients", "Throughput", "Mean (ms)", "P95 (ms)", "P99 (ms)", "Success")
			fmt.Println(strings.Repeat("-", 80))
		}

		// Warmup once before matrix
		if config.Warmup > 0 {
			runWarmup(track, images, config.Warmup, true, config)
		}

		var matrixResults []MatrixResult

		for _, clients := range clientCounts {
			// Run concurrent test for this client count
			result := runConcurrentTest(track, images, clients, config.Duration, fmt.Sprintf("matrix_%d", clients), config)

			successRate := float64(result.SuccessRequests) / float64(result.TotalRequests) * 100

			matrixResults = append(matrixResults, MatrixResult{
				Clients:     clients,
				Throughput:  result.Throughput,
				MeanLatency: result.MeanLatency,
				P95Latency:  result.P95Latency,
				P99Latency:  result.P99Latency,
				Success:     successRate,
			})

			if !config.Quiet {
				fmt.Printf("%-10d | %-12.1f | %-12.2f | %-12.2f | %-12.2f | %-10.1f%%\n",
					clients, result.Throughput, result.MeanLatency, result.P95Latency, result.P99Latency, successRate)
			}

			// Append each result separately with client count in track ID
			result.TrackID = fmt.Sprintf("%s@%d", track.ID, clients)
			results = append(results, result)
		}

		// Print summary for this track
		if !config.Quiet {
			fmt.Println(strings.Repeat("-", 80))
			// Find peak throughput
			var peakTP float64
			var peakClients int
			for _, mr := range matrixResults {
				if mr.Throughput > peakTP {
					peakTP = mr.Throughput
					peakClients = mr.Clients
				}
			}
			fmt.Printf("Peak throughput: %.1f req/sec @ %d clients\n", peakTP, peakClients)
		}
	}

	return results, nil
}

func runConcurrentTest(track Track, images [][]byte, clients int, duration int, testType string, config *BenchmarkConfig) *BenchmarkResults {
	var (
		totalRequests   int64
		successRequests int64
		failedRequests  int64
		latencies       []float64
		latenciesMutex  sync.Mutex
		wg              sync.WaitGroup
	)

	// Check if this is a batch track
	trackConfig, isBatchTrack := trackConfigs[track.ID]
	batchSize := 1
	if isBatchTrack {
		batchSize = trackConfig.BatchSize
	}

	client := createHTTPClient(clients)
	startTime := time.Now()
	endTime := startTime.Add(time.Duration(duration) * time.Second)

	// Launch workers
	for i := 0; i < clients; i++ {
		wg.Add(1)
		go func(clientID int) {
			defer wg.Done()

			requestCount := 0
			for time.Now().Before(endTime) {
				if isBatchTrack {
					// Batch mode: send multiple images per request
					batchImages := make([][]byte, batchSize)
					for j := 0; j < batchSize; j++ {
						batchImages[j] = images[(requestCount*batchSize+j)%len(images)]
					}

					latency, processedCount, ok := sendBatchRequest(client, track.URL, batchImages)

					// Count each image in the batch as a request
					atomic.AddInt64(&totalRequests, int64(batchSize))

					if ok {
						atomic.AddInt64(&successRequests, int64(processedCount))
						atomic.AddInt64(&failedRequests, int64(batchSize-processedCount))
						// Record per-image latency for fair comparison
						perImageLatency := latency / float64(processedCount)
						latenciesMutex.Lock()
						for k := 0; k < processedCount; k++ {
							latencies = append(latencies, perImageLatency)
						}
						latenciesMutex.Unlock()
					} else {
						atomic.AddInt64(&failedRequests, int64(batchSize))
					}
				} else {
					// Single image mode
					imageData := images[requestCount%len(images)]

					latency, ok := sendSingleRequest(client, track.URL, imageData, config)

					atomic.AddInt64(&totalRequests, 1)

					if ok {
						atomic.AddInt64(&successRequests, 1)
						latenciesMutex.Lock()
						latencies = append(latencies, latency)
						latenciesMutex.Unlock()
					} else {
						atomic.AddInt64(&failedRequests, 1)
					}
				}

				requestCount++
			}
		}(i)
	}

	wg.Wait()
	actualDuration := time.Since(startTime).Seconds()

	return calculateResults(track, testType, successRequests, failedRequests, latencies, actualDuration)
}

func processAllImagesConcurrent(track Track, images [][]byte, workers int, config *BenchmarkConfig) *BenchmarkResults {
	var (
		success        int64
		failed         int64
		latencies      []float64
		latenciesMutex sync.Mutex
		wg             sync.WaitGroup
	)

	client := createHTTPClient(workers)
	imageChan := make(chan []byte, len(images))

	// Fill channel with images
	for _, img := range images {
		imageChan <- img
	}
	close(imageChan)

	startTime := time.Now()

	// Launch workers
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for img := range imageChan {
				latency, ok := sendSingleRequest(client, track.URL, img, config)

				if ok {
					atomic.AddInt64(&success, 1)
					latenciesMutex.Lock()
					latencies = append(latencies, latency)
					latenciesMutex.Unlock()
				} else {
					atomic.AddInt64(&failed, 1)
				}
			}
		}()
	}

	wg.Wait()
	duration := time.Since(startTime).Seconds()

	return calculateResults(track, "all_images_concurrent", success, failed, latencies, duration)
}

func runBurstPattern(track Track, images [][]byte, config *BenchmarkConfig) *BenchmarkResults {
	// Burst pattern: high load for 5 seconds, then idle for interval seconds
	burstDuration := 5
	totalCycles := config.Duration / (burstDuration + config.BurstInterval)

	var allLatencies []float64
	var totalSuccess, totalFailed int64

	startTime := time.Now()

	for i := 0; i < totalCycles; i++ {
		// Burst phase
		result := runConcurrentTest(track, images, config.Clients, burstDuration, "burst", config)
		allLatencies = append(allLatencies, extractLatencies(result)...)
		totalSuccess += result.SuccessRequests
		totalFailed += result.FailedRequests

		// Idle phase
		if i < totalCycles-1 {
			time.Sleep(time.Duration(config.BurstInterval) * time.Second)
		}
	}

	duration := time.Since(startTime).Seconds()

	return calculateResults(track, "variable_burst", totalSuccess, totalFailed, allLatencies, duration)
}

func runRampPattern(track Track, images [][]byte, config *BenchmarkConfig) *BenchmarkResults {
	// Gradually increase load from rampStep to clients
	steps := config.Clients / config.RampStep
	durationPerStep := config.Duration / steps

	var allLatencies []float64
	var totalSuccess, totalFailed int64

	startTime := time.Now()

	for step := 1; step <= steps; step++ {
		currentClients := step * config.RampStep

		result := runConcurrentTest(track, images, currentClients, durationPerStep, "ramp", config)
		allLatencies = append(allLatencies, extractLatencies(result)...)
		totalSuccess += result.SuccessRequests
		totalFailed += result.FailedRequests
	}

	duration := time.Since(startTime).Seconds()

	return calculateResults(track, "variable_ramp", totalSuccess, totalFailed, allLatencies, duration)
}

func findOptimalClients(track Track, images [][]byte, quiet bool, config *BenchmarkConfig) int {
	testDuration := 10 // seconds per test
	clientCounts := []int{16, 32, 64, 128, 256}

	var bestClients int
	var bestThroughput float64

	for _, clients := range clientCounts {
		if !quiet {
			fmt.Printf("  Testing %d clients...\n", clients)
		}

		result := runConcurrentTest(track, images, clients, testDuration, "optimization", config)

		if result.Throughput > bestThroughput {
			bestThroughput = result.Throughput
			bestClients = clients
		}

		// If throughput is decreasing, we've passed the optimal point
		if result.Throughput < bestThroughput*0.95 {
			break
		}
	}

	return bestClients
}

func extractLatencies(result *BenchmarkResults) []float64 {
	// This is a simplified version - in real implementation we'd store raw latencies
	// For now, we approximate using available statistics
	count := int(result.SuccessRequests)
	latencies := make([]float64, count)
	for i := 0; i < count; i++ {
		latencies[i] = result.MeanLatency
	}
	return latencies
}

func sendSingleRequest(client *http.Client, url string, imageData []byte, config ...*BenchmarkConfig) (float64, bool) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	part, err := writer.CreateFormFile("image", "image.jpg")
	if err != nil {
		return 0, false
	}

	if _, writeErr := part.Write(imageData); writeErr != nil {
		return 0, false
	}

	if closeErr := writer.Close(); closeErr != nil {
		return 0, false
	}

	// Add query parameters if resize enabled
	finalURL := url
	if len(config) > 0 && config[0].EnableResize {
		finalURL = fmt.Sprintf("%s?resize=true&max_size=%d", url, config[0].MaxSize)
	}

	req, err := http.NewRequest("POST", finalURL, &buf)
	if err != nil {
		return 0, false
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	start := time.Now()
	resp, err := client.Do(req)
	latency := time.Since(start).Seconds() * 1000

	if err != nil {
		return 0, false
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != 200 {
		return 0, false
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, false
	}

	var detResp DetectionResponse
	if err := json.Unmarshal(body, &detResp); err != nil {
		return 0, false
	}

	return latency, true
}

// BatchResponse represents response from batch endpoint
type BatchResponse struct {
	Status        string `json:"status"`
	BatchSize     int    `json:"batch_size"`
	ThroughputIPS float64 `json:"throughput_ips"`
}

// sendBatchRequest sends multiple images in a single request
// Returns: latency in ms, number of images processed, success
func sendBatchRequest(client *http.Client, url string, imagesData [][]byte) (float64, int, bool) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add all images to the multipart form
	for i, imageData := range imagesData {
		part, err := writer.CreateFormFile("images", fmt.Sprintf("image_%d.jpg", i))
		if err != nil {
			return 0, 0, false
		}
		if _, writeErr := part.Write(imageData); writeErr != nil {
			return 0, 0, false
		}
	}

	if closeErr := writer.Close(); closeErr != nil {
		return 0, 0, false
	}

	req, err := http.NewRequest("POST", url, &buf)
	if err != nil {
		return 0, 0, false
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	start := time.Now()
	resp, err := client.Do(req)
	latency := time.Since(start).Seconds() * 1000

	if err != nil {
		return 0, 0, false
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		fmt.Printf("Batch request failed: %d - %s\n", resp.StatusCode, string(body))
		return 0, 0, false
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, 0, false
	}

	var batchResp BatchResponse
	if err := json.Unmarshal(body, &batchResp); err != nil {
		return 0, 0, false
	}

	return latency, batchResp.BatchSize, true
}

func runWarmup(track Track, images [][]byte, count int, quiet bool, config *BenchmarkConfig) {
	if !quiet {
		fmt.Printf("  Warming up (%d requests)...\n", count)
	}

	client := createHTTPClient(1)
	for i := 0; i < count; i++ {
		sendSingleRequest(client, track.URL, images[i%len(images)], config)
	}

	time.Sleep(1 * time.Second)
}

func loadImages(dir string, limit int) ([][]byte, error) {
	images := make([][]byte, 0, limit)

	// Use filepath.Walk for recursive directory search
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Skip files we can't access
		}
		if info.IsDir() {
			return nil // Continue into subdirectories
		}
		if len(images) >= limit {
			return filepath.SkipAll // Stop once we have enough images
		}

		ext := strings.ToLower(filepath.Ext(info.Name()))
		if ext != ".jpg" && ext != ".jpeg" {
			return nil
		}

		data, err := os.ReadFile(path)
		if err != nil {
			return nil // Skip files we can't read
		}

		images = append(images, data)
		return nil
	})

	if err != nil && err != filepath.SkipAll {
		return nil, err
	}

	if len(images) == 0 {
		return nil, fmt.Errorf("no JPEG images found in %s", dir)
	}

	return images, nil
}

func filterTracks(filter string, port int) []Track {
	var tracks []Track
	if filter == trackAll {
		tracks = make([]Track, len(allTracks))
		copy(tracks, allTracks)
	} else {
		for _, track := range allTracks {
			if track.ID == filter {
				tracks = append(tracks, track)
			}
		}
	}

	// Apply custom port if specified (non-default)
	if port != 4603 {
		for i := range tracks {
			tracks[i].URL = strings.Replace(tracks[i].URL, ":4603/", fmt.Sprintf(":%d/", port), 1)
		}
	}

	return tracks
}

func checkAvailability(tracks []Track, quiet bool) []Track {
	if !quiet {
		fmt.Println("Checking track availability...")
	}

	var available []Track
	client := &http.Client{Timeout: 5 * time.Second}

	for _, track := range tracks {
		baseURL := getBaseURL(track.URL)
		resp, err := client.Get(baseURL + "/health")

		if err == nil && resp.StatusCode == 200 {
			available = append(available, track)
			if !quiet {
				fmt.Printf("  ✓ Track %s: %s\n", track.ID, track.Name)
			}
			_ = resp.Body.Close()
		} else {
			if !quiet {
				fmt.Printf("  ✗ Track %s: %s (unavailable)\n", track.ID, track.Name)
			}
		}
	}

	fmt.Println()
	return available
}

func getBaseURL(urlStr string) string {
	// Extract scheme://host:port from URL
	// Example: http://localhost:4603/pytorch/predict/small -> http://localhost:4603
	if idx := strings.Index(urlStr, "://"); idx != -1 {
		// Find the first slash after scheme://
		rest := urlStr[idx+3:]
		if slashIdx := strings.Index(rest, "/"); slashIdx != -1 {
			return urlStr[:idx+3+slashIdx]
		}
	}
	return urlStr
}

func createHTTPClient(maxConns int) *http.Client {
	return &http.Client{
		Timeout: 30 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        maxConns * 2,
			MaxIdleConnsPerHost: maxConns * 2,
			IdleConnTimeout:     90 * time.Second,
		},
	}
}

func calculateResults(track Track, testType string, success, failed int64, latencies []float64, duration float64) *BenchmarkResults {
	result := &BenchmarkResults{
		TrackID:         track.ID,
		TrackName:       track.Name,
		TestMode:        testType,
		TotalRequests:   success + failed,
		SuccessRequests: success,
		FailedRequests:  failed,
		TotalDuration:   duration,
		Throughput:      float64(success) / duration,
	}

	if len(latencies) == 0 {
		return result
	}

	sort.Float64s(latencies)

	var sum float64
	for _, lat := range latencies {
		sum += lat
	}

	result.MeanLatency = sum / float64(len(latencies))
	result.MinLatency = latencies[0]
	result.MaxLatency = latencies[len(latencies)-1]
	result.MedianLatency = percentile(latencies, 50)
	result.P95Latency = percentile(latencies, 95)
	result.P99Latency = percentile(latencies, 99)

	return result
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}

	index := (p / 100.0) * float64(len(sorted)-1)
	lower := int(index)
	upper := lower + 1

	if upper >= len(sorted) {
		return sorted[len(sorted)-1]
	}

	weight := index - float64(lower)
	return sorted[lower]*(1-weight) + sorted[upper]*weight
}

func outputResults(results []*BenchmarkResults, config *BenchmarkConfig) error {
	if config.JSONOutput {
		return outputJSON(results, config.OutputFile)
	}

	printResultsTable(results)

	// Fetch and display Triton model statistics (skip for Track A which is PyTorch only)
	if config.TrackFilter != "A" {
		tritonURL := getTritonBaseURL(config.Port)
		stats, err := fetchTritonStats(tritonURL)
		if err != nil {
			fmt.Printf("\n⚠ Could not fetch Triton stats: %v\n", err)
		} else {
			printTritonStats(stats, config.TrackFilter)
		}
	}

	return saveJSON(results, config.OutputFile)
}

func printResultsTable(results []*BenchmarkResults) {
	fmt.Println(strings.Repeat("=", 120))
	fmt.Println("BENCHMARK RESULTS")
	fmt.Println(strings.Repeat("=", 120))

	fmt.Printf("%-15s %-25s %-12s %-10s %-10s %-10s %-10s\n",
		"Track", "Name", "Throughput", "P50 (ms)", "P95 (ms)", "P99 (ms)", "Success")
	fmt.Println(strings.Repeat("-", 120))

	for _, r := range results {
		successRate := float64(r.SuccessRequests) / float64(r.TotalRequests) * 100

		fmt.Printf("%-15s %-25s %10.1f rps %9.2f %9.2f %9.2f %9.1f%%\n",
			r.TrackID,
			truncate(r.TrackName, 25),
			r.Throughput,
			r.MedianLatency,
			r.P95Latency,
			r.P99Latency,
			successRate)
	}

	// Print speedup comparison if we have Track A
	var baselineResult *BenchmarkResults
	for _, r := range results {
		if r.TrackID == "A" {
			baselineResult = r
			break
		}
	}

	if baselineResult != nil {
		fmt.Println()
		fmt.Println("Speedup vs Track A (PyTorch Baseline):")
		fmt.Println(strings.Repeat("-", 120))

		for _, r := range results {
			speedup := r.Throughput / baselineResult.Throughput
			fmt.Printf("%-15s %-25s %11.2fx\n",
				r.TrackID,
				truncate(r.TrackName, 25),
				speedup)
		}
	}

	fmt.Println(strings.Repeat("=", 120))
}

func outputJSON(results []*BenchmarkResults, filename string) error {
	data, err := json.MarshalIndent(map[string]interface{}{
		"timestamp": time.Now().Format(time.RFC3339),
		"results":   results,
	}, "", "  ")

	if err != nil {
		return err
	}

	fmt.Println(string(data))
	return nil
}

func saveJSON(results []*BenchmarkResults, filename string) error {
	dir := filepath.Dir(filename)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	// Generate timestamped filename
	// Extract base name and extension
	ext := filepath.Ext(filename)
	baseName := filename[:len(filename)-len(ext)]

	// Create timestamp in format: 2025-12-07_143045
	timestamp := time.Now().Format("2006-01-02_150405")

	// Build timestamped filename: results/benchmark_2025-12-07_143045.json
	timestampedFilename := fmt.Sprintf("%s_%s%s", baseName, timestamp, ext)

	data, err := json.MarshalIndent(map[string]interface{}{
		"timestamp": time.Now().Format(time.RFC3339),
		"results":   results,
	}, "", "  ")

	if err != nil {
		return err
	}

	// Save timestamped version
	if err := os.WriteFile(timestampedFilename, data, 0644); err != nil {
		return err
	}

	// Also save to default filename (latest)
	if err := os.WriteFile(filename, data, 0644); err != nil {
		return err
	}

	fmt.Printf("\n✓ Results saved to: %s\n", timestampedFilename)
	fmt.Printf("✓ Latest results:   %s\n", filename)
	return nil
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

func printHeader(config *BenchmarkConfig) {
	fmt.Println(strings.Repeat("=", 120))
	fmt.Println("TRITON YOLO BENCHMARK SUITE v" + version)
	fmt.Println(strings.Repeat("=", 120))
	fmt.Printf("Mode: %s\n", getModeString(config.Mode))
	fmt.Printf("Images: %s (limit: %d)\n", config.ImageDir, config.ImageLimit)
	if config.Mode == ModeFullConcurrency || config.Mode == ModeAllImages || config.Mode == ModeVariableLoad {
		fmt.Printf("Clients: %d\n", config.Clients)
	}
	if config.Mode != ModeSingleImage && config.Mode != ModeImageSet {
		fmt.Printf("Duration: %d seconds\n", config.Duration)
	}
	if config.TrackFilter != trackAll {
		fmt.Printf("Track filter: %s\n", config.TrackFilter)
	}
	fmt.Println(strings.Repeat("=", 120))
	fmt.Println()
}

func getModeString(mode TestMode) string {
	switch mode {
	case ModeSingleImage:
		return "Single Image"
	case ModeImageSet:
		return "Image Set"
	case ModeQuickConcurrency:
		return "Quick Concurrency"
	case ModeFullConcurrency:
		return "Full Concurrency"
	case ModeAllImages:
		return "All Images"
	case ModeThroughputSustained:
		return "Sustained Throughput"
	case ModeVariableLoad:
		return "Variable Load"
	default:
		return "Unknown"
	}
}

// =============================================================================
// Triton Model Statistics
// =============================================================================

// TritonStatsResponse represents the response from /v2/models/stats
type TritonStatsResponse struct {
	ModelStats []ModelStats `json:"model_stats"`
}

// ModelStats represents stats for a single model
type ModelStats struct {
	Name           string         `json:"name"`
	Version        string         `json:"version"`
	InferenceStats InferenceStats `json:"inference_stats"`
	BatchStats     []BatchStats   `json:"batch_stats"`
}

// InferenceStats holds timing information
type InferenceStats struct {
	Success       StatMetric `json:"success"`
	Queue         StatMetric `json:"queue"`
	ComputeInfer  StatMetric `json:"compute_infer"`
	ComputeInput  StatMetric `json:"compute_input"`
	ComputeOutput StatMetric `json:"compute_output"`
}

// StatMetric holds count and nanoseconds
type StatMetric struct {
	Count int64 `json:"count"`
	Ns    int64 `json:"ns"`
}

// BatchStats holds batch size distribution
type BatchStats struct {
	BatchSize    int        `json:"batch_size"`
	ComputeInfer StatMetric `json:"compute_infer"`
}

// fetchTritonStats fetches model statistics from Triton server
func fetchTritonStats(tritonURL string) (*TritonStatsResponse, error) {
	client := &http.Client{Timeout: 30 * time.Second}

	resp, err := client.Get(tritonURL + "/v2/models/stats")
	if err != nil {
		return nil, fmt.Errorf("failed to fetch stats: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("stats endpoint returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %v", err)
	}

	var stats TritonStatsResponse
	if err := json.Unmarshal(body, &stats); err != nil {
		return nil, fmt.Errorf("failed to parse stats JSON: %v", err)
	}

	return &stats, nil
}

// printTritonStats displays Triton model statistics in a formatted table
func printTritonStats(stats *TritonStatsResponse, trackFilter string) {
	fmt.Println()
	fmt.Println(strings.Repeat("=", 120))
	fmt.Println("TRITON MODEL STATISTICS")
	fmt.Println(strings.Repeat("=", 120))

	// Filter models based on track
	relevantModels := filterRelevantModels(stats.ModelStats, trackFilter)

	if len(relevantModels) == 0 {
		fmt.Println("No model statistics available")
		return
	}

	for _, model := range relevantModels {
		if model.InferenceStats.Success.Count == 0 {
			continue
		}

		count := model.InferenceStats.Success.Count
		totalNs := model.InferenceStats.Success.Ns
		queueNs := model.InferenceStats.Queue.Ns
		computeNs := model.InferenceStats.ComputeInfer.Ns

		avgMs := float64(totalNs) / float64(count) / 1e6
		queueMs := float64(queueNs) / float64(count) / 1e6
		computeMs := float64(computeNs) / float64(count) / 1e6

		fmt.Printf("\n%s:\n", model.Name)
		fmt.Printf("  Inferences: %d\n", count)
		fmt.Printf("  Avg total: %.2fms (queue: %.2fms, compute: %.2fms)\n", avgMs, queueMs, computeMs)
		fmt.Printf("  Total queue time: %.1fs, Total compute: %.1fs\n", float64(queueNs)/1e9, float64(computeNs)/1e9)

		// Batch distribution
		if len(model.BatchStats) > 0 {
			fmt.Println("  Batch distribution:")

			var totalSamples int64
			var totalExecutions int64

			// Sort batch stats by batch size
			sortedBatches := make([]BatchStats, len(model.BatchStats))
			copy(sortedBatches, model.BatchStats)
			sort.Slice(sortedBatches, func(i, j int) bool {
				return sortedBatches[i].BatchSize < sortedBatches[j].BatchSize
			})

			for _, b := range sortedBatches {
				if b.ComputeInfer.Count > 0 {
					samples := int64(b.BatchSize) * b.ComputeInfer.Count
					totalSamples += samples
					totalExecutions += b.ComputeInfer.Count
					fmt.Printf("    batch=%d: %d executions (%d samples)\n",
						b.BatchSize, b.ComputeInfer.Count, samples)
				}
			}

			if totalExecutions > 0 {
				avgBatch := float64(totalSamples) / float64(totalExecutions)
				fmt.Printf("  Average batch size: %.2f\n", avgBatch)
			}
		}
	}

	fmt.Println()
}

// filterRelevantModels filters models based on track
func filterRelevantModels(models []ModelStats, trackFilter string) []ModelStats {
	var relevant []ModelStats

	// Define model prefixes for each track
	trackPrefixes := map[string][]string{
		"A":              {}, // PyTorch - no Triton models
		"B":              {"yolov11_small_trt"},
		"C":              {"yolov11_small_trt_end2end"},
		"D_streaming":    {"yolo_preprocess_dali_streaming", "yolov11_small_trt_end2end_streaming", "yolov11_small_gpu_e2e_streaming"},
		"D_balanced":     {"yolo_preprocess_dali", "yolov11_small_trt_end2end", "yolov11_small_gpu_e2e"},
		"D_batch":        {"yolo_preprocess_dali_batch", "yolov11_small_trt_end2end_batch", "yolov11_small_gpu_e2e_batch"},
		"E":              {"yolo_clip_preprocess_dali", "yolov11_small_trt_end2end", "mobileclip", "yolo_clip_ensemble"},
		"E_full":         {"yolo_clip_preprocess_dali", "yolov11_small_trt_end2end", "mobileclip", "box_embedding", "yolo_mobileclip_ensemble"},
		"E_faces":        {"quad_preprocess_dali", "scrfd", "arcface", "face_pipeline"},
		"E_faces_detect": {"quad_preprocess_dali", "scrfd", "face_pipeline"},
		"E_quad":         {"quad_preprocess_dali", "yolov11_small_trt_end2end", "mobileclip", "scrfd", "arcface", "face_pipeline", "yolo_face_clip_ensemble"},
	}

	// Get prefixes for the current track
	prefixes, ok := trackPrefixes[trackFilter]
	if !ok || trackFilter == trackAll {
		// Return all models with activity
		for _, m := range models {
			if m.InferenceStats.Success.Count > 0 {
				relevant = append(relevant, m)
			}
		}
		return relevant
	}

	// Filter by prefix
	for _, m := range models {
		if m.InferenceStats.Success.Count == 0 {
			continue
		}

		for _, prefix := range prefixes {
			if strings.HasPrefix(m.Name, prefix) {
				relevant = append(relevant, m)
				break
			}
		}
	}

	return relevant
}

// getTritonBaseURL extracts Triton base URL from track URL
func getTritonBaseURL(port int) string {
	// Map API port to Triton HTTP port
	// 4603 (yolo-api) -> 4600 (triton-api)
	// 4613 (yolo-api-trackd) -> 4610 (triton-api-trackd)
	tritonPort := 4600
	if port == 4613 {
		tritonPort = 4610
	}
	return fmt.Sprintf("http://localhost:%d", tritonPort)
}
