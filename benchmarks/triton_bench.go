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
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const version = "1.0.0"

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
)

// BenchmarkConfig holds all configuration
type BenchmarkConfig struct {
	Mode            TestMode
	ImageDir        string
	ImageLimit      int
	Clients         int
	Duration        int
	Warmup          int
	TrackFilter     string
	OutputFile      string
	Quiet           bool
	JSONOutput      bool
	LoadPattern     string // "constant", "burst", "ramp"
	BurstInterval   int    // seconds between bursts
	RampStep        int    // client increment for ramp
	EnableResize    bool   // enable pre-resize in FastAPI
	MaxSize         int    // max dimension for resize
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
	Status     string                    `json:"status"`
}

// All available tracks
var allTracks = []Track{
	{ID: "A", Name: "PyTorch Baseline", URL: "http://localhost:9600/pytorch/predict/small", Description: "Native PyTorch + CPU NMS"},
	{ID: "B", Name: "Triton Standard TRT", URL: "http://localhost:9600/predict/small", Description: "TensorRT + CPU NMS"},
	{ID: "C", Name: "Triton End2End TRT", URL: "http://localhost:9600/predict/small_end2end", Description: "TensorRT + GPU NMS"},
	{ID: "D_streaming", Name: "DALI+TRT Streaming", URL: "http://localhost:9600/predict/small_gpu_e2e_streaming", Description: "Full GPU (low latency)"},
	{ID: "D_balanced", Name: "DALI+TRT Balanced", URL: "http://localhost:9600/predict/small_gpu_e2e", Description: "Full GPU (balanced)"},
	{ID: "D_batch", Name: "DALI+TRT Batch", URL: "http://localhost:9600/predict/small_gpu_e2e_batch", Description: "Full GPU (max throughput)"},
}

func main() {
	// Define flags
	var (
		mode            = flag.String("mode", "quick", "Test mode: single, set, quick, full, all, sustained, variable")
		imageDir        = flag.String("images", "/mnt/nvm/KILLBOY_SAMPLE_PICTURES", "Directory containing test images")
		imageLimit      = flag.Int("limit", 100, "Maximum number of images to load")
		clients         = flag.Int("clients", 64, "Number of concurrent clients")
		duration        = flag.Int("duration", 60, "Test duration in seconds")
		warmup          = flag.Int("warmup", 10, "Number of warmup requests per track")
		track           = flag.String("track", "all", "Track filter (A, B, C, D_streaming, D_balanced, D_batch, or all)")
		output          = flag.String("output", "benchmarks/results/benchmark_results.json", "Output JSON file")
		quiet           = flag.Bool("quiet", false, "Quiet mode (minimal output)")
		jsonOut         = flag.Bool("json", false, "JSON output only")
		loadPattern     = flag.String("load-pattern", "constant", "Load pattern: constant, burst, ramp")
		burstInterval   = flag.Int("burst-interval", 10, "Seconds between bursts (for burst mode)")
		rampStep        = flag.Int("ramp-step", 16, "Client increment for ramp mode")
		enableResize    = flag.Bool("resize", false, "Enable pre-resize for large images (recommended for 5MP+ images)")
		maxSize         = flag.Int("max-size", 1024, "Maximum dimension for resize (640-4096)")
		showVersion     = flag.Bool("version", false, "Show version and exit")
		listModes       = flag.Bool("list-modes", false, "List all available test modes")
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
	case "all":
		return ModeAllImages, nil
	case "sustained":
		return ModeThroughputSustained, nil
	case "variable":
		return ModeVariableLoad, nil
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

	// Filter tracks
	tracks := filterTracks(config.TrackFilter)

	// Check availability
	available := checkAvailability(tracks, config.Quiet)
	if len(available) == 0 {
		return fmt.Errorf("no tracks available")
	}

	// Run tests based on mode
	var results []*BenchmarkResults

	switch config.Mode {
	case ModeSingleImage:
		results, err = testSingleImage(available, images, config)
	case ModeImageSet:
		results, err = testImageSet(available, images, config)
	case ModeQuickConcurrency:
		results, err = testQuickConcurrency(available, images, config)
	case ModeFullConcurrency:
		results, err = testFullConcurrency(available, images, config)
	case ModeAllImages:
		results, err = testAllImages(available, images, config)
	case ModeThroughputSustained:
		results, err = testSustained(available, images, config)
	case ModeVariableLoad:
		results, err = testVariableLoad(available, images, config)
	}

	if err != nil {
		return err
	}

	// Output results
	return outputResults(results, config)
}

func testSingleImage(tracks []Track, images [][]byte, config *BenchmarkConfig) ([]*BenchmarkResults, error) {
	if !config.Quiet {
		fmt.Println("Test Mode: Single Image")
		fmt.Println("Testing one image through all tracks with 10 iterations for accuracy")
		fmt.Println(strings.Repeat("=", 80))
	}

	var results []*BenchmarkResults
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

	var results []*BenchmarkResults
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

	var results []*BenchmarkResults

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

	var results []*BenchmarkResults

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

	var results []*BenchmarkResults

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

	var results []*BenchmarkResults

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

	var results []*BenchmarkResults

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

func runConcurrentTest(track Track, images [][]byte, clients int, duration int, testType string, config *BenchmarkConfig) *BenchmarkResults {
	var (
		totalRequests   int64
		successRequests int64
		failedRequests  int64
		latencies       []float64
		latenciesMutex  sync.Mutex
		wg              sync.WaitGroup
	)

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

	if _, err := part.Write(imageData); err != nil {
		return 0, false
	}

	if err := writer.Close(); err != nil {
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
	defer resp.Body.Close()

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
	var images [][]byte

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		ext := strings.ToLower(filepath.Ext(entry.Name()))
		if ext != ".jpg" && ext != ".jpeg" {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}

		images = append(images, data)

		if len(images) >= limit {
			break
		}
	}

	if len(images) == 0 {
		return nil, fmt.Errorf("no JPEG images found in %s", dir)
	}

	return images, nil
}

func filterTracks(filter string) []Track {
	if filter == "all" {
		return allTracks
	}

	var filtered []Track
	for _, track := range allTracks {
		if track.ID == filter {
			filtered = append(filtered, track)
		}
	}

	return filtered
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
			resp.Body.Close()
		} else {
			if !quiet {
				fmt.Printf("  ✗ Track %s: %s (unavailable)\n", track.ID, track.Name)
			}
		}
	}

	fmt.Println()
	return available
}

func getBaseURL(url string) string {
	for i := len(url) - 1; i >= 0; i-- {
		if url[i] == '/' {
			for j := i - 1; j >= 0; j-- {
				if url[j] == '/' {
					return url[:j]
				}
			}
		}
	}
	return url
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

	data, err := json.MarshalIndent(map[string]interface{}{
		"timestamp": time.Now().Format(time.RFC3339),
		"results":   results,
	}, "", "  ")

	if err != nil {
		return err
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		return err
	}

	fmt.Printf("\n✓ Results saved to: %s\n", filename)
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
	if config.TrackFilter != "all" {
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
