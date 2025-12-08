#!/usr/bin/env python3
"""
Compare Detection Outputs Across All Tracks (A, B, C, D, E)

This script compares detection outputs from:
- Track A: PyTorch Baseline (CPU preprocessing + CPU NMS)
- Track B: TensorRT + CPU NMS (2x speedup)
- Track C: CPU Letterbox + TensorRT End2End (CPU preprocess + GPU NMS)
- Track D: DALI GPU + TensorRT End2End (GPU preprocess + GPU NMS)
- Track E: DALI GPU + YOLO + MobileCLIP (GPU preprocess + GPU NMS + embeddings)

The goal is to verify that all tracks produce equivalent detection results
despite using different preprocessing methods.

Track E also returns embeddings - the script validates embedding norms are ~1.0.

Usage:
    # Run inside container (all tracks)
    docker compose exec yolo-api python /app/tests/compare_tracks.py

    # Test specific tracks
    docker compose exec yolo-api python /app/tests/compare_tracks.py --tracks A,B,C,E

    # Test Track E embeddings specifically
    docker compose exec yolo-api python /app/tests/compare_tracks.py --tracks E,E_full

    # Use different IoU threshold
    docker compose exec yolo-api python /app/tests/compare_tracks.py --iou-threshold 0.7

    # Run from host (using curl)
    python tests/compare_tracks.py --host localhost --port 4603
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests


# Add paths for imports (works from /app or /app/tests)
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from tests.detection_comparison_utils import (
    ComparisonMetrics,
    calculate_comparison_metrics,
    parse_detections,
)


@dataclass
class TrackResult:
    """Results from a single track inference."""

    track: str
    detections: list[dict]
    inference_time_ms: float
    preprocessing: str | None = None
    nms_location: str | None = None
    error: str | None = None
    # Track E specific fields
    embedding_norm: float | None = None
    has_box_embeddings: bool = False
    num_box_embeddings: int = 0


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding quality."""

    embedding_norm: float
    embedding_valid: bool  # norm should be ~1.0 for normalized embeddings
    has_box_embeddings: bool
    num_box_embeddings: int


class TrackComparator:
    """Compare detection outputs across all tracks (A, B, C, D, E)."""

    # Track descriptions for documentation
    TRACK_DESCRIPTIONS = {
        'A': 'PyTorch Baseline (CPU preprocess + CPU NMS)',
        'B': 'TensorRT + CPU NMS (2x speedup)',
        'C': 'CPU Letterbox + TensorRT End2End (CPU preprocess + GPU NMS)',
        'D': 'DALI GPU + TensorRT End2End (GPU preprocess + GPU NMS)',
        'E': 'Track E Simple: DALI + YOLO + MobileCLIP global embedding',
        'E_full': 'Track E Full: DALI + YOLO + MobileCLIP + per-box embeddings',
        'E_detect': 'Track E Detect: DALI + YOLO only (no embeddings)',
    }

    def __init__(
        self,
        host: str = 'localhost',
        port_main: int = 4603,
        port_trackd: int = 4613,
        verbose: bool = True,
    ):
        """
        Initialize comparator with API endpoints.

        Args:
            host: API host
            port_main: Port for Track A/C/E (yolo-api)
            port_trackd: Port for Track D (yolo-api-trackd)
            verbose: Print progress messages
        """
        self.host = host
        self.port_main = port_main
        self.port_trackd = port_trackd
        self.verbose = verbose

        # Endpoint URLs for all tracks
        self.endpoints = {
            # Detection-only tracks
            'A': f'http://{host}:{port_main}/pytorch/predict/small',
            'B': f'http://{host}:{port_main}/predict/small',
            'C': f'http://{host}:{port_main}/predict/small_end2end',
            'D': f'http://{host}:{port_main}/predict/small_gpu_e2e_batch',
            # Track E variants (detection + embeddings)
            'E': f'http://{host}:{port_main}/track_e/predict',
            'E_full': f'http://{host}:{port_main}/track_e/predict_full',
            'E_detect': f'http://{host}:{port_main}/track_e/detect',
        }

        # Tracks that return embeddings
        self.embedding_tracks = {'E', 'E_full'}

        if self.verbose:
            print('Track Endpoints:')
            for track, url in self.endpoints.items():
                desc = self.TRACK_DESCRIPTIONS.get(track, '')
                print(f'  {track}: {url}')
                if desc:
                    print(f'       ({desc})')
            print()

    def _call_endpoint(self, track: str, image_path: str) -> TrackResult:
        """
        Call a track endpoint with an image.

        Args:
            track: Track identifier (A, C, D, E, E_full, E_detect)
            image_path: Path to image file

        Returns:
            TrackResult with detections, timing, and optional embeddings
        """
        url = self.endpoints.get(track)
        if not url:
            return TrackResult(
                track=track, detections=[], inference_time_ms=0, error=f'Unknown track: {track}'
            )

        try:
            with open(image_path, 'rb') as f:
                files = {'image': (Path(image_path).name, f, 'image/jpeg')}

                start = time.perf_counter()
                response = requests.post(url, files=files, timeout=60)
                elapsed = (time.perf_counter() - start) * 1000

                if response.status_code != 200:
                    return TrackResult(
                        track=track,
                        detections=[],
                        inference_time_ms=elapsed,
                        error=f'HTTP {response.status_code}: {response.text[:200]}',
                    )

                data = response.json()

                # Parse embedding info for Track E variants
                embedding_norm = data.get('embedding_norm')
                has_box_embeddings = (
                    'box_embeddings' in data and len(data.get('box_embeddings', [])) > 0
                )
                num_box_embeddings = (
                    len(data.get('box_embeddings', [])) if has_box_embeddings else 0
                )

                return TrackResult(
                    track=track,
                    detections=data.get('detections', []),
                    inference_time_ms=data.get('inference_time_ms')
                    or data.get('total_time_ms')
                    or elapsed,
                    preprocessing=data.get('preprocessing'),
                    nms_location=data.get('nms_location'),
                    embedding_norm=embedding_norm,
                    has_box_embeddings=has_box_embeddings,
                    num_box_embeddings=num_box_embeddings,
                )

        except requests.exceptions.ConnectionError as e:
            return TrackResult(
                track=track, detections=[], inference_time_ms=0, error=f'Connection error: {e}'
            )
        except Exception as e:
            return TrackResult(track=track, detections=[], inference_time_ms=0, error=str(e))

    def _validate_embedding(self, result: TrackResult) -> EmbeddingMetrics | None:
        """Validate embedding quality for Track E variants."""
        if result.embedding_norm is None:
            return None

        # Embedding norm should be close to 1.0 for L2-normalized embeddings
        embedding_valid = 0.9 <= result.embedding_norm <= 1.1

        return EmbeddingMetrics(
            embedding_norm=result.embedding_norm,
            embedding_valid=embedding_valid,
            has_box_embeddings=result.has_box_embeddings,
            num_box_embeddings=result.num_box_embeddings,
        )

    def compare_on_image(
        self, image_path: str, iou_threshold: float = 0.5, tracks: list[str] | None = None
    ) -> dict:
        """
        Compare all tracks on a single image.

        Args:
            image_path: Path to image file
            iou_threshold: IoU threshold for matching detections
            tracks: List of tracks to test (default: ["A", "B", "C", "D", "E"])

        Returns:
            Dictionary with comparison results including detection metrics and embedding info
        """
        if tracks is None:
            tracks = ['A', 'B', 'C', 'D', 'E']

        if self.verbose:
            print(f'\nProcessing: {Path(image_path).name}')

        # Run inference on all tracks
        results = {}
        embedding_metrics = {}

        for track in tracks:
            result = self._call_endpoint(track, image_path)
            results[track] = result

            # Validate embeddings for Track E variants
            if track in self.embedding_tracks:
                embedding_metrics[track] = self._validate_embedding(result)

            if self.verbose:
                if result.error:
                    print(f'  Track {track}: ERROR - {result.error[:50]}')
                else:
                    msg = f'  Track {track}: {len(result.detections)} detections ({result.inference_time_ms:.1f}ms)'
                    if result.embedding_norm is not None:
                        msg += f' [emb_norm={result.embedding_norm:.4f}]'
                    if result.has_box_embeddings:
                        msg += f' [box_emb={result.num_box_embeddings}]'
                    print(msg)

        # Use Track A as baseline if available, else first available track
        baseline_track = 'A' if 'A' in results and not results['A'].error else None

        if baseline_track is None:
            # Fallback to first available track (prefer detection-only tracks)
            for t in ['C', 'D', 'E_detect', 'E', 'E_full']:
                if t in results and not results[t].error:
                    baseline_track = t
                    break

        if baseline_track is None:
            return {
                'image': str(image_path),
                'error': 'All tracks failed',
                'results': {t: asdict(r) for t, r in results.items()},
            }

        # Parse baseline detections
        baseline_dets = parse_detections(results[baseline_track].detections)

        # Compare each track against baseline
        metrics = {}
        for track, result in results.items():
            if result.error:
                metrics[track] = None
                continue

            if track == baseline_track:
                # Baseline against itself is perfect
                metrics[track] = ComparisonMetrics(
                    num_matches=len(baseline_dets),
                    num_reference=len(baseline_dets),
                    num_test=len(baseline_dets),
                    precision=1.0,
                    recall=1.0,
                    f1_score=1.0,
                    mean_iou=1.0,
                    mean_conf_diff=0.0,
                    mean_box_diff=0.0,
                )
            else:
                test_dets = parse_detections(result.detections)
                metrics[track] = calculate_comparison_metrics(
                    baseline_dets, test_dets, iou_threshold
                )

        # Print comparison summary
        if self.verbose:
            for track, m in metrics.items():
                if m is None:
                    continue
                if track != baseline_track:
                    print(
                        f'  {track} vs {baseline_track}: F1={m.f1_score:.3f}, IoU={m.mean_iou:.3f}, '
                        f'ConfDiff={m.mean_conf_diff:.4f}, BoxDiff={m.mean_box_diff:.1f}px'
                    )

        return {
            'image': str(image_path),
            'baseline_track': baseline_track,
            'results': {
                t: {
                    'track': r.track,
                    'num_detections': len(r.detections),
                    'detections': r.detections,
                    'inference_time_ms': r.inference_time_ms,
                    'preprocessing': r.preprocessing,
                    'nms_location': r.nms_location,
                    'error': r.error,
                    'embedding_norm': r.embedding_norm,
                    'has_box_embeddings': r.has_box_embeddings,
                    'num_box_embeddings': r.num_box_embeddings,
                }
                for t, r in results.items()
            },
            'metrics': {t: asdict(m) if m else None for t, m in metrics.items()},
            'embedding_metrics': {
                t: asdict(m) if m else None for t, m in embedding_metrics.items()
            },
        }

    def compare_on_dataset(
        self,
        image_dir: str,
        iou_threshold: float = 0.5,
        max_images: int | None = None,
        tracks: list[str] | None = None,
    ) -> dict:
        """
        Compare tracks on a directory of images.

        Args:
            image_dir: Directory containing test images
            iou_threshold: IoU threshold for matching
            max_images: Maximum number of images to process
            tracks: List of tracks to test

        Returns:
            Aggregated comparison results
        """
        if tracks is None:
            tracks = ['A', 'B', 'C', 'D', 'E']

        # Find all images
        image_dir = Path(image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = sorted(
            [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]
        )

        if max_images:
            image_files = image_files[:max_images]

        print(f'\n{"=" * 80}')
        print(f'Comparing Tracks {", ".join(tracks)} on {len(image_files)} images')
        print(f'{"=" * 80}')

        results = []
        for image_path in image_files:
            result = self.compare_on_image(str(image_path), iou_threshold, tracks)
            results.append(result)

        # Aggregate metrics per track
        aggregated = {}
        for track in tracks:
            track_metrics = [
                r['metrics'].get(track)
                for r in results
                if r.get('metrics', {}).get(track) is not None
            ]

            if not track_metrics:
                aggregated[track] = None
                continue

            aggregated[track] = {
                'precision': sum(m['precision'] for m in track_metrics) / len(track_metrics),
                'recall': sum(m['recall'] for m in track_metrics) / len(track_metrics),
                'f1_score': sum(m['f1_score'] for m in track_metrics) / len(track_metrics),
                'mean_iou': sum(m['mean_iou'] for m in track_metrics) / len(track_metrics),
                'mean_conf_diff': sum(m['mean_conf_diff'] for m in track_metrics)
                / len(track_metrics),
                'mean_box_diff': sum(m['mean_box_diff'] for m in track_metrics)
                / len(track_metrics),
                'total_matches': sum(m['num_matches'] for m in track_metrics),
                'total_reference': sum(m['num_reference'] for m in track_metrics),
                'total_test': sum(m['num_test'] for m in track_metrics),
                'num_images': len(track_metrics),
            }

        # Aggregate timing
        avg_times = {}
        for track in tracks:
            times = [
                r['results'][track]['inference_time_ms']
                for r in results
                if r.get('results', {}).get(track, {}).get('error') is None
            ]
            avg_times[track] = sum(times) / len(times) if times else 0

        # Aggregate embedding metrics for Track E variants
        embedding_summary = {}
        for track in tracks:
            if track not in self.embedding_tracks:
                continue

            emb_metrics = [
                r['embedding_metrics'].get(track)
                for r in results
                if r.get('embedding_metrics', {}).get(track) is not None
            ]

            if not emb_metrics:
                embedding_summary[track] = None
                continue

            valid_count = sum(1 for m in emb_metrics if m.get('embedding_valid', False))
            embedding_summary[track] = {
                'avg_embedding_norm': sum(m['embedding_norm'] for m in emb_metrics)
                / len(emb_metrics),
                'valid_embeddings': valid_count,
                'total_images': len(emb_metrics),
                'validity_rate': valid_count / len(emb_metrics) if emb_metrics else 0,
                'has_box_embeddings': any(m.get('has_box_embeddings', False) for m in emb_metrics),
                'total_box_embeddings': sum(m.get('num_box_embeddings', 0) for m in emb_metrics),
            }

        return {
            'num_images': len(results),
            'iou_threshold': iou_threshold,
            'tracks_tested': tracks,
            'individual_results': results,
            'aggregated_metrics': aggregated,
            'average_times_ms': avg_times,
            'embedding_summary': embedding_summary,
        }


def print_summary(results: dict):
    """Print summary of comparison results."""
    print(f'\n{"=" * 80}')
    print('TRACK COMPARISON SUMMARY')
    print(f'{"=" * 80}')

    print(f'\nDataset: {results["num_images"]} images')
    print(f'IoU Threshold: {results["iou_threshold"]}')
    print(f'Tracks Tested: {", ".join(results["tracks_tested"])}')

    print(
        f'\n{"Track":<12} {"Prec":<8} {"Recall":<8} {"F1":<8} {"IoU":<8} {"ConfΔ":<10} {"BoxΔ(px)":<10} {"Time(ms)":<10}'
    )
    print('=' * 90)

    metrics = results['aggregated_metrics']
    times = results['average_times_ms']

    for track in results['tracks_tested']:
        m = metrics.get(track)
        t = times.get(track, 0)

        if m is None:
            print(
                f'Track {track:<6} {"FAILED":<8} {"-":<8} {"-":<8} {"-":<8} {"-":<10} {"-":<10} {"-":<10}'
            )
        else:
            print(
                f'Track {track:<6} '
                f'{m["precision"]:>7.3f} '
                f'{m["recall"]:>7.3f} '
                f'{m["f1_score"]:>7.3f} '
                f'{m["mean_iou"]:>7.3f} '
                f'{m["mean_conf_diff"]:>9.4f} '
                f'{m["mean_box_diff"]:>9.2f} '
                f'{t:>9.1f}'
            )

    # Embedding summary for Track E variants
    embedding_summary = results.get('embedding_summary', {})
    if embedding_summary:
        print(f'\n{"=" * 80}')
        print('EMBEDDING VALIDATION (Track E)')
        print(f'{"=" * 80}')

        for track, emb in embedding_summary.items():
            if emb is None:
                print(f'\nTrack {track}: No embedding data')
                continue

            print(f'\nTrack {track}:')
            print(f'  Avg Embedding Norm: {emb["avg_embedding_norm"]:.4f} (should be ~1.0)')
            print(
                f'  Valid Embeddings: {emb["valid_embeddings"]}/{emb["total_images"]} ({emb["validity_rate"] * 100:.1f}%)'
            )

            if emb['has_box_embeddings']:
                print(f'  Box Embeddings: {emb["total_box_embeddings"]} total across all images')

            # Status
            if emb['validity_rate'] >= 0.99:
                print('  Status: EXCELLENT - All embeddings properly normalized')
            elif emb['validity_rate'] >= 0.95:
                print('  Status: GOOD - Most embeddings properly normalized')
            else:
                print('  Status: WARNING - Some embeddings have abnormal norms')

    # Analysis
    print(f'\n{"=" * 80}')
    print('DETECTION ANALYSIS')
    print(f'{"=" * 80}')

    baseline = 'A'
    if metrics.get(baseline) is None:
        baseline = next((t for t in results['tracks_tested'] if metrics.get(t)), None)

    if baseline is None:
        print('ERROR: No valid baseline track found!')
        return

    print(f'\nBaseline: Track {baseline}')

    for track in results['tracks_tested']:
        if track == baseline:
            continue

        m = metrics.get(track)
        if m is None:
            print(f'\nTrack {track}: FAILED - No results')
            continue

        f1 = m['f1_score']
        iou = m['mean_iou']
        conf_diff = m['mean_conf_diff']
        box_diff = m['mean_box_diff']

        print(f'\nTrack {track} vs Track {baseline}:')

        # F1 Score analysis
        if f1 >= 0.99:
            print(f'  F1 Score: {f1:.3f} - EXCELLENT (≥99% match)')
        elif f1 >= 0.95:
            print(f'  F1 Score: {f1:.3f} - GOOD (≥95% match)')
        elif f1 >= 0.90:
            print(f'  F1 Score: {f1:.3f} - ACCEPTABLE (≥90% match)')
        else:
            print(f'  F1 Score: {f1:.3f} - WARNING (<90% match)')

        # IoU analysis
        if iou >= 0.95:
            print(f'  Mean IoU: {iou:.3f} - EXCELLENT box alignment')
        elif iou >= 0.85:
            print(f'  Mean IoU: {iou:.3f} - GOOD box alignment')
        else:
            print(f'  Mean IoU: {iou:.3f} - WARNING: Box misalignment detected')

        # Confidence diff analysis
        if conf_diff <= 0.02:
            print(f'  Conf Diff: {conf_diff:.4f} - EXCELLENT (<2% difference)')
        elif conf_diff <= 0.05:
            print(f'  Conf Diff: {conf_diff:.4f} - ACCEPTABLE (<5% difference)')
        else:
            print(f'  Conf Diff: {conf_diff:.4f} - WARNING (>5% difference)')

        # Box center diff analysis
        if box_diff <= 5.0:
            print(f'  Box Diff: {box_diff:.2f}px - EXCELLENT (<5px)')
        elif box_diff <= 15.0:
            print(f'  Box Diff: {box_diff:.2f}px - ACCEPTABLE (<15px)')
        else:
            print(f'  Box Diff: {box_diff:.2f}px - WARNING (>15px)')

    # Speed comparison
    print(f'\n{"=" * 80}')
    print('PERFORMANCE')
    print(f'{"=" * 80}')

    baseline_time = times.get(baseline, 1)
    for track in results['tracks_tested']:
        t = times.get(track, 0)
        if t > 0 and baseline_time > 0:
            speedup = baseline_time / t
            print(f'  Track {track}: {t:.1f}ms ({speedup:.2f}x vs Track {baseline})')


def main():
    parser = argparse.ArgumentParser(
        description='Compare detection outputs across all tracks (A, B, C, D, E)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Tracks:
  A        PyTorch Baseline (CPU preprocess + CPU NMS)
  B        TensorRT + CPU NMS (2x speedup)
  C        CPU Letterbox + TensorRT End2End (CPU preprocess + GPU NMS)
  D        DALI GPU + TensorRT End2End (GPU preprocess + GPU NMS)
  E        Track E Simple: DALI + YOLO + MobileCLIP global embedding
  E_full   Track E Full: DALI + YOLO + MobileCLIP + per-box embeddings
  E_detect Track E Detect: DALI + YOLO only (no embeddings)

Examples:
  # Compare all main tracks
  python compare_tracks.py --tracks A,B,C,D,E

  # Compare only detection tracks (faster)
  python compare_tracks.py --tracks A,B,C,D

  # Test Track E embedding variants
  python compare_tracks.py --tracks E,E_full

  # Quick test with 2 images
  python compare_tracks.py --tracks A,B,C,E --max-images 2
""",
    )
    parser.add_argument(
        '--images', type=str, default='/app/test_images', help='Directory containing test images'
    )
    parser.add_argument(
        '--host', type=str, default='localhost', help='API host (default: localhost)'
    )
    parser.add_argument(
        '--port',
        '--port-main',
        type=int,
        default=4603,
        dest='port_main',
        help='Port for all tracks (default: 4603)',
    )
    parser.add_argument(
        '--port-trackd', type=int, default=4603, help='Deprecated: Track D now uses main port'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for matching detections (default: 0.5)',
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to process (default: all)',
    )
    parser.add_argument(
        '--tracks',
        type=str,
        default='A,B,C,D,E',
        help='Comma-separated list of tracks to compare (default: A,B,C,D,E)',
    )
    parser.add_argument(
        '--output', type=str, default=None, help='Output JSON file for detailed results'
    )
    parser.add_argument('--quiet', action='store_true', help='Suppress per-image output')

    args = parser.parse_args()

    tracks = [t.strip() for t in args.tracks.split(',')]

    # Initialize comparator
    comparator = TrackComparator(
        host=args.host,
        port_main=args.port_main,
        port_trackd=args.port_trackd,
        verbose=not args.quiet,
    )

    # Run comparison
    results = comparator.compare_on_dataset(
        args.images, iou_threshold=args.iou_threshold, max_images=args.max_images, tracks=tracks
    )

    # Print summary
    print_summary(results)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f'\nDetailed results saved to: {output_path}')


if __name__ == '__main__':
    main()
