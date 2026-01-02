#!/usr/bin/env python3
"""
Aggregate and compare isolated track benchmark results.

Reads JSON results from isolated benchmarks and generates:
1. Comparison table (console output)
2. Speedup calculations (vs Track A baseline)
3. Statistical summary (JSON)
4. CSV export for external analysis
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def load_results(input_dir: str) -> dict[str, Any]:
    """Load all benchmark result files from directory."""
    results = {}
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f'Error: Directory not found: {input_dir}', file=sys.stderr)
        return results

    # Find all benchmark JSON files
    for filepath in sorted(input_path.glob('benchmark_*_*.json')):
        try:
            with open(filepath) as f:
                data = json.load(f)

            # Extract track from filename: benchmark_D_batch_20241208_143045.json
            parts = filepath.stem.split('_')
            # Handle track names with underscores (e.g., D_batch, E_full)
            # Format: benchmark_<track>_<timestamp>
            track_name = '_'.join(parts[1:-2])

            # The benchmark tool saves results in a specific format
            if 'results' in data and isinstance(data['results'], list):
                for result in data['results']:
                    track_id = result.get('track_id', track_name)
                    # Skip matrix results (contain @)
                    if '@' in track_id:
                        continue
                    if track_id not in results:
                        results[track_id] = result
                        results[track_id]['_source_file'] = str(filepath)
            elif 'throughput_rps' in data:
                # Direct result format
                results[track_name] = data
                results[track_name]['_source_file'] = str(filepath)

        except json.JSONDecodeError as e:
            print(f'Warning: Invalid JSON in {filepath}: {e}', file=sys.stderr)
        except Exception as e:
            print(f'Warning: Could not load {filepath}: {e}', file=sys.stderr)

    return results


def calculate_speedup(results: dict[str, Any], baseline: str = 'A') -> dict[str, float]:
    """Calculate speedup relative to baseline track."""
    speedups = {}

    if baseline not in results:
        print(f"Warning: Baseline track '{baseline}' not found in results", file=sys.stderr)
        if results:
            min_rps = min(r.get('throughput_rps', r.get('Throughput', 0)) for r in results.values())
            baseline_rps = min_rps if min_rps > 0 else 1.0
        else:
            return speedups
    else:
        baseline_rps = results[baseline].get(
            'throughput_rps', results[baseline].get('Throughput', 1.0)
        )

    if baseline_rps <= 0:
        baseline_rps = 1.0

    for track_id, result in results.items():
        track_rps = result.get('throughput_rps', result.get('Throughput', 0.0))
        speedups[track_id] = track_rps / baseline_rps

    return speedups


def generate_summary(results: dict[str, Any], baseline: str = 'A') -> dict[str, Any]:
    """Generate comprehensive summary of benchmark results."""
    speedups = calculate_speedup(results, baseline)

    summary = {
        'generated_at': datetime.now().isoformat(),
        'baseline_track': baseline,
        'track_count': len(results),
        'tracks': {},
    }

    # Sort tracks by throughput (descending)
    def get_throughput(item):
        return item[1].get('throughput_rps', item[1].get('Throughput', 0))

    sorted_tracks = sorted(results.items(), key=get_throughput, reverse=True)

    for track_id, result in sorted_tracks:
        total = max(result.get('total_requests', result.get('TotalRequests', 1)), 1)
        success = result.get('success_requests', result.get('SuccessRequests', 0))

        summary['tracks'][track_id] = {
            'throughput_rps': round(result.get('throughput_rps', result.get('Throughput', 0)), 2),
            'mean_latency_ms': round(
                result.get('mean_latency_ms', result.get('MeanLatency', 0)), 2
            ),
            'p50_latency_ms': round(
                result.get('median_latency_ms', result.get('MedianLatency', 0)), 2
            ),
            'p95_latency_ms': round(result.get('p95_latency_ms', result.get('P95Latency', 0)), 2),
            'p99_latency_ms': round(result.get('p99_latency_ms', result.get('P99Latency', 0)), 2),
            'min_latency_ms': round(result.get('min_latency_ms', result.get('MinLatency', 0)), 2),
            'max_latency_ms': round(result.get('max_latency_ms', result.get('MaxLatency', 0)), 2),
            'total_requests': total,
            'success_requests': success,
            'failed_requests': result.get('failed_requests', result.get('FailedRequests', 0)),
            'success_rate_pct': round((success / total) * 100, 2),
            'speedup_vs_baseline': round(speedups.get(track_id, 0), 2),
            'duration_sec': result.get('total_duration_sec', result.get('TotalDuration', 0)),
        }

    # Add ranking
    for i, (track_id, _) in enumerate(sorted_tracks, 1):
        summary['tracks'][track_id]['rank'] = i

    return summary


def print_comparison_table(results: dict[str, Any], speedups: dict[str, float]):
    """Print formatted comparison table to console."""
    print()
    print('=' * 130)
    print('ISOLATED TRACK BENCHMARK COMPARISON')
    print('=' * 130)
    print()

    # Header
    print(
        f'{"Rank":<5} {"Track":<14} {"Throughput":>12} {"Mean (ms)":>11} {"P50 (ms)":>10} '
        f'{"P95 (ms)":>10} {"P99 (ms)":>10} {"Success":>10} {"Speedup":>10}'
    )
    print('-' * 130)

    # Sort by throughput (descending)
    def get_throughput(item):
        return item[1].get('throughput_rps', item[1].get('Throughput', 0))

    sorted_results = sorted(results.items(), key=get_throughput, reverse=True)

    for rank, (track_id, result) in enumerate(sorted_results, 1):
        throughput = result.get('throughput_rps', result.get('Throughput', 0))
        mean_lat = result.get('mean_latency_ms', result.get('MeanLatency', 0))
        p50_lat = result.get('median_latency_ms', result.get('MedianLatency', 0))
        p95_lat = result.get('p95_latency_ms', result.get('P95Latency', 0))
        p99_lat = result.get('p99_latency_ms', result.get('P99Latency', 0))
        success = result.get('success_requests', result.get('SuccessRequests', 0))
        total = max(result.get('total_requests', result.get('TotalRequests', 1)), 1)
        success_rate = (success / total) * 100
        speedup = speedups.get(track_id, 0)

        print(
            f'{rank:<5} {track_id:<14} {throughput:>10.1f} rps {mean_lat:>9.2f} {p50_lat:>10.2f} '
            f'{p95_lat:>10.2f} {p99_lat:>10.2f} {success_rate:>9.1f}% {speedup:>9.2f}x'
        )

    print('=' * 130)
    print()

    # Print summary insights
    if sorted_results:
        fastest_track, fastest_result = sorted_results[0]
        slowest_track, slowest_result = sorted_results[-1]

        fastest_tp = fastest_result.get('throughput_rps', fastest_result.get('Throughput', 0))
        slowest_tp = slowest_result.get('throughput_rps', slowest_result.get('Throughput', 0))

        print('Summary:')
        print(f'  Fastest track: {fastest_track} ({fastest_tp:.1f} RPS)')
        print(f'  Slowest track: {slowest_track} ({slowest_tp:.1f} RPS)')

        if len(sorted_results) >= 2:
            max_speedup = max(speedups.values()) if speedups else 1
            print(f'  Max speedup: {max_speedup:.2f}x (vs baseline)')
        print()


def export_csv(results: dict[str, Any], speedups: dict[str, float], output_path: str):
    """Export results to CSV for external analysis."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                'Rank',
                'Track',
                'Throughput_RPS',
                'Mean_Latency_ms',
                'P50_Latency_ms',
                'P95_Latency_ms',
                'P99_Latency_ms',
                'Min_Latency_ms',
                'Max_Latency_ms',
                'Total_Requests',
                'Success_Requests',
                'Failed_Requests',
                'Success_Rate_Pct',
                'Speedup_vs_Baseline',
                'Duration_sec',
            ]
        )

        # Sort by throughput
        def get_throughput(item):
            return item[1].get('throughput_rps', item[1].get('Throughput', 0))

        sorted_results = sorted(results.items(), key=get_throughput, reverse=True)

        for rank, (track_id, result) in enumerate(sorted_results, 1):
            total = max(result.get('total_requests', result.get('TotalRequests', 1)), 1)
            success = result.get('success_requests', result.get('SuccessRequests', 0))

            writer.writerow(
                [
                    rank,
                    track_id,
                    round(result.get('throughput_rps', result.get('Throughput', 0)), 2),
                    round(result.get('mean_latency_ms', result.get('MeanLatency', 0)), 2),
                    round(result.get('median_latency_ms', result.get('MedianLatency', 0)), 2),
                    round(result.get('p95_latency_ms', result.get('P95Latency', 0)), 2),
                    round(result.get('p99_latency_ms', result.get('P99Latency', 0)), 2),
                    round(result.get('min_latency_ms', result.get('MinLatency', 0)), 2),
                    round(result.get('max_latency_ms', result.get('MaxLatency', 0)), 2),
                    total,
                    success,
                    result.get('failed_requests', result.get('FailedRequests', 0)),
                    round((success / total) * 100, 2),
                    round(speedups.get(track_id, 0), 2),
                    result.get('total_duration_sec', result.get('TotalDuration', 0)),
                ]
            )

    print(f'CSV exported to: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate and compare isolated track benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input benchmarks/isolated
  %(prog)s --input benchmarks/isolated --csv comparison.csv
  %(prog)s --input benchmarks/isolated --baseline B
        """,
    )
    parser.add_argument(
        '--input',
        '-i',
        default='isolated',
        help='Directory containing benchmark JSON files (default: isolated)',
    )
    parser.add_argument('--output', '-o', help='Output JSON file for summary (optional)')
    parser.add_argument('--csv', help='Export results to CSV file (optional)')
    parser.add_argument(
        '--baseline', '-b', default='A', help='Baseline track for speedup calculation (default: A)'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true', help='Suppress table output (only generate files)'
    )

    args = parser.parse_args()

    # Handle relative paths from benchmarks directory
    script_dir = Path(__file__).parent
    input_path = args.input
    if not Path(input_path).is_absolute():
        input_path = script_dir / input_path

    # Load results
    results = load_results(str(input_path))

    if not results:
        print(f'No benchmark results found in: {input_path}', file=sys.stderr)
        print('\nTo run isolated benchmarks:', file=sys.stderr)
        print('  make bench-isolated-all', file=sys.stderr)
        print('  # or', file=sys.stderr)
        print('  make bench-isolated-track TRACK=B', file=sys.stderr)
        sys.exit(1)

    print(f'Loaded {len(results)} track results from {input_path}')

    # Calculate speedups
    speedups = calculate_speedup(results, args.baseline)

    # Print comparison table
    if not args.quiet:
        print_comparison_table(results, speedups)

    # Generate and save summary
    if args.output:
        summary = generate_summary(results, args.baseline)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'Summary saved to: {args.output}')

    # Export CSV
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        export_csv(results, speedups, args.csv)


if __name__ == '__main__':
    main()
