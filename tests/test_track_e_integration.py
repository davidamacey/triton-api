#!/usr/bin/env python3
"""
Track E: Integration Test Suite

Comprehensive end-to-end testing for the visual search pipeline.

Tests:
1. Index creation
2. Image ingestion (single + batch)
3. Image-to-image search
4. Object-to-object search
5. Text-to-image search
6. Index statistics
7. Performance benchmarking

Run from: yolo-api container
    docker compose exec yolo-api python /app/scripts/track_e/test_integration.py

Requirements:
- Triton server running with Track E models loaded
- OpenSearch service running
- Test images in /app/test_images
"""

import asyncio
import sys
import time
from pathlib import Path

import httpx
import numpy as np


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TrackEIntegrationTest:
    """
    Integration test suite for Track E visual search.
    """

    def __init__(
        self,
        api_url: str = 'http://localhost:9600',
        test_images_dir: Path = Path('/app/test_images'),
    ):
        """
        Initialize test suite.

        Args:
            api_url: FastAPI service URL
            test_images_dir: Directory containing test images
        """
        self.api_url = api_url
        self.test_images_dir = test_images_dir
        self.client = httpx.AsyncClient(timeout=60.0)
        self.ingested_images = []

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    def print_header(self, text: str):
        """Print formatted test header."""
        print('\n' + '=' * 80)
        print(f'TEST: {text}')
        print('=' * 80)

    def print_result(self, test_name: str, passed: bool, message: str = ''):
        """Print test result."""
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f'{status}: {test_name}')
        if message:
            print(f'  → {message}')

    async def test_health_check(self) -> bool:
        """Test API health endpoint."""
        self.print_header('Health Check')

        try:
            response = await self.client.get(f'{self.api_url}/health')
            passed = response.status_code == 200

            self.print_result('API Health Check', passed, f'Status: {response.status_code}')

            return passed

        except Exception as e:
            self.print_result('API Health Check', False, str(e))
            return False

    async def test_create_index(self, force_recreate: bool = True) -> bool:
        """Test index creation."""
        self.print_header('Index Creation')

        try:
            response = await self.client.post(
                f'{self.api_url}/track_e/index/create', params={'force_recreate': force_recreate}
            )

            passed = response.status_code == 200
            data = response.json()

            self.print_result('Create Index', passed, f'Message: {data.get("message", "N/A")}')

            return passed

        except Exception as e:
            self.print_result('Create Index', False, str(e))
            return False

    async def test_ingest_single_image(self, image_path: Path) -> bool:
        """Test single image ingestion."""
        self.print_header(f'Single Image Ingestion: {image_path.name}')

        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/jpeg')}
                params = {
                    'image_id': f'test_{image_path.stem}',
                    'metadata': '{"source": "integration_test"}',
                }

                response = await self.client.post(
                    f'{self.api_url}/track_e/ingest', files=files, params=params
                )

            passed = response.status_code == 200
            data = response.json()

            if passed:
                self.ingested_images.append(data['image_id'])

            self.print_result(
                'Ingest Single Image',
                passed,
                f'ID: {data.get("image_id", "N/A")}, '
                f'Detections: {data.get("num_detections", 0)}, '
                f'Norm: {data.get("global_embedding_norm", 0):.4f}',
            )

            return passed

        except Exception as e:
            self.print_result('Ingest Single Image', False, str(e))
            return False

    async def test_ingest_multiple_images(self, max_images: int = 5) -> bool:
        """Test batch image ingestion."""
        self.print_header(f'Batch Ingestion ({max_images} images)')

        # Find test images
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(self.test_images_dir.glob(f'*{ext}'))
            if len(image_files) >= max_images:
                break

        image_files = image_files[:max_images]

        if not image_files:
            self.print_result('Batch Ingestion', False, 'No test images found')
            return False

        try:
            success_count = 0

            for img_path in image_files:
                with open(img_path, 'rb') as f:
                    files = {'file': (img_path.name, f, 'image/jpeg')}
                    params = {'metadata': '{"source": "batch_test"}'}

                    response = await self.client.post(
                        f'{self.api_url}/track_e/ingest', files=files, params=params
                    )

                    if response.status_code == 200:
                        data = response.json()
                        self.ingested_images.append(data['image_id'])
                        success_count += 1

            passed = success_count == len(image_files)

            self.print_result(
                'Batch Ingestion',
                passed,
                f'{success_count}/{len(image_files)} images ingested successfully',
            )

            return passed

        except Exception as e:
            self.print_result('Batch Ingestion', False, str(e))
            return False

    async def test_image_to_image_search(self, query_image: Path) -> bool:
        """Test image-to-image search."""
        self.print_header(f'Image-to-Image Search: {query_image.name}')

        try:
            with open(query_image, 'rb') as f:
                files = {'file': (query_image.name, f, 'image/jpeg')}
                params = {'top_k': 5}

                response = await self.client.post(
                    f'{self.api_url}/track_e/search/image', files=files, params=params
                )

            passed = response.status_code == 200
            data = response.json()

            if passed:
                print(f'\n  Results ({data.get("total_results", 0)} matches):')
                for i, result in enumerate(data.get('results', [])[:3], 1):
                    print(f'    {i}. {result["image_id"]} - Score: {result["score"]:.4f}')

                print(f'  Search Time: {data.get("search_time_ms", 0):.2f}ms')

            self.print_result(
                'Image-to-Image Search',
                passed,
                f'Found {data.get("total_results", 0)} results in {data.get("search_time_ms", 0):.2f}ms',
            )

            return passed

        except Exception as e:
            self.print_result('Image-to-Image Search', False, str(e))
            return False

    async def test_text_to_image_search(self, query_text: str) -> bool:
        """Test text-to-image search."""
        self.print_header(f"Text-to-Image Search: '{query_text}'")

        try:
            response = await self.client.post(
                f'{self.api_url}/track_e/search/text',
                json={'query_text': query_text},
                params={'top_k': 5},
            )

            passed = response.status_code == 200
            data = response.json()

            if passed:
                print(f'\n  Results ({data.get("total_results", 0)} matches):')
                for i, result in enumerate(data.get('results', [])[:3], 1):
                    print(f'    {i}. {result["image_id"]} - Score: {result["score"]:.4f}')

                print(f'  Search Time: {data.get("search_time_ms", 0):.2f}ms')

            self.print_result(
                'Text-to-Image Search',
                passed,
                f'Found {data.get("total_results", 0)} results in {data.get("search_time_ms", 0):.2f}ms',
            )

            return passed

        except Exception as e:
            self.print_result('Text-to-Image Search', False, str(e))
            return False

    async def test_object_search(self, query_image: Path) -> bool:
        """Test object-to-object search."""
        self.print_header(f'Object-to-Object Search: {query_image.name}')

        try:
            with open(query_image, 'rb') as f:
                files = {'file': (query_image.name, f, 'image/jpeg')}
                params = {'top_k': 5}

                response = await self.client.post(
                    f'{self.api_url}/track_e/search/object', files=files, params=params
                )

            passed = response.status_code == 200
            data = response.json()

            if passed:
                print(f'\n  Results ({data.get("total_results", 0)} matches):')
                for i, result in enumerate(data.get('results', [])[:3], 1):
                    matched_objs = len(result.get('matched_objects', []))
                    print(
                        f'    {i}. {result["image_id"]} - '
                        f'Score: {result["score"]:.4f}, '
                        f'Objects: {matched_objs}'
                    )

                print(f'  Search Time: {data.get("search_time_ms", 0):.2f}ms')

            self.print_result(
                'Object-to-Object Search',
                passed,
                f'Found {data.get("total_results", 0)} results in {data.get("search_time_ms", 0):.2f}ms',
            )

            return passed

        except Exception as e:
            self.print_result('Object-to-Object Search', False, str(e))
            return False

    async def test_index_stats(self) -> bool:
        """Test index statistics retrieval."""
        self.print_header('Index Statistics')

        try:
            response = await self.client.get(f'{self.api_url}/track_e/index/stats')

            passed = response.status_code == 200
            data = response.json()

            self.print_result(
                'Index Statistics',
                passed,
                f'Documents: {data.get("total_documents", 0)}, '
                f'Size: {data.get("index_size_mb", 0):.2f} MB',
            )

            return passed

        except Exception as e:
            self.print_result('Index Statistics', False, str(e))
            return False

    async def test_performance_benchmark(self, num_queries: int = 10) -> bool:
        """Test search performance with multiple queries."""
        self.print_header(f'Performance Benchmark ({num_queries} queries)')

        # Find query image
        query_images = list(self.test_images_dir.glob('*.jpg'))
        if not query_images:
            self.print_result('Performance Benchmark', False, 'No query images found')
            return False

        query_image = query_images[0]

        try:
            latencies = []

            for i in range(num_queries):
                start_time = time.time()

                with open(query_image, 'rb') as f:
                    files = {'file': (query_image.name, f, 'image/jpeg')}
                    params = {'top_k': 10}

                    response = await self.client.post(
                        f'{self.api_url}/track_e/search/image', files=files, params=params
                    )

                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)

                if response.status_code != 200:
                    self.print_result('Performance Benchmark', False, f'Query {i + 1} failed')
                    return False

            # Calculate statistics
            avg_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)

            print(f'\n  Latency Statistics ({num_queries} queries):')
            print(f'    Average: {avg_latency:.2f}ms')
            print(f'    P50:     {p50_latency:.2f}ms')
            print(f'    P95:     {p95_latency:.2f}ms')
            print(f'    P99:     {p99_latency:.2f}ms')
            print(f'    Min:     {min_latency:.2f}ms')
            print(f'    Max:     {max_latency:.2f}ms')

            passed = avg_latency < 500  # Should be under 500ms on average

            self.print_result(
                'Performance Benchmark',
                passed,
                f'Average latency: {avg_latency:.2f}ms (target: <500ms)',
            )

            return passed

        except Exception as e:
            self.print_result('Performance Benchmark', False, str(e))
            return False

    async def run_all_tests(self):
        """Run complete integration test suite."""
        print('\n' + '=' * 80)
        print('TRACK E: INTEGRATION TEST SUITE')
        print('=' * 80)
        print(f'API URL: {self.api_url}')
        print(f'Test Images: {self.test_images_dir}')

        results = {}

        # Find test images
        test_images = list(self.test_images_dir.glob('*.jpg'))
        if not test_images:
            print('\n✗ ERROR: No test images found!')
            return None

        query_image = test_images[0]

        # Run tests in sequence
        results['health_check'] = await self.test_health_check()
        results['create_index'] = await self.test_create_index(force_recreate=True)
        results['ingest_single'] = await self.test_ingest_single_image(query_image)
        results['ingest_batch'] = await self.test_ingest_multiple_images(max_images=5)

        # Wait for indexing to complete
        await asyncio.sleep(2)

        results['image_search'] = await self.test_image_to_image_search(query_image)
        results['text_search'] = await self.test_text_to_image_search('a person walking')
        results['object_search'] = await self.test_object_search(query_image)
        results['index_stats'] = await self.test_index_stats()
        results['performance'] = await self.test_performance_benchmark(num_queries=10)

        # Summary
        print('\n' + '=' * 80)
        print('TEST SUMMARY')
        print('=' * 80)

        total_tests = len(results)
        passed_tests = sum(1 for v in results.values() if v)

        for test_name, passed in results.items():
            status = '✓ PASS' if passed else '✗ FAIL'
            print(f'{status}: {test_name}')

        print('\n' + '-' * 80)
        print(f'Total: {passed_tests}/{total_tests} tests passed')
        print('=' * 80)

        return passed_tests == total_tests


async def main():
    import argparse

    parser = argparse.ArgumentParser(description='Track E Integration Test Suite')
    parser.add_argument(
        '--api_url', type=str, default='http://localhost:9600', help='FastAPI service URL'
    )
    parser.add_argument(
        '--test_images',
        type=str,
        default='/app/test_images',
        help='Directory containing test images',
    )

    args = parser.parse_args()

    # Validate test images directory
    test_images_dir = Path(args.test_images)
    if not test_images_dir.exists():
        print(f'✗ ERROR: Test images directory not found: {test_images_dir}')
        sys.exit(1)

    # Run tests
    test_suite = TrackEIntegrationTest(api_url=args.api_url, test_images_dir=test_images_dir)

    try:
        all_passed = await test_suite.run_all_tests()
        sys.exit(0 if all_passed else 1)

    finally:
        await test_suite.close()


if __name__ == '__main__':
    asyncio.run(main())
