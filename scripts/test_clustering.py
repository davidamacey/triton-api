#!/usr/bin/env python3
"""
Test script for FAISS IVF clustering with killboy sample images.

This script:
1. Creates OpenSearch indexes
2. Ingests images from killboy sample folder
3. Trains FAISS clustering
4. Creates mosaic visualizations of clusters
5. Tests similarity search

Usage:
    python scripts/test_clustering.py --images /mnt/nvm/KILLBOY_SAMPLE_PICTURES --limit 200
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import httpx
from PIL import Image


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API base URL
API_BASE = 'http://localhost:4603'


async def check_health():
    """Check API and OpenSearch health."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(f'{API_BASE}/health')
            health = resp.json()
            logger.info(f'API Status: {health["status"]}')
            logger.info(f'GPU: {health["performance"]["gpu_name"]}')
            return True
        except Exception as e:
            logger.error(f'Health check failed: {e}')
            return False


async def create_indexes(force_recreate: bool = False):
    """Create OpenSearch indexes."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f'{API_BASE}/track_e/index/create', params={'force_recreate': force_recreate}
        )
        result = resp.json()
        logger.info(f'Index creation: {result}')
        return result


async def ingest_image(client: httpx.AsyncClient, image_path: Path, image_id: str):
    """Ingest a single image."""
    with open(image_path, 'rb') as f:
        files = {'file': (image_path.name, f, 'image/jpeg')}
        params = {
            'image_id': image_id,
            'image_path': str(image_path),
        }
        resp = await client.post(
            f'{API_BASE}/track_e/ingest',
            files=files,
            params=params,
            timeout=60,
        )
        return resp.json()


async def ingest_images(image_dir: Path, limit: int | None = None, concurrency: int = 64):
    """Ingest all images from directory using concurrent requests."""
    images = sorted(image_dir.glob('*.jpg'))[:limit] if limit else sorted(image_dir.glob('*.jpg'))
    logger.info(f'Found {len(images)} images to ingest with {concurrency} concurrent workers')

    results = {
        'success': 0,
        'failed': 0,
        'vehicles': 0,
        'people': 0,
    }
    results_lock = asyncio.Lock()
    progress_count = 0
    progress_lock = asyncio.Lock()

    start_time = time.time()

    semaphore = asyncio.Semaphore(concurrency)

    async def ingest_with_semaphore(client: httpx.AsyncClient, image_path: Path):
        nonlocal progress_count
        async with semaphore:
            try:
                image_id = f'killboy_{image_path.stem}'
                result = await ingest_image(client, image_path, image_id)

                async with results_lock:
                    if result.get('status') == 'success':
                        results['success'] += 1
                        indexed = result.get('indexed', {})
                        results['vehicles'] += indexed.get('vehicles', 0)
                        results['people'] += indexed.get('people', 0)
                    else:
                        results['failed'] += 1

                async with progress_lock:
                    progress_count += 1
                    if progress_count % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = progress_count / elapsed
                        logger.info(f'Progress: {progress_count}/{len(images)} ({rate:.1f} img/s)')

            except Exception as e:
                async with results_lock:
                    results['failed'] += 1
                logger.error(f'Error ingesting {image_path.name}: {e}')

    # Use connection pooling with limits
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency + 10)
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:
        tasks = [ingest_with_semaphore(client, img) for img in images]
        await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    rate = len(images) / elapsed if elapsed > 0 else 0
    logger.info(f'Ingestion complete in {elapsed:.1f}s ({rate:.1f} img/s)')
    logger.info(f'Results: {results}')
    return results


async def get_index_stats():
    """Get index statistics."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f'{API_BASE}/track_e/index/stats')
        return resp.json()


async def train_clusters(index_name: str, n_clusters: int | None = None):
    """Train FAISS clustering for an index."""
    async with httpx.AsyncClient(timeout=300) as client:
        params = {}
        if n_clusters:
            params['n_clusters'] = n_clusters

        logger.info(f'Training clusters for {index_name}...')
        resp = await client.post(
            f'{API_BASE}/track_e/clusters/train/{index_name}',
            params=params,
            timeout=300,
        )
        result = resp.json()
        logger.info(f'Training result: {result}')
        return result


async def get_cluster_stats(index_name: str):
    """Get cluster statistics."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f'{API_BASE}/track_e/clusters/stats/{index_name}')
        return resp.json()


async def get_cluster_members(index_name: str, cluster_id: int, size: int = 50):
    """Get members of a cluster."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f'{API_BASE}/track_e/clusters/{index_name}/{cluster_id}', params={'size': size}
        )
        return resp.json()


async def list_albums(min_size: int = 3):
    """List auto-generated albums."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f'{API_BASE}/track_e/albums', params={'min_size': min_size})
        return resp.json()


async def search_similar(image_path: Path, top_k: int = 10):
    """Search for similar images."""
    async with httpx.AsyncClient(timeout=60) as client:
        with open(image_path, 'rb') as f:
            files = {'image': (image_path.name, f, 'image/jpeg')}
            resp = await client.post(
                f'{API_BASE}/track_e/search/image',
                files=files,
                params={'top_k': top_k},
            )
            return resp.json()


def create_mosaic(
    image_paths: list[str], output_path: Path, title: str, cols: int = 5, thumb_size: int = 200
):
    """Create a mosaic image from a list of image paths."""
    # Filter to existing files
    valid_paths = [p for p in image_paths if Path(p).exists()]

    if not valid_paths:
        logger.warning(f'No valid images for mosaic: {title}')
        return None

    n_images = len(valid_paths)
    rows = (n_images + cols - 1) // cols

    # Create canvas
    canvas_width = cols * thumb_size
    canvas_height = rows * thumb_size
    canvas = Image.new('RGB', (canvas_width, canvas_height), (40, 40, 40))

    for i, img_path in enumerate(valid_paths):
        try:
            row = i // cols
            col = i % cols

            img = Image.open(img_path)
            img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)

            # Center in cell
            x = col * thumb_size + (thumb_size - img.width) // 2
            y = row * thumb_size + (thumb_size - img.height) // 2

            canvas.paste(img, (x, y))
        except Exception as e:
            logger.warning(f'Error loading {img_path}: {e}')

    # Add title
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
    except Exception:
        font = ImageFont.load_default()

    draw.rectangle([(0, 0), (canvas_width, 30)], fill=(0, 0, 0))
    draw.text((10, 5), title, fill=(255, 255, 255), font=font)

    canvas.save(output_path, quality=90)
    logger.info(f'Saved mosaic: {output_path}')
    return output_path


async def create_cluster_mosaics(
    output_dir: Path,
    index_name: str = 'global',
    max_clusters: int = 10,
    images_per_cluster: int = 20,
):
    """Create mosaic images for top clusters."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get albums
    albums = await list_albums(min_size=3)
    if albums.get('status') != 'success':
        logger.error(f'Failed to list albums: {albums}')
        return

    cluster_list = albums.get('albums', [])[:max_clusters]
    logger.info(f'Creating mosaics for {len(cluster_list)} clusters')

    for cluster in cluster_list:
        cluster_id = cluster['cluster_id']
        count = cluster['count']
        avg_dist = cluster.get('avg_distance', 0)

        # Get cluster members
        members = await get_cluster_members(index_name, cluster_id, size=images_per_cluster)
        if members.get('status') != 'success':
            continue

        image_paths = [m.get('image_path', '') for m in members.get('members', [])]

        title = f'Cluster {cluster_id} ({count} images, avg_dist={avg_dist:.3f})'
        output_path = output_dir / f'cluster_{cluster_id:04d}.jpg'

        create_mosaic(image_paths, output_path, title, cols=5, thumb_size=200)


async def create_search_mosaic(query_image: Path, output_dir: Path, top_k: int = 20):
    """Create mosaic showing query image and similar results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search for similar
    results = await search_similar(query_image, top_k=top_k)

    if results.get('status') != 'success':
        logger.error(f'Search failed: {results}')
        return None

    # Build image list: query first, then results
    image_paths = [str(query_image)]
    image_paths.extend(r.get('image_path', '') for r in results.get('results', []))

    title = f'Query: {query_image.name} → {len(results.get("results", []))} similar'
    output_path = output_dir / f'search_{query_image.stem}.jpg'

    create_mosaic(image_paths, output_path, title, cols=5, thumb_size=200)
    return output_path


async def create_vehicle_cluster_mosaics(output_dir: Path, max_clusters: int = 10):
    """Create mosaic images for vehicle clusters."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get cluster stats for vehicles index
    stats = await get_cluster_stats('vehicles')
    if stats.get('status') != 'success':
        logger.warning(f'No vehicle clusters yet: {stats}')
        return

    clusters = stats.get('opensearch_clusters', [])[:max_clusters]
    logger.info(f'Creating mosaics for {len(clusters)} vehicle clusters')

    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        count = cluster['count']

        # Get cluster members
        members = await get_cluster_members('vehicles', cluster_id, size=20)
        if members.get('status') != 'success':
            continue

        image_paths = [m.get('image_path', '') for m in members.get('members', [])]

        title = f'Vehicle Cluster {cluster_id} ({count} detections)'
        output_path = output_dir / f'vehicle_cluster_{cluster_id:04d}.jpg'

        create_mosaic(image_paths, output_path, title, cols=5, thumb_size=200)


async def main():
    parser = argparse.ArgumentParser(description='Test FAISS clustering with killboy images')
    parser.add_argument(
        '--images',
        type=Path,
        default=Path('/mnt/nvm/KILLBOY_SAMPLE_PICTURES'),
        help='Path to image directory',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('/mnt/nvm/repos/triton-api/cluster_test_output'),
        help='Output directory for mosaics',
    )
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images to ingest')
    parser.add_argument(
        '--skip-ingest', action='store_true', help='Skip ingestion (use existing data)'
    )
    parser.add_argument(
        '--skip-train', action='store_true', help='Skip training (use existing clusters)'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=32,
        help='Number of clusters for training (default: 32 for small dataset)',
    )
    args = parser.parse_args()

    # Check health
    if not await check_health():
        logger.error('API not healthy, exiting')
        sys.exit(1)

    # Create indexes
    if not args.skip_ingest:
        await create_indexes(force_recreate=True)

    # Ingest images
    if not args.skip_ingest:
        if not args.images.exists():
            logger.error(f'Image directory not found: {args.images}')
            sys.exit(1)

        await ingest_images(args.images, limit=args.limit)

    # Show index stats
    stats = await get_index_stats()
    logger.info(f'Index stats: {stats}')

    # Train clustering
    if not args.skip_train:
        # Train global clustering
        await train_clusters('global', n_clusters=args.n_clusters)

        # Train vehicle clustering (fewer clusters for smaller dataset)
        vehicle_stats = stats.get('indexes', {}).get('visual_search_vehicles', {})
        if vehicle_stats.get('total_documents', 0) > 10:
            await train_clusters('vehicles', n_clusters=min(16, args.n_clusters // 2))

    # Show cluster stats
    cluster_stats = await get_cluster_stats('global')
    logger.info(f'Global cluster stats: {cluster_stats}')

    # List albums
    albums = await list_albums(min_size=2)
    logger.info(f'Albums: {albums.get("total_albums", 0)} with 2+ images')

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Create cluster mosaics
    logger.info('Creating cluster mosaics...')
    await create_cluster_mosaics(args.output / 'clusters', max_clusters=15)

    # Create vehicle cluster mosaics
    logger.info('Creating vehicle cluster mosaics...')
    await create_vehicle_cluster_mosaics(args.output / 'vehicles', max_clusters=10)

    # Create search result mosaics for a few sample images
    logger.info('Creating search result mosaics...')
    sample_images = sorted(args.images.glob('*.jpg'))[:5]
    for img in sample_images:
        await create_search_mosaic(img, args.output / 'searches', top_k=15)

    logger.info(f'\nDone! Check output at: {args.output}')
    logger.info('  - clusters/: Global image clusters (albums)')
    logger.info('  - vehicles/: Vehicle detection clusters')
    logger.info('  - searches/: Similarity search results')


if __name__ == '__main__':
    asyncio.run(main())
