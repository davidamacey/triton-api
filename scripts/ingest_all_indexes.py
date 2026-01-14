#!/usr/bin/env python3
"""
Multi-Index Ingestion Pipeline

Ingests images into all visual search indexes:
- visual_search_global: Whole image embeddings (MobileCLIP)
- visual_search_vehicles: Vehicle detections (car, truck, bus, motorcycle, boat)
- visual_search_people: Person detections (MobileCLIP appearance)
- visual_search_faces: Face detections (ArcFace identity)

Uses the unified /track_e/faces/full endpoint which runs:
- YOLO for object detection
- SCRFD for face detection
- MobileCLIP for global + per-box embeddings
- ArcFace for face identity embeddings

Usage:
    docker compose exec yolo-api python /app/scripts/ingest_all_indexes.py \
        --images-dir /app/test_images/faces/lfw-deepfunneled \
        --max-images 200 \
        --recreate-indexes \
        --create-mosaics

    docker compose exec yolo-api python /app/scripts/ingest_all_indexes.py \
        --images-dir /mnt/user_data/killboy \
        --max-images 500 \
        --create-mosaics
"""

import argparse
import asyncio
import contextlib
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image


# Add src to path (works both inside container and externally)
script_dir = Path(__file__).parent.absolute()
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, '/app')  # Container path


try:
    from src.clients.opensearch import IndexName, OpenSearchClient

    HAS_OPENSEARCH = True
except ImportError:
    HAS_OPENSEARCH = False
    OpenSearchClient = None
    # Define IndexName enum for external use
    from enum import Enum

    class IndexName(Enum):
        GLOBAL = 'visual_search_global'
        VEHICLES = 'visual_search_vehicles'
        PEOPLE = 'visual_search_people'
        FACES = 'visual_search_faces'
        OCR = 'visual_search_ocr'


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_BASE = 'http://localhost:4603'  # External API port
BATCH_SIZE = 16
MAX_WORKERS = 16  # Parallel workers for higher throughput

# COCO class mappings
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 8: 'boat'}
PERSON_CLASS = 0


def find_images(directory: Path, max_images: int = 1000) -> list[Path]:
    """Recursively find image files."""
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = []

    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                images.append(Path(root) / f)
                if len(images) >= max_images:
                    return images

    return sorted(images)


def ingest_single_image(image_path: Path) -> dict | None:
    """Ingest single image using /track_e/ingest endpoint which handles all indexing."""
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Use the /ingest endpoint which runs full pipeline and indexes to OpenSearch
        response = requests.post(
            f'{API_BASE}/track_e/ingest',
            files={'file': (image_path.name, image_bytes, 'image/jpeg')},
            params={
                'image_path': str(image_path),
                'skip_duplicates': 'false',
                'detect_near_duplicates': 'false',
                'enable_ocr': 'false',
            },
            timeout=120,
        )

        if response.status_code != 200:
            logger.debug(f'API error for {image_path}: {response.text[:100]}')
            return {'status': 'error', 'error': response.text[:100]}

        data = response.json()
        indexed = data.get('indexed', {})

        return {
            'image_path': str(image_path),
            'status': data.get('status', 'error'),
            'image_id': data.get('image_id'),
            'num_detections': data.get('num_detections', 0),
            'num_faces': data.get('num_faces', 0),
            'global': indexed.get('global', False),
            'vehicles': indexed.get('vehicles', 0),
            'people': indexed.get('people', 0),
            'faces': indexed.get('faces', 0),
        }

    except Exception as e:
        logger.debug(f'Failed to ingest {image_path}: {e}')
        return {'status': 'error', 'error': str(e)}


def ingest_batch_images(image_paths: list[Path]) -> list[dict]:
    """Ingest batch of images using /track_e/ingest_batch endpoint for higher throughput."""
    try:
        # Prepare files for multipart upload
        files = []
        paths_str = []
        for img_path in image_paths:
            with open(img_path, 'rb') as f:
                image_bytes = f.read()
            files.append(('files', (img_path.name, image_bytes, 'image/jpeg')))
            paths_str.append(str(img_path))

        # Use batch endpoint
        response = requests.post(
            f'{API_BASE}/track_e/ingest_batch',
            files=files,
            params={
                'image_paths': ','.join(paths_str),
                'skip_duplicates': 'false',
                'detect_near_duplicates': 'false',
                'enable_ocr': 'false',  # OCR is slow, disable for bulk ingest
            },
            timeout=300,
        )

        if response.status_code != 200:
            logger.warning(f'Batch API error: {response.text[:200]}')
            return [{'status': 'error', 'error': response.text[:100]} for _ in image_paths]

        data = response.json()

        # Batch endpoint returns summary, not per-image results
        # Convert to per-image format for consistency
        indexed = data.get('indexed', {})
        processed = data.get('processed', 0)
        duplicates = data.get('duplicates', 0)
        errors = data.get('errors', 0)

        # Create result entries
        results = []
        success_count = processed  # Images that were processed and indexed

        for i, img_path in enumerate(image_paths):
            # Mark as success if within the processed count
            is_success = i < success_count
            results.append(
                {
                    'image_path': str(img_path),
                    'status': 'success' if is_success else 'duplicate',
                    'image_id': None,
                    'num_detections': 0,
                    'num_faces': 0,
                    'global': is_success,
                    'vehicles': indexed.get('vehicles', 0) // max(1, processed)
                    if is_success
                    else 0,
                    'people': indexed.get('people', 0) // max(1, processed) if is_success else 0,
                    'faces': indexed.get('faces', 0) // max(1, processed) if is_success else 0,
                }
            )

        # Override totals with batch summary (more accurate)
        if results:
            results[0]['_batch_summary'] = {
                'processed': processed,
                'duplicates': duplicates,
                'errors': errors,
                'indexed': indexed,
                'timing': data.get('timing', {}),
            }

        return results

    except Exception as e:
        logger.warning(f'Batch ingestion failed: {e}')
        return [{'status': 'error', 'error': str(e)} for _ in image_paths]


async def ingest_to_all_indexes(
    opensearch: OpenSearchClient,
    image_data: list[dict],
) -> dict:
    """Ingest detections to all appropriate indexes."""
    stats = {
        'global': 0,
        'vehicles': 0,
        'people': 0,
        'faces': 0,
        'images': 0,
    }

    global_actions = []
    vehicle_actions = []
    people_actions = []
    face_actions = []

    for item in image_data:
        image_id = item['image_id']
        image_path = item['image_path']
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Directory name as category/person name
        category = Path(image_path).parent.name

        # 1. Global index - whole image embedding
        if item.get('global_embedding'):
            global_actions.append(
                {
                    '_index': IndexName.GLOBAL.value,
                    '_id': image_id,
                    '_source': {
                        'image_id': image_id,
                        'image_path': image_path,
                        'embedding': item['global_embedding'],
                        'category': category,
                        'indexed_at': timestamp,
                    },
                }
            )
            stats['global'] += 1

        # 2. Process YOLO detections
        for det_idx, det in enumerate(item.get('detections', [])):
            class_id = det.get('class_id', -1)
            box = det.get('box', [0, 0, 1, 1])
            confidence = det.get('confidence', 0.0)

            # Get embedding for this detection
            embedding = None
            if det_idx < len(item.get('box_embeddings', [])):
                embedding = item['box_embeddings'][det_idx]

            if embedding is None:
                continue

            det_id = f'{image_id}_det_{det_idx}'

            # Vehicle detection
            if class_id in VEHICLE_CLASSES:
                vehicle_actions.append(
                    {
                        '_index': IndexName.VEHICLES.value,
                        '_id': det_id,
                        '_source': {
                            'detection_id': det_id,
                            'image_id': image_id,
                            'image_path': image_path,
                            'embedding': embedding,
                            'box': box,
                            'class_id': class_id,
                            'class_name': VEHICLE_CLASSES[class_id],
                            'confidence': confidence,
                            'category': category,
                            'indexed_at': timestamp,
                        },
                    }
                )
                stats['vehicles'] += 1

            # Person detection
            elif class_id == PERSON_CLASS:
                people_actions.append(
                    {
                        '_index': IndexName.PEOPLE.value,
                        '_id': det_id,
                        '_source': {
                            'detection_id': det_id,
                            'image_id': image_id,
                            'image_path': image_path,
                            'embedding': embedding,
                            'box': box,
                            'confidence': confidence,
                            'category': category,
                            'indexed_at': timestamp,
                        },
                    }
                )
                stats['people'] += 1

        # 3. Face detections with ArcFace embeddings
        for face_idx, (face, embedding) in enumerate(
            zip(item.get('faces', []), item.get('face_embeddings', []), strict=False)
        ):
            if not embedding:
                continue

            face_id = f'{image_id}_face_{face_idx}'
            face_actions.append(
                {
                    '_index': IndexName.FACES.value,
                    '_id': face_id,
                    '_source': {
                        'face_id': face_id,
                        'image_id': image_id,
                        'image_path': image_path,
                        'embedding': embedding,
                        'box': face.get('box', [0, 0, 1, 1]),
                        'landmarks': face.get('landmarks', []),
                        'confidence': face.get('score', 0.0),
                        'quality': face.get('quality', 0.0),
                        'person_name': category,
                        'indexed_at': timestamp,
                    },
                }
            )
            stats['faces'] += 1

        stats['images'] += 1

    # Bulk insert to each index
    for index_name, actions in [
        ('global', global_actions),
        ('vehicles', vehicle_actions),
        ('people', people_actions),
        ('faces', face_actions),
    ]:
        if actions:
            # Batch into chunks of 500
            for i in range(0, len(actions), 500):
                batch = actions[i : i + 500]
                await opensearch.client.bulk(body=batch, refresh=False)
            logger.info(f'Ingested {len(actions)} to {index_name}')

    # Refresh all indexes
    for index in [IndexName.GLOBAL, IndexName.VEHICLES, IndexName.PEOPLE, IndexName.FACES]:
        with contextlib.suppress(Exception):
            await opensearch.client.indices.refresh(index=index.value)

    return stats


async def cluster_index(
    opensearch: OpenSearchClient,
    index_name: IndexName,
    n_clusters: int = 20,
) -> dict:
    """Cluster embeddings in an index using FAISS."""
    import faiss

    logger.info(f'Clustering {index_name.value}...')

    # Get all embeddings
    ids = []
    embeddings = []

    response = await opensearch.client.search(
        index=index_name.value,
        body={
            'query': {'match_all': {}},
            '_source': ['embedding'],
            'size': 10000,
        },
        scroll='5m',
    )

    scroll_id = response['_scroll_id']
    hits = response['hits']['hits']

    while hits:
        for hit in hits:
            ids.append(hit['_id'])
            embeddings.append(hit['_source']['embedding'])

        response = await opensearch.client.scroll(scroll_id=scroll_id, scroll='5m')
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']

    await opensearch.client.clear_scroll(scroll_id=scroll_id)

    if len(embeddings) < 10:
        logger.warning(f'Not enough embeddings in {index_name.value} to cluster')
        return {'clusters': 0, 'items': len(embeddings)}

    embeddings = np.array(embeddings, dtype=np.float32)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create IVF index
    d = embeddings.shape[1]
    n_clusters = min(n_clusters, len(embeddings) // 5)
    n_clusters = max(n_clusters, 2)

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)

    # Assign clusters
    _, cluster_ids = index.search(embeddings, 1)
    cluster_ids = cluster_ids.flatten()

    # Update documents with cluster IDs
    for doc_id, cluster_id in zip(ids, cluster_ids, strict=False):
        await opensearch.client.update(
            index=index_name.value,
            id=doc_id,
            body={
                'doc': {
                    'cluster_id': int(cluster_id),
                    'clustered_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                }
            },
        )

    await opensearch.client.indices.refresh(index=index_name.value)

    return {'clusters': n_clusters, 'items': len(embeddings)}


async def create_mosaics_for_index(
    opensearch: OpenSearchClient,
    index_name: IndexName,
    output_dir: Path,
    item_size: int = 100,
    max_items_per_mosaic: int = 36,
) -> int:
    """Create mosaic visualizations for clustered items."""
    # Get clustered items
    response = await opensearch.client.search(
        index=index_name.value,
        body={
            'query': {'exists': {'field': 'cluster_id'}},
            '_source': ['cluster_id', 'image_path', 'box', 'category', 'person_name', 'class_name'],
            'size': 10000,
        },
    )

    hits = response['hits']['hits']
    if not hits:
        return 0

    # Group by cluster
    clusters = {}
    for hit in hits:
        source = hit['_source']
        cid = source.get('cluster_id', -1)
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(source)

    output_dir.mkdir(parents=True, exist_ok=True)
    mosaics_created = 0

    for cluster_id, items in clusters.items():
        if len(items) < 2:
            continue

        selected_items = items[:max_items_per_mosaic]

        # Calculate mosaic dimensions
        n = len(selected_items)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        mosaic_w = cols * item_size
        mosaic_h = rows * item_size
        mosaic = Image.new('RGB', (mosaic_w, mosaic_h), (50, 50, 50))

        for i, item in enumerate(selected_items):
            try:
                img = Image.open(item['image_path'])
                w, h = img.size
                box = item.get('box', [0, 0, 1, 1])

                # Crop detection
                x1 = int(box[0] * w)
                y1 = int(box[1] * h)
                x2 = int(box[2] * w)
                y2 = int(box[3] * h)

                # Add padding
                pad = int((x2 - x1) * 0.1)
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)

                crop = img.crop((x1, y1, x2, y2))
                crop = crop.resize((item_size, item_size), Image.Resampling.LANCZOS)

                col = i % cols
                row = i // cols
                mosaic.paste(crop, (col * item_size, row * item_size))

            except Exception as e:
                logger.debug(f'Failed to add item to mosaic: {e}')

        # Get cluster label
        categories = set()
        for item in items:
            cat = (
                item.get('person_name')
                or item.get('category')
                or item.get('class_name')
                or 'unknown'
            )
            categories.add(cat)

        label = '_'.join(sorted(categories)[:3])[:30]
        mosaic_path = output_dir / f'cluster_{cluster_id:03d}_n{len(items)}_{label}.jpg'
        mosaic.save(mosaic_path, quality=90)
        mosaics_created += 1

    return mosaics_created


async def main(args):
    """Main pipeline."""
    logger.info('=' * 70)
    logger.info('Multi-Index Ingestion Pipeline')
    logger.info('=' * 70)

    opensearch = OpenSearchClient() if HAS_OPENSEARCH else None

    try:
        # Step 1: Create indexes via API (works from external clients)
        if args.recreate_indexes:
            logger.info('\nStep 1: Recreating all indexes via API...')
            try:
                # Delete all indexes first
                resp = requests.delete(f'{API_BASE}/track_e/index', timeout=30)
                if resp.status_code == 200:
                    logger.info('All indexes deleted')
                # Recreate indexes
                resp = requests.post(f'{API_BASE}/track_e/index/create', timeout=30)
                if resp.status_code == 200:
                    logger.info('All indexes recreated via API')
                else:
                    logger.warning(f'Index creation returned: {resp.text[:100]}')
            except Exception as e:
                logger.warning(f'Could not recreate indexes via API: {e}')
                logger.info('Indexes will be created automatically during ingestion')
        else:
            logger.info('\nStep 1: Indexes will be created automatically during ingestion')

        # Step 2: Find images
        images_dir = Path(args.images_dir)
        if not images_dir.exists():
            logger.error(f'Directory not found: {images_dir}')
            return 1

        logger.info(f'\nStep 2: Finding images in {images_dir}...')
        images = find_images(images_dir, args.max_images)
        logger.info(f'Found {len(images)} images')

        if not images:
            return 1

        # Step 3: Ingest images via /track_e/ingest_batch endpoint for high throughput
        logger.info('\nStep 3: Ingesting images via /track_e/ingest_batch endpoint...')
        logger.info(f'Using batch size {BATCH_SIZE} with {MAX_WORKERS} parallel workers')

        start_time = time.time()
        results = []
        processed = 0

        # Split images into batches
        batches = [images[i : i + BATCH_SIZE] for i in range(0, len(images), BATCH_SIZE)]
        logger.info(f'Processing {len(batches)} batches of up to {BATCH_SIZE} images each')

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(ingest_batch_images, batch): batch for batch in batches}

            for future, batch in futures.items():
                batch_results = future.result()
                if batch_results:
                    results.extend(batch_results)

                processed += len(batch)
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                logger.info(f'Processed {processed}/{len(images)} images ({rate:.1f} img/sec)')

        elapsed = time.time() - start_time

        # Calculate stats from batch summaries (more accurate than per-image)
        total_processed = 0
        total_duplicates = 0
        total_errors = 0
        stats = {
            'global': 0,
            'vehicles': 0,
            'people': 0,
            'faces': 0,
            'ocr': 0,
        }

        for r in results:
            summary = r.get('_batch_summary')
            if summary:
                total_processed += summary.get('processed', 0)
                total_duplicates += summary.get('duplicates', 0)
                total_errors += summary.get('errors', 0)
                indexed = summary.get('indexed', {})
                stats['global'] += indexed.get('global', 0)
                stats['vehicles'] += indexed.get('vehicles', 0)
                stats['people'] += indexed.get('people', 0)
                stats['faces'] += indexed.get('faces', 0)
                stats['ocr'] += indexed.get('ocr', 0)

        success_count = total_processed

        logger.info(
            f'\nIngestion complete in {elapsed:.1f}s ({len(images) / elapsed:.1f} images/sec):'
        )
        logger.info(f'  Processed: {success_count}/{len(images)} images')
        logger.info(f'  Duplicates skipped: {total_duplicates}')
        logger.info(f'  Errors: {total_errors}')
        logger.info(
            f'  Indexed - Global: {stats["global"]}, Vehicles: {stats["vehicles"]}, People: {stats["people"]}, Faces: {stats["faces"]}'
        )

        # Step 5: Cluster each index (optional, requires OpenSearch access)
        if not args.skip_clustering:
            logger.info('\nStep 5: Clustering all indexes...')
            cluster_stats = {}

            for index_name, n_clusters in [
                (IndexName.GLOBAL, 30),
                (IndexName.VEHICLES, 20),
                (IndexName.PEOPLE, 30),
                (IndexName.FACES, 40),
            ]:
                try:
                    result = await cluster_index(opensearch, index_name, n_clusters)
                    cluster_stats[index_name.value] = result
                    logger.info(
                        f'  {index_name.value}: {result["clusters"]} clusters, {result["items"]} items'
                    )
                except Exception as e:
                    logger.warning(f'  {index_name.value}: clustering failed - {e}')
        else:
            logger.info('\nStep 5: Skipping clustering (--skip-clustering)')

        # Step 6: Create mosaics
        if args.create_mosaics:
            logger.info('\nStep 6: Creating mosaic visualizations...')
            output_base = Path(args.output_dir) / 'clusters'

            for index_name, item_size in [
                (IndexName.GLOBAL, 150),
                (IndexName.VEHICLES, 120),
                (IndexName.PEOPLE, 100),
                (IndexName.FACES, 112),
            ]:
                output_dir = output_base / index_name.value.replace('visual_search_', '')
                count = await create_mosaics_for_index(
                    opensearch, index_name, output_dir, item_size
                )
                logger.info(f'  {index_name.value}: {count} mosaics')

        # Summary
        logger.info('\n' + '=' * 70)
        logger.info('PIPELINE COMPLETE')
        logger.info('=' * 70)
        logger.info(f'Images processed: {success_count}/{len(images)}')
        logger.info(f'Total time: {elapsed:.1f}s ({len(images) / elapsed:.1f} images/sec)')
        for idx, count in stats.items():
            logger.info(f'  {idx}: {count}')

        if args.create_mosaics:
            logger.info(f'\nMosaics saved to: {Path(args.output_dir) / "clusters"}')

        return 0

    except Exception as e:
        logger.error(f'Pipeline failed: {e}')
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Index Ingestion Pipeline')
    parser.add_argument('--images-dir', type=str, required=True, help='Images directory')
    parser.add_argument('--max-images', type=int, default=5000, help='Max images to process')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for ingestion')
    parser.add_argument('--workers', type=int, default=16, help='Number of parallel workers')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--recreate-indexes', action='store_true', help='Recreate all indexes')
    parser.add_argument('--skip-clustering', action='store_true', help='Skip clustering step')
    parser.add_argument(
        '--create-mosaics', action='store_true', help='Create mosaic visualizations'
    )

    args = parser.parse_args()

    # Allow overriding globals from command line
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    if args.workers:
        MAX_WORKERS = args.workers

    sys.exit(asyncio.run(main(args)))
