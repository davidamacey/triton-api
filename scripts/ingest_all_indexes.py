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
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image


# Add src to path
sys.path.insert(0, '/app')

import contextlib

from src.clients.opensearch import IndexName, OpenSearchClient


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_BASE = 'http://localhost:8000'
BATCH_SIZE = 16
MAX_WORKERS = 4

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


def process_image_full(image_path: Path) -> dict | None:
    """Process image through unified pipeline to get all detections."""
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Use the unified faces/full endpoint for YOLO + SCRFD + MobileCLIP + ArcFace
        response = requests.post(
            f'{API_BASE}/track_e/faces/full',
            files={'image': (image_path.name, image_bytes, 'image/jpeg')},
            timeout=60,
        )

        if response.status_code != 200:
            logger.debug(f'API error for {image_path}: {response.text[:100]}')
            return None

        data = response.json()

        return {
            'image_path': str(image_path),
            'image_id': str(uuid.uuid4())[:8],
            # YOLO detections
            'num_dets': data.get('num_dets', 0),
            'detections': data.get('detections', []),
            'global_embedding': data.get('global_embedding'),
            'box_embeddings': data.get('box_embeddings', []),
            # Face detections
            'num_faces': data.get('num_faces', 0),
            'faces': data.get('faces', []),
            'face_embeddings': data.get('face_embeddings', []),
            # Image metadata
            'image_width': data.get('image', {}).get('width', 0),
            'image_height': data.get('image', {}).get('height', 0),
        }

    except Exception as e:
        logger.debug(f'Failed to process {image_path}: {e}')
        return None


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

    opensearch = OpenSearchClient()
    await opensearch.connect()

    try:
        # Step 1: Create indexes
        if args.recreate_indexes:
            logger.info('\nStep 1: Creating all indexes...')
            await opensearch.create_all_indexes(force_recreate=True)
            logger.info('All indexes created')
        else:
            logger.info('\nStep 1: Ensuring indexes exist...')
            await opensearch.create_all_indexes(force_recreate=False)

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

        # Step 3: Process images
        logger.info('\nStep 3: Processing images through unified pipeline...')
        image_data = []
        processed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_image_full, img): img for img in images}

            for future in futures:
                result = future.result()
                if result:
                    image_data.append(result)

                processed += 1
                if processed % 20 == 0:
                    logger.info(f'Processed {processed}/{len(images)} images')

        logger.info(f'Successfully processed {len(image_data)}/{len(images)} images')

        if not image_data:
            logger.error('No images processed successfully')
            return 1

        # Step 4: Ingest to indexes
        logger.info('\nStep 4: Ingesting to all indexes...')
        stats = await ingest_to_all_indexes(opensearch, image_data)

        logger.info('Ingestion complete:')
        logger.info(f'  Global embeddings: {stats["global"]}')
        logger.info(f'  Vehicle detections: {stats["vehicles"]}')
        logger.info(f'  Person detections: {stats["people"]}')
        logger.info(f'  Face detections: {stats["faces"]}')

        # Step 5: Cluster each index
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
        logger.info(f'Images processed: {len(image_data)}')
        for idx, count in stats.items():
            if idx != 'images':
                logger.info(f'  {idx}: {count}')

        if args.create_mosaics:
            logger.info(f'\nMosaics saved to: {Path(args.output_dir) / "clusters"}')

        return 0

    finally:
        await opensearch.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Index Ingestion Pipeline')
    parser.add_argument('--images-dir', type=str, required=True, help='Images directory')
    parser.add_argument('--max-images', type=int, default=200, help='Max images to process')
    parser.add_argument('--output-dir', type=str, default='/app/outputs', help='Output directory')
    parser.add_argument('--recreate-indexes', action='store_true', help='Recreate all indexes')
    parser.add_argument(
        '--create-mosaics', action='store_true', help='Create mosaic visualizations'
    )

    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))
