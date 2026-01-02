#!/usr/bin/env python3
"""
Face Ingestion and Clustering Pipeline

Ingests faces from images, clusters them using FAISS, and creates mosaic visualizations.

Usage:
    docker compose exec yolo-api python /app/scripts/ingest_and_cluster_faces.py \
        --images-dir /app/test_images/faces/lfw-deepfunneled \
        --max-images 500 \
        --create-mosaics

    docker compose exec yolo-api python /app/scripts/ingest_and_cluster_faces.py \
        --images-dir /mnt/user_data/killboy \
        --max-images 1000 \
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

from src.clients.opensearch import IndexName, OpenSearchClient
from src.services.clustering import ClusteringService


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_BASE = 'http://localhost:8000'
BATCH_SIZE = 16
MAX_WORKERS = 4


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


def extract_face_embeddings(image_path: Path) -> dict | None:
    """Extract face embeddings from an image using the API."""
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        response = requests.post(
            f'{API_BASE}/track_e/faces/recognize',
            files={'image': (image_path.name, image_bytes, 'image/jpeg')},
            timeout=30,
        )

        if response.status_code != 200:
            return None

        data = response.json()
        if data.get('num_faces', 0) == 0:
            return None

        return {
            'num_faces': data['num_faces'],
            'faces': data['faces'],
            'embeddings': data['embeddings'],
            'image_path': str(image_path),
        }

    except Exception as e:
        logger.debug(f'Failed to process {image_path}: {e}')
        return None


async def ingest_faces_to_opensearch(
    opensearch: OpenSearchClient,
    face_data: list[dict],
    batch_size: int = 100,
) -> dict:
    """Ingest face embeddings into OpenSearch."""
    total_faces = 0
    total_images = 0

    for i in range(0, len(face_data), batch_size):
        batch = face_data[i : i + batch_size]

        actions = []
        for item in batch:
            image_path = item['image_path']
            image_id = str(uuid.uuid4())[:8]

            for face_idx, (face, embedding) in enumerate(
                zip(item['faces'], item['embeddings'], strict=False)
            ):
                face_id = f'{image_id}_face_{face_idx}'
                doc = {
                    'face_id': face_id,
                    'image_id': image_id,
                    'image_path': image_path,
                    'embedding': embedding,
                    'box': face.get('box', [0, 0, 1, 1]),
                    'landmarks': face.get('landmarks', []),
                    'confidence': face.get('score', 0.0),
                    'quality': face.get('quality', 0.0),
                    'person_name': Path(image_path).parent.name,  # Use directory as person name
                    'indexed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                }

                actions.append(
                    {
                        '_index': IndexName.FACES.value,
                        '_id': face_id,
                        '_source': doc,
                    }
                )
                total_faces += 1

            total_images += 1

        if actions:
            await opensearch.client.bulk(body=actions, refresh=False)
            logger.info(f'Ingested batch {i // batch_size + 1}: {len(actions)} faces')

    # Final refresh
    await opensearch.client.indices.refresh(index=IndexName.FACES.value)

    return {
        'total_images': total_images,
        'total_faces': total_faces,
    }


async def get_all_face_embeddings(opensearch: OpenSearchClient) -> tuple[list[str], np.ndarray]:
    """Retrieve all face embeddings from OpenSearch."""
    face_ids = []
    embeddings = []

    # Scroll through all documents
    response = await opensearch.client.search(
        index=IndexName.FACES.value,
        body={
            'query': {'match_all': {}},
            '_source': ['face_id', 'embedding'],
            'size': 10000,
        },
        scroll='5m',
    )

    scroll_id = response['_scroll_id']
    hits = response['hits']['hits']

    while hits:
        for hit in hits:
            face_ids.append(hit['_id'])
            embeddings.append(hit['_source']['embedding'])

        response = await opensearch.client.scroll(scroll_id=scroll_id, scroll='5m')
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']

    # Clear scroll
    await opensearch.client.clear_scroll(scroll_id=scroll_id)

    return face_ids, np.array(embeddings, dtype=np.float32)


async def cluster_faces(
    opensearch: OpenSearchClient,
    _clustering: ClusteringService,
    n_clusters: int = 50,
) -> dict:
    """Cluster face embeddings using FAISS IVF."""
    logger.info('Retrieving face embeddings from OpenSearch...')
    face_ids, embeddings = await get_all_face_embeddings(opensearch)

    if len(embeddings) == 0:
        logger.warning('No faces found in index')
        return {'clusters': 0, 'faces': 0}

    logger.info(f'Clustering {len(embeddings)} faces into {n_clusters} clusters...')

    # Use clustering service
    n_clusters = min(n_clusters, len(embeddings) // 2)  # Ensure enough samples per cluster
    n_clusters = max(n_clusters, 2)

    # Create FAISS index and cluster
    import faiss

    d = embeddings.shape[1]  # 512 for ArcFace
    nlist = n_clusters

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Train IVF index
    quantizer = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    index.train(embeddings)
    index.add(embeddings)

    # Assign clusters
    _, cluster_ids = index.search(embeddings, 1)
    cluster_ids = cluster_ids.flatten()

    # Update OpenSearch with cluster assignments
    logger.info('Updating cluster assignments in OpenSearch...')
    for i, (face_id, cluster_id) in enumerate(zip(face_ids, cluster_ids, strict=False)):
        await opensearch.client.update(
            index=IndexName.FACES.value,
            id=face_id,
            body={
                'doc': {
                    'cluster_id': int(cluster_id),
                    'clustered_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                }
            },
        )

        if (i + 1) % 100 == 0:
            logger.info(f'Updated {i + 1}/{len(face_ids)} cluster assignments')

    await opensearch.client.indices.refresh(index=IndexName.FACES.value)

    # Count faces per cluster
    cluster_counts = {}
    for cid in cluster_ids:
        cluster_counts[int(cid)] = cluster_counts.get(int(cid), 0) + 1

    return {
        'n_clusters': n_clusters,
        'n_faces': len(face_ids),
        'cluster_counts': cluster_counts,
    }


async def create_cluster_mosaics(
    opensearch: OpenSearchClient,
    output_dir: Path,
    max_faces_per_mosaic: int = 36,
    face_size: int = 112,
) -> dict:
    """Create mosaic images for each cluster."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all faces grouped by cluster
    response = await opensearch.client.search(
        index=IndexName.FACES.value,
        body={
            'query': {'exists': {'field': 'cluster_id'}},
            '_source': ['face_id', 'image_path', 'box', 'cluster_id', 'person_name'],
            'size': 10000,
        },
    )

    hits = response['hits']['hits']
    if not hits:
        logger.warning('No clustered faces found')
        return {'mosaics_created': 0}

    # Group by cluster
    clusters = {}
    for hit in hits:
        source = hit['_source']
        cid = source['cluster_id']
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(source)

    logger.info(f'Creating mosaics for {len(clusters)} clusters...')

    mosaics_created = 0
    for cluster_id, faces in clusters.items():
        if len(faces) < 2:
            continue  # Skip single-face clusters

        # Sort by person_name to group same people together
        selected_faces = sorted(faces, key=lambda x: x.get('person_name', ''))[
            :max_faces_per_mosaic
        ]

        # Calculate mosaic dimensions
        n_faces = len(selected_faces)
        cols = int(np.ceil(np.sqrt(n_faces)))
        rows = int(np.ceil(n_faces / cols))

        # Create mosaic
        mosaic_w = cols * face_size
        mosaic_h = rows * face_size
        mosaic = Image.new('RGB', (mosaic_w, mosaic_h), (50, 50, 50))

        for i, face in enumerate(selected_faces):
            try:
                image_path = face['image_path']
                box = face.get('box', [0, 0, 1, 1])

                # Load and crop face
                img = Image.open(image_path)
                w, h = img.size
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

                face_crop = img.crop((x1, y1, x2, y2))
                face_crop = face_crop.resize((face_size, face_size), Image.Resampling.LANCZOS)

                # Place in mosaic
                col = i % cols
                row = i // cols
                mosaic.paste(face_crop, (col * face_size, row * face_size))

            except Exception as e:
                logger.debug(f'Failed to add face to mosaic: {e}')

        # Save mosaic with cluster info
        person_names = {f.get('person_name', 'unknown') for f in faces}
        mosaic_path = output_dir / f'cluster_{cluster_id:03d}_n{len(faces)}.jpg'
        mosaic.save(mosaic_path, quality=90)

        # Save cluster metadata
        meta_path = output_dir / f'cluster_{cluster_id:03d}_n{len(faces)}.txt'
        with open(meta_path, 'w') as f:
            f.write(f'Cluster ID: {cluster_id}\n')
            f.write(f'Face count: {len(faces)}\n')
            f.write(f'Person names: {", ".join(sorted(person_names)[:10])}\n')
            f.write('\nFaces:\n')
            for face in faces:
                f.write(f'  - {face["face_id"]}: {face.get("person_name", "unknown")}\n')

        mosaics_created += 1

    logger.info(f'Created {mosaics_created} cluster mosaics')
    return {'mosaics_created': mosaics_created}


async def main(args):
    """Main pipeline."""
    logger.info('=' * 60)
    logger.info('Face Ingestion and Clustering Pipeline')
    logger.info('=' * 60)

    # Initialize clients
    opensearch = OpenSearchClient()
    await opensearch.connect()

    clustering = ClusteringService()

    try:
        # Step 1: Create/recreate faces index
        if args.recreate_index:
            logger.info('\nStep 1: Creating faces index...')
            await opensearch.create_faces_index(force_recreate=True)
            logger.info('Faces index created')
        else:
            # Ensure index exists
            exists = await opensearch.client.indices.exists(index=IndexName.FACES.value)
            if not exists:
                await opensearch.create_faces_index(force_recreate=False)
                logger.info('Faces index created')

        # Step 2: Find and process images
        images_dir = Path(args.images_dir)
        if not images_dir.exists():
            logger.error(f'Images directory not found: {images_dir}')
            return 1

        logger.info(f'\nStep 2: Finding images in {images_dir}...')
        images = find_images(images_dir, args.max_images)
        logger.info(f'Found {len(images)} images')

        if not images:
            logger.warning('No images found')
            return 1

        # Step 3: Extract face embeddings
        logger.info('\nStep 3: Extracting face embeddings...')
        face_data = []
        processed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(extract_face_embeddings, img): img for img in images}

            for future in futures:
                result = future.result()
                if result and result['num_faces'] > 0:
                    face_data.append(result)
                else:
                    failed += 1

                processed += 1
                if processed % 50 == 0:
                    logger.info(
                        f'Processed {processed}/{len(images)} images, {len(face_data)} with faces'
                    )

        logger.info(f'Extracted faces from {len(face_data)} images ({failed} failed or no faces)')

        if not face_data:
            logger.warning('No faces found in any images')
            return 1

        # Step 4: Ingest to OpenSearch
        logger.info('\nStep 4: Ingesting faces to OpenSearch...')
        ingest_result = await ingest_faces_to_opensearch(opensearch, face_data)
        logger.info(
            f'Ingested {ingest_result["total_faces"]} faces from {ingest_result["total_images"]} images'
        )

        # Step 5: Cluster faces
        logger.info('\nStep 5: Clustering faces...')
        cluster_result = await cluster_faces(opensearch, clustering, n_clusters=args.n_clusters)
        logger.info(
            f'Created {cluster_result["n_clusters"]} clusters from {cluster_result["n_faces"]} faces'
        )

        # Step 6: Create mosaics
        if args.create_mosaics:
            logger.info('\nStep 6: Creating cluster mosaics...')
            output_dir = Path(args.output_dir) / 'face_clusters'
            mosaic_result = await create_cluster_mosaics(opensearch, output_dir)
            logger.info(f'Created {mosaic_result["mosaics_created"]} mosaics in {output_dir}')

        # Summary
        logger.info('\n' + '=' * 60)
        logger.info('PIPELINE COMPLETE')
        logger.info('=' * 60)
        logger.info(f'Images processed: {len(images)}')
        logger.info(f'Images with faces: {len(face_data)}')
        logger.info(f'Faces ingested: {ingest_result["total_faces"]}')
        logger.info(f'Clusters created: {cluster_result["n_clusters"]}')

        if args.create_mosaics:
            logger.info(f'Mosaics saved to: {output_dir}')

        return 0

    finally:
        await opensearch.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Ingestion and Clustering Pipeline')
    parser.add_argument('--images-dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--max-images', type=int, default=500, help='Maximum images to process')
    parser.add_argument('--n-clusters', type=int, default=50, help='Number of clusters')
    parser.add_argument('--output-dir', type=str, default='/app/outputs', help='Output directory')
    parser.add_argument(
        '--create-mosaics', action='store_true', help='Create mosaic visualizations'
    )
    parser.add_argument('--recreate-index', action='store_true', help='Recreate the faces index')

    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))
