#!/usr/bin/env python3
"""
Create mosaic visualizations for clustered items.

Usage:
    python scripts/create_mosaics.py --output-dir ./outputs/mosaics
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_BASE = 'http://localhost:4603'
OPENSEARCH_BASE = 'http://localhost:4607'


def get_cluster_items(index_name: str, cluster_id: int, max_items: int = 36) -> list[dict]:
    """Get items from a specific cluster."""
    query = {
        'query': {'term': {'cluster_id': cluster_id}},
        '_source': [
            'image_id',
            'image_path',
            'box',
            'cluster_id',
            'confidence',
            'class_name',
            'person_name',
            'face_id',
        ],
        'size': max_items,
    }

    resp = requests.post(f'{OPENSEARCH_BASE}/{index_name}/_search', json=query)
    if resp.status_code != 200:
        return []

    hits = resp.json().get('hits', {}).get('hits', [])
    return [h['_source'] for h in hits]


def get_all_clusters(index_name: str) -> list[int]:
    """Get all cluster IDs in an index."""
    query = {'size': 0, 'aggs': {'clusters': {'terms': {'field': 'cluster_id', 'size': 1000}}}}

    resp = requests.post(f'{OPENSEARCH_BASE}/{index_name}/_search', json=query)
    if resp.status_code != 200:
        return []

    buckets = resp.json().get('aggregations', {}).get('clusters', {}).get('buckets', [])
    return [b['key'] for b in buckets]


def create_mosaic(items: list[dict], output_path: Path, item_size: int = 112) -> bool:
    """Create a mosaic image from cluster items."""
    if len(items) < 2:
        return False

    n = len(items)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    mosaic_w = cols * item_size
    mosaic_h = rows * item_size
    mosaic = Image.new('RGB', (mosaic_w, mosaic_h), (50, 50, 50))

    for i, item in enumerate(items):
        try:
            img_path = item.get('image_path', '')
            if not img_path or not Path(img_path).exists():
                continue

            img = Image.open(img_path)
            w, h = img.size
            box = item.get('box', [0, 0, 1, 1])

            # Crop detection area
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

    mosaic.save(output_path, quality=90)
    return True


def process_index(index_name: str, output_dir: Path, item_size: int = 112) -> int:
    """Create mosaics for all clusters in an index."""
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = get_all_clusters(index_name)
    logger.info(f'Found {len(clusters)} clusters in {index_name}')

    mosaics_created = 0
    for cluster_id in clusters:
        items = get_cluster_items(index_name, cluster_id)
        if len(items) < 2:
            continue

        # Create filename with cluster info
        output_path = output_dir / f'cluster_{cluster_id:03d}_n{len(items)}.jpg'

        if create_mosaic(items, output_path, item_size):
            mosaics_created += 1

    return mosaics_created


def main(args):
    print('=' * 60)
    print('Cluster Mosaic Generator')
    print('=' * 60)

    output_base = Path(args.output_dir)

    indexes = [
        ('visual_search_global', 'global', 150),
        ('visual_search_vehicles', 'vehicles', 120),
        ('visual_search_people', 'people', 100),
        ('visual_search_faces', 'faces', 112),
    ]

    total_mosaics = 0
    for index_name, folder, item_size in indexes:
        output_dir = output_base / folder
        logger.info(f'\nProcessing {index_name}...')
        count = process_index(index_name, output_dir, item_size)
        logger.info(f'Created {count} mosaics for {folder}')
        total_mosaics += count

    print('\n' + '=' * 60)
    print(f'COMPLETE: Created {total_mosaics} mosaics')
    print(f'Output: {output_base}')
    print('=' * 60)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create cluster mosaics')
    parser.add_argument(
        '--output-dir', type=str, default='./outputs/mosaics', help='Output directory'
    )

    args = parser.parse_args()
    sys.exit(main(args))
