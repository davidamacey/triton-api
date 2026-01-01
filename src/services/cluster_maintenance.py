"""
Cluster Maintenance Service.

Background tasks for automatic cluster rebalancing based on ingestion patterns.

Usage:
    # One-time check and rebalance
    from src.services.cluster_maintenance import check_and_rebalance_all
    await check_and_rebalance_all(opensearch_client)

    # Or use the ClusterMaintenanceService for more control
    service = ClusterMaintenanceService(opensearch_client)
    await service.check_and_rebalance('global')

Scheduling (example with APScheduler):
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        check_and_rebalance_all,
        'interval',
        hours=6,  # Adjust based on ingestion volume
        args=[opensearch_client],
    )
    scheduler.start()
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.clients.opensearch import IndexName, OpenSearchClient
from src.services.clustering import ClusterIndex, get_clustering_service


logger = logging.getLogger(__name__)


class IngestionPattern(str, Enum):
    """Ingestion pattern for determining rebalance frequency."""

    LOW_VOLUME = 'low_volume'  # < 1K images/day
    MEDIUM_VOLUME = 'medium_volume'  # 1K - 10K images/day
    HIGH_VOLUME = 'high_volume'  # 10K - 100K images/day
    VERY_HIGH_VOLUME = 'very_high_volume'  # 100K+ images/day


@dataclass
class RebalanceConfig:
    """Configuration for automatic rebalancing."""

    # Thresholds for triggering rebalance
    new_data_threshold: float = 0.5  # Rebalance when 50% new data added
    imbalance_threshold: float = 10.0  # Rebalance when max/min ratio > 10
    empty_cluster_threshold: float = 0.1  # Rebalance when > 10% clusters empty

    # Minimum time between rebalances (prevents thrashing)
    min_rebalance_interval_hours: float = 1.0

    # Whether to rebalance automatically or just report
    auto_rebalance: bool = True


# Default configs by ingestion pattern
PATTERN_CONFIGS: dict[IngestionPattern, RebalanceConfig] = {
    IngestionPattern.LOW_VOLUME: RebalanceConfig(
        new_data_threshold=0.5,
        min_rebalance_interval_hours=24.0,
    ),
    IngestionPattern.MEDIUM_VOLUME: RebalanceConfig(
        new_data_threshold=0.4,
        min_rebalance_interval_hours=6.0,
    ),
    IngestionPattern.HIGH_VOLUME: RebalanceConfig(
        new_data_threshold=0.3,
        min_rebalance_interval_hours=2.0,
    ),
    IngestionPattern.VERY_HIGH_VOLUME: RebalanceConfig(
        new_data_threshold=0.2,
        min_rebalance_interval_hours=0.5,
    ),
}


# Mapping from IndexName to ClusterIndex
INDEX_MAPPING = {
    IndexName.GLOBAL: ClusterIndex.GLOBAL,
    IndexName.VEHICLES: ClusterIndex.VEHICLES,
    IndexName.PEOPLE: ClusterIndex.PEOPLE,
    IndexName.FACES: ClusterIndex.FACES,
}


class ClusterMaintenanceService:
    """
    Service for automatic cluster maintenance and rebalancing.

    Monitors cluster health and triggers rebalancing when needed based on:
    - Amount of new data since last training
    - Cluster size imbalance
    - Empty cluster ratio

    Can be run as a scheduled background task or triggered manually.
    """

    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        config: RebalanceConfig | None = None,
        pattern: IngestionPattern = IngestionPattern.MEDIUM_VOLUME,
    ):
        """
        Initialize the maintenance service.

        Args:
            opensearch_client: OpenSearch client for data operations
            config: Custom rebalance configuration (overrides pattern default)
            pattern: Ingestion pattern for default configuration
        """
        self.opensearch = opensearch_client
        self.config = config or PATTERN_CONFIGS[pattern]
        self.clustering = get_clustering_service()

        # Track last rebalance times
        self._last_rebalance: dict[ClusterIndex, datetime] = {}

    async def check_all_indexes(self) -> dict[str, Any]:
        """
        Check all indexes for rebalancing needs.

        Returns:
            Dict with status for each index
        """
        results = {}

        for index_name, cluster_index in INDEX_MAPPING.items():
            try:
                result = await self.check_index(cluster_index, index_name)
                results[cluster_index.value] = result
            except Exception as e:
                logger.error(f'Failed to check {cluster_index.value}: {e}')
                results[cluster_index.value] = {'status': 'error', 'error': str(e)}

        return results

    async def check_index(
        self,
        cluster_index: ClusterIndex,
        opensearch_index: IndexName,  # noqa: ARG002 - kept for API consistency
    ) -> dict[str, Any]:
        """
        Check a single index for rebalancing needs.

        Returns:
            Dict with check results and recommendation
        """
        # Check if clustering is trained for this index
        if not self.clustering.is_trained(cluster_index):
            return {
                'status': 'not_trained',
                'needs_rebalance': False,
                'recommendation': 'Train clusters first with POST /track_e/clusters/train/{index}',
            }

        # Get balance status from clustering service
        balance = await self.clustering.check_balance(cluster_index)

        # Check minimum interval
        can_rebalance = self._can_rebalance(cluster_index)

        return {
            'status': 'checked',
            'index_name': cluster_index.value,
            'is_balanced': balance.is_balanced,
            'needs_rebalance': balance.needs_rebalance,
            'can_rebalance': can_rebalance,
            'imbalance_ratio': round(balance.imbalance_ratio, 2),
            'empty_ratio': round(balance.empty_ratio, 4),
            'vectors_since_training': balance.vectors_since_training,
            'reason': balance.reason,
            'last_rebalance': self._last_rebalance.get(cluster_index, None),
        }

    async def check_and_rebalance(
        self,
        cluster_index: ClusterIndex,
        opensearch_index: IndexName,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Check an index and rebalance if needed.

        Args:
            cluster_index: FAISS cluster index
            opensearch_index: OpenSearch index
            force: Force rebalance even if not needed

        Returns:
            Dict with results
        """
        # First check
        check_result = await self.check_index(cluster_index, opensearch_index)

        if check_result['status'] == 'not_trained':
            return check_result

        needs_rebalance = force or (
            check_result['needs_rebalance'] and check_result['can_rebalance']
        )

        if not needs_rebalance:
            check_result['action'] = 'none'
            return check_result

        if not self.config.auto_rebalance and not force:
            check_result['action'] = 'recommended'
            check_result['message'] = 'Rebalance recommended but auto_rebalance is disabled'
            return check_result

        # Perform rebalance
        logger.info(f'Rebalancing {cluster_index.value}: {check_result.get("reason", "forced")}')

        try:
            rebalance_result = await self._rebalance(cluster_index, opensearch_index)
            self._last_rebalance[cluster_index] = datetime.now(UTC)

            return {
                'status': 'rebalanced',
                'index_name': cluster_index.value,
                'action': 'rebalanced',
                'previous_state': check_result,
                'rebalance_result': rebalance_result,
            }

        except Exception as e:
            logger.error(f'Rebalance failed for {cluster_index.value}: {e}')
            return {
                'status': 'error',
                'index_name': cluster_index.value,
                'action': 'failed',
                'error': str(e),
                'previous_state': check_result,
            }

    async def check_and_rebalance_all(self, force: bool = False) -> dict[str, Any]:
        """
        Check and rebalance all indexes.

        Args:
            force: Force rebalance even if not needed

        Returns:
            Dict with results for each index
        """
        results = {}
        rebalanced = 0
        errors = 0

        for index_name, cluster_index in INDEX_MAPPING.items():
            try:
                result = await self.check_and_rebalance(cluster_index, index_name, force=force)
                results[cluster_index.value] = result

                if result.get('action') == 'rebalanced':
                    rebalanced += 1
                elif result.get('status') == 'error':
                    errors += 1

            except Exception as e:
                logger.error(f'Failed to process {cluster_index.value}: {e}')
                results[cluster_index.value] = {'status': 'error', 'error': str(e)}
                errors += 1

        return {
            'status': 'complete',
            'indexes_checked': len(INDEX_MAPPING),
            'indexes_rebalanced': rebalanced,
            'errors': errors,
            'results': results,
        }

    async def _rebalance(
        self,
        cluster_index: ClusterIndex,
        opensearch_index: IndexName,
    ) -> dict[str, Any]:
        """
        Perform the actual rebalancing.

        Extracts all embeddings and re-trains the FAISS index.
        """
        import time

        start_time = time.time()

        # Extract embeddings from OpenSearch
        embeddings, doc_ids = await self.opensearch.get_all_embeddings(
            index_name=opensearch_index,
        )

        if len(embeddings) == 0:
            return {'status': 'skipped', 'reason': 'No embeddings in index'}

        # Re-train FAISS index
        stats = await self.clustering.train_index(
            index_name=cluster_index,
            embeddings=embeddings,
        )

        # Assign all documents to new clusters
        assignments = self.clustering.assign_clusters_batch(cluster_index, embeddings)
        cluster_ids = [a.cluster_id for a in assignments]
        cluster_distances = [a.distance for a in assignments]

        # Update OpenSearch with new assignments
        updated = await self.opensearch.update_cluster_assignments(
            index_name=opensearch_index,
            doc_ids=doc_ids,
            cluster_ids=cluster_ids,
            cluster_distances=cluster_distances,
        )

        elapsed = time.time() - start_time

        return {
            'status': 'success',
            'n_vectors': stats.n_vectors,
            'n_clusters': stats.n_clusters,
            'documents_updated': updated,
            'elapsed_seconds': round(elapsed, 2),
        }

    def _can_rebalance(self, cluster_index: ClusterIndex) -> bool:
        """Check if enough time has passed since last rebalance."""
        last = self._last_rebalance.get(cluster_index)
        if last is None:
            return True

        elapsed_hours = (datetime.now(UTC) - last).total_seconds() / 3600
        return elapsed_hours >= self.config.min_rebalance_interval_hours


# =============================================================================
# Convenience Functions
# =============================================================================


async def check_and_rebalance_all(
    opensearch_client: OpenSearchClient,
    pattern: IngestionPattern = IngestionPattern.MEDIUM_VOLUME,
    force: bool = False,
) -> dict[str, Any]:
    """
    Convenience function to check and rebalance all indexes.

    Use this in scheduled tasks or cron jobs.

    Args:
        opensearch_client: OpenSearch client
        pattern: Ingestion pattern for configuration
        force: Force rebalance even if not needed

    Returns:
        Results for all indexes

    Example:
        # In a scheduled task
        from src.services.cluster_maintenance import check_and_rebalance_all
        from src.clients.opensearch import OpenSearchClient

        client = OpenSearchClient()
        results = await check_and_rebalance_all(client)
        print(f"Rebalanced {results['indexes_rebalanced']} indexes")
    """
    service = ClusterMaintenanceService(opensearch_client, pattern=pattern)
    return await service.check_and_rebalance_all(force=force)


async def run_maintenance_loop(
    opensearch_client: OpenSearchClient,
    interval_hours: float = 6.0,
    pattern: IngestionPattern = IngestionPattern.MEDIUM_VOLUME,
):
    """
    Run continuous maintenance loop.

    This is a simple async loop for running maintenance. For production,
    consider using APScheduler or Celery for more robust scheduling.

    Args:
        opensearch_client: OpenSearch client
        interval_hours: Hours between maintenance runs
        pattern: Ingestion pattern for configuration

    Example:
        # Run in background
        import asyncio
        asyncio.create_task(run_maintenance_loop(client, interval_hours=6))
    """
    service = ClusterMaintenanceService(opensearch_client, pattern=pattern)
    interval_seconds = interval_hours * 3600

    logger.info(f'Starting cluster maintenance loop (interval={interval_hours}h)')

    while True:
        try:
            logger.info('Running scheduled cluster maintenance check...')
            results = await service.check_and_rebalance_all()
            logger.info(
                f'Maintenance complete: {results["indexes_rebalanced"]} rebalanced, '
                f'{results["errors"]} errors'
            )
        except Exception as e:
            logger.error(f'Maintenance loop error: {e}')

        await asyncio.sleep(interval_seconds)
