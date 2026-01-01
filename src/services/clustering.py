"""
FAISS-based clustering service for visual search.

Industry-standard clustering using Inverted File (IVF) indexes,
the same approach used by Google, Meta, and other Fortune 100 companies
for billion-scale visual search.

Key Features:
- GPU-accelerated training and assignment
- Incremental cluster assignment (no re-training needed for new items)
- Persistent index storage to disk
- Automatic rebalancing when cluster distribution becomes uneven
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


class ClusterIndex(str, Enum):
    """FAISS cluster index names matching OpenSearch indexes."""

    GLOBAL = 'global'
    VEHICLES = 'vehicles'
    PEOPLE = 'people'
    FACES = 'faces'


@dataclass
class ClusterConfig:
    """Configuration for a category's clustering."""

    n_clusters: int
    nprobe: int  # Number of clusters to search during queries
    rebalance_threshold: float  # Ratio of new items that triggers rebalance


# Default configurations based on expected data distribution
# Rule of thumb: n_clusters ≈ sqrt(n_vectors) to n_vectors/10 for fine separation
# More clusters = better separation but slightly more memory
DEFAULT_CONFIGS: dict[ClusterIndex, ClusterConfig] = {
    ClusterIndex.GLOBAL: ClusterConfig(n_clusters=1024, nprobe=32, rebalance_threshold=0.5),
    ClusterIndex.VEHICLES: ClusterConfig(
        n_clusters=512, nprobe=24, rebalance_threshold=0.5
    ),  # Higher for brand/type separation
    ClusterIndex.PEOPLE: ClusterConfig(n_clusters=512, nprobe=24, rebalance_threshold=0.5),
    ClusterIndex.FACES: ClusterConfig(
        n_clusters=2048, nprobe=64, rebalance_threshold=0.5
    ),  # Highest for face identity
}


@dataclass
class ClusterStats:
    """Statistics about a FAISS cluster index."""

    index_name: str
    n_clusters: int
    n_vectors: int
    avg_cluster_size: float
    min_cluster_size: int
    max_cluster_size: int
    empty_clusters: int
    trained_at: str | None
    is_trained: bool


@dataclass
class ClusterBalance:
    """Cluster balance assessment."""

    index_name: str
    is_balanced: bool
    imbalance_ratio: float  # max_size / min_size (excluding empty)
    empty_ratio: float  # empty_clusters / n_clusters
    vectors_since_training: int
    needs_rebalance: bool
    reason: str | None


@dataclass
class ClusterAssignment:
    """Result of cluster assignment."""

    cluster_id: int
    distance: float  # Distance to centroid (lower = closer to cluster center)


class ClusteringService:
    """
    FAISS-based clustering for all visual search indexes.

    Uses IVF (Inverted File) indexes which allow:
    - One-time training on initial data
    - O(1) cluster assignment for new items
    - Efficient search by probing only relevant clusters

    This is the industry-standard approach for billion-scale systems.
    """

    def __init__(
        self,
        index_dir: str | Path = 'faiss_indexes',
        embedding_dim: int = 512,
        use_gpu: bool = True,
    ):
        """
        Initialize the clustering service.

        Args:
            index_dir: Directory for persistent FAISS index storage
            embedding_dim: Dimension of embeddings (512 for MobileCLIP)
            use_gpu: Whether to use GPU acceleration
        """
        self.index_dir = Path(index_dir)
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.configs = DEFAULT_CONFIGS.copy()

        # Lazy-loaded FAISS
        self._faiss = None
        self._gpu_resources = None

        # In-memory index cache
        self._indexes: dict[ClusterIndex, object] = {}
        self._training_metadata: dict[ClusterIndex, dict] = {}

        # Ensure index directory exists
        self.index_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f'ClusteringService initialized: index_dir={self.index_dir}, gpu={use_gpu}')

    @property
    def faiss(self):
        """Lazy-load FAISS to avoid import errors if not installed."""
        if self._faiss is None:
            try:
                import faiss

                self._faiss = faiss
                logger.info(f'FAISS loaded: GPU available = {faiss.get_num_gpus() > 0}')
            except ImportError as e:
                raise ImportError(
                    'faiss-gpu is required for clustering. Install with: pip install faiss-gpu'
                ) from e
        return self._faiss

    @property
    def gpu_resources(self):
        """Get GPU resources for FAISS (lazy-loaded)."""
        if self._gpu_resources is None and self.use_gpu:
            if self.faiss.get_num_gpus() > 0:
                self._gpu_resources = self.faiss.StandardGpuResources()
                logger.info('FAISS GPU resources initialized')
            else:
                logger.warning('No GPU available for FAISS, falling back to CPU')
        return self._gpu_resources

    # =========================================================================
    # TRAINING
    # =========================================================================

    async def train_index(
        self,
        index_name: ClusterIndex,
        embeddings: np.ndarray,
        n_clusters: int | None = None,
    ) -> ClusterStats:
        """
        Train a FAISS IVF index from embeddings.

        This is called once initially, then periodically for rebalancing.
        Training time scales with embedding count but is typically fast:
        - 100K: ~2s
        - 1M: ~15s
        - 10M: ~120s

        Args:
            index_name: Which index to train
            embeddings: Training embeddings (N x embedding_dim)
            n_clusters: Number of clusters (uses default if None)

        Returns:
            ClusterStats with training results
        """
        n_vectors = len(embeddings)

        if n_clusters is None:
            # Auto-calculate optimal clusters: between sqrt(n) and n/10
            # More clusters = finer separation (better for brand/type grouping)
            default_n = self.configs[index_name].n_clusters
            min_clusters = max(16, int(np.sqrt(n_vectors)))  # At least sqrt(n)
            max_clusters = max(min_clusters, n_vectors // 10)  # Up to n/10
            # Use default if it's within range, otherwise auto-scale
            if min_clusters <= default_n <= max_clusters:
                n_clusters = default_n
            else:
                n_clusters = min(max_clusters, max(min_clusters, default_n))
            logger.info(f'Auto-calculated n_clusters={n_clusters} for {n_vectors} vectors')

        # Ensure we don't have more clusters than vectors
        n_clusters = min(n_clusters, n_vectors)

        logger.info(f'Training {index_name} index: {n_vectors} vectors → {n_clusters} clusters')

        # Validate input
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f'Expected shape (N, {self.embedding_dim}), got {embeddings.shape}')

        # Ensure float32 for FAISS
        embeddings = embeddings.astype(np.float32)

        # Normalize embeddings for cosine similarity (FAISS uses inner product)
        embeddings = self._normalize(embeddings)

        # Run training in executor to avoid blocking
        loop = asyncio.get_event_loop()
        index = await loop.run_in_executor(
            None,
            self._train_index_sync,
            index_name,
            embeddings,
            n_clusters,
        )

        # Store index and metadata
        self._indexes[index_name] = index
        self._training_metadata[index_name] = {
            'trained_at': datetime.now(UTC).isoformat(),
            'n_vectors': n_vectors,
            'n_clusters': n_clusters,
        }

        # Save to disk
        self._save_index(index_name)

        return self._compute_stats(index_name)

    def _train_index_sync(
        self,
        index_name: ClusterIndex,
        embeddings: np.ndarray,
        n_clusters: int,
    ):
        """Synchronous training (runs in executor)."""
        # Create IVF index with flat quantizer
        # IndexFlatIP = flat index using inner product (for normalized vectors = cosine)
        quantizer = self.faiss.IndexFlatIP(self.embedding_dim)
        index = self.faiss.IndexIVFFlat(
            quantizer,
            self.embedding_dim,
            n_clusters,
            self.faiss.METRIC_INNER_PRODUCT,
        )

        # Move to GPU for faster training
        if self.use_gpu and self.gpu_resources is not None:
            index = self.faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
            logger.info(f'Training {index_name} on GPU')
        else:
            logger.info(f'Training {index_name} on CPU')

        # Set verbose to see training progress
        index.verbose = True

        # Train on embeddings (learns cluster centroids via k-means)
        # Default is 25 iterations; we don't need to change as FAISS auto-optimizes
        index.train(embeddings)

        # Add all vectors to the index
        index.add(embeddings)

        # Move back to CPU for storage
        if self.use_gpu and self.gpu_resources is not None:
            index = self.faiss.index_gpu_to_cpu(index)

        return index

    # =========================================================================
    # INCREMENTAL ASSIGNMENT
    # =========================================================================

    def assign_cluster(
        self,
        index_name: ClusterIndex,
        embedding: np.ndarray,
    ) -> ClusterAssignment:
        """
        Assign a single embedding to its nearest cluster.

        Time complexity: O(n_clusters) ≈ 0.1ms for 1024 clusters

        Args:
            index_name: Which index to use
            embedding: Single embedding vector (embedding_dim,)

        Returns:
            ClusterAssignment with cluster_id and distance
        """
        index = self._get_index(index_name)

        # Reshape and normalize
        embedding = embedding.reshape(1, -1).astype(np.float32)
        embedding = self._normalize(embedding)

        # Search quantizer for nearest centroid
        distances, cluster_ids = index.quantizer.search(embedding, 1)

        return ClusterAssignment(
            cluster_id=int(cluster_ids[0, 0]),
            distance=float(1.0 - distances[0, 0]),  # Convert IP to distance
        )

    def assign_clusters_batch(
        self,
        index_name: ClusterIndex,
        embeddings: np.ndarray,
    ) -> list[ClusterAssignment]:
        """
        Batch assign embeddings to clusters.

        Time complexity: O(n_clusters * batch_size) ≈ 1ms for 100 embeddings

        Args:
            index_name: Which index to use
            embeddings: Batch of embeddings (N x embedding_dim)

        Returns:
            List of ClusterAssignment for each embedding
        """
        index = self._get_index(index_name)

        # Normalize
        embeddings = embeddings.astype(np.float32)
        embeddings = self._normalize(embeddings)

        # Search quantizer for nearest centroids
        distances, cluster_ids = index.quantizer.search(embeddings, 1)

        return [
            ClusterAssignment(
                cluster_id=int(cluster_ids[i, 0]),
                distance=float(1.0 - distances[i, 0]),
            )
            for i in range(len(embeddings))
        ]

    # =========================================================================
    # CLUSTER SEARCH
    # =========================================================================

    def search_similar_clusters(
        self,
        index_name: ClusterIndex,
        query_embedding: np.ndarray,
        n_clusters: int | None = None,
    ) -> list[int]:
        """
        Find clusters most similar to a query embedding.

        Use this to narrow down which clusters to search in OpenSearch.

        Args:
            index_name: Which index to use
            query_embedding: Query embedding vector
            n_clusters: Number of clusters to return (uses nprobe default if None)

        Returns:
            List of cluster IDs ordered by similarity
        """
        index = self._get_index(index_name)

        if n_clusters is None:
            n_clusters = self.configs[index_name].nprobe

        # Reshape and normalize
        query = query_embedding.reshape(1, -1).astype(np.float32)
        query = self._normalize(query)

        # Search quantizer
        _, cluster_ids = index.quantizer.search(query, n_clusters)

        return cluster_ids[0].tolist()

    def get_cluster_centroid(
        self,
        index_name: ClusterIndex,
        cluster_id: int,
    ) -> np.ndarray:
        """
        Get the centroid (representative vector) for a cluster.

        Args:
            index_name: Which index to use
            cluster_id: Cluster ID

        Returns:
            Centroid embedding vector
        """
        index = self._get_index(index_name)

        # Extract centroid from quantizer
        return index.quantizer.reconstruct(cluster_id)

    # =========================================================================
    # REBALANCING
    # =========================================================================

    async def check_balance(self, index_name: ClusterIndex) -> ClusterBalance:
        """
        Check if clusters need rebalancing.

        Rebalancing is recommended when:
        - Max cluster > 10x average cluster size
        - Empty clusters > 10% of total
        - More than rebalance_threshold new data since training
        """
        index = self._get_index(index_name)
        config = self.configs[index_name]

        # Get cluster sizes
        cluster_sizes = self._get_cluster_sizes(index)
        non_empty_sizes = [s for s in cluster_sizes if s > 0]

        n_clusters = len(cluster_sizes)
        empty_clusters = cluster_sizes.count(0)
        empty_ratio = empty_clusters / n_clusters if n_clusters > 0 else 0

        # Calculate imbalance
        if non_empty_sizes:
            min_size = min(non_empty_sizes)
            max_size = max(non_empty_sizes)
            imbalance_ratio = max_size / min_size if min_size > 0 else float('inf')
        else:
            min_size = max_size = 0
            imbalance_ratio = 1.0

        # Check for vectors added since training
        metadata = self._training_metadata.get(index_name, {})
        trained_vectors = metadata.get('n_vectors', 0)
        current_vectors = index.ntotal
        vectors_since_training = current_vectors - trained_vectors

        # Determine if rebalance needed
        needs_rebalance = False
        reason = None

        if imbalance_ratio > 10:
            needs_rebalance = True
            reason = f'High imbalance: max cluster is {imbalance_ratio:.1f}x larger than min'
        elif empty_ratio > 0.1:
            needs_rebalance = True
            reason = f'Too many empty clusters: {empty_ratio:.1%}'
        elif (
            trained_vectors > 0
            and vectors_since_training / trained_vectors > config.rebalance_threshold
        ):
            needs_rebalance = True
            reason = f'Significant new data: {vectors_since_training} vectors since training'

        return ClusterBalance(
            index_name=index_name.value,
            is_balanced=not needs_rebalance,
            imbalance_ratio=imbalance_ratio,
            empty_ratio=empty_ratio,
            vectors_since_training=vectors_since_training,
            needs_rebalance=needs_rebalance,
            reason=reason,
        )

    async def rebalance(
        self,
        index_name: ClusterIndex,
        embeddings: np.ndarray,
    ) -> ClusterStats:
        """
        Rebalance clusters by re-training from current data.

        Args:
            index_name: Which index to rebalance
            embeddings: All current embeddings to re-cluster

        Returns:
            New ClusterStats after rebalancing
        """
        logger.info(f'Rebalancing {index_name} index with {len(embeddings)} embeddings')

        # Re-train with current data
        return await self.train_index(index_name, embeddings)

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _save_index(self, index_name: ClusterIndex) -> None:
        """Save index to disk."""
        index = self._indexes.get(index_name)
        if index is None:
            return

        index_path = self.index_dir / f'{index_name.value}.index'
        self.faiss.write_index(index, str(index_path))

        # Save metadata
        metadata_path = self.index_dir / f'{index_name.value}.meta.npy'
        np.save(metadata_path, self._training_metadata.get(index_name, {}))

        logger.info(f'Saved {index_name} index to {index_path}')

    def load_index(self, index_name: ClusterIndex) -> bool:
        """
        Load index from disk.

        Returns:
            True if loaded successfully, False if not found
        """
        index_path = self.index_dir / f'{index_name.value}.index'
        metadata_path = self.index_dir / f'{index_name.value}.meta.npy'

        if not index_path.exists():
            logger.warning(f'No saved index found for {index_name}')
            return False

        try:
            index = self.faiss.read_index(str(index_path))
            self._indexes[index_name] = index

            if metadata_path.exists():
                self._training_metadata[index_name] = np.load(
                    metadata_path, allow_pickle=True
                ).item()

            logger.info(f'Loaded {index_name} index from {index_path}')
            return True
        except Exception as e:
            logger.error(f'Failed to load {index_name} index: {e}')
            return False

    def load_all_indexes(self) -> dict[ClusterIndex, bool]:
        """Load all indexes from disk."""
        results = {}
        for index_name in ClusterIndex:
            results[index_name] = self.load_index(index_name)
        return results

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def _compute_stats(self, index_name: ClusterIndex) -> ClusterStats:
        """Compute statistics for an index."""
        index = self._indexes.get(index_name)
        if index is None:
            return ClusterStats(
                index_name=index_name.value,
                n_clusters=0,
                n_vectors=0,
                avg_cluster_size=0,
                min_cluster_size=0,
                max_cluster_size=0,
                empty_clusters=0,
                trained_at=None,
                is_trained=False,
            )

        cluster_sizes = self._get_cluster_sizes(index)
        n_clusters = len(cluster_sizes)
        n_vectors = index.ntotal

        non_empty = [s for s in cluster_sizes if s > 0]
        empty_clusters = cluster_sizes.count(0)

        metadata = self._training_metadata.get(index_name, {})

        return ClusterStats(
            index_name=index_name.value,
            n_clusters=n_clusters,
            n_vectors=n_vectors,
            avg_cluster_size=n_vectors / n_clusters if n_clusters > 0 else 0,
            min_cluster_size=min(non_empty) if non_empty else 0,
            max_cluster_size=max(non_empty) if non_empty else 0,
            empty_clusters=empty_clusters,
            trained_at=metadata.get('trained_at'),
            is_trained=True,
        )

    def get_stats(self, index_name: ClusterIndex) -> ClusterStats:
        """Get statistics for an index."""
        return self._compute_stats(index_name)

    def get_all_stats(self) -> dict[ClusterIndex, ClusterStats]:
        """Get statistics for all indexes."""
        return {name: self._compute_stats(name) for name in ClusterIndex}

    def is_trained(self, index_name: ClusterIndex) -> bool:
        """Check if an index has been trained."""
        return index_name in self._indexes and self._indexes[index_name].is_trained

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _get_index(self, index_name: ClusterIndex):
        """Get index, loading from disk if necessary."""
        if index_name not in self._indexes and not self.load_index(index_name):
            raise ValueError(f'Index {index_name} not found. Train it first with train_index().')
        return self._indexes[index_name]

    def _get_cluster_sizes(self, index) -> list[int]:
        """Get the size of each cluster in the index."""
        # For IVF indexes, we can get list sizes
        return [index.invlists.list_size(i) for i in range(index.nlist)]

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return embeddings / norms


# Singleton instance for app-wide use
_clustering_service: ClusteringService | None = None


def get_clustering_service(
    index_dir: str | Path = 'faiss_indexes',
    use_gpu: bool = True,
) -> ClusteringService:
    """Get or create the global ClusteringService instance."""
    global _clustering_service  # noqa: PLW0603 - singleton pattern
    if _clustering_service is None:
        _clustering_service = ClusteringService(index_dir=index_dir, use_gpu=use_gpu)
    return _clustering_service
