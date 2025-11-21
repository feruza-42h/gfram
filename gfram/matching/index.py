"""
Face Indexing and Matching using FAISS.

Provides efficient similarity search for face embeddings.
"""

import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pickle
import logging

logger = logging.getLogger(__name__)


class FaceIndex:
    """
    Face database with efficient similarity search.

    Uses FAISS for fast nearest neighbor search.
    """

    def __init__(
            self,
            dimension: int,
            index_type: str = "HNSW",
            metric: str = "cosine"
    ):
        """
        Initialize face index.

        Args:
            dimension: Embedding dimension.
            index_type: FAISS index type ('Flat', 'HNSW', 'IVF').
            metric: Distance metric ('euclidean', 'cosine').
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric

        # Create FAISS index
        self.index = self._create_index()

        # Store names and metadata
        self.names = []
        self.metadata = []
        self.name_to_ids = {}  # Maps name to list of IDs

        logger.info(f"FaceIndex created: dim={dimension}, type={index_type}, metric={metric}")

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on type and metric."""
        if self.metric == "cosine":
            # For cosine similarity, we'll normalize vectors and use L2
            if self.index_type == "Flat":
                index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "HNSW":
                index = faiss.IndexHNSWFlat(self.dimension, 32)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 50
            elif self.index_type == "IVF":
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
        else:  # euclidean
            if self.index_type == "Flat":
                index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "HNSW":
                index = faiss.IndexHNSWFlat(self.dimension, 32)
            elif self.index_type == "IVF":
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")

        return index

    def add(
            self,
            name: str,
            embeddings: np.ndarray,
            metadata: Optional[Dict] = None
    ):
        """
        Add face embeddings to the index.

        Args:
            name: Person's name/identifier.
            embeddings: Embedding vectors (N, D) or (D,).
            metadata: Optional metadata dictionary.
        """
        # Ensure embeddings is 2D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Normalize if using cosine metric
        if self.metric == "cosine":
            embeddings = self._normalize(embeddings)

        # Convert to float32
        embeddings = embeddings.astype('float32')

        # Get current size before adding
        start_id = len(self.names)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store names and metadata
        for i in range(len(embeddings)):
            self.names.append(name)
            self.metadata.append(metadata or {})

        # Update name mapping
        ids = list(range(start_id, start_id + len(embeddings)))
        if name in self.name_to_ids:
            self.name_to_ids[name].extend(ids)
        else:
            self.name_to_ids[name] = ids

        logger.info(f"Added {len(embeddings)} embeddings for '{name}'")

    def search(
            self,
            query: np.ndarray,
            k: int = 5,
            threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for similar faces.

        Args:
            query: Query embedding vector.
            k: Number of results to return.
            threshold: Optional distance threshold.

        Returns:
            List of matches with name, distance, and metadata.
        """
        if self.index.ntotal == 0:
            return []

        # Ensure query is 2D
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Normalize if using cosine metric
        if self.metric == "cosine":
            query = self._normalize(query)

        # Convert to float32
        query = query.astype('float32')

        # Search
        distances, indices = self.index.search(query, min(k, self.index.ntotal))

        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            # Convert distance to similarity
            if self.metric == "cosine":
                similarity = 1.0 - dist
            else:
                # For Euclidean, convert to similarity (inverse exponential)
                similarity = np.exp(-dist)

            # Apply threshold
            if threshold is not None and similarity < threshold:
                continue

            result = {
                'name': self.names[idx],
                'distance': float(dist),
                'similarity': float(similarity),
                'metadata': self.metadata[idx],
                'id': int(idx)
            }
            results.append(result)

        return results

    def remove(self, name: str) -> int:
        """
        Remove all embeddings for a person.

        Note: FAISS doesn't support removal, so this rebuilds the index.

        Args:
            name: Person's name to remove.

        Returns:
            Number of embeddings removed.
        """
        if name not in self.name_to_ids:
            logger.warning(f"Name '{name}' not found in index")
            return 0

        # Get IDs to remove
        ids_to_remove = set(self.name_to_ids[name])

        # Collect all embeddings except those to remove
        all_embeddings = []
        new_names = []
        new_metadata = []
        new_name_to_ids = {}

        for i in range(len(self.names)):
            if i not in ids_to_remove:
                # Reconstruct from index (FAISS stores vectors)
                vec = self.index.reconstruct(int(i))
                all_embeddings.append(vec)
                new_names.append(self.names[i])
                new_metadata.append(self.metadata[i])

        # Rebuild name_to_ids mapping
        for i, n in enumerate(new_names):
            if n in new_name_to_ids:
                new_name_to_ids[n].append(i)
            else:
                new_name_to_ids[n] = [i]

        # Recreate index
        self.index = self._create_index()
        self.names = []
        self.metadata = []
        self.name_to_ids = {}

        # Add back
        if all_embeddings:
            all_embeddings = np.array(all_embeddings)
            self.index.add(all_embeddings)
            self.names = new_names
            self.metadata = new_metadata
            self.name_to_ids = new_name_to_ids

        removed_count = len(ids_to_remove)
        logger.info(f"Removed {removed_count} embeddings for '{name}'")
        return removed_count

    def get_stats(self) -> Dict:
        """Get index statistics."""
        unique_names = len(self.name_to_ids)
        total_embeddings = self.index.ntotal

        return {
            'total_embeddings': int(total_embeddings),
            'unique_identities': unique_names,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
        }

    def save(self, path: str):
        """
        Save index to disk.

        Args:
            path: Directory path to save index.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))

        # Save metadata
        metadata = {
            'names': self.names,
            'metadata': self.metadata,
            'name_to_ids': self.name_to_ids,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
        }

        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Index saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'FaceIndex':
        """
        Load index from disk.

        Args:
            path: Directory path containing saved index.

        Returns:
            Loaded FaceIndex instance.
        """
        path = Path(path)

        # Load metadata
        with open(path / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        # Create instance
        instance = cls(
            dimension=metadata['dimension'],
            index_type=metadata['index_type'],
            metric=metadata['metric']
        )

        # Load FAISS index
        instance.index = faiss.read_index(str(path / "faiss.index"))

        # Restore metadata
        instance.names = metadata['names']
        instance.metadata = metadata['metadata']
        instance.name_to_ids = metadata['name_to_ids']

        logger.info(f"Index loaded from {path}")
        return instance

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms

    def __len__(self) -> int:
        """Return number of embeddings in index."""
        return self.index.ntotal

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"FaceIndex(embeddings={stats['total_embeddings']}, "
            f"identities={stats['unique_identities']}, "
            f"dim={stats['dimension']}, "
            f"type={stats['index_type']})"
        )


class DistanceMetrics:
    """
    Common distance metrics for face matching.
    """

    @staticmethod
    def euclidean(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance."""
        return np.linalg.norm(a - b)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance (1 - cosine similarity)."""
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return 1.0 - sim

    @staticmethod
    def manhattan(a: np.ndarray, b: np.ndarray) -> float:
        """Manhattan (L1) distance."""
        return np.sum(np.abs(a - b))

    @staticmethod
    def chebyshev(a: np.ndarray, b: np.ndarray) -> float:
        """Chebyshev (L-infinity) distance."""
        return np.max(np.abs(a - b))