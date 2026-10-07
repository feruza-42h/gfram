"""
Face indexing and matching.

Exact similarity search for face embeddings (numpy).
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pickle
import logging

logger = logging.getLogger(__name__)


class FaceIndex:
    """
    Face database with exact similarity search (numpy).

    Exact search over a dense matrix is fast for the database sizes a face
    recognizer handles (thousands of faces) and avoids loading FAISS next to
    PyTorch: on macOS both ship their own OpenMP runtime and the process aborts.
    """

    def __init__(
            self,
            dimension: int,
            index_type: str = "Flat",
            metric: str = "cosine"
    ):
        """
        Initialize face index.

        Args:
            dimension: Embedding dimension.
            index_type: Kept for compatibility ('Flat', 'HNSW', 'IVF'); search is always exact.
            metric: Distance metric ('euclidean', 'cosine').
        """
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"Unknown metric: {metric}")
        if index_type not in ("Flat", "HNSW", "IVF"):
            raise ValueError(f"Unknown index type: {index_type}")
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric

        self.vectors = np.zeros((0, dimension), dtype=np.float32)
        self.names = []
        self.metadata = []
        self.name_to_ids = {}

        logger.info(f"FaceIndex created: dim={dimension}, type={index_type}, metric={metric}")

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
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {embeddings.shape[1]}")
        if self.metric == "cosine":
            embeddings = self._normalize(embeddings)

        start_id = len(self.names)
        self.vectors = np.vstack([self.vectors, embeddings])
        for _ in range(len(embeddings)):
            self.names.append(name)
            self.metadata.append(metadata or {})
        self.name_to_ids.setdefault(name, []).extend(range(start_id, start_id + len(embeddings)))

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
            threshold: Optional similarity threshold.

        Returns:
            List of matches with name, similarity, and metadata (best first).
        """
        if len(self.names) == 0:
            return []

        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self.metric == "cosine":
            scores = (self.vectors @ self._normalize(query)[0])
            order = np.argsort(-scores)
        else:
            scores = ((self.vectors - query) ** 2).sum(axis=1)  # squared L2, as FAISS IndexFlatL2
            order = np.argsort(scores)

        results = []
        for idx in order[:min(k, len(order))]:
            score = float(scores[idx])
            similarity = score if self.metric == "cosine" else float(np.exp(-score))
            if threshold is not None and similarity < threshold:
                continue
            results.append({
                'name': self.names[idx],
                'distance': float(1.0 - similarity) if self.metric == "cosine" else score,
                'similarity': similarity,
                'metadata': self.metadata[idx],
                'id': int(idx)
            })
        return results

    def remove(self, name: str) -> int:
        """
        Remove all embeddings for a person.

        Args:
            name: Person's name to remove.

        Returns:
            Number of embeddings removed.
        """
        if name not in self.name_to_ids:
            logger.warning(f"Name '{name}' not found in index")
            return 0

        drop = set(self.name_to_ids[name])
        keep = [i for i in range(len(self.names)) if i not in drop]
        self.vectors = self.vectors[keep]
        self.names = [self.names[i] for i in keep]
        self.metadata = [self.metadata[i] for i in keep]
        self.name_to_ids = {}
        for i, n in enumerate(self.names):
            self.name_to_ids.setdefault(n, []).append(i)

        logger.info(f"Removed {len(drop)} embeddings for '{name}'")
        return len(drop)

    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            'total_embeddings': len(self.names),
            'unique_identities': len(self.name_to_ids),
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
        np.save(path / "embeddings.npy", self.vectors)

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
        Load index from disk (also reads indexes saved by older FAISS-based versions
        when FAISS is installed).

        Args:
            path: Directory path containing saved index.

        Returns:
            Loaded FaceIndex instance.
        """
        path = Path(path)
        with open(path / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        instance = cls(metadata['dimension'], metadata['index_type'], metadata['metric'])
        if (path / "embeddings.npy").exists():
            instance.vectors = np.load(path / "embeddings.npy").astype(np.float32)
        elif (path / "faiss.index").exists():
            try:
                import faiss
            except ImportError as e:
                raise ImportError(f"{path} was saved by an older FAISS-based gfram; "
                                  "install faiss-cpu once to read it, then save it again") from e
            legacy = faiss.read_index(str(path / "faiss.index"))
            instance.vectors = legacy.reconstruct_n(0, legacy.ntotal).astype(np.float32)
        instance.names = metadata['names']
        instance.metadata = metadata['metadata']
        instance.name_to_ids = metadata['name_to_ids']

        logger.info(f"Index loaded from {path}: {len(instance.names)} embeddings")
        return instance

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms

    def __len__(self) -> int:
        """Return number of embeddings in index."""
        return len(self.names)

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