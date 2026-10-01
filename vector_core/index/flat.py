"""Flat Vector Index (Brute-Force k-Nearest Neighbors)

Serves as the golden baseline for search recall and latency comparisons.
"""

from typing import List, Tuple, Optional
import numpy as np
from vector_core.metrics.distance import (
    l2_distance,
    cosine_distance,
    inner_product_distance,
)


class FlatIndex:
    """Stores raw vectors in memory and performs linear scan exact search.

    Guarantees 100% search recall (ground truth).
    """

    def __init__(self, dim: int, metric: str = "l2"):
        self.dim = dim
        self.metric = metric.lower()
        self.vectors: Optional[np.ndarray] = None
        self.ids: List[int] = []

        if self.metric not in ["l2", "cosine", "ip"]:
            raise ValueError("Metric must be 'l2', 'cosine', or 'ip'")

    def add(self, ids: List[int], vectors: np.ndarray) -> None:
        """Adds vectors and their corresponding integer IDs into the index."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dim}"
            )
        if len(ids) != len(vectors):
            raise ValueError(
                f"Number of IDs ({len(ids)}) must match number of vectors ({len(vectors)})"
            )

        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

        self.ids.extend(ids)

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
        """Scans all indexed vectors and returns the top-k nearest IDs and distance scores."""
        if self.vectors is None or len(self.ids) == 0:
            return [], []

        query = np.ascontiguousarray(query, dtype=np.float32)

        if self.metric == "l2":
            distances = l2_distance(query, self.vectors)
        elif self.metric == "cosine":
            distances = cosine_distance(query, self.vectors)
        elif self.metric == "ip":
            distances = inner_product_distance(query, self.vectors)

        k = min(k, len(self.ids))
        top_k_indices = np.argsort(distances)[:k]

        result_ids = [self.ids[idx] for idx in top_k_indices]
        result_distances = [float(distances[idx]) for idx in top_k_indices]

        return result_ids, result_distances

    def batch_search(
        self, queries: np.ndarray, k: int = 5
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """Performs batch top-k search across multiple query vectors."""
        all_ids = []
        all_distances = []
        for q in queries:
            ids, dists = self.search(q, k=k)
            all_ids.append(ids)
            all_distances.append(dists)
        return all_ids, all_distances

    def count(self) -> int:
        return len(self.ids)