"""
Flat Vector Index (Brute-Force k-Nearest Neighbors)
Serves as the golden baseline for search recall and latency comparisons.
"""

import numpy as np
from typing import List, Tuple, Optional
from vector_core.metrics.distance import l2_distance, cosine_distance


class FlatIndex:
    """
    Stores raw vectors in memory and performs linear scan exact search.
    Guarantees 100% search recall (ground truth).
    """
    def __init__(self, dim: int, metric: str = "l2"):
        self.dim = dim
        self.metric = metric.lower()
        self.vectors: Optional[np.ndarray] = None
        self.ids: List[int] = []

        if self.metric not in ["l2", "cosine"]:
            raise ValueError("Metric must be either 'l2' or 'cosine'")

    def add(self, ids: List[int], vectors: np.ndarray) -> None:
        """Adds vectors and their corresponding integer IDs into the index."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dim}")

        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

        self.ids.extend(ids)

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Scans all indexed vectors and returns the top-k nearest IDs and distance scores.
        """
        if self.vectors is None or len(self.ids) == 0:
            return [], []

        query = np.ascontiguousarray(query, dtype=np.float32)

        if self.metric == "l2":
            distances = l2_distance(query, self.vectors)
            top_k_indices = np.argsort(distances)[:k]
        else:
            distances = cosine_distance(query, self.vectors)
            top_k_indices = np.argsort(distances)[:k]

        result_ids = [self.ids[idx] for idx in top_k_indices]
        result_distances = [float(distances[idx]) for idx in top_k_indices]

        return result_ids, result_distances

    def count(self) -> int:
        return len(self.ids)