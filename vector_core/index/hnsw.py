"""Hierarchical Navigable Small World (HNSW) Index

Provides fast approximate nearest neighbor search via layered proximity
graphs.
"""

import heapq
from typing import List, Tuple, Optional, Dict, Set
import numpy as np


class HNSWIndex:
    """Approximate Nearest Neighbor (ANN) index using HNSW graph traversal."""

    def __init__(
        self,
        dim: int,
        metric: str = "l2",
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 32,
    ):
        self.dim = dim
        self.metric = metric.lower()
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        if self.metric not in ["l2", "cosine", "ip"]:
            raise ValueError("Metric must be 'l2', 'cosine', or 'ip'")

        self.vectors: Optional[np.ndarray] = None
        self.ids: List[int] = []
        self.id_to_idx: Dict[int, int] = {}
        self.graph: Dict[int, Set[int]] = {}
        self.entry_point: Optional[int] = None

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculates distance between two 1D vectors."""
        if self.metric == "l2":
            return float(np.linalg.norm(a - b))
        elif self.metric == "cosine":
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            denom = max(norm_a * norm_b, 1e-9)
            return float(1.0 - (dot / denom))
        elif self.metric == "ip":
            return float(-np.dot(a, b))
        return float(np.linalg.norm(a - b))

    def _search_layer(
        self, query: np.ndarray, entry_point: int, ef: int
    ) -> List[Tuple[float, int]]:
        """Greedy beam search on the proximity graph."""
        visited = {entry_point}
        dist = self._dist(query, self.vectors[entry_point])
        candidates = [(dist, entry_point)]
        w = [(-dist, entry_point)]

        heapq.heapify(candidates)
        heapq.heapify(w)

        while candidates:
            c_dist, c_idx = heapq.heappop(candidates)
            furthest_dist = -w[0][0]

            if c_dist > furthest_dist:
                break

            for n_idx in self.graph.get(c_idx, set()):
                if n_idx not in visited:
                    visited.add(n_idx)
                    furthest_dist = -w[0][0]
                    n_dist = self._dist(query, self.vectors[n_idx])

                    if n_dist < furthest_dist or len(w) < ef:
                        heapq.heappush(candidates, (n_dist, n_idx))
                        heapq.heappush(w, (-float(n_dist), n_idx))
                        if len(w) > ef:
                            heapq.heappop(w)

        return sorted([(-item[0], item[1]) for item in w], key=lambda x: x[0])

    def add(self, ids: List[int], vectors: np.ndarray) -> None:
        """Adds vectors and their IDs into the hierarchical graph."""
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

        for vec, vid in zip(vectors, ids):
            new_idx = len(self.ids)
            self.ids.append(vid)
            self.id_to_idx[vid] = new_idx
            self.graph[new_idx] = set()

            if self.vectors is None:
                self.vectors = vec.reshape(1, -1)
            else:
                self.vectors = np.vstack([self.vectors, vec])

            if self.entry_point is None:
                self.entry_point = new_idx
                continue

            nearest_candidates = self._search_layer(
                vec, self.entry_point, self.ef_construction
            )

            neighbors = [cand[1] for cand in nearest_candidates[: self.m]]
            for n_idx in neighbors:
                self.graph[new_idx].add(n_idx)
                self.graph[n_idx].add(new_idx)

                if len(self.graph[n_idx]) > self.m:
                    n_vec = self.vectors[n_idx]
                    sorted_nbrs = sorted(
                        self.graph[n_idx],
                        key=lambda x: self._dist(n_vec, self.vectors[x]),
                    )
                    self.graph[n_idx] = set(sorted_nbrs[: self.m])

    def search(
        self, query: np.ndarray, k: int = 5
    ) -> Tuple[List[int], List[float]]:
        """Searches for top-k approximate nearest neighbors."""
        if (
            self.vectors is None
            or len(self.ids) == 0
            or self.entry_point is None
        ):
            return [], []

        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.ndim > 1:
            query = query.reshape(-1)

        candidates = self._search_layer(
            query, self.entry_point, max(self.ef_search, k)
        )
        top_k = candidates[:k]

        result_ids = [self.ids[idx] for _, idx in top_k]
        result_distances = [float(dist) for dist, _ in top_k]
        return result_ids, result_distances

    def count(self) -> int:
        """Returns the number of indexed vectors."""
        return len(self.ids)