"""
Hierarchical Navigable Small World (HNSW) Graph Index
Implements greedy graph traversal and dynamic edge linking for sub-millisecond approximate nearest neighbor search.
"""

import heapq
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from vector_core.metrics.distance import l2_distance, cosine_distance


class HNSWIndex:
    """
    Graph-based approximate nearest neighbor search index.
    """
    def __init__(self, dim: int, metric: str = "l2", m: int = 16, ef_construction: int = 64, ef_search: int = 32):
        self.dim = dim
        self.metric = metric.lower()
        self.m = m                          # Max bidirectional connections per node
        self.ef_construction = ef_construction  # Search beam width during construction
        self.ef_search = ef_search              # Search beam width during query time

        self.vectors: Optional[np.ndarray] = None
        self.ids: List[int] = []
        self.id_to_idx: Dict[int, int] = {}
        self.graph: Dict[int, Set[int]] = {}   # Adjacency list: internal_idx -> set of neighbor internal_indices
        self.entry_point: Optional[int] = None  # Internal index of graph entry node

    def _compute_dist(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        if self.metric == "l2":
            return l2_distance(query, candidates)
        return cosine_distance(query, candidates)

    def _search_layer(self, query: np.ndarray, entry_idx: int, ef: int) -> List[Tuple[float, int]]:
        """
        Greedy beam search traversal starting from entry_idx.
        Returns the top 'ef' closest candidates as (distance, internal_idx) pairs.
        """
        dist_entry = float(self._compute_dist(query, self.vectors[entry_idx : entry_idx + 1])[0])
        
        visited: Set[int] = {entry_idx}
        # Min-heap for candidate exploration: (distance, internal_idx)
        candidates = [(dist_entry, entry_idx)]
        # Max-heap for best results: (-distance, internal_idx)
        w = [(-dist_entry, entry_idx)]

        while candidates:
            c_dist, c_idx = heapq.heappop(candidates)
            furthest_dist = -w[0][0]

            if c_dist > furthest_dist:
                break

            neighbors = self.graph.get(c_idx, set())
            unvisited = [n for n in neighbors if n not in visited]
            
            if not unvisited:
                continue

            for n_idx in unvisited:
                visited.add(n_idx)

            neighbor_vectors = self.vectors[unvisited]
            n_dists = self._compute_dist(query, neighbor_vectors)

            for n_idx, n_dist in zip(unvisited, n_dists):
                furthest_dist = -w[0][0]
                if n_dist < furthest_dist or len(w) < ef:
                    heapq.heappush(candidates, (float(n_dist), n_idx))
                    heapq.heappush(w, (-float(n_dist), n_idx))
                    if len(w) > ef:
                        heapq.heappop(w)

        # Convert back to sorted ascending distances: (distance, internal_idx)
        return sorted([(-item[0], item[1]) for item in w], key=lambda x: x[0])

    def add(self, ids: List[int], vectors: np.ndarray) -> None:
        """Inserts vectors into the graph and constructs neighbor connections."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        for vec, vid in zip(vectors, ids):
            new_idx = len(self.ids)
            self.ids.append(vid)
            self.id_to_idx[vid] = new_idx
            self.graph[new_idx] = set()

            if self.vectors is None:
                self.vectors = vec.reshape(1, -1)
            else:
                self.vectors = np.vstack([self.vectors, vec])

            # First node becomes initial graph entry point
            if self.entry_point is None:
                self.entry_point = new_idx
                continue

            # Traverse graph to find nearest neighbors for the new node
            nearest_candidates = self._search_layer(vec, self.entry_point, self.ef_construction)
            
            # Connect the new node to top-M closest neighbors
            neighbors = [cand[1] for cand in nearest_candidates[: self.m]]
            for n_idx in neighbors:
                self.graph[new_idx].add(n_idx)
                self.graph[n_idx].add(new_idx)

                # Prune edges if a neighbor exceeds max connection capacity M
                if len(self.graph[n_idx]) > self.m:
                    n_vec = self.vectors[n_idx]
                    connected = list(self.graph[n_idx])
                    dists = self._compute_dist(n_vec, self.vectors[connected])
                    closest_m = np.argsort(dists)[: self.m]
                    self.graph[n_idx] = {connected[i] for i in closest_m}

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
        """Searches the graph for the k-nearest vectors."""
        if self.vectors is None or self.entry_point is None:
            return [], []

        query = np.ascontiguousarray(query, dtype=np.float32)
        candidates = self._search_layer(query, self.entry_point, max(self.ef_search, k))
        top_k = candidates[:k]

        result_ids = [self.ids[cand[1]] for cand in top_k]
        result_distances = [cand[0] for cand in top_k]

        return result_ids, result_distances

    def count(self) -> int:
        return len(self.ids)