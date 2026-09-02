import heapq
import pickle
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from vector_core.metrics.distance import cosine_distance, l2_distance


class HNSWIndex:

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

    self.vectors: Optional[np.ndarray] = None
    self.ids: List[int] = []
    self.id_to_idx: Dict[int, int] = {}
    self.graph: Dict[int, Set[int]] = {}
    self.entry_point: Optional[int] = None

  def _compute_dist(
      self, query: np.ndarray, candidates: np.ndarray
  ) -> np.ndarray:
    if self.metric == "l2":
      return l2_distance(query, candidates)
    return cosine_distance(query, candidates)

  def _search_layer(
      self, query: np.ndarray, entry_idx: int, ef: int
  ) -> List[Tuple[float, int]]:
    dist_entry = float(
        self._compute_dist(query, self.vectors[entry_idx : entry_idx + 1])[0]
    )

    visited: Set[int] = {entry_idx}
    candidates = [(dist_entry, entry_idx)]
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

    return sorted([(-item[0], item[1]) for item in w], key=lambda x: x[0])

  def add(self, ids: List[int], vectors: np.ndarray) -> None:
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
          connected = list(self.graph[n_idx])
          dists = self._compute_dist(n_vec, self.vectors[connected])
          closest_m = np.argsort(dists)[: self.m]
          self.graph[n_idx] = {connected[i] for i in closest_m}

  def search(
      self, query: np.ndarray, k: int = 5
  ) -> Tuple[List[int], List[float]]:
    if self.vectors is None or self.entry_point is None:
      return [], []

    query = np.ascontiguousarray(query, dtype=np.float32)
    candidates = self._search_layer(
        query, self.entry_point, max(self.ef_search, k)
    )
    top_k = candidates[:k]

    result_ids = [self.ids[cand[1]] for cand in top_k]
    result_distances = [cand[0] for cand in top_k]

    return result_ids, result_distances

  def count(self) -> int:
    return len(self.ids)

  def save_index(self, filepath: str) -> None:
    data = {
        "dim": self.dim,
        "metric": self.metric,
        "m": self.m,
        "ef_construction": self.ef_construction,
        "ef_search": self.ef_search,
        "vectors": self.vectors,
        "ids": self.ids,
        "id_to_idx": self.id_to_idx,
        "graph": self.graph,
        "entry_point": self.entry_point,
    }
    with open(filepath, "wb") as f:
      pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

  @classmethod
  def load_index(cls, filepath: str):
    with open(filepath, "rb") as f:
      data = pickle.load(f)

    instance = cls(
        dim=data["dim"],
        metric=data["metric"],
        m=data["m"],
        ef_construction=data["ef_construction"],
        ef_search=data["ef_search"],
    )
    instance.vectors = data["vectors"]
    instance.ids = data["ids"]
    instance.id_to_idx = data["id_to_idx"]
    instance.graph = data["graph"]
    instance.entry_point = data["entry_point"]
    return instance