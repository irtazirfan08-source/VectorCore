import time
import numpy as np
from vector_core.index.hnsw import HNSWIndex


def run_benchmark():
  dim = 64
  num_vectors = 2000
  num_queries = 50
  k = 5

  print(f"Generating {num_vectors} vectors with dimension {dim}...")
  data = np.random.randn(num_vectors, dim).astype(np.float32)
  ids = list(range(num_vectors))
  queries = np.random.randn(num_queries, dim).astype(np.float32)

  index = HNSWIndex(
      dim=dim, metric="l2", m=16, ef_construction=64, ef_search=32
  )
  index.add(ids, data)

  t0 = time.perf_counter()
  ground_truth = []
  for q in queries:
    dists = np.linalg.norm(data - q, axis=1)
    top_indices = np.argsort(dists)[:k]
    ground_truth.append(set(top_indices))
  linear_time = time.perf_counter() - t0
  linear_qps = num_queries / linear_time

  t0 = time.perf_counter()
  hnsw_results = []
  for q in queries:
    res_ids, _ = index.search(q, k=k)
    hnsw_results.append(set(res_ids))
  hnsw_time = time.perf_counter() - t0
  hnsw_qps = num_queries / hnsw_time

  recalls = [
      len(h.intersection(gt)) / k for h, gt in zip(hnsw_results, ground_truth)
  ]
  avg_recall = np.mean(recalls) * 100
  speedup = hnsw_qps / linear_qps

  print(
      f"Brute-Force Linear Scan : {linear_qps:.2f} QPS (Latency:"
      f" {linear_time/num_queries*1000:.2f} ms/query)"
  )
  print(
      f"HNSW Graph Traversal    : {hnsw_qps:.2f} QPS (Latency:"
      f" {hnsw_time/num_queries*1000:.2f} ms/query)"
  )
  print(f"Speedup                 : {speedup:.2f}x")
  print(f"Recall@{k}                : {avg_recall:.2f}%")


if __name__ == "__main__":
  run_benchmark()