"""
VectorCore Performance Benchmark Suite
Compares Linear Scan (FlatIndex) vs HNSW Graph Index on Latency, QPS, and Recall.
"""

import time
import numpy as np
from tabulate import tabulate
from vector_core.index.flat import FlatIndex
from vector_core.index.hnsw import HNSWIndex

dim = 128
num_vectors = 5000
num_queries = 100
k = 10

print("=" * 65)
print(f" VectorCore Benchmark: {num_vectors} Vectors ({dim}-D), {num_queries} Queries (k={k})")
print("=" * 65)

# Generate dataset and test queries
np.random.seed(42)
dataset = np.random.randn(num_vectors, dim).astype(np.float32)
queries = np.random.randn(num_queries, dim).astype(np.float32)
vector_ids = list(range(1, num_vectors + 1))

# 1. Benchmark Flat Index
flat_index = FlatIndex(dim=dim, metric="l2")
t0 = time.perf_counter()
flat_index.add(vector_ids, dataset)
flat_build_time = time.perf_counter() - t0

t0 = time.perf_counter()
flat_results = [flat_index.search(q, k=k)[0] for q in queries]
flat_search_time = time.perf_counter() - t0
flat_qps = num_queries / flat_search_time
flat_avg_lat_ms = (flat_search_time / num_queries) * 1000

# 2. Benchmark HNSW Index
hnsw_index = HNSWIndex(dim=dim, metric="l2", m=16, ef_construction=64, ef_search=32)
t0 = time.perf_counter()
hnsw_index.add(vector_ids, dataset)
hnsw_build_time = time.perf_counter() - t0

t0 = time.perf_counter()
hnsw_results = [hnsw_index.search(q, k=k)[0] for q in queries]
hnsw_search_time = time.perf_counter() - t0
hnsw_qps = num_queries / hnsw_search_time
hnsw_avg_lat_ms = (hnsw_search_time / num_queries) * 1000

# 3. Calculate Recall against Exact Ground Truth
total_matches = sum(len(set(h).intersection(set(f))) for h, f in zip(hnsw_results, flat_results))
hnsw_recall = (total_matches / (num_queries * k)) * 100.0

# 4. Display Results Table
headers = ["Index Type", "Build Time (s)", "Avg Latency (ms)", "Throughput (QPS)", f"Recall@{k}"]
table = [
    ["Flat (Brute-Force)", f"{flat_build_time:.3f}s", f"{flat_avg_lat_ms:.3f} ms", f"{flat_qps:.1f} queries/s", "100.0%"],
    ["HNSW (Graph ANN)", f"{hnsw_build_time:.3f}s", f"{hnsw_avg_lat_ms:.3f} ms", f"{hnsw_qps:.1f} queries/s", f"{hnsw_recall:.1f}%"]
]

print("\n" + tabulate(table, headers=headers, tablefmt="fancy_grid"))
speedup = flat_avg_lat_ms / hnsw_avg_lat_ms if hnsw_avg_lat_ms > 0 else 1.0
print(f"\n⚡ HNSW Query Speedup: {speedup:.2f}x faster search latency at {hnsw_recall:.1f}% Recall@{k}.\n")