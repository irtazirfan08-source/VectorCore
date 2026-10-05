"""Empirical Benchmark: FlatIndex (Exact) vs HNSWIndex (Approximate)

Evaluates Indexing Time, Latency (ms), Queries Per Second (QPS), and Recall@k.
"""

import time
from typing import Dict, Any, List
import numpy as np
from vector_core.index.flat import FlatIndex
from vector_core.index.hnsw import HNSWIndex


def compute_recall_at_k(
    ground_truth: List[List[int]], predictions: List[List[int]], k: int
) -> float:
    """Computes average Recall@k across all evaluated queries."""
    recalls = []
    for gt, pred in zip(ground_truth, predictions):
        intersection = set(pred[:k]).intersection(set(gt[:k]))
        recalls.append(len(intersection) / float(k))
    return float(np.mean(recalls))


def run_benchmark(
    num_vectors: int = 2000,
    dim: int = 64,
    num_queries: int = 100,
    k: int = 5,
    metric: str = "l2",
) -> Dict[str, Any]:
    print(
        f"\n======================================================================"
    )
    print(
        f" Benchmark Setup: N={num_vectors:,} | Dim={dim} | Queries={num_queries} | k={k} | Metric={metric}"
    )
    print(
        f"======================================================================"
    )

    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vector_ids = list(range(1, num_vectors + 1))
    queries = np.random.randn(num_queries, dim).astype(np.float32)

    # ---------------- 1. Exact FlatIndex (Ground Truth) ----------------
    print("[1/2] Indexing & Querying FlatIndex (Ground Truth)...")
    flat_index = FlatIndex(dim=dim, metric=metric)

    t0 = time.perf_counter()
    flat_index.add(ids=vector_ids, vectors=vectors)
    flat_build_time = time.perf_counter() - t0

    flat_results = []
    t0 = time.perf_counter()
    for q in queries:
        ids, _ = flat_index.search(query=q, k=k)
        flat_results.append(ids)
    flat_search_time = time.perf_counter() - t0

    flat_qps = num_queries / flat_search_time
    flat_latency_ms = (flat_search_time / num_queries) * 1000.0

    # ---------------- 2. Approximate HNSWIndex ----------------
    print("[2/2] Indexing & Querying HNSWIndex...")
    hnsw_index = HNSWIndex(
        dim=dim, metric=metric, m=16, ef_construction=64, ef_search=32
    )

    t0 = time.perf_counter()
    hnsw_index.add(ids=vector_ids, vectors=vectors)
    hnsw_build_time = time.perf_counter() - t0

    hnsw_results = []
    t0 = time.perf_counter()
    for q in queries:
        ids, _ = hnsw_index.search(query=q, k=k)
        hnsw_results.append(ids)
    hnsw_search_time = time.perf_counter() - t0

    hnsw_qps = num_queries / hnsw_search_time
    hnsw_latency_ms = (hnsw_search_time / num_queries) * 1000.0
    hnsw_recall = (
        compute_recall_at_k(flat_results, hnsw_results, k=k) * 100.0
    )

    # ---------------- Output Summary Table ----------------
    print("\n" + "-" * 70)
    print(
        f"{'Index Type':<16} | {'Build (s)':<10} | {'Latency (ms)':<14} | {'QPS':<10} | {'Recall@' + str(k):<10}"
    )
    print("-" * 70)
    print(
        f"{'FlatIndex':<16} | {flat_build_time:<10.4f} | {flat_latency_ms:<14.3f} | {flat_qps:<10.1f} | {'100.0%':<10}"
    )
    print(
        f"{'HNSWIndex':<16} | {hnsw_build_time:<10.4f} | {hnsw_latency_ms:<14.3f} | {hnsw_qps:<10.1f} | {hnsw_recall:<9.1f}%"
    )
    print("-" * 70 + "\n")

    return {
        "flat": {
            "build_time": flat_build_time,
            "latency_ms": flat_latency_ms,
            "qps": flat_qps,
            "recall": 100.0,
        },
        "hnsw": {
            "build_time": hnsw_build_time,
            "latency_ms": hnsw_latency_ms,
            "qps": hnsw_qps,
            "recall": hnsw_recall,
        },
    }


if __name__ == "__main__":
    run_benchmark(num_vectors=2000, dim=64, num_queries=100, k=5)