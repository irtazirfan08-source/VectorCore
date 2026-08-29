# VectorCore: Low-Level Vector Search & HNSW Indexing Engine

VectorCore is a lightweight, zero-dependency vector search engine built from scratch in Python and NumPy. It implements SIMD-friendly vector distance metrics, an exact brute-force baseline index, and a Hierarchical Navigable Small World (HNSW) graph index with binary disk serialization.

## Key Features

* **Vectorized Metric Kernels**: Optimized Euclidean (L2) and Cosine distance implementations.
* **HNSW Graph Index**: Approximate Nearest Neighbor (ANN) search using greedy graph traversal with configurable `ef_construction` and `ef_search` beam width.
* **Exact Flat Index**: Linear-scan baseline providing 100% ground-truth recall validation.
* **Binary Serialization**: Zero-copy disk persistence protocol (`.vcore`) preserving index topologies and high-dimensional vector embeddings.

## Benchmark Results

Evaluated on 5,000 vectors (128 dimensions) queried with 100 randomized vectors at $k=10$:

| Index Type | Build Time | Avg Latency | Throughput (QPS) | Recall@10 |
| :--- | :--- | :--- | :--- | :--- |
| **Flat (Brute-Force)** | 0.000s | 1.298 ms | 770.3 queries/s | 100.0% |
| **HNSW (Graph ANN)** | 6.830s | **0.533 ms** | **1876.5 queries/s** | 61.3% |

* **Performance Gain**: 2.44x faster search latency over brute-force linear scanning.

## Quickstart

### 1. Installation
```bash
git clone [https://github.com/irtazirfan08-source/VectorCore.git](https://github.com/irtazirfan08-source/VectorCore.git)
cd VectorCore
pip install -r requirements.txt