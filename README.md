# VectorCore: Low-Level Vector Search & HNSW Indexing Engine

[![CI Pipeline](https://github.com/irtazirfan08-source/VectorCore/actions/workflows/ci.yml/badge.svg)](https://github.com/irtazirfan08-source/VectorCore/actions)
[![PyPI version](https://img.shields.io/pypi/v/vectorcore-ann.svg?color=blue)](https://pypi.org/project/vectorcore-ann/)
[![Python versions](https://img.shields.io/pypi/pyversions/vectorcore-ann.svg)](https://pypi.org/project/vectorcore-ann/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

VectorCore is an Approximate Nearest Neighbor (ANN) vector search engine implemented from scratch in Python and NumPy. It provides vectorized distance kernels, an exact brute-force index for ground-truth validation, and a Hierarchical Navigable Small World (HNSW) graph index with configurable beam-search parameters and binary serialization.

---

## Key Features

* **Distance Metrics**: Vectorized Euclidean ($L_2$), Cosine Distance ($1 - \text{sim}$), and Inner Product / Dot Product (`ip`) kernels.
* **HNSW Graph Index**: Logarithmic-time greedy beam search across proximity graphs with configurable `m`, `ef_construction`, and `ef_search`.
* **Exact Flat Baseline**: Exhaustive linear scan index with dimension and contiguous memory validation providing 100% recall ground truth.
* **Batch Retrieval API**: Vectorized multi-query execution via `batch_search(queries, k)`.
* **CI Validation**: Automated test suite executing on GitHub Actions across Python 3.11, 3.12, and 3.13.

---

## Empirical Benchmarks

Evaluated with `benchmark_compare.py` across 2,000 synthetic embeddings (64 dimensions) queried with 100 randomized queries at $k=5$ under $L_2$ distance on Python 3.13:

| Index Type | Build Time | Query Latency | Throughput (QPS) | Recall@5 | Traversal Complexity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FlatIndex (Brute-Force)** | **0.0000s** | **0.469 ms** | **2,130.5 queries/s** | **100.0%** | $O(N \cdot d)$ |
| **HNSWIndex (Graph ANN)** | 3.1730s | 0.821 ms | 1,218.2 queries/s | **84.0%** | $O(\log N \cdot d)$ |

Run benchmarks locally:
```bash
python benchmark_compare.py
