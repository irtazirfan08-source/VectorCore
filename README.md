# VectorCore: Low-Level Vector Search & HNSW Indexing Engine

[![PyPI version](https://img.shields.io/pypi/v/vectorcore-ann.svg?color=blue)](https://pypi.org/project/vectorcore-ann/)
[![Python versions](https://img.shields.io/pypi/pyversions/vectorcore-ann.svg)](https://pypi.org/project/vectorcore-ann/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

VectorCore is a lightweight, zero-dependency Approximate Nearest Neighbor (ANN) vector search engine built from scratch in Python and NumPy. It implements SIMD-friendly vector distance metrics, an exact brute-force baseline index, and a Hierarchical Navigable Small World (HNSW) graph index with binary disk serialization.

---

## Key Features

* **Vectorized Metric Kernels**: Optimized Euclidean ($L_2$) and Cosine distance metric routines.
* **HNSW Graph Index**: Fast Approximate Nearest Neighbor (ANN) greedy graph traversal with configurable `ef_construction`, `M`, and `ef_search` beam width parameters.
* **Exact Flat Index**: Exhaustive linear-scan baseline providing 100% ground-truth recall validation.
* **Binary Serialization**: Zero-copy disk persistence protocol (`.vcore`) preserving graph topologies and vector payload matrices.
* **Production Distribution**: Published on PyPI as `vectorcore-ann` with automated CI/CD releases.

---

## Benchmark Results

Evaluated on 5,000 vectors (128 dimensions) queried with 100 randomized vectors at $k=10$:

| Index Type | Build Time | Avg Latency | Throughput (QPS) | Recall@10 |
| :--- | :--- | :--- | :--- | :--- |
| **Flat (Brute-Force)** | 0.000s | 1.298 ms | 770.3 queries/s | 100.0% |
| **HNSW (Graph ANN)** | 6.830s | **0.533 ms** | **1876.5 queries/s** | 61.3% |

* **Performance Gain**: **2.44x faster** search latency over brute-force linear scanning.

---

## Installation

### Via PyPI (Recommended)

```bash
pip install vectorcore-ann