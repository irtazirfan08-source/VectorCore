import numpy as np
import pytest
from vector_core.index.hnsw import HNSWIndex
from vector_core.index.flat import FlatIndex


def test_hnsw_construction_and_properties():
    dim = 64
    num_vectors = 200
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    ids = list(range(1, num_vectors + 1))

    hnsw = HNSWIndex(dim=dim, metric="l2", m=16, ef_construction=64, ef_search=32)
    hnsw.add(ids=ids, vectors=vectors)

    assert hnsw.count() == num_vectors
    assert hnsw.entry_point is not None


def test_hnsw_self_query_exact_match():
    dim = 64
    num_vectors = 150
    k = 5
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    ids = list(range(1, num_vectors + 1))

    hnsw = HNSWIndex(dim=dim, metric="l2", m=16, ef_construction=64, ef_search=32)
    hnsw.add(ids=ids, vectors=vectors)

    # Querying an existing vector should return itself as rank #1
    query_vector = vectors[0]
    matched_ids, distances = hnsw.search(query=query_vector, k=k)

    assert len(matched_ids) == k
    assert matched_ids[0] == 1
    assert pytest.approx(distances[0], abs=1e-4) == 0.0


def test_hnsw_recall_against_flat_ground_truth():
    dim = 64
    num_vectors = 300
    k = 5
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    ids = list(range(1, num_vectors + 1))

    # 1. Exact ground truth with FlatIndex
    flat = FlatIndex(dim=dim, metric="l2")
    flat.add(ids=ids, vectors=vectors)

    # 2. Approximate search with HNSWIndex
    hnsw = HNSWIndex(dim=dim, metric="l2", m=16, ef_construction=64, ef_search=32)
    hnsw.add(ids=ids, vectors=vectors)

    # Arbitrary unseen query vector
    query_vector = np.random.randn(dim).astype(np.float32)

    flat_ids, _ = flat.search(query=query_vector, k=k)
    hnsw_ids, _ = hnsw.search(query=query_vector, k=k)

    # Calculate Recall@k
    intersection = set(hnsw_ids).intersection(set(flat_ids))
    recall = len(intersection) / k

    # HNSW should achieve high recall against exact scan
    assert recall >= 0.8


def test_hnsw_cosine_metric():
    dim = 32
    hnsw = HNSWIndex(dim=dim, metric="cosine", m=16, ef_construction=32, ef_search=16)

    v1 = np.ones(dim, dtype=np.float32)
    v2 = -np.ones(dim, dtype=np.float32)
    hnsw.add(ids=[101, 102], vectors=np.vstack([v1, v2]))

    matched_ids, distances = hnsw.search(query=v1, k=2)
    assert matched_ids[0] == 101
    assert pytest.approx(distances[0], abs=1e-4) == 0.0


def test_hnsw_empty_and_dimension_validation():
    hnsw = HNSWIndex(dim=16, metric="l2")

    # Search on empty index
    empty_ids, empty_dists = hnsw.search(query=np.ones(16, dtype=np.float32), k=5)
    assert empty_ids == []
    assert empty_dists == []

    # Dimension mismatch
    with pytest.raises(ValueError):
        hnsw.add(ids=[1], vectors=np.random.randn(1, 32).astype(np.float32))