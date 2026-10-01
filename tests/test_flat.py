import numpy as np
import pytest
from vector_core.index.flat import FlatIndex


def test_flat_index_l2_exact_retrieval():
    dim = 64
    num_vectors = 200
    k = 5

    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    ids = list(range(1, num_vectors + 1))

    index = FlatIndex(dim=dim, metric="l2")
    index.add(ids=ids, vectors=vectors)

    assert index.count() == num_vectors

    # Self-query: distance to itself must be 0.0 and rank #1
    query = vectors[0]
    matched_ids, distances = index.search(query=query, k=k)

    assert matched_ids[0] == 1
    assert pytest.approx(distances[0], abs=1e-5) == 0.0
    assert len(matched_ids) == k


def test_flat_index_cosine_metric():
    dim = 32
    index = FlatIndex(dim=dim, metric="cosine")

    v1 = np.ones(dim, dtype=np.float32)
    v2 = -np.ones(dim, dtype=np.float32)
    index.add(ids=[101, 102], vectors=np.vstack([v1, v2]))

    # Same vector -> cosine distance should be 0.0
    matched_ids, distances = index.search(query=v1, k=2)
    assert matched_ids[0] == 101
    assert pytest.approx(distances[0], abs=1e-5) == 0.0
    # Opposite vector -> cosine distance should be 2.0
    assert matched_ids[1] == 102
    assert pytest.approx(distances[1], abs=1e-5) == 2.0


def test_flat_index_inner_product():
    dim = 16
    index = FlatIndex(dim=dim, metric="ip")

    v1 = np.array([1.0] * dim, dtype=np.float32)
    v2 = np.array([2.0] * dim, dtype=np.float32)
    index.add(ids=[1, 2], vectors=np.vstack([v1, v2]))

    # Query with ones vector: dot product with v2 (2.0*16 = 32) > v1 (1.0*16 = 16)
    matched_ids, distances = index.search(query=v1, k=2)
    assert matched_ids[0] == 2
    assert matched_ids[1] == 1


def test_flat_index_batch_search():
    dim = 16
    index = FlatIndex(dim=dim, metric="l2")
    vectors = np.random.randn(20, dim).astype(np.float32)
    ids = list(range(100, 120))
    index.add(ids=ids, vectors=vectors)

    queries = vectors[:3]
    batch_ids, batch_dists = index.batch_search(queries=queries, k=3)

    assert len(batch_ids) == 3
    assert len(batch_dists) == 3
    for i in range(3):
        assert batch_ids[i][0] == ids[i]
        assert pytest.approx(batch_dists[i][0], abs=1e-5) == 0.0


def test_flat_index_validation_errors():
    index = FlatIndex(dim=8, metric="l2")

    with pytest.raises(ValueError, match="does not match index dimension"):
        index.add(ids=[1], vectors=np.random.randn(1, 12).astype(np.float32))

    with pytest.raises(ValueError, match="Metric must be"):
        FlatIndex(dim=8, metric="manhattan")