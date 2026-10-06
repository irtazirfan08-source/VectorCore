import os
import numpy as np
import pytest
from vector_core.index.hnsw import HNSWIndex
from vector_core.storage.serializer import IndexSerializer


def test_hnsw_serialization_roundtrip_l2(tmp_path):
    dim = 64
    num_vectors = 300
    save_path = str(tmp_path / "hnsw_test_l2.vcore")

    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vector_ids = list(range(1, num_vectors + 1))

    # Build and serialize
    index = HNSWIndex(dim=dim, metric="l2", m=16, ef_construction=64, ef_search=32)
    index.add(ids=vector_ids, vectors=vectors)
    IndexSerializer.save(index, save_path)

    # Verify file exists on disk and is non-empty
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0

    # Deserialize back from disk
    loaded_index = IndexSerializer.load(save_path)

    # Validate structural properties
    assert loaded_index.count() == index.count()
    assert loaded_index.dim == index.dim
    assert loaded_index.metric == index.metric
    assert loaded_index.entry_point == index.entry_point

    # Validate search consistency on indexed vectors
    query = vectors[0]
    orig_ids, orig_dists = index.search(query, k=5)
    loaded_ids, loaded_dists = loaded_index.search(query, k=5)

    assert orig_ids == loaded_ids
    np.testing.assert_allclose(orig_dists, loaded_dists, rtol=1e-5, atol=1e-5)


def test_hnsw_serialization_unseen_query_cosine(tmp_path):
    dim = 32
    num_vectors = 150
    save_path = str(tmp_path / "hnsw_test_cosine.vcore")

    np.random.seed(99)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vector_ids = list(range(1, num_vectors + 1))

    # Build and serialize cosine index
    index = HNSWIndex(dim=dim, metric="cosine", m=16, ef_construction=32, ef_search=16)
    index.add(ids=vector_ids, vectors=vectors)
    IndexSerializer.save(index, save_path)

    loaded_index = IndexSerializer.load(save_path)

    # Verify identical nearest neighbor routing on unseen query
    unseen_query = np.random.randn(dim).astype(np.float32)
    orig_ids, orig_dists = index.search(unseen_query, k=3)
    loaded_ids, loaded_dists = loaded_index.search(unseen_query, k=3)

    assert orig_ids == loaded_ids
    np.testing.assert_allclose(orig_dists, loaded_dists, rtol=1e-5, atol=1e-5)