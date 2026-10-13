import os
import numpy as np
import pytest
from vector_core.index.hnsw import HNSWIndex
from vector_core.storage.serializer import IndexSerializer


def test_persistence_custom_id_mapping(tmp_path):
    """Validates that non-contiguous custom ID ranges (e.g. 100..150)

    and graph topology survive binary serialization with exact search parity.
    """
    dim = 16
    num_items = 50
    dump_path = str(tmp_path / "persistence_custom_ids.vcore")

    np.random.seed(42)
    vectors = np.random.randn(num_items, dim).astype(np.float32)
    custom_ids = list(range(100, 100 + num_items))

    index = HNSWIndex(dim=dim, metric="l2", m=16, ef_construction=64, ef_search=32)
    index.add(ids=custom_ids, vectors=vectors)

    query = vectors[0]
    orig_ids, orig_dists = index.search(query, k=5)

    IndexSerializer.save(index, dump_path)
    assert os.path.exists(dump_path)

    restored_index = IndexSerializer.load(dump_path)
    restored_ids, restored_dists = restored_index.search(query, k=5)

    assert orig_ids == restored_ids, "IDs mismatch after restoring index"
    np.testing.assert_allclose(
        orig_dists, restored_dists, rtol=1e-5, atol=1e-5,
        err_msg="Distances mismatch after restoring index"
    )
    assert restored_index.count() == num_items
    assert restored_index.id_to_idx[100] == 0


def test_persistence_inner_product_metric(tmp_path):
    """Validates persistence roundtrip for Inner Product (ip) distance metric."""
    dim = 32
    num_items = 80
    dump_path = str(tmp_path / "persistence_ip.vcore")

    np.random.seed(7)
    vectors = np.random.randn(num_items, dim).astype(np.float32)
    ids = list(range(1, num_items + 1))

    index = HNSWIndex(dim=dim, metric="ip", m=16, ef_construction=32, ef_search=16)
    index.add(ids=ids, vectors=vectors)

    query = np.random.randn(dim).astype(np.float32)
    orig_ids, orig_dists = index.search(query, k=5)

    IndexSerializer.save(index, dump_path)
    restored_index = IndexSerializer.load(dump_path)

    restored_ids, restored_dists = restored_index.search(query, k=5)

    assert restored_index.metric == "ip"
    assert orig_ids == restored_ids
    np.testing.assert_allclose(orig_dists, restored_dists, rtol=1e-5, atol=1e-5)


def test_persistence_empty_index_roundtrip(tmp_path):
    """Validates saving and restoring an empty HNSW index without crashes."""
    dim = 64
    dump_path = str(tmp_path / "persistence_empty.vcore")

    index = HNSWIndex(dim=dim, metric="l2")
    IndexSerializer.save(index, dump_path)

    restored_index = IndexSerializer.load(dump_path)
    assert restored_index.count() == 0
    assert restored_index.vectors is None
    assert restored_index.entry_point is None

    empty_ids, empty_dists = restored_index.search(np.zeros(dim, dtype=np.float32), k=5)
    assert empty_ids == []
    assert empty_dists == []