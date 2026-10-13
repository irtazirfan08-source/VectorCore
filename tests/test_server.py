"""Integration tests for the VectorCore FastAPI REST API."""

import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

import vector_core.server.app as server_module
from vector_core.server.app import app


@pytest.fixture(autouse=True)
def reset_server_index():
    """Reset the global index instance before each test run."""
    server_module.index = None
    yield
    server_module.index = None


@pytest.fixture
def client():
    """Provides a fresh FastAPI test client."""
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "VectorCore"}


def test_stats_uninitialized(client):
    response = client.get("/v1/vectors/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "uninitialized"
    assert data["count"] == 0
    assert data["dim"] is None


def test_insert_and_stats(client):
    vectors = [[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i] for i in range(10)]
    vector_ids = list(range(1, 11))

    payload = {"vector_ids": vector_ids, "vectors": vectors}
    response = client.post("/v1/vectors/insert", json=payload)
    assert response.status_code == 201
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["inserted"] == 10
    assert res_data["total_count"] == 10
    assert res_data["dimension"] == 4

    stats_resp = client.get("/v1/vectors/stats")
    assert stats_resp.status_code == 200
    stats_data = stats_resp.json()
    assert stats_data["status"] == "ready"
    assert stats_data["count"] == 10
    assert stats_data["dim"] == 4


def test_insert_validation_mismatched_lengths(client):
    payload = {
        "vector_ids": [1, 2],
        "vectors": [[0.1, 0.2, 0.3]],
    }
    response = client.post("/v1/vectors/insert", json=payload)
    assert response.status_code == 400
    assert "must match" in response.json()["detail"]


def test_search_exact_match(client):
    np.random.seed(42)
    dim = 16
    vectors = np.random.randn(20, dim).astype(np.float32).tolist()
    vector_ids = list(range(100, 120))

    client.post("/v1/vectors/insert", json={"vector_ids": vector_ids, "vectors": vectors})

    search_payload = {"query_vector": vectors[0], "k": 3}
    search_resp = client.post("/v1/vectors/search", json=search_payload)
    assert search_resp.status_code == 200
    data = search_resp.json()

    assert data["ids"][0] == 100
    assert pytest.approx(data["distances"][0], abs=1e-5) == 0.0
    assert len(data["results"]) == 3
    assert data["total_indexed"] == 20


def test_search_empty_index_error(client):
    payload = {"query_vector": [1.0, 2.0, 3.0], "k": 3}
    response = client.post("/v1/vectors/search", json=payload)
    assert response.status_code == 400
    assert "Index is empty" in response.json()["detail"]


def test_search_dimension_mismatch_error(client):
    vectors = [[1.0, 2.0, 3.0, 4.0]]
    client.post("/v1/vectors/insert", json={"vector_ids": [1], "vectors": vectors})

    payload = {"query_vector": [1.0, 2.0, 3.0], "k": 1}
    response = client.post("/v1/vectors/search", json=payload)
    assert response.status_code == 400
    assert "dimension" in response.json()["detail"]


def test_batch_search_success(client):
    np.random.seed(42)
    dim = 8
    vectors = np.random.randn(15, dim).astype(np.float32).tolist()
    vector_ids = list(range(1, 16))
    client.post("/v1/vectors/insert", json={"vector_ids": vector_ids, "vectors": vectors})

    query_batch = [vectors[0], vectors[1]]
    payload = {"query_vectors": query_batch, "k": 2}
    response = client.post("/v1/vectors/batch_search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total_queries"] == 2
    assert data["total_indexed"] == 15
    assert len(data["batch_results"]) == 2

    assert data["batch_results"][0]["ids"][0] == 1
    assert pytest.approx(data["batch_results"][0]["distances"][0], abs=1e-5) == 0.0
    assert data["batch_results"][1]["ids"][0] == 2
    assert pytest.approx(data["batch_results"][1]["distances"][0], abs=1e-5) == 0.0


def test_batch_search_empty_queries_error(client):
    vectors = [[1.0, 2.0, 3.0, 4.0]]
    client.post("/v1/vectors/insert", json={"vector_ids": [1], "vectors": vectors})

    payload = {"query_vectors": [], "k": 1}
    response = client.post("/v1/vectors/batch_search", json=payload)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_index_save_and_load_roundtrip(client, tmp_path):
    save_file = str(tmp_path / "server_hnsw.vcore")
    dim = 16
    vectors = [[float(i + j) for j in range(dim)] for i in range(5)]
    vector_ids = [10, 20, 30, 40, 50]

    # Insert data
    client.post("/v1/vectors/insert", json={"vector_ids": vector_ids, "vectors": vectors})

    # Save to disk over HTTP
    save_resp = client.post("/v1/index/save", json={"filepath": save_file})
    assert save_resp.status_code == 200
    assert os.path.exists(save_file)
    assert save_resp.json()["count"] == 5

    # Wipe in-memory index
    server_module.index = None

    # Verify index is uninitialized
    stats_before = client.get("/v1/vectors/stats").json()
    assert stats_before["status"] == "uninitialized"

    # Reload index over HTTP
    load_resp = client.post("/v1/index/load", json={"filepath": save_file})
    assert load_resp.status_code == 200
    assert load_resp.json()["count"] == 5

    # Verify search parity after restore
    search_resp = client.post("/v1/vectors/search", json={"query_vector": vectors[0], "k": 1})
    assert search_resp.status_code == 200
    assert search_resp.json()["ids"][0] == 10


def test_index_save_uninitialized_error(client, tmp_path):
    save_file = str(tmp_path / "error.vcore")
    resp = client.post("/v1/index/save", json={"filepath": save_file})
    assert resp.status_code == 400
    assert "uninitialized" in resp.json()["detail"]


def test_index_load_missing_file_error(client):
    resp = client.post("/v1/index/load", json={"filepath": "/non/existent/file.vcore"})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]