"""Integration tests for the VectorCore FastAPI REST API."""

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
        "vectors": [[0.1, 0.2, 0.3]],  # 2 IDs, but only 1 vector
    }
    response = client.post("/v1/vectors/insert", json=payload)
    assert response.status_code == 400
    assert "must match" in response.json()["detail"]


def test_search_exact_match(client):
    np.random.seed(42)
    dim = 16
    vectors = np.random.randn(20, dim).astype(np.float32).tolist()
    vector_ids = list(range(100, 120))

    # Insert dataset
    client.post("/v1/vectors/insert", json={"vector_ids": vector_ids, "vectors": vectors})

    # Query with exact first vector
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

    # Search with 3-dimensional vector on 4-dimensional index
    payload = {"query_vector": [1.0, 2.0, 3.0], "k": 1}
    response = client.post("/v1/vectors/search", json=payload)
    assert response.status_code == 400
    assert "dimension" in response.json()["detail"]