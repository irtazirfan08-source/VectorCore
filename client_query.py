"""End-to-end lifecycle demonstration client for VectorCore REST API."""

import json
import os
import urllib.request
import numpy as np

BASE_URL = "http://127.0.0.1:8001"


def post_json(endpoint: str, data: dict) -> dict:
    """Helper to send HTTP POST requests with JSON payload."""
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(endpoint: str) -> dict:
    """Helper to send HTTP GET requests."""
    url = f"{BASE_URL}{endpoint}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    print("=== [1] Verifying Service Health & Initial Stats ===")
    health = get_json("/health")
    print(f"Health Status: {health}")
    stats = get_json("/v1/vectors/stats")
    print(f"Initial State: {stats}\n")

    print("=== [2] Ingesting 50 Synthetic 128-Dimensional Vectors ===")
    np.random.seed(42)
    dim = 128
    vectors = np.random.randn(50, dim).tolist()
    vector_ids = list(range(1001, 1051))

    insert_payload = {"vector_ids": vector_ids, "vectors": vectors}
    insert_res = post_json("/v1/vectors/insert", insert_payload)
    print(f"Insert Status: {insert_res}\n")

    print("=== [3] Querying Top-3 Nearest Neighbors for Vector #1001 ===")
    search_payload = {"query_vector": vectors[0], "k": 3}
    search_res = post_json("/v1/vectors/search", search_payload)
    print("Single Query Matches:")
    print(json.dumps(search_res["results"], indent=2))
    print()

    print("=== [4] Executing Batch Search for First 2 Vectors ===")
    batch_payload = {"query_vectors": [vectors[0], vectors[1]], "k": 2}
    batch_res = post_json("/v1/vectors/batch_search", batch_payload)
    print(f"Batch Search Processed {batch_res['total_queries']} queries.")
    for idx, batch_item in enumerate(batch_res["batch_results"]):
        print(f"  Query {idx} Top Match: ID={batch_item['ids'][0]}, Distance={batch_item['distances'][0]:.6f}")
    print()

    print("=== [5] Persisting Index Snapshot to Disk ===")
    dump_dir = os.path.abspath("./storage_dump")
    os.makedirs(dump_dir, exist_ok=True)
    snapshot_path = os.path.join(dump_dir, "client_demo_index.vcore")

    save_payload = {"filepath": snapshot_path}
    save_res = post_json("/v1/index/save", save_payload)
    print(f"Saved snapshot to: {save_res['filepath']} (Count: {save_res['count']})\n")

    print("=== [6] Restoring Index Snapshot from Disk ===")
    load_payload = {"filepath": snapshot_path}
    load_res = post_json("/v1/index/load", load_payload)
    print(f"Restored index: {load_res}\n")

    print("=== [7] Validating Search Parity After Restoration ===")
    verify_res = post_json("/v1/vectors/search", search_payload)
    print(f"Top neighbor after restore: ID={verify_res['ids'][0]}, Distance={verify_res['distances'][0]:.6f}")
    assert verify_res["ids"][0] == 1001, "Error: Index state did not match after restoration!"
    print("\nAll lifecycle operations completed successfully.")


if __name__ == "__main__":
    main()