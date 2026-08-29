import urllib.request
import json
import numpy as np

# 1. Insert 50 synthetic 128-dimensional vectors via HTTP POST
print("=== [1] Inserting 50 Vectors into VectorCore Server ===")
np.random.seed(42)
vectors = np.random.randn(50, 128).tolist()
vector_ids = list(range(1001, 1051))

insert_data = json.dumps({"vector_ids": vector_ids, "vectors": vectors}).encode("utf-8")
insert_req = urllib.request.Request(
    "http://127.0.0.1:8001/v1/vectors/insert",
    data=insert_data,
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(insert_req) as response:
    print(response.read().decode("utf-8"))

# 2. Search for the nearest neighbors of vector #1001 (should return #1001 with distance 0.0)
print("\n=== [2] Querying Top-3 Nearest Neighbors ===")
search_data = json.dumps({"query_vector": vectors[0], "k": 3}).encode("utf-8")
search_req = urllib.request.Request(
    "http://127.0.0.1:8001/v1/vectors/search",
    data=search_data,
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(search_req) as response:
    result = json.loads(response.read().decode("utf-8"))
    print(json.dumps(result, indent=2))