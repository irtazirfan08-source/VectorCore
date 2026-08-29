import numpy as np
from vector_core.index.hnsw import HNSWIndex
from vector_core.index.flat import FlatIndex

dim = 128
num_vectors = 1000
k = 5

np.random.seed(42)
synthetic_vectors = np.random.randn(num_vectors, dim).astype(np.float32)
vector_ids = list(range(1, num_vectors + 1))

print("=== [1] Building HNSW Graph Index (1,000 Vectors) ===")
hnsw = HNSWIndex(dim=dim, metric="l2", m=16, ef_construction=64, ef_search=32)
hnsw.add(ids=vector_ids, vectors=synthetic_vectors)
print(f"HNSW Graph Built. Total Nodes: {hnsw.count()} | Entry Point Node: #{hnsw.entry_point}\n")

# Run search on HNSW
query_vector = synthetic_vectors[0]
print("=== [2] Running HNSW Approximate Top-5 Search ===")
hnsw_ids, hnsw_dists = hnsw.search(query=query_vector, k=k)
for rank, (vid, dist) in enumerate(zip(hnsw_ids, hnsw_dists), start=1):
    print(f"Rank {rank}: Vector ID={vid:<4} | L2 Distance = {dist:.6f}")

# Cross-validate recall against Flat Exact Index
print("\n=== [3] Cross-Validating Recall with Exact Flat Scan ===")
flat = FlatIndex(dim=dim, metric="l2")
flat.add(ids=vector_ids, vectors=synthetic_vectors)
flat_ids, _ = flat.search(query=query_vector, k=k)

recall = len(set(hnsw_ids).intersection(set(flat_ids))) / k * 100
print(f"Ground Truth Top-5 IDs : {flat_ids}")
print(f"HNSW Returned Top-5 IDs: {hnsw_ids}")
print(f"Search Recall @ {k}       : {recall:.1f}%")