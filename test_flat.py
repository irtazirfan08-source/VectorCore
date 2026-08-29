import numpy as np
from vector_core.index.flat import FlatIndex

# Initialize a 128-dimensional Flat Index using L2 distance
dim = 128
num_vectors = 1000
k = 5

index = FlatIndex(dim=dim, metric="l2")

# Generate 1,000 synthetic 128-dimensional embeddings
np.random.seed(42)
synthetic_vectors = np.random.randn(num_vectors, dim).astype(np.float32)
vector_ids = list(range(1, num_vectors + 1))

print("=== [1] Populating Flat Vector Index ===")
index.add(ids=vector_ids, vectors=synthetic_vectors)
print(f"Total Indexed Vectors: {index.count()} | Dimensions: {dim}\n")

# Run search for target vector #1 (distance to itself must be 0.0)
query_vector = synthetic_vectors[0]
print("=== [2] Running Exact Top-5 Search ===")
matched_ids, distances = index.search(query=query_vector, k=k)

for rank, (vid, dist) in enumerate(zip(matched_ids, distances), start=1):
    print(f"Rank {rank}: Vector ID={vid:<4} | L2 Distance = {dist:.6f}")