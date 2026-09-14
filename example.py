import numpy as np
from vector_core.index.hnsw import HNSWIndex

# 1. Initialize 64-dim HNSW index
dim = 64
index = HNSWIndex(dim=dim, metric="euclidean")

# 2. Insert synthetic embeddings
vectors = np.random.randn(500, dim).astype(np.float32)
ids = list(range(500))
index.add(ids=ids, vectors=vectors)

# 3. Query nearest neighbors
query = np.random.randn(dim).astype(np.float32)
nearest_ids, distances = index.search(query=query, k=3)

print(f"Top neighbor IDs: {nearest_ids}")
print(f"Distances: {distances}")