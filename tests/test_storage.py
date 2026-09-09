import numpy as np
import os
from vector_core.index.hnsw import HNSWIndex
from vector_core.storage.serializer import IndexSerializer

dim = 64
num_vectors = 500
save_path = "storage_dump/hnsw_test.vcore"

np.random.seed(42)
vectors = np.random.randn(num_vectors, dim).astype(np.float32)
vector_ids = list(range(1, num_vectors + 1))

# 1. Build and save index
print("=== [1] Building and Saving HNSW Index to Disk ===")
index = HNSWIndex(dim=dim, metric="l2")
index.add(ids=vector_ids, vectors=vectors)
IndexSerializer.save(index, save_path)
file_size_kb = os.path.getsize(save_path) / 1024
print(f"Saved {index.count()} vectors to '{save_path}' ({file_size_kb:.2f} KB)\n")

# 2. Load index back from disk
print("=== [2] Loading Index from Disk ===")
loaded_index = IndexSerializer.load(save_path)
print(f"Loaded successfully. Total nodes in graph: {loaded_index.count()}\n")

# 3. Verify search consistency
query = vectors[0]
ids_original, _ = index.search(query, k=3)
ids_loaded, _ = loaded_index.search(query, k=3)

print(f"Original Index Search: {ids_original}")
print(f"Loaded Index Search  : {ids_loaded}")
assert ids_original == ids_loaded, "Mismatch between original and loaded index results!"
print("✅ Disk persistence verification passed: Search outputs match 100%.")