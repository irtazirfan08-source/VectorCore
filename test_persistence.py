import os
import numpy as np
from vector_core.index.hnsw import HNSWIndex


def run():
  dim = 16
  num_items = 50
  vectors = np.random.randn(num_items, dim).astype(np.float32)
  ids = list(range(100, 100 + num_items))

  index = HNSWIndex(dim=dim, metric="l2")
  index.add(ids, vectors)

  query = vectors[0]
  orig_ids, orig_dists = index.search(query, k=5)

  dump_path = "test_dump.bin"
  index.save_index(dump_path)

  restored_index = HNSWIndex.load_index(dump_path)
  restored_ids, restored_dists = restored_index.search(query, k=5)

  assert orig_ids == restored_ids, "IDs mismatch after loading"
  assert np.allclose(orig_dists, restored_dists), (
      "Distances mismatch after loading"
  )
  print("Persistence verification passed successfully.")

  if os.path.exists(dump_path):
    os.remove(dump_path)


if __name__ == "__main__":
  run()