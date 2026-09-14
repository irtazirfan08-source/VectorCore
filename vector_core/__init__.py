"""VectorCore: Lightweight Approximate Nearest Neighbor search engine."""

from vector_core.index.hnsw import HNSWIndex
from vector_core.index.flat import FlatIndex

__version__ = "0.1.0"
__all__ = ["HNSWIndex", "FlatIndex"]