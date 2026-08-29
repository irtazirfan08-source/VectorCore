"""
Binary Index Serializer & Disk Persistence
Saves and loads HNSW/Flat vector indices to and from disk.
"""

import pickle
from pathlib import Path
from typing import Union
from vector_core.index.flat import FlatIndex
from vector_core.index.hnsw import HNSWIndex


class IndexSerializer:
    """Handles binary serialization and disk I/O for vector indices."""

    @staticmethod
    def save(index: Union[FlatIndex, HNSWIndex], filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filepath: str) -> Union[FlatIndex, HNSWIndex]:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")
        with open(path, "rb") as f:
            index = pickle.load(f)
        return index