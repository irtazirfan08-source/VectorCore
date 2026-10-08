"""FastAPI REST server for VectorCore HNSW search engine."""

from typing import List, Optional, Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn

from vector_core.index.hnsw import HNSWIndex

app = FastAPI(
    title="VectorCore API",
    version="0.1.0",
    description="High-performance vector search engine powered by HNSW indexing.",
)

# Global index instance (initialized dynamically on first insertion)
index: Optional[HNSWIndex] = None
INDEX_METRIC = "l2"
INDEX_M = 16
INDEX_EF_CONSTRUCTION = 64
INDEX_EF_SEARCH = 32


# ---------------- Pydantic Request / Response Schemas ----------------


class InsertRequest(BaseModel):
    vector_ids: List[int] = Field(..., description="List of unique integer IDs")
    vectors: List[List[float]] = Field(..., description="Matrix of float vectors")


class SearchRequest(BaseModel):
    query_vector: List[float] = Field(..., description="Query vector matching index dimension")
    k: int = Field(default=5, ge=1, description="Number of nearest neighbors to retrieve")


class MatchResult(BaseModel):
    id: int
    distance: float


class SearchResponse(BaseModel):
    ids: List[int]
    distances: List[float]
    results: List[MatchResult]
    total_indexed: int


# ---------------- API Endpoints ----------------


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    """Health status probe."""
    return {"status": "ok", "service": "VectorCore"}


@app.get("/v1/vectors/stats", status_code=status.HTTP_200_OK)
def get_stats() -> Dict[str, Any]:
    """Returns the current state and size of the vector index."""
    if index is None:
        return {"status": "uninitialized", "count": 0, "dim": None}
    return {
        "status": "ready",
        "count": index.count(),
        "dim": index.dim,
        "metric": index.metric,
        "m": index.m,
        "ef_construction": index.ef_construction,
        "ef_search": index.ef_search,
    }


@app.post("/v1/vectors/insert", status_code=status.HTTP_201_CREATED)
def insert_vectors(payload: InsertRequest) -> Dict[str, Any]:
    """Inserts a batch of vectors and IDs into the index."""
    global index

    if not payload.vector_ids or not payload.vectors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="vector_ids and vectors cannot be empty",
        )

    if len(payload.vector_ids) != len(payload.vectors):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Number of vector_ids ({len(payload.vector_ids)}) must match "
                f"number of vectors ({len(payload.vectors)})"
            ),
        )

    vec_array = np.array(payload.vectors, dtype=np.float32)
    dim = vec_array.shape[1]

    # Initialize index on first batch if not yet created
    if index is None:
        index = HNSWIndex(
            dim=dim,
            metric=INDEX_METRIC,
            m=INDEX_M,
            ef_construction=INDEX_EF_CONSTRUCTION,
            ef_search=INDEX_EF_SEARCH,
        )
    elif index.dim != dim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Vector dimension {dim} does not match initialized index dimension {index.dim}",
        )

    index.add(ids=payload.vector_ids, vectors=vec_array)

    return {
        "status": "success",
        "inserted": len(payload.vector_ids),
        "total_count": index.count(),
        "dimension": index.dim,
    }


@app.post("/v1/vectors/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
def search_vectors(payload: SearchRequest) -> SearchResponse:
    """Searches for top-k nearest neighbors of the query vector."""
    if index is None or index.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Index is empty. Insert vectors prior to querying.",
        )

    query_arr = np.array(payload.query_vector, dtype=np.float32)
    if query_arr.ndim != 1 or query_arr.shape[0] != index.dim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query vector dimension ({query_arr.shape[0]}) does not match index dimension ({index.dim})",
        )

    ids, dists = index.search(query=query_arr, k=payload.k)
    matches = [MatchResult(id=int(i), distance=float(d)) for i, d in zip(ids, dists)]

    return SearchResponse(
        ids=[int(i) for i in ids],
        distances=[float(d) for d in dists],
        results=matches,
        total_indexed=index.count(),
    )


if __name__ == "__main__":
    uvicorn.run("vector_core.server.app:app", host="127.0.0.1", port=8001, reload=False)