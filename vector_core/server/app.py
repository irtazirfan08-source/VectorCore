"""
VectorCore FastAPI Server
Exposes HTTP endpoints for inserting high-dimensional vectors and querying nearest neighbors via HNSW.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

from vector_core.index.hnsw import HNSWIndex
from vector_core.storage.serializer import IndexSerializer

app = FastAPI(title="VectorCore Search Engine API", version="1.0.0")

# Initialize global HNSW index (128-D vectors, L2 distance)
DIMENSION = 128
index = HNSWIndex(dim=DIMENSION, metric="l2", m=16, ef_construction=64, ef_search=32)


class InsertRequest(BaseModel):
    vector_ids: List[int]
    vectors: List[List[float]]


class SearchRequest(BaseModel):
    query_vector: List[float]
    k: int = 5


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "total_indexed_vectors": index.count(),
        "dimension": index.dim,
        "metric": index.metric
    }


@app.post("/v1/vectors/insert")
def insert_vectors(req: InsertRequest):
    if len(req.vector_ids) != len(req.vectors):
        raise HTTPException(status_code=400, detail="Length of vector_ids must match length of vectors.")
    
    vec_array = np.array(req.vectors, dtype=np.float32)
    if vec_array.shape[1] != DIMENSION:
        raise HTTPException(status_code=400, detail=f"Vectors must have dimension {DIMENSION}.")

    index.add(ids=req.vector_ids, vectors=vec_array)
    return {
        "message": f"Successfully indexed {len(req.vector_ids)} vectors.",
        "total_vectors": index.count()
    }


@app.post("/v1/vectors/search")
def search_vectors(req: SearchRequest):
    if len(req.query_vector) != DIMENSION:
        raise HTTPException(status_code=400, detail=f"Query vector must have dimension {DIMENSION}.")

    q_array = np.array(req.query_vector, dtype=np.float32)
    result_ids, distances = index.search(query=q_array, k=req.k)

    results = [{"id": vid, "distance": round(float(dist), 6)} for vid, dist in zip(result_ids, distances)]
    return {
        "query_k": req.k,
        "results": results
    }