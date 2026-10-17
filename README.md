# VectorCore

[![CI Pipeline](https://github.com/irtazirfan08-source/VectorCore/actions/workflows/ci.yml/badge.svg)](https://github.com/irtazirfan08-source/VectorCore/actions/workflows/ci.yml)
![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A lightweight, high-performance Approximate Nearest Neighbor (ANN) vector search engine and REST microservice built from scratch in Python and NumPy, featuring Hierarchical Navigable Small World (HNSW) graphs, zero-dependency persistence, and a containerized FastAPI service.

---

## Architecture Overview

```mermaid
flowchart TD
    Client[HTTP Client / CLI] -->|REST / Commands| API[FastAPI Layer - app.py]
    
    subgraph VectorCore Engine
        API --> IndexHandler{Route Handler}
        IndexHandler -->|Insert / Search / Batch| HNSW[HNSWIndex Graph Engine]
        IndexHandler -->|Persistence Save / Load| Serializer[IndexSerializer]
        
        HNSW --> Metric[Distance Metrics: L2 / Cosine / IP]
        Serializer --> BinaryStore[(Disk: .vcore Binary Format)]
    end