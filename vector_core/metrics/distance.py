"""Optimized distance metrics for vector similarity search."""

import numpy as np


def l2_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes Euclidean (L2) distance between query vector (1D)

    and candidate vector matrix (2D).
    """
    return np.linalg.norm(b - a, axis=1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes Cosine Similarity between query vector (1D)

    and candidate vector matrix (2D).
    """
    dot_product = np.dot(b, a)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b, axis=1)

    denominator = norm_a * norm_b
    denominator = np.where(denominator == 0, 1e-9, denominator)

    return dot_product / denominator


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes Cosine Distance (1.0 - Cosine Similarity)."""
    return 1.0 - cosine_similarity(a, b)


def inner_product_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes Inner Product (dot product) distance.

    Returns negative dot product so smaller values represent greater similarity.
    """
    return -np.dot(b, a)