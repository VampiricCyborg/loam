"""Distance functions.

Two metrics, both reduced to a single batched numpy operation:

``cosine``
    Vectors are L2-normalized when they enter the index, so the cosine
    *distance* between a query ``q`` and a stored vector ``x`` is simply
    ``1 - dot(q, x)``. Normalizing once at insert time keeps the hot path a
    dot product instead of a dot product plus two norms.

``l2``
    Squared Euclidean distance. The square root is monotonic, so omitting it
    changes no ordering and saves a pass over the data. Any distance value
    reported by Loam under this metric is therefore a *squared* distance.

Every function here takes one query and a matrix of candidates, and returns a
1-D array of distances. That shape is deliberate: the inner loop of
SEARCH-LAYER needs the distance from ``q`` to a whole neighbor list at once.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Metric = Literal["cosine", "l2"]

METRICS: tuple[str, ...] = ("cosine", "l2")


def check_metric(metric: str) -> Metric:
    """Validate a metric name, returning it unchanged."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    return metric  # type: ignore[return-value]


def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows of ``x``, leaving zero rows untouched.

    Works on a single vector (1-D) or a batch (2-D). A zero vector has no
    direction, so it is passed through rather than turned into NaN; it will sit
    at cosine distance 1.0 from everything, which is the honest answer.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr if norm < eps else (arr / norm).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return (arr / norms).astype(np.float32)


def prepare(x: np.ndarray, metric: Metric) -> np.ndarray:
    """Transform vectors into the form the index stores them in."""
    arr = np.ascontiguousarray(x, dtype=np.float32)
    return normalize(arr) if metric == "cosine" else arr


def cosine_batch(q: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine distance from pre-normalized ``q`` to pre-normalized rows.

    No ``astype`` here: float32 inputs give a float32 product, and subtracting
    a Python float leaves the dtype alone. An explicit cast would be a wasted
    copy on the hottest line in the project.
    """
    return 1.0 - matrix @ q


def l2_batch(q: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from ``q`` to each row of ``matrix``."""
    diff = matrix - q
    return np.einsum("ij,ij->i", diff, diff)


def pairwise_distance(matrix: np.ndarray, metric: Metric) -> np.ndarray:
    """Full ``(n, n)`` distance matrix between the rows of ``matrix``.

    Algorithm 4 repeatedly asks "how far is this candidate from the ones I have
    already kept?". Answering that one pair at a time is a swarm of tiny numpy
    calls whose overhead dwarfs their arithmetic. Computing the whole candidate
    block once, as a single matmul, and then reading rows out of it is roughly
    an order of magnitude faster for the candidate-set sizes HNSW uses
    (``ef_construction`` of a few hundred at most).
    """
    n = matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=np.float32)
    gram = matrix @ matrix.T
    if metric == "cosine":
        return 1.0 - gram
    sq = np.einsum("ij,ij->i", matrix, matrix)
    d = sq[:, None] + sq[None, :] - 2.0 * gram
    return np.maximum(d, 0.0, out=d)  # clamp float error on the diagonal


def batch_distance(q: np.ndarray, matrix: np.ndarray, metric: Metric) -> np.ndarray:
    """Distance from one query to every row of ``matrix``."""
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    return cosine_batch(q, matrix) if metric == "cosine" else l2_batch(q, matrix)


def pair_distance(a: np.ndarray, b: np.ndarray, metric: Metric) -> float:
    """Distance between two single vectors, in the index's stored form."""
    if metric == "cosine":
        return float(1.0 - np.dot(a, b))
    diff = a - b
    return float(np.dot(diff, diff))
