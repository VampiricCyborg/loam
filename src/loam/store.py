"""The vector store.

All vectors live in one contiguous ``float32`` matrix, and a node's ID is its
row index. Nothing else in Loam owns vector data.

This is the single most important performance decision in a pure-Python HNSW.
SEARCH-LAYER needs the distance from the query to a node's entire neighbor
list; with the vectors in one matrix that is ``store.take(neighbor_ids) @ q``,
one numpy call, instead of a Python loop over per-node arrays. The interpreter
overhead per distance drops from microseconds to nanoseconds.

The matrix grows by doubling, so ``n`` appends cost ``O(n)`` copies amortized
and the caller never has to declare capacity up front.
"""

from __future__ import annotations

import numpy as np

from .distance import Metric, batch_distance, check_metric, prepare


class VectorStore:
    """A growable, preallocated ``float32`` matrix of row-indexed vectors."""

    def __init__(self, dim: int, metric: Metric = "cosine", capacity: int = 1024) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)
        self.metric = check_metric(metric)
        self._data = np.zeros((max(1, capacity), self.dim), dtype=np.float32)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    @property
    def vectors(self) -> np.ndarray:
        """A view of the populated rows, in stored (e.g. normalized) form."""
        return self._data[: self._size]

    def _ensure_capacity(self, needed: int) -> None:
        capacity = self._data.shape[0]
        if needed <= capacity:
            return
        while capacity < needed:
            capacity *= 2
        grown = np.zeros((capacity, self.dim), dtype=np.float32)
        grown[: self._size] = self._data[: self._size]
        self._data = grown

    def add(self, vectors: np.ndarray) -> np.ndarray:
        """Append one or many vectors; returns the IDs assigned to them."""
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {arr.shape}")

        count = arr.shape[0]
        self._ensure_capacity(self._size + count)
        self._data[self._size : self._size + count] = prepare(arr, self.metric)
        ids = np.arange(self._size, self._size + count, dtype=np.int64)
        self._size += count
        return ids

    def get(self, node_id: int) -> np.ndarray:
        """One stored vector, by ID."""
        if not 0 <= node_id < self._size:
            raise IndexError(f"node {node_id} out of range (size {self._size})")
        return self._data[node_id]

    def take(self, node_ids: np.ndarray | list[int]) -> np.ndarray:
        """Gather the rows for ``node_ids`` into a fresh contiguous matrix.

        A Python list is handed straight to numpy rather than converted with
        ``np.asarray`` first; numpy does the conversion inside the indexing
        call, and skipping the extra round trip is measurable on a path that
        runs once per expanded node.
        """
        return self._data[node_ids]

    def prepare_query(self, q: np.ndarray) -> np.ndarray:
        """Put a raw query vector into the same form as the stored rows."""
        arr = np.asarray(q, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.dim:
            raise ValueError(f"expected a query of dim {self.dim}, got {arr.shape[0]}")
        return prepare(arr, self.metric)

    def distances_to(self, q: np.ndarray, node_ids: np.ndarray | list[int]) -> np.ndarray:
        """Distance from a prepared query to each of ``node_ids``, batched."""
        return batch_distance(q, self.take(node_ids), self.metric)

    def distances_to_all(self, q: np.ndarray) -> np.ndarray:
        """Distance from a prepared query to every stored vector."""
        return batch_distance(q, self.vectors, self.metric)
