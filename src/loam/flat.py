"""The exact oracle.

``FlatIndex`` is brute force: it computes the distance from the query to every
stored vector and sorts. That is ``O(n*d)`` per query and it is exactly the
cost HNSW exists to avoid.

It is here because no approximate number in this repo is reported without a
ground truth computed on the same data. Recall is meaningless otherwise, and
"the neighbors looked plausible" is not a measurement. Every recall figure in
the README and in the test suite is checked against this class.

It is also not slow in the way you might expect: one ``(n, d) @ (d,)`` matmul
is a single BLAS call, so for the dataset sizes Loam benchmarks at, computing
ground truth for a thousand queries takes seconds.
"""

from __future__ import annotations

import numpy as np

from .distance import Metric
from .store import VectorStore


class FlatIndex:
    """Exact k-nearest-neighbor search by full scan."""

    def __init__(self, dim: int, metric: Metric = "cosine") -> None:
        self.store = VectorStore(dim, metric)

    def __len__(self) -> int:
        return len(self.store)

    @property
    def dim(self) -> int:
        return self.store.dim

    @property
    def metric(self) -> Metric:
        return self.store.metric

    def add(self, vectors: np.ndarray) -> np.ndarray:
        """Add one or many vectors; returns their IDs."""
        return self.store.add(vectors)

    def search(self, q: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Exact top-``k`` for one query, as ``(ids, distances)``."""
        if len(self.store) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        query = self.store.prepare_query(q)
        dists = self.store.distances_to_all(query)
        k = min(k, dists.shape[0])

        # argpartition finds the k smallest in O(n); only those k are sorted.
        part = np.argpartition(dists, k - 1)[:k]
        order = part[np.argsort(dists[part], kind="stable")]
        return order.astype(np.int64), dists[order]

    def search_batch(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Exact top-``k`` for a batch of queries, as ``(ids, distances)``.

        Chunked so the intermediate distance matrix stays bounded regardless of
        how many queries are asked for at once.
        """
        arr = np.asarray(queries, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        n = len(self.store)
        if n == 0:
            empty_i = np.empty((arr.shape[0], 0), dtype=np.int64)
            return empty_i, np.empty((arr.shape[0], 0), dtype=np.float32)

        k = min(k, n)
        ids = np.empty((arr.shape[0], k), dtype=np.int64)
        dists = np.empty((arr.shape[0], k), dtype=np.float32)

        chunk = max(1, min(arr.shape[0], 4_000_000 // max(1, n)))
        for start in range(0, arr.shape[0], chunk):
            stop = min(start + chunk, arr.shape[0])
            block = np.stack([self.store.prepare_query(row) for row in arr[start:stop]])

            if self.metric == "cosine":
                d = 1.0 - block @ self.store.vectors.T
            else:
                sq_data = np.einsum("ij,ij->i", self.store.vectors, self.store.vectors)
                sq_query = np.einsum("ij,ij->i", block, block)[:, None]
                d = sq_query + sq_data[None, :] - 2.0 * (block @ self.store.vectors.T)

            part = np.argpartition(d, k - 1, axis=1)[:, :k]
            rows = np.arange(part.shape[0])[:, None]
            order = np.argsort(d[rows, part], axis=1, kind="stable")
            sorted_ids = part[rows, order]
            ids[start:stop] = sorted_ids
            dists[start:stop] = d[rows, sorted_ids]

        return ids, dists
