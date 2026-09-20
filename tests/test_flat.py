"""The oracle has to be right, or nothing else measured here means anything.

``FlatIndex`` is checked against ``numpy.argsort`` over a distance computed a
different way: the tests spell out ``1 - cos`` and squared Euclidean by hand
rather than calling ``loam.distance``, so a sign error or a stale
normalization in the library cannot agree with itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from loam import FlatIndex


def reference_cosine(data: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine distance, written out independently of ``loam.distance``."""
    d = data / np.linalg.norm(data, axis=1, keepdims=True)
    q = query / np.linalg.norm(query)
    return 1.0 - d @ q


def reference_l2(data: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance, written out independently."""
    return np.sum((data - query) ** 2, axis=1)


REFERENCE = {"cosine": reference_cosine, "l2": reference_l2}


@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_matches_argsort(metric: str) -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((500, 16)).astype(np.float32)
    queries = rng.standard_normal((20, 16)).astype(np.float32)

    index = FlatIndex(dim=16, metric=metric)
    index.add(data)

    for query in queries:
        ids, dists = index.search(query, k=10)
        expected = REFERENCE[metric](data.astype(np.float64), query.astype(np.float64))
        assert ids.tolist() == np.argsort(expected, kind="stable")[:10].tolist()
        np.testing.assert_allclose(dists, expected[ids], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_batch_matches_single(metric: str) -> None:
    """``search_batch`` is an optimization, so it must agree with ``search``."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((300, 8)).astype(np.float32)
    queries = rng.standard_normal((25, 8)).astype(np.float32)

    index = FlatIndex(dim=8, metric=metric)
    index.add(data)
    batch_ids, batch_dists = index.search_batch(queries, k=7)

    for i, query in enumerate(queries):
        ids, dists = index.search(query, k=7)
        assert batch_ids[i].tolist() == ids.tolist()
        np.testing.assert_allclose(batch_dists[i], dists, rtol=1e-4, atol=1e-5)


def test_distances_are_sorted_ascending() -> None:
    rng = np.random.default_rng(2)
    index = FlatIndex(dim=12, metric="cosine")
    index.add(rng.standard_normal((200, 12)).astype(np.float32))
    _, dists = index.search(rng.standard_normal(12).astype(np.float32), k=20)
    assert np.all(np.diff(dists) >= -1e-6)


def test_k_larger_than_dataset_is_clamped() -> None:
    rng = np.random.default_rng(3)
    index = FlatIndex(dim=4, metric="l2")
    index.add(rng.standard_normal((5, 4)).astype(np.float32))
    ids, dists = index.search(rng.standard_normal(4).astype(np.float32), k=50)
    assert ids.shape == (5,) and dists.shape == (5,)
    assert sorted(ids.tolist()) == [0, 1, 2, 3, 4]


def test_exact_match_has_zero_distance() -> None:
    """A stored vector is its own nearest neighbor, at distance ~0."""
    rng = np.random.default_rng(4)
    data = rng.standard_normal((100, 10)).astype(np.float32)
    for metric in ("cosine", "l2"):
        index = FlatIndex(dim=10, metric=metric)
        index.add(data)
        ids, dists = index.search(data[42], k=1)
        assert ids[0] == 42
        assert abs(float(dists[0])) < 1e-4


def test_empty_index_returns_nothing() -> None:
    index = FlatIndex(dim=3, metric="cosine")
    ids, dists = index.search(np.ones(3, dtype=np.float32), k=5)
    assert ids.shape == (0,) and dists.shape == (0,)


def test_wrong_dimension_is_rejected() -> None:
    index = FlatIndex(dim=4, metric="cosine")
    index.add(np.ones((2, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        index.search(np.ones(5, dtype=np.float32), k=1)


def test_zero_vector_does_not_produce_nan() -> None:
    """A zero vector has no direction; it must not poison the index."""
    data = np.zeros((3, 4), dtype=np.float32)
    data[1] = [1.0, 0.0, 0.0, 0.0]
    index = FlatIndex(dim=4, metric="cosine")
    index.add(data)
    _, dists = index.search(np.array([1.0, 0, 0, 0], dtype=np.float32), k=3)
    assert not np.isnan(dists).any()
