"""Same data, same seed, same graph, bit for bit.

Determinism is not a nicety here, it is what makes the benchmarks in the
README reproducible and the ablations meaningful. An ablation compares two
builds and attributes the difference to one changed flag; that attribution is
only valid if everything else about the build is fixed.

Two sources of nondeterminism are easy to introduce by accident and are both
guarded below: the level draw (which must come from the seeded generator and
nothing else) and iteration order over sets, which Python does not promise to
keep stable across processes for arbitrary objects.
"""

from __future__ import annotations

import numpy as np
import pytest

from loam import HNSWIndex


def graph_signature(index: HNSWIndex) -> list[list[tuple[int, tuple[int, ...]]]]:
    """A hashable, order-insensitive snapshot of the full adjacency."""
    return [
        sorted((node, tuple(sorted(neighbors))) for node, neighbors in graph.items())
        for graph in index.layers
    ]


def build(seed: int | None, data: np.ndarray, **kwargs) -> HNSWIndex:
    index = HNSWIndex(dim=data.shape[1], M=8, ef_construction=50, seed=seed, **kwargs)
    index.add(data)
    return index


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((400, 12)).astype(np.float32)


def test_same_seed_gives_identical_graph(data: np.ndarray) -> None:
    a, b = build(42, data), build(42, data)
    assert graph_signature(a) == graph_signature(b)
    assert a.entry_point == b.entry_point
    assert a.max_layer == b.max_layer
    assert a.layer_sizes() == b.layer_sizes()


def test_same_seed_gives_identical_answers(data: np.ndarray) -> None:
    a, b = build(42, data), build(42, data)
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.standard_normal(12).astype(np.float32)
        ids_a, dists_a = a.search(q, k=10, ef=32)
        ids_b, dists_b = b.search(q, k=10, ef=32)
        assert ids_a.tolist() == ids_b.tolist()
        np.testing.assert_array_equal(dists_a, dists_b)


def test_different_seeds_give_different_graphs(data: np.ndarray) -> None:
    """A guard against the seed being ignored entirely."""
    assert graph_signature(build(1, data)) != graph_signature(build(2, data))


def test_instrumentation_is_reproducible(data: np.ndarray) -> None:
    """The work counters must be a property of the graph, not of timing."""
    index = build(42, data)
    q = np.random.default_rng(3).standard_normal(12).astype(np.float32)
    first = index.search(q, k=10, ef=32, return_stats=True)[2]
    second = index.search(q, k=10, ef=32, return_stats=True)[2]
    assert first.distance_computations == second.distance_computations
    assert first.visited == second.visited
    assert first.hops == second.hops


def test_batch_add_matches_incremental_insert(data: np.ndarray) -> None:
    """``add`` is a loop over ``insert``, and must stay one."""
    batched = build(42, data)
    one_at_a_time = HNSWIndex(dim=12, M=8, ef_construction=50, seed=42)
    for row in data:
        one_at_a_time.insert(row)
    assert graph_signature(batched) == graph_signature(one_at_a_time)


def test_ablation_flags_change_the_graph(data: np.ndarray) -> None:
    """Each ablation flag must actually alter the build it claims to."""
    baseline = build(42, data)
    assert graph_signature(build(42, data, selection="simple")) != graph_signature(baseline)
    assert graph_signature(build(42, data, hierarchical=False)) != graph_signature(baseline)


def test_stats_counters_start_at_zero(data: np.ndarray) -> None:
    """A fresh QueryStats must not carry work from a previous query."""
    index = build(42, data)
    q = np.random.default_rng(4).standard_normal(12).astype(np.float32)
    _, _, stats = index.search(q, k=5, ef=16, return_stats=True)
    assert stats.distance_computations > 0
    assert stats.total_hops > 0
    assert stats.visited >= stats.total_hops
