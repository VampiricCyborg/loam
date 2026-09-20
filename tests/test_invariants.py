"""Structural invariants of the graph, checked with hypothesis.

Recall tests tell you the index is *useful*. These tell you it is *well
formed*, which is a different and more basic claim: a graph can return decent
neighbors while quietly violating a degree bound or orphaning a layer, and the
damage only shows up as mysterious recall loss at a larger ``n``.

The five properties, and why each one matters:

* **Degree bounds.** Exceeding ``M_max`` (or ``M_max0`` on layer 0) means the
  shrinking step in Algorithm 1 is not firing. Search cost per hop is
  proportional to degree, so an unbounded graph silently becomes brute force.
* **Layer nesting.** A node on layer ``l`` must exist on every layer below it,
  or the descent can hand down an entry point that does not exist where it
  lands.
* **No self-loops.** A node linked to itself wastes a slot in its degree
  budget and can never advance a search.
* **Valid neighbor IDs.** Every ID must name a real node.
* **Entry point on the top layer.** Algorithm 5 starts there. If the entry
  point is not on the highest populated layer, the upper layers are dead
  weight that every query still pays to traverse.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from loam import HNSWIndex

# Builds are the expensive part, so the search space is kept narrow and the
# example count low. These are structural properties: they either hold for all
# inputs or fail on almost any of them.
SLOW = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow],
)


def build(
    n: int,
    dim: int,
    M: int,
    seed: int,
    selection: str = "heuristic",
    metric: str = "cosine",
    hierarchical: bool = True,
    clustered: bool = False,
) -> HNSWIndex:
    rng = np.random.default_rng(seed)
    if clustered:
        centers = rng.standard_normal((max(2, n // 20), dim)).astype(np.float32)
        data = centers[rng.integers(0, centers.shape[0], n)] + 0.05 * rng.standard_normal(
            (n, dim)
        ).astype(np.float32)
    else:
        data = rng.standard_normal((n, dim)).astype(np.float32)

    index = HNSWIndex(
        dim=dim,
        metric=metric,  # type: ignore[arg-type]
        M=M,
        ef_construction=max(16, 2 * M),
        selection=selection,  # type: ignore[arg-type]
        hierarchical=hierarchical,
        seed=seed,
    )
    index.add(data.astype(np.float32))
    return index


def assert_invariants(index: HNSWIndex) -> None:
    """Every structural claim Loam makes about a finished graph."""
    n = len(index)
    if n == 0:
        return

    for layer, graph in enumerate(index.layers):
        cap = index.M_max0 if layer == 0 else index.M_max
        for node, neighbors in graph.items():
            assert len(neighbors) <= cap, (
                f"node {node} has {len(neighbors)} links on layer {layer}, cap is {cap}"
            )
            assert node not in neighbors, f"node {node} links to itself on layer {layer}"
            assert len(set(neighbors)) == len(neighbors), (
                f"node {node} has duplicate links on layer {layer}"
            )
            for neighbor in neighbors:
                assert 0 <= neighbor < n, f"node {node} links to unknown id {neighbor}"
                assert neighbor in graph, (
                    f"node {node} links to {neighbor}, which is absent from layer {layer}"
                )

    # Layer nesting: presence on a layer implies presence on every layer below.
    for layer in range(len(index.layers) - 1, 0, -1):
        below = index.layers[layer - 1]
        for node in index.layers[layer]:
            assert node in below, f"node {node} is on layer {layer} but not on layer {layer - 1}"

    # Layer 0 holds everything.
    assert len(index.layers[0]) == n

    # The entry point sits on the top populated layer.
    assert index.entry_point is not None
    populated = [i for i, graph in enumerate(index.layers) if graph]
    assert index.max_layer == max(populated)
    assert index.entry_point in index.layers[index.max_layer]


@given(
    n=st.integers(min_value=1, max_value=120),
    dim=st.integers(min_value=2, max_value=12),
    M=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=2**16),
)
@SLOW
def test_invariants_hold_for_heuristic_selection(n: int, dim: int, M: int, seed: int) -> None:
    assert_invariants(build(n, dim, M, seed, selection="heuristic"))


@given(
    n=st.integers(min_value=1, max_value=120),
    dim=st.integers(min_value=2, max_value=12),
    M=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=2**16),
)
@SLOW
def test_invariants_hold_for_simple_selection(n: int, dim: int, M: int, seed: int) -> None:
    assert_invariants(build(n, dim, M, seed, selection="simple"))


@given(
    n=st.integers(min_value=1, max_value=100),
    M=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=2**16),
    metric=st.sampled_from(["cosine", "l2"]),
)
@SLOW
def test_invariants_hold_on_clustered_data(n: int, M: int, seed: int, metric: str) -> None:
    """Clustered data is where the heuristic's pruning is most aggressive."""
    assert_invariants(build(n, 6, M, seed, metric=metric, clustered=True))


@given(
    n=st.integers(min_value=1, max_value=100),
    M=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=2**16),
)
@SLOW
def test_flat_graph_has_exactly_one_layer(n: int, M: int, seed: int) -> None:
    """``hierarchical=False`` must collapse everything onto layer 0."""
    index = build(n, 6, M, seed, hierarchical=False)
    assert_invariants(index)
    assert index.max_layer == 0
    assert index.layer_sizes() == [n]


@given(
    n=st.integers(min_value=2, max_value=120),
    M=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=2**16),
)
@SLOW
def test_search_returns_valid_distinct_ids(n: int, M: int, seed: int) -> None:
    """Whatever a search returns must be usable: real, distinct, sorted."""
    index = build(n, 6, M, seed)
    rng = np.random.default_rng(seed + 1)
    ids, dists = index.search(rng.standard_normal(6).astype(np.float32), k=5, ef=20)

    assert ids.shape[0] == min(5, n)
    assert len(set(ids.tolist())) == ids.shape[0]
    assert all(0 <= int(i) < n for i in ids)
    assert np.all(np.diff(dists) >= -1e-6)


def test_layer_populations_decay() -> None:
    """The level distribution should thin out by roughly a factor of M.

    Loose bounds on purpose: this is a random draw, so the test guards against
    a broken ``mL`` or a missing logarithm, not against ordinary variance.
    """
    index = build(3000, 8, M=16, seed=7)
    sizes = index.layer_sizes()
    assert sizes[0] == 3000
    assert len(sizes) >= 2
    for lower, upper in zip(sizes, sizes[1:]):
        assert upper < lower
    assert 3000 / 16 * 0.5 < sizes[1] < 3000 / 16 * 2.0
