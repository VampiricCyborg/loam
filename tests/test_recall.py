"""Recall floors: a regression guard, not a quality claim.

Each threshold below was *measured* before it was written down. The procedure
was: run the configuration across five seeds, take the worst recall observed,
and round down with headroom. The number in the assertion is therefore a floor
the implementation has already cleared with room to spare, so the test fails
when construction genuinely regresses rather than when a random draw is
unlucky.

That headroom is the point. A floor set at the measured value would flake; a
floor set far below it would catch nothing. If you change graph construction
and one of these fails, do not lower the threshold. Find out what the change
did to the graph first.

Everything here is checked against ``FlatIndex`` on the same vectors. There is
no stored ground truth anywhere in the suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from loam import FlatIndex, HNSWIndex
from loam.bench import recall_at_k

K = 10


def measure(
    data: np.ndarray,
    queries: np.ndarray,
    ef: int,
    metric: str = "cosine",
    seed: int = 0,
    **kwargs,
) -> float:
    """Build, query, and score against the exact oracle on the same data."""
    index = HNSWIndex(dim=data.shape[1], metric=metric, seed=seed, **kwargs)  # type: ignore[arg-type]
    index.add(data)

    oracle = FlatIndex(dim=data.shape[1], metric=metric)  # type: ignore[arg-type]
    oracle.add(data)
    exact, _ = oracle.search_batch(queries, k=K)

    approx = np.stack([index.search(q, k=K, ef=ef)[0] for q in queries])
    return recall_at_k(approx, exact, K)


@pytest.fixture(scope="module")
def uniform() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return (
        rng.standard_normal((1200, 16)).astype(np.float32),
        rng.standard_normal((150, 16)).astype(np.float32),
    )


@pytest.fixture(scope="module")
def clustered() -> tuple[np.ndarray, np.ndarray]:
    """Forty tight blobs in ten dimensions: the heuristic's proving ground."""
    rng = np.random.default_rng(2)
    centers = rng.standard_normal((40, 10)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    data = centers[rng.integers(0, 40, 1500)] + 0.08 * rng.standard_normal((1500, 10))
    queries = centers[rng.integers(0, 40, 150)] + 0.08 * rng.standard_normal((150, 10))
    return data.astype(np.float32), queries.astype(np.float32)


# Worst over seeds 0-4: ef=16 -> 0.9793, ef=32 -> 0.9993, ef=64 -> 1.0000.
@pytest.mark.parametrize("ef, floor", [(16, 0.95), (32, 0.98), (64, 0.99)])
def test_recall_floor_on_uniform_data(uniform, ef: int, floor: float) -> None:
    data, queries = uniform
    recall = measure(data, queries, ef=ef, M=16, ef_construction=100)
    assert recall >= floor, f"recall@{K} at ef={ef} was {recall:.4f}, floor is {floor}"


def test_recall_improves_monotonically_with_ef(uniform) -> None:
    """The core trade-off: more beam must not buy less accuracy.

    A non-monotonic curve is the signature of a bug in the stopping condition
    or in the eviction of the result set, and it is not something a single
    threshold would catch.
    """
    data, queries = uniform
    recalls = [measure(data, queries, ef=ef, M=16, ef_construction=100) for ef in (8, 16, 32, 64)]
    for lower, higher in zip(recalls, recalls[1:]):
        assert higher >= lower - 1e-9, f"recall went down as ef grew: {recalls}"


# Worst over seeds 0-4 on harder 32-d data: ef=32 -> 0.9873, ef=64 -> 1.0000.
@pytest.mark.parametrize("ef, floor", [(32, 0.95), (64, 0.99)])
def test_recall_floor_on_l2(uniform, ef: int, floor: float) -> None:
    """The L2 path is separate code and needs its own floor."""
    data, queries = uniform
    recall = measure(data, queries, ef=ef, metric="l2", M=16, ef_construction=100)
    assert recall >= floor, f"L2 recall@{K} at ef={ef} was {recall:.4f}, floor is {floor}"


# Worst over seeds 0-4: 1.0000 at every ef. The heuristic handles this case
# outright, which is exactly the claim `loam ablate selection` tests at scale.
@pytest.mark.parametrize("ef, floor", [(16, 0.95), (64, 0.99)])
def test_recall_floor_on_clustered_data(clustered, ef: int, floor: float) -> None:
    """Clustered data is the case simple selection struggles with.

    The heuristic clears it outright, so these floors are not lower than the
    uniform ones. They exist to catch a regression that would turn Algorithm 4
    back into Algorithm 3 in practice.
    """
    data, queries = clustered
    recall = measure(data, queries, ef=ef, M=16, ef_construction=100)
    assert recall >= floor, f"clustered recall@{K} at ef={ef} was {recall:.4f}, floor is {floor}"


def test_flat_graph_still_answers_well(uniform) -> None:
    """``hierarchical=False`` removes the layers, not the accuracy.

    A flat NSW graph should stay competitive at a generous ``ef``. If this
    fails, the hierarchy ablation is not measuring what it claims to, because
    the flat variant is simply broken.
    """
    data, queries = uniform
    recall = measure(data, queries, ef=64, M=16, ef_construction=100, hierarchical=False)
    assert recall >= 0.97, f"flat NSW recall@{K} at ef=64 was {recall:.4f}"


def test_simple_selection_still_works_on_uniform_data(uniform) -> None:
    """Algorithm 3 is not broken, it is just less robust than Algorithm 4.

    On uniform data the two should be close. The ablation's claim is about
    *clustered* data, and a test that showed simple selection failing here
    would mean something else had gone wrong.
    """
    data, queries = uniform
    recall = measure(data, queries, ef=64, M=16, ef_construction=100, selection="simple")
    assert recall >= 0.97, f"simple-selection recall@{K} at ef=64 was {recall:.4f}"


def test_searching_for_a_stored_vector_finds_it(uniform) -> None:
    """The strongest single-query check: a point must retrieve itself.

    If an indexed vector cannot find itself at a wide beam, it is unreachable
    in the graph, which no aggregate recall number would isolate.
    """
    data, _ = uniform
    index = HNSWIndex(dim=data.shape[1], M=16, ef_construction=100, seed=0)
    index.add(data)

    rng = np.random.default_rng(9)
    misses = [i for i in rng.choice(data.shape[0], 60, replace=False) if index.search(data[i], k=1, ef=64)[0][0] != i]
    assert not misses, f"{len(misses)} indexed vectors could not retrieve themselves: {misses[:5]}"


def test_recall_is_one_when_ef_covers_the_dataset(uniform) -> None:
    """With a beam as wide as the data, an exhaustive search is exact."""
    data, queries = uniform
    small, small_q = data[:200], queries[:20]
    recall = measure(small, small_q, ef=200, M=16, ef_construction=100)
    assert recall == pytest.approx(1.0)
