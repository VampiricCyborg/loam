"""Per-query instrumentation.

Production ANN libraries report latency and leave you to guess why. Loam
counts the work instead, so a change in ``ef`` or in the neighbor-selection
strategy shows up as a change in *distance computations* and *hops*, not just
in milliseconds that also depend on your CPU's mood.

The three numbers, and what each one tells you:

``distance_computations``
    Every vector-to-query distance evaluated, including the batched ones. This
    is the algorithm-level cost, independent of hardware. For ``FlatIndex`` it
    equals ``n``; a good HNSW run on the same data lands orders of magnitude
    below that, and the ratio is the whole point of the index.

``visited``
    Distinct nodes popped or marked seen. Tracks how much of the graph the
    beam actually touched.

``hops_per_layer``
    Nodes expanded on each layer, highest layer first. Upper layers should show
    a handful of hops (greedy, ``ef = 1``); layer 0 carries the beam search and
    dominates the total.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryStats:
    """Work performed while answering a single query."""

    distance_computations: int = 0
    visited: int = 0
    hops: dict[int, int] = field(default_factory=dict)
    entry_point: int | None = None
    top_layer: int = 0

    def add_distances(self, n: int) -> None:
        self.distance_computations += n

    def add_visited(self, n: int = 1) -> None:
        self.visited += n

    def add_hop(self, layer: int, n: int = 1) -> None:
        self.hops[layer] = self.hops.get(layer, 0) + n

    @property
    def hops_per_layer(self) -> list[tuple[int, int]]:
        """Hops as ``(layer, count)`` pairs, highest layer first."""
        return sorted(self.hops.items(), reverse=True)

    @property
    def total_hops(self) -> int:
        return sum(self.hops.values())

    def merge(self, other: "QueryStats") -> None:
        """Fold another stats object into this one (used across layers)."""
        self.distance_computations += other.distance_computations
        self.visited += other.visited
        for layer, count in other.hops.items():
            self.hops[layer] = self.hops.get(layer, 0) + count

    def __str__(self) -> str:
        hops = ", ".join(f"L{layer}:{count}" for layer, count in self.hops_per_layer)
        return (
            f"QueryStats(dist_comps={self.distance_computations}, "
            f"visited={self.visited}, hops=[{hops}])"
        )


@dataclass
class LayerTrace:
    """What happened on one layer of a single traced search."""

    layer: int
    ef: int
    entry_points: list[tuple[int, float]] = field(default_factory=list)
    expansions: list[tuple[int, float, int]] = field(default_factory=list)
    result: list[tuple[float, int]] = field(default_factory=list)

    @property
    def best(self) -> tuple[float, int] | None:
        return self.result[0] if self.result else None


@dataclass
class SearchTrace:
    """The full path one query took through the graph.

    Recorded only when explicitly asked for (``loam trace``), because keeping
    it costs allocations on the hot path.
    """

    layers: list[LayerTrace] = field(default_factory=list)
    entry_point: int | None = None
    top_layer: int = 0
    ef: int = 0
    k: int = 0
    stats: QueryStats | None = None
