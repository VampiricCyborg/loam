"""Hierarchical Navigable Small World graphs.

Every method below is named after an algorithm in Malkov & Yashunin
(arXiv:1603.09320), so the code can be read next to the paper:

===========================================  ==================================
Paper                                        Loam
===========================================  ==================================
Algorithm 1  INSERT                          ``HNSWIndex.insert``
Algorithm 2  SEARCH-LAYER                    ``HNSWIndex._search_layer``
Algorithm 3  SELECT-NEIGHBORS-SIMPLE         ``HNSWIndex._select_neighbors_simple``
Algorithm 4  SELECT-NEIGHBORS-HEURISTIC      ``HNSWIndex._select_neighbors_heuristic``
Algorithm 5  K-NN-SEARCH                     ``HNSWIndex.knn_search``
===========================================  ==================================

Data layout
-----------
``self.layers[level][node_id] -> list[int]`` is the adjacency, plain Python
lists so invariants are easy to assert and the graph is easy to print. Vectors
themselves live in a VectorStore, one row per node.

Inside SEARCH-LAYER the candidate set ``C`` is a min-heap and the dynamic
result list ``W`` is a max-heap (distances negated), which is exactly the pair
of structures Algorithm 2 needs: cheapest-first expansion, and cheap access to
the current furthest result for the stopping condition.
"""

from __future__ import annotations

import math
import random
from heapq import heapify, heappop, heappush, heapreplace
from typing import Callable, Iterable, Literal

import numpy as np

from .distance import Metric, batch_distance, check_metric, pairwise_distance
from .stats import LayerTrace, QueryStats, SearchTrace
from .store import VectorStore

Selection = Literal["simple", "heuristic"]


class HNSWIndex:
    """An HNSW graph index.

    Parameters follow the paper's defaults. ``M`` is the number of links a new
    node establishes per layer; ``M_max0`` (``2*M``) is the degree cap on layer
    0, where the graph must stay dense enough to be accurate; ``mL = 1/ln(M)``
    is the level-generation normalizer that makes the layer populations decay
    exponentially, so layer ``l`` holds roughly ``n / M**l`` nodes.

    ``ef_construction`` is the beam width used while *building*: it buys graph
    quality once, at build time. ``ef`` at search time buys recall per query.

    Set ``hierarchical=False`` to force every node onto layer 0. The result is
    a flat navigable small world graph, which is the ablation for the "Hub
    Highway" claim of arXiv:2412.01940.
    """

    def __init__(
        self,
        dim: int,
        metric: Metric = "cosine",
        M: int = 16,
        ef_construction: int = 200,
        M_max0: int | None = None,
        mL: float | None = None,
        selection: Selection = "heuristic",
        extend_candidates: bool = False,
        keep_pruned_connections: bool = True,
        hierarchical: bool = True,
        seed: int | None = None,
    ) -> None:
        if M < 2:
            raise ValueError(f"M must be at least 2, got {M}")
        if selection not in ("simple", "heuristic"):
            raise ValueError(f"selection must be 'simple' or 'heuristic', got {selection!r}")

        self.store = VectorStore(dim, check_metric(metric))
        self.M = int(M)
        self.M_max = int(M)
        self.M_max0 = int(M_max0) if M_max0 is not None else 2 * int(M)
        self.mL = float(mL) if mL is not None else 1.0 / math.log(M)
        self.ef_construction = int(ef_construction)
        self.selection: Selection = selection
        self.extend_candidates = bool(extend_candidates)
        self.keep_pruned_connections = bool(keep_pruned_connections)
        self.hierarchical = bool(hierarchical)
        self.seed = seed

        self._rng = random.Random(seed)
        self.layers: list[dict[int, list[int]]] = []
        self.entry_point: int | None = None
        self.max_layer: int = -1
        self.build_distance_computations: int = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.store)

    @property
    def dim(self) -> int:
        return self.store.dim

    @property
    def metric(self) -> Metric:
        return self.store.metric

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def layer_sizes(self) -> list[int]:
        """Node count per layer, layer 0 first."""
        return [len(graph) for graph in self.layers]

    def neighbors(self, node_id: int, layer: int = 0) -> list[int]:
        """The adjacency list of ``node_id`` on ``layer``."""
        if not 0 <= layer < len(self.layers):
            return []
        return list(self.layers[layer].get(node_id, ()))

    def node_level(self, node_id: int) -> int:
        """The highest layer ``node_id`` appears on."""
        for layer in range(len(self.layers) - 1, -1, -1):
            if node_id in self.layers[layer]:
                return layer
        return -1

    # ------------------------------------------------------------------
    # Level generation
    # ------------------------------------------------------------------

    def _draw_level(self) -> int:
        """``l = floor(-ln(U(0,1)) * mL)``, the paper's level distribution.

        This is the skip-list idea: an exponentially decaying share of nodes
        reaches each successive layer, so the expected number of layers is
        ``O(log n)`` and the upper layers stay sparse enough that the greedy
        descent through them is cheap.
        """
        if not self.hierarchical:
            return 0
        u = self._rng.random()
        while u <= 0.0:  # -ln(0) is undefined, so redraw rather than clamp
            u = self._rng.random()
        return int(-math.log(u) * self.mL)

    def _ensure_layers(self, level: int) -> None:
        while len(self.layers) <= level:
            self.layers.append({})

    # ------------------------------------------------------------------
    # Algorithm 2: SEARCH-LAYER
    # ------------------------------------------------------------------

    def _search_layer(
        self,
        q: np.ndarray,
        entry_points: Iterable[int],
        ef: int,
        layer: int,
        stats: QueryStats | None = None,
        trace: SearchTrace | None = None,
    ) -> list[tuple[float, int]]:
        """Algorithm 2. Beam search of width ``ef`` within a single layer.

        Returns the dynamic list ``W`` as a **max-heap** of ``(-distance, id)``,
        negated so ``W[0]`` is the current furthest result, which is what both
        the stopping condition and the eviction step need.

        The loop is the paper's verbatim: pop the nearest candidate ``c``; if it
        is further from ``q`` than the furthest element of ``W``, every
        remaining candidate is too, so stop. Otherwise expand ``c``'s unvisited
        neighbors and admit each one that either beats the furthest result or
        fills a still-hungry beam.
        """
        graph = self.layers[layer]
        entries = [e for e in entry_points if e in graph]
        if not entries:
            return []

        entry_dists = self.store.distances_to(q, entries)
        if stats is not None:
            stats.add_distances(len(entries))
            stats.add_visited(len(entries))

        visited: set[int] = set(entries)
        # C: min-heap of candidates to expand. W: max-heap of the best results.
        candidates: list[tuple[float, int]] = [
            (float(d), int(e)) for d, e in zip(entry_dists, entries)
        ]
        heapify(candidates)
        results: list[tuple[float, int]] = [(-d, e) for d, e in candidates]
        heapify(results)
        while len(results) > ef:
            heappop(results)

        layer_trace = None
        if trace is not None:
            layer_trace = LayerTrace(
                layer=layer,
                ef=ef,
                entry_points=[(e, d) for d, e in candidates],
            )

        store = self.store
        while candidates:
            dist_c, c = heappop(candidates)
            if dist_c > -results[0][0]:
                break  # every remaining candidate is further still

            if stats is not None:
                stats.add_hop(layer)

            neighbors = [e for e in graph.get(c, ()) if e not in visited]
            if layer_trace is not None:
                layer_trace.expansions.append((c, dist_c, len(neighbors)))
            if not neighbors:
                continue

            visited.update(neighbors)
            # One numpy call for the whole neighbor list: the hot path.
            dists = store.distances_to(q, neighbors)
            if stats is not None:
                stats.add_distances(len(neighbors))
                stats.add_visited(len(neighbors))

            # A plain Python loop, deliberately. Screening these with a numpy
            # mask was measured and is slower: adjacency lists are at most
            # ``M_max0`` long, and at that size a numpy call costs more than
            # the interpreted loop it would replace.
            for d, e in zip(dists.tolist(), neighbors):
                if len(results) < ef:
                    heappush(candidates, (d, e))
                    heappush(results, (-d, e))
                elif d < -results[0][0]:
                    heappush(candidates, (d, e))
                    heapreplace(results, (-d, e))

        if layer_trace is not None and trace is not None:
            layer_trace.result = sorted((-nd, n) for nd, n in results)
            trace.layers.append(layer_trace)

        return results

    # ------------------------------------------------------------------
    # Algorithms 3 and 4: SELECT-NEIGHBORS
    # ------------------------------------------------------------------

    def _select_neighbors(
        self,
        q_vec: np.ndarray,
        candidates: list[tuple[float, int]],
        M: int,
        layer: int,
    ) -> list[int]:
        """Dispatch to the configured selection strategy."""
        if self.selection == "simple":
            return self._select_neighbors_simple(candidates, M)
        return self._select_neighbors_heuristic(q_vec, candidates, M, layer)

    @staticmethod
    def _select_neighbors_simple(candidates: list[tuple[float, int]], M: int) -> list[int]:
        """Algorithm 3. Keep the ``M`` nearest candidates, nothing more.

        Cheap, and perfectly adequate on uniform data. Its failure mode is
        clustered data: every link points back into the node's own cluster, so
        the graph fragments at cluster boundaries and greedy search has no edge
        with which to cross them.
        """
        return [node for _, node in sorted(candidates)[:M]]

    def _select_neighbors_heuristic(
        self,
        q_vec: np.ndarray,
        candidates: list[tuple[float, int]],
        M: int,
        layer: int,
    ) -> list[int]:
        """Algorithm 4. Keep candidates that are *diverse*, not merely close.

        Walking candidates nearest-first, ``e`` is kept only if it is closer to
        ``q`` than it is to any element already kept. A candidate lying in the
        same direction as one already selected gets pruned, so the surviving
        links spread out around ``q`` and at least one of them tends to point
        across a cluster boundary. That is what keeps the graph navigable on
        clustered data (paper, Fig. 7).

        Implementation note: the naive form asks "is ``e`` closer to any ``r``?"
        once per candidate, which is a Python loop over pairs. Instead, each
        time an element is selected we compute its distance to every remaining
        candidate in one batched call and fold it into a running
        ``min_to_selected`` array. That turns ``O(|C| * M)`` scalar distances
        into ``M`` numpy calls.
        """
        if not candidates:
            return []

        pool = {int(n): float(d) for d, n in candidates}  # dedupes entry points
        if self.extend_candidates:
            graph = self.layers[layer]
            extra = sorted(
                {adj for node in list(pool) for adj in graph.get(node, ()) if adj not in pool}
            )
            if extra:
                extra_d = self.store.distances_to(q_vec, extra)
                self.build_distance_computations += len(extra)
                pool.update({int(n): float(d) for n, d in zip(extra, extra_d)})

        # Nearest-first, ties broken by ID so the build stays deterministic.
        ordered = sorted((d, n) for n, d in pool.items())
        ids = np.array([n for _, n in ordered], dtype=np.int64)
        d_to_q = np.array([d for d, _ in ordered], dtype=np.float32)
        vectors = self.store.take(ids)

        # Candidate-to-candidate distances, needed for the diversity test.
        # Computing the whole block up front is a single matmul but does
        # ``O(n^2)`` arithmetic; computing one row per *selected* element does
        # only ``O(M*n)`` but pays numpy call overhead ``M`` times. Which wins
        # depends on what fraction of the pool survives. Shrinking a neighbor
        # list keeps nearly all of a short pool, so the block is cheaper;
        # picking ``M`` out of ``ef_construction`` candidates keeps a small
        # fraction, so the rows are. Both were measured.
        block = pairwise_distance(vectors, self.metric) if 2 * M >= ids.shape[0] else None

        min_to_selected = np.full(ids.shape[0], np.inf, dtype=np.float32)
        selected: list[int] = []
        discarded: list[int] = []

        for i in range(ids.shape[0]):
            if len(selected) >= M:
                discarded.extend(int(n) for n in ids[i:])
                break
            if d_to_q[i] < min_to_selected[i]:
                selected.append(int(ids[i]))
                tail = min_to_selected[i + 1 :]
                if tail.shape[0]:
                    row = (
                        block[i, i + 1 :]
                        if block is not None
                        else batch_distance(vectors[i], vectors[i + 1 :], self.metric)
                    )
                    # Counted as algorithmic work: the paper compares the
                    # element against each remaining candidate here.
                    self.build_distance_computations += tail.shape[0]
                    np.minimum(tail, row, out=tail)
            else:
                discarded.append(int(ids[i]))

        if self.keep_pruned_connections:
            # Pruned candidates are still the nearest things around, so refilling
            # with them spends the degree budget rather than wasting it.
            for node in discarded:
                if len(selected) >= M:
                    break
                selected.append(node)

        return selected

    # ------------------------------------------------------------------
    # Algorithm 1: INSERT
    # ------------------------------------------------------------------

    def add(
        self,
        vectors: np.ndarray,
        progress: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Insert many vectors, one at a time. Returns the assigned IDs."""
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        ids = np.empty(arr.shape[0], dtype=np.int64)
        for i in range(arr.shape[0]):
            ids[i] = self.insert(arr[i])
            if progress is not None and (i + 1) % 128 == 0:
                progress(128)
        if progress is not None:
            progress(arr.shape[0] % 128)
        return ids

    def insert(self, vector: np.ndarray) -> int:
        """Algorithm 1. Insert one element and wire it into the graph.

        Two descents. From the top layer down to ``l+1`` the new node is not a
        resident, so the search is purely navigational: greedy, ``ef = 1``, each
        layer handing its closest node down as the next entry point. From
        ``min(L, l)`` down to 0 the node *is* a resident, so each layer runs a
        full ``ef_construction`` search, picks ``M`` neighbors out of the
        result, links bidirectionally, and shrinks any neighbor that has now
        exceeded its degree cap.
        """
        node_id = int(self.store.add(vector)[0])
        level = self._draw_level()
        self._ensure_layers(max(level, self.max_layer, 0))

        for lc in range(level + 1):
            self.layers[lc].setdefault(node_id, [])

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_layer = level
            return node_id

        q = self.store.get(node_id)
        L = self.max_layer
        ep: list[int] = [self.entry_point]

        # Phase 1: greedy descent through the layers the new node does not join.
        for lc in range(L, level, -1):
            W = self._search_layer(q, ep, ef=1, layer=lc)
            self.build_distance_computations += 1
            if W:
                ep = [max(W)[1]]  # the max of (-dist, id) is the nearest element

        # Phase 2: connect on every layer the new node does join.
        for lc in range(min(L, level), -1, -1):
            W = self._search_layer(q, ep, ef=self.ef_construction, layer=lc)
            candidates = [(-nd, n) for nd, n in W if n != node_id]
            self.build_distance_computations += len(candidates)
            if not candidates:
                continue

            neighbors = self._select_neighbors(q, candidates, self.M, lc)
            graph = self.layers[lc]
            graph[node_id] = list(neighbors)

            M_max = self.M_max0 if lc == 0 else self.M_max
            for e in neighbors:
                conn = graph.setdefault(e, [])
                if node_id not in conn:
                    conn.append(node_id)
                if len(conn) > M_max:
                    e_vec = self.store.get(e)
                    e_dists = self.store.distances_to(e_vec, conn)
                    self.build_distance_computations += len(conn)
                    e_candidates = [(float(d), int(n)) for d, n in zip(e_dists, conn)]
                    graph[e] = self._select_neighbors(e_vec, e_candidates, M_max, lc)

            ep = [n for _, n in candidates]

        if level > L:
            self.entry_point = node_id
            self.max_layer = level
        return node_id

    # ------------------------------------------------------------------
    # Algorithm 5: K-NN-SEARCH
    # ------------------------------------------------------------------

    def knn_search(
        self,
        q: np.ndarray,
        k: int = 10,
        ef: int | None = None,
        stats: QueryStats | None = None,
        trace: SearchTrace | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Algorithm 5. Top-``k`` neighbors of ``q``, as ``(ids, distances)``.

        Sink through the sparse upper layers greedily (``ef = 1``, one closest
        node handed down each time), then run one wide beam search on layer 0.
        All of the recall is bought in that last step; the descent only decides
        *where* the beam starts.
        """
        if self.entry_point is None or len(self.store) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        ef = self.ef_construction if ef is None else int(ef)
        ef = max(ef, k)
        query = self.store.prepare_query(q)

        if stats is not None:
            stats.entry_point = self.entry_point
            stats.top_layer = self.max_layer
        if trace is not None:
            trace.entry_point = self.entry_point
            trace.top_layer = self.max_layer
            trace.ef = ef
            trace.k = k

        ep: list[int] = [self.entry_point]
        for lc in range(self.max_layer, 0, -1):
            W = self._search_layer(query, ep, ef=1, layer=lc, stats=stats, trace=trace)
            if W:
                ep = [max(W)[1]]

        W = self._search_layer(query, ep, ef=ef, layer=0, stats=stats, trace=trace)
        best = sorted((-nd, n) for nd, n in W)[:k]
        ids = np.array([n for _, n in best], dtype=np.int64)
        dists = np.array([d for d, _ in best], dtype=np.float32)
        return ids, dists

    def search(
        self,
        q: np.ndarray,
        k: int = 10,
        ef: int | None = None,
        return_stats: bool = False,
    ):
        """Public search. As ``knn_search``, with optional instrumentation.

        With ``return_stats=True`` the third return value is a ``QueryStats``
        recording the distance computations, nodes visited and hops per layer
        that this one query cost.
        """
        if not return_stats:
            return self.knn_search(q, k=k, ef=ef)
        stats = QueryStats()
        ids, dists = self.knn_search(q, k=k, ef=ef, stats=stats)
        return ids, dists, stats

    def search_traced(self, q: np.ndarray, k: int = 10, ef: int | None = None):
        """Search while recording the full per-layer path, for ``loam trace``."""
        trace = SearchTrace()
        stats = QueryStats()
        ids, dists = self.knn_search(q, k=k, ef=ef, stats=stats, trace=trace)
        trace.stats = stats
        return ids, dists, trace
