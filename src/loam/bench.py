"""The benchmark harness: recall against an oracle, plus the work it cost.

The measurement rules this module enforces, because they are the difference
between a benchmark and a marketing number:

1. **Ground truth is computed, never assumed.** ``FlatIndex`` runs on exactly
   the vectors the HNSW index holds. See ``datasets`` for why the shipped
   ANN-Benchmarks ground truth cannot be used on a subsample.
2. **Recall and speed are reported together, always.** Either one alone is
   meaningless: ``ef=1`` is wonderfully fast and wrong, and the flat index has
   perfect recall and is unusably slow. The output of a sweep is a *curve*.
3. **There is a warm-up pass.** The first queries pay import, allocation and
   cache costs that have nothing to do with the algorithm.
4. **Queries are timed one at a time, single-threaded.** Loam does not batch
   queries, so QPS here is the reciprocal of mean per-query latency.
"""

from __future__ import annotations

import csv
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .datasets import Dataset
from .hnsw import HNSWIndex, Selection


@dataclass
class BenchRow:
    """One row of a sweep: a parameter setting and everything it produced."""

    ef: int
    recall: float
    qps: float
    mean_distance_computations: float
    mean_visited: float
    mean_hops: float
    mean_latency_ms: float
    k: int
    n: int
    dim: int
    M: int
    ef_construction: int
    selection: str
    hierarchical: bool
    metric: str
    dataset: str
    build_seconds: float
    index: str = "loam-hnsw"


@dataclass
class BenchResult:
    """A completed sweep, plus the build it was measured against."""

    dataset: str
    n: int
    dim: int
    k: int
    metric: str
    M: int
    ef_construction: int
    selection: str
    hierarchical: bool
    build_seconds: float
    num_queries: int
    layer_sizes: list[int] = field(default_factory=list)
    rows: list[BenchRow] = field(default_factory=list)

    def write_csv(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(asdict(self.rows[0]).keys()) if self.rows else []
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(asdict(row))
        return out


def machine_description() -> str:
    """A one-line machine label, so a benchmark number can be placed."""
    cpu = platform.processor() or platform.machine()
    return f"{platform.system()} {platform.release()}, {cpu}, Python {platform.python_version()}"


def recall_at_k(approx: np.ndarray, exact: np.ndarray, k: int) -> float:
    """Mean ``|approx ∩ exact| / k`` over queries.

    Set intersection, not positional equality: returning the right ten
    neighbors in a different order is a correct answer, and ties in distance
    make position unstable anyway.
    """
    if approx.shape[0] == 0:
        return 0.0
    hits = sum(
        len(set(approx[i][:k].tolist()) & set(exact[i][:k].tolist()))
        for i in range(approx.shape[0])
    )
    return hits / (k * approx.shape[0])


def build_index(
    dataset: Dataset,
    M: int = 16,
    ef_construction: int = 200,
    selection: Selection = "heuristic",
    hierarchical: bool = True,
    seed: int = 42,
    progress: object | None = None,
) -> tuple[HNSWIndex, float]:
    """Build an HNSW index over ``dataset.train``, returning it and its time."""
    index = HNSWIndex(
        dim=dataset.dim,
        metric=dataset.metric,
        M=M,
        ef_construction=ef_construction,
        selection=selection,
        hierarchical=hierarchical,
        seed=seed,
    )
    start = time.perf_counter()
    index.add(dataset.train, progress=progress)  # type: ignore[arg-type]
    return index, time.perf_counter() - start


def evaluate(
    index: HNSWIndex,
    dataset: Dataset,
    ef: int,
    k: int,
    warmup: int = 10,
) -> tuple[float, float, float, float, float, float]:
    """Run every query at one ``ef``.

    Returns ``(recall, qps, mean_dist_comps, mean_visited, mean_hops,
    mean_latency_ms)``.

    Instrumentation is collected on the same pass that is timed. The counters
    are a handful of integer increments per expanded node, which is noise next
    to the numpy calls around them, and keeping one pass means the reported
    work and the reported latency describe the same queries.
    """
    if dataset.ground_truth is None:
        dataset.compute_ground_truth(k)

    queries = dataset.test
    for i in range(min(warmup, queries.shape[0])):
        index.knn_search(queries[i], k=k, ef=ef)

    from .stats import QueryStats

    approx = np.full((queries.shape[0], k), -1, dtype=np.int64)
    total = QueryStats()

    start = time.perf_counter()
    for i in range(queries.shape[0]):
        stats = QueryStats()
        ids, _ = index.knn_search(queries[i], k=k, ef=ef, stats=stats)
        approx[i, : ids.shape[0]] = ids
        total.merge(stats)
    elapsed = time.perf_counter() - start

    num = queries.shape[0]
    recall = recall_at_k(approx, dataset.ground_truth, k)
    return (
        recall,
        num / elapsed if elapsed > 0 else float("inf"),
        total.distance_computations / num,
        total.visited / num,
        total.total_hops / num,
        1000.0 * elapsed / num,
    )


def sweep(
    dataset: Dataset,
    ef_values: list[int],
    k: int = 10,
    M: int = 16,
    ef_construction: int = 200,
    selection: Selection = "heuristic",
    hierarchical: bool = True,
    seed: int = 42,
    index: HNSWIndex | None = None,
    build_seconds: float | None = None,
    progress: object | None = None,
) -> BenchResult:
    """Build once, then measure recall and speed at each ``ef``."""
    if index is None:
        index, build_seconds = build_index(
            dataset,
            M=M,
            ef_construction=ef_construction,
            selection=selection,
            hierarchical=hierarchical,
            seed=seed,
            progress=progress,
        )
    assert build_seconds is not None

    if dataset.ground_truth is None:
        dataset.compute_ground_truth(k)

    result = BenchResult(
        dataset=dataset.name,
        n=dataset.n,
        dim=dataset.dim,
        k=k,
        metric=dataset.metric,
        M=M,
        ef_construction=ef_construction,
        selection=selection,
        hierarchical=hierarchical,
        build_seconds=build_seconds,
        num_queries=dataset.num_queries,
        layer_sizes=index.layer_sizes(),
    )

    for ef in ef_values:
        recall, qps, dc, visited, hops, latency = evaluate(index, dataset, ef=ef, k=k)
        result.rows.append(
            BenchRow(
                ef=ef,
                recall=recall,
                qps=qps,
                mean_distance_computations=dc,
                mean_visited=visited,
                mean_hops=hops,
                mean_latency_ms=latency,
                k=k,
                n=dataset.n,
                dim=dataset.dim,
                M=M,
                ef_construction=ef_construction,
                selection=selection,
                hierarchical=hierarchical,
                metric=dataset.metric,
                dataset=dataset.name,
                build_seconds=build_seconds,
            )
        )
    return result


def measure_flat(dataset: Dataset, k: int = 10, warmup: int = 5) -> BenchRow:
    """Time the exact oracle on the same queries, as the curve's endpoint.

    Recall is 1.0 by definition and distance computations are exactly ``n``.
    The row exists so the trade-off has a visible right-hand end: this is what
    perfect recall costs.
    """
    from .flat import FlatIndex

    flat = FlatIndex(dim=dataset.dim, metric=dataset.metric)
    flat.add(dataset.train)

    for i in range(min(warmup, dataset.num_queries)):
        flat.search(dataset.test[i], k=k)

    start = time.perf_counter()
    for i in range(dataset.num_queries):
        flat.search(dataset.test[i], k=k)
    elapsed = time.perf_counter() - start

    return BenchRow(
        ef=0,
        recall=1.0,
        qps=dataset.num_queries / elapsed,
        mean_distance_computations=float(dataset.n),
        mean_visited=float(dataset.n),
        mean_hops=0.0,
        mean_latency_ms=1000.0 * elapsed / dataset.num_queries,
        k=k,
        n=dataset.n,
        dim=dataset.dim,
        M=0,
        ef_construction=0,
        selection="n/a",
        hierarchical=False,
        metric=dataset.metric,
        dataset=dataset.name,
        build_seconds=0.0,
        index="flat-exact",
    )


def measure_hnswlib(
    dataset: Dataset,
    ef_values: list[int],
    k: int = 10,
    M: int = 16,
    ef_construction: int = 200,
    seed: int = 42,
) -> list[BenchRow]:
    """Optional: the same data and parameters through hnswlib.

    A sanity check, not a race. If Loam's recall curve sits far below hnswlib's
    at identical ``M``, ``ef_construction`` and ``ef``, something in Loam's
    graph construction is wrong. The speed gap is expected and uninteresting;
    the recall gap is the signal.
    """
    try:
        import hnswlib  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "hnswlib is not installed. It is published as a source distribution, "
            "so it needs a C++ compiler: `uv sync --extra reference`."
        ) from exc

    if dataset.ground_truth is None:
        dataset.compute_ground_truth(k)

    space = "cosine" if dataset.metric == "cosine" else "l2"
    index = hnswlib.Index(space=space, dim=dataset.dim)
    index.init_index(
        max_elements=dataset.n, ef_construction=ef_construction, M=M, random_seed=seed
    )
    index.set_num_threads(1)

    start = time.perf_counter()
    index.add_items(dataset.train, np.arange(dataset.n))
    build_seconds = time.perf_counter() - start

    rows: list[BenchRow] = []
    for ef in ef_values:
        index.set_ef(max(ef, k))
        for i in range(min(10, dataset.num_queries)):
            index.knn_query(dataset.test[i], k=k)

        approx = np.empty((dataset.num_queries, k), dtype=np.int64)
        start = time.perf_counter()
        for i in range(dataset.num_queries):
            labels, _ = index.knn_query(dataset.test[i], k=k)
            approx[i] = labels[0]
        elapsed = time.perf_counter() - start

        rows.append(
            BenchRow(
                ef=ef,
                recall=recall_at_k(approx, dataset.ground_truth, k),
                qps=dataset.num_queries / elapsed,
                mean_distance_computations=float("nan"),  # hnswlib exposes no counter
                mean_visited=float("nan"),
                mean_hops=float("nan"),
                mean_latency_ms=1000.0 * elapsed / dataset.num_queries,
                k=k,
                n=dataset.n,
                dim=dataset.dim,
                M=M,
                ef_construction=ef_construction,
                selection="heuristic",
                hierarchical=True,
                metric=dataset.metric,
                dataset=dataset.name,
                build_seconds=build_seconds,
                index="hnswlib",
            )
        )
    return rows


def plot(result: BenchResult, path: Path | str, extra: list[BenchRow] | None = None) -> Path:
    """Write the recall-vs-QPS curve.

    Up and to the right is better. QPS is on a log axis because it spans orders
    of magnitude between a narrow beam and the exact oracle, and the
    interesting part of the recall axis is the top: the difference between
    0.95 and 0.99 recall matters far more than the difference between 0.2 and
    0.24.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=140)

    recalls = [row.recall for row in result.rows]
    qps = [row.qps for row in result.rows]
    ax.plot(recalls, qps, marker="o", linewidth=2, color="#2E6F40", label="Loam HNSW")
    for row in result.rows:
        ax.annotate(
            f"ef={row.ef}",
            (row.recall, row.qps),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="#2E6F40",
        )

    for row in extra or []:
        style = {"flat-exact": ("*", "#B4531F", 14), "hnswlib": ("s", "#1F4E79", 7)}
        marker, color, size = style.get(row.index, ("^", "#666666", 7))
        ax.plot(
            row.recall,
            row.qps,
            marker=marker,
            markersize=size,
            linestyle="none",
            color=color,
            label=row.index if row.index not in ax.get_legend_handles_labels()[1] else None,
        )

    ax.set_yscale("log")
    ax.set_xlabel(f"recall@{result.k}")
    ax.set_ylabel("queries per second (1 thread)")
    ax.set_title(
        f"{result.dataset}: n={result.n:,}, dim={result.dim}, "
        f"M={result.M}, ef_construction={result.ef_construction}"
    )
    ax.grid(True, which="both", alpha=0.25, linestyle=":")
    ax.legend(loc="lower left", frameon=False)
    fig.text(0.01, 0.01, machine_description(), fontsize=6, color="#888888")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
