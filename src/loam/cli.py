"""The ``loam`` command line: fetch, bench, ablate, trace.

Every command prints the numbers it measured and, where a number is destined
for the README, the command that produced it. That is the repo's one rule:
a benchmark figure without its command does not belong in the documentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from . import __version__, datasets
from .bench import (
    BenchResult,
    build_index,
    machine_description,
    measure_flat,
    measure_hnswlib,
    plot,
    sweep,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Loam: a readable, instrumented HNSW vector index in pure Python.",
)
console = Console()

DEFAULT_EF = "16,32,64,128,256"


def _parse_ef(value: str) -> list[int]:
    try:
        efs = [int(part) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise typer.BadParameter(f"--ef must be a comma-separated list of integers: {value}") from exc
    if not efs:
        raise typer.BadParameter("--ef needs at least one value")
    return sorted(set(efs))


def _load(
    dataset: str,
    n: int | None,
    queries: int | None,
    k: int,
    dim: int,
    clusters: int,
    spread: float,
    seed: int,
    data_dir: Path,
) -> datasets.Dataset:
    """Load, subsample, and recompute ground truth, with progress printed."""
    with console.status(f"loading {dataset} ..."):
        ds = datasets.load(
            dataset,
            n=n,
            queries=queries,
            dim=dim,
            clusters=clusters,
            spread=spread,
            data_dir=data_dir,
            seed=seed,
        )
    console.print(f"[dim]{ds.describe()}[/dim]")

    with console.status(f"computing exact ground truth (k={k}) with FlatIndex ..."):
        ds.compute_ground_truth(k)
    console.print(
        f"[dim]ground truth recomputed on the subsample "
        f"({ds.num_queries:,} queries x k={k})[/dim]"
    )
    return ds


def _build_with_progress(ds: datasets.Dataset, label: str, **kwargs):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as bar:
        task = bar.add_task(label, total=ds.n)
        index, seconds = build_index(ds, progress=lambda done: bar.advance(task, done), **kwargs)
    console.print(
        f"[dim]{label}: {seconds:.1f}s  "
        f"({1000 * seconds / max(1, ds.n):.2f} ms/insert)  "
        f"layer sizes {index.layer_sizes()}[/dim]"
    )
    return index, seconds


def _results_table(result: BenchResult, extra=None) -> Table:
    table = Table(title=None, header_style="bold")
    table.add_column("index")
    table.add_column("ef", justify="right")
    table.add_column(f"recall@{result.k}", justify="right")
    table.add_column("QPS", justify="right")
    table.add_column("mean dist comps", justify="right")
    table.add_column("mean hops", justify="right")
    table.add_column("latency (ms)", justify="right")

    for row in result.rows:
        table.add_row(
            row.index,
            str(row.ef),
            f"{row.recall:.4f}",
            f"{row.qps:.1f}",
            f"{row.mean_distance_computations:.1f}",
            f"{row.mean_hops:.1f}",
            f"{row.mean_latency_ms:.3f}",
        )
    for row in extra or []:
        comps = (
            "n/a"
            if row.mean_distance_computations != row.mean_distance_computations
            else f"{row.mean_distance_computations:.1f}"
        )
        table.add_row(
            f"[dim]{row.index}[/dim]",
            "-" if row.ef == 0 else str(row.ef),
            f"{row.recall:.4f}",
            f"{row.qps:.1f}",
            comps,
            "n/a" if row.mean_hops != row.mean_hops else f"{row.mean_hops:.1f}",
            f"{row.mean_latency_ms:.3f}",
        )
    return table


@app.command()
def version() -> None:
    """Print the Loam version and the machine description."""
    console.print(f"loam {__version__}")
    console.print(f"[dim]{machine_description()}[/dim]")


@app.command()
def fetch(
    name: Annotated[str, typer.Argument(help="ANN-Benchmarks dataset name")] = "glove-25-angular",
    data_dir: Annotated[Path, typer.Option(help="Where to store HDF5 files")] = Path("data"),
    force: Annotated[bool, typer.Option(help="Re-download even if present")] = False,
) -> None:
    """Download an ANN-Benchmarks HDF5 dataset."""
    console.print(f"fetching [bold]{name}[/bold] into {data_dir}/ ...")
    path = datasets.fetch(name, data_dir=data_dir, force=force)
    size_mb = path.stat().st_size / 1e6
    console.print(f"[green]ok[/green] {path} ({size_mb:.1f} MB)")
    console.print(
        "[dim]note: the shipped `neighbors` array is ground truth for the FULL train set. "
        "Loam subsamples, so it recomputes ground truth with FlatIndex instead.[/dim]"
    )


@app.command()
def bench(
    dataset: Annotated[str, typer.Argument(help="Dataset name, or 'clustered' / 'random'")] = "glove-25-angular",
    n: Annotated[Optional[int], typer.Option(help="Train vectors to index")] = None,
    queries: Annotated[Optional[int], typer.Option(help="Queries to run")] = None,
    k: Annotated[int, typer.Option(help="Neighbors per query")] = 10,
    m: Annotated[int, typer.Option("--M", "-M", help="Links per node per layer")] = 16,
    ef_construction: Annotated[int, typer.Option(help="Build-time beam width")] = 200,
    ef: Annotated[str, typer.Option(help="Comma-separated search beam widths")] = DEFAULT_EF,
    selection: Annotated[str, typer.Option(help="simple | heuristic")] = "heuristic",
    no_hierarchy: Annotated[bool, typer.Option("--no-hierarchy", help="Collapse to a flat NSW graph")] = False,
    dim: Annotated[int, typer.Option(help="Dimensions (synthetic datasets only)")] = 32,
    clusters: Annotated[int, typer.Option(help="Clusters (clustered dataset only)")] = 100,
    spread: Annotated[float, typer.Option(help="Cluster spread (clustered only)")] = 0.08,
    seed: Annotated[int, typer.Option(help="Seed for sampling and graph construction")] = 42,
    flat: Annotated[bool, typer.Option(help="Also time the exact oracle")] = True,
    reference: Annotated[Optional[str], typer.Option(help="Also run 'hnswlib'")] = None,
    out: Annotated[Optional[Path], typer.Option(help="CSV output path")] = None,
    make_plot: Annotated[bool, typer.Option("--plot", help="Write a recall-vs-QPS plot")] = False,
    data_dir: Annotated[Path, typer.Option(help="Where HDF5 files live")] = Path("data"),
) -> None:
    """Sweep ``ef`` and report recall against the exact oracle."""
    ef_values = _parse_ef(ef)
    ds = _load(dataset, n, queries, k, dim, clusters, spread, seed, data_dir)

    index, build_seconds = _build_with_progress(
        ds,
        "building HNSW",
        M=m,
        ef_construction=ef_construction,
        selection=selection,  # type: ignore[arg-type]
        hierarchical=not no_hierarchy,
        seed=seed,
    )
    result = sweep(
        ds,
        ef_values,
        k=k,
        M=m,
        ef_construction=ef_construction,
        selection=selection,  # type: ignore[arg-type]
        hierarchical=not no_hierarchy,
        seed=seed,
        index=index,
        build_seconds=build_seconds,
    )

    extra = []
    if flat:
        with console.status("timing the exact oracle ..."):
            extra.append(measure_flat(ds, k=k))
    if reference == "hnswlib":
        with console.status("running hnswlib on the same data ..."):
            try:
                extra.extend(
                    measure_hnswlib(ds, ef_values, k=k, M=m, ef_construction=ef_construction, seed=seed)
                )
            except RuntimeError as exc:
                console.print(f"[yellow]skipping reference:[/yellow] {exc}")
    elif reference:
        raise typer.BadParameter(f"unknown reference {reference!r}; only 'hnswlib' is supported")

    console.print()
    console.print(_results_table(result, extra))
    console.print(f"[dim]build: {build_seconds:.1f}s  |  {machine_description()}[/dim]")

    if out:
        result.rows.extend(extra)
        path = result.write_csv(out)
        console.print(f"[green]wrote[/green] {path}")
        if make_plot:
            plot_path = plot(result, Path(out).with_suffix(".png"), extra=extra)
            console.print(f"[green]wrote[/green] {plot_path}")
    elif make_plot:
        console.print("[yellow]--plot needs --out to know where to write[/yellow]")


ablate = typer.Typer(no_args_is_help=True, help="One-flag ablations of a single design decision.")
app.add_typer(ablate, name="ablate")


def _ablation_table(title: str, label_a: str, rows_a, label_b: str, rows_b, k: int) -> Table:
    table = Table(title=title, header_style="bold")
    table.add_column("ef", justify="right")
    table.add_column(f"{label_a} recall@{k}", justify="right")
    table.add_column(f"{label_b} recall@{k}", justify="right")
    table.add_column("difference", justify="right")
    table.add_column(f"{label_a} QPS", justify="right")
    table.add_column(f"{label_b} QPS", justify="right")

    for a, b in zip(rows_a, rows_b):
        delta = b.recall - a.recall
        colour = "green" if delta > 0.005 else ("red" if delta < -0.005 else "dim")
        table.add_row(
            str(a.ef),
            f"{a.recall:.4f}",
            f"{b.recall:.4f}",
            f"[{colour}]{delta:+.4f}[/{colour}]",
            f"{a.qps:.1f}",
            f"{b.qps:.1f}",
        )
    return table


@ablate.command("selection")
def ablate_selection(
    dataset: Annotated[str, typer.Option(help="Dataset; 'clustered' is the interesting one")] = "clustered",
    n: Annotated[int, typer.Option()] = 20000,
    queries: Annotated[Optional[int], typer.Option()] = None,
    k: Annotated[int, typer.Option()] = 10,
    m: Annotated[int, typer.Option("--M", "-M")] = 16,
    ef_construction: Annotated[int, typer.Option()] = 200,
    ef: Annotated[str, typer.Option()] = DEFAULT_EF,
    dim: Annotated[int, typer.Option()] = 10,
    clusters: Annotated[int, typer.Option()] = 100,
    spread: Annotated[float, typer.Option()] = 0.08,
    seed: Annotated[int, typer.Option()] = 42,
    out: Annotated[Optional[Path], typer.Option(help="CSV output path")] = None,
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
) -> None:
    """Algorithm 3 (simple) vs Algorithm 4 (heuristic) neighbor selection.

    The paper's Fig. 7 reports that the diversity heuristic matters most on
    clustered data, where nearest-M selection links every node only inside its
    own cluster and leaves the graph with no edges to cross between them.
    """
    ef_values = _parse_ef(ef)
    ds = _load(dataset, n, queries, k, dim, clusters, spread, seed, data_dir)

    results = {}
    for variant in ("simple", "heuristic"):
        index, seconds = _build_with_progress(
            ds,
            f"building HNSW ({variant})",
            M=m,
            ef_construction=ef_construction,
            selection=variant,  # type: ignore[arg-type]
            seed=seed,
        )
        results[variant] = sweep(
            ds, ef_values, k=k, M=m, ef_construction=ef_construction,
            selection=variant, seed=seed, index=index, build_seconds=seconds,  # type: ignore[arg-type]
        )

    console.print()
    console.print(
        _ablation_table(
            f"Neighbor selection on {ds.name} (n={ds.n:,}, dim={ds.dim})",
            "simple", results["simple"].rows,
            "heuristic", results["heuristic"].rows,
            k,
        )
    )
    console.print(f"[dim]{machine_description()}[/dim]")

    if out:
        combined = results["simple"]
        combined.rows = list(results["simple"].rows) + list(results["heuristic"].rows)
        console.print(f"[green]wrote[/green] {combined.write_csv(out)}")


@ablate.command("hierarchy")
def ablate_hierarchy(
    dataset: Annotated[str, typer.Option()] = "glove-25-angular",
    n: Annotated[int, typer.Option()] = 20000,
    queries: Annotated[Optional[int], typer.Option()] = None,
    k: Annotated[int, typer.Option()] = 10,
    m: Annotated[int, typer.Option("--M", "-M")] = 16,
    ef_construction: Annotated[int, typer.Option()] = 200,
    ef: Annotated[str, typer.Option()] = DEFAULT_EF,
    dim: Annotated[int, typer.Option()] = 32,
    clusters: Annotated[int, typer.Option()] = 100,
    spread: Annotated[float, typer.Option()] = 0.08,
    seed: Annotated[int, typer.Option()] = 42,
    out: Annotated[Optional[Path], typer.Option(help="CSV output path")] = None,
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
) -> None:
    """HNSW vs a flat NSW graph: is the "H" doing any work?

    *Down with the Hierarchy* (arXiv:2412.01940) reports that on
    high-dimensional data a flat graph matches HNSW's recall and latency,
    because a "highway" of hub nodes provides the long-range connectivity the
    upper layers were supposed to provide.
    """
    ef_values = _parse_ef(ef)
    ds = _load(dataset, n, queries, k, dim, clusters, spread, seed, data_dir)

    results = {}
    for hierarchical in (True, False):
        label = "HNSW" if hierarchical else "flat NSW"
        index, seconds = _build_with_progress(
            ds, f"building {label}", M=m, ef_construction=ef_construction,
            hierarchical=hierarchical, seed=seed,
        )
        results[label] = sweep(
            ds, ef_values, k=k, M=m, ef_construction=ef_construction,
            hierarchical=hierarchical, seed=seed, index=index, build_seconds=seconds,
        )

    console.print()
    console.print(
        _ablation_table(
            f"Hierarchy on {ds.name} (n={ds.n:,}, dim={ds.dim})",
            "HNSW", results["HNSW"].rows,
            "flat NSW", results["flat NSW"].rows,
            k,
        )
    )
    console.print(f"[dim]{machine_description()}[/dim]")

    if out:
        combined = results["HNSW"]
        combined.rows = list(results["HNSW"].rows) + list(results["flat NSW"].rows)
        console.print(f"[green]wrote[/green] {combined.write_csv(out)}")


@app.command()
def trace(
    dataset: Annotated[str, typer.Argument()] = "glove-25-angular",
    n: Annotated[int, typer.Option()] = 5000,
    query_index: Annotated[int, typer.Option(help="Which test query to trace")] = 0,
    k: Annotated[int, typer.Option()] = 10,
    ef: Annotated[int, typer.Option()] = 32,
    m: Annotated[int, typer.Option("--M", "-M")] = 16,
    ef_construction: Annotated[int, typer.Option()] = 200,
    dim: Annotated[int, typer.Option()] = 32,
    clusters: Annotated[int, typer.Option()] = 100,
    spread: Annotated[float, typer.Option()] = 0.08,
    seed: Annotated[int, typer.Option()] = 42,
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
) -> None:
    """Print the path a single query takes through each layer."""
    from .trace import trace_query

    ds = _load(dataset, n, None, k, dim, clusters, spread, seed, data_dir)
    if not 0 <= query_index < ds.num_queries:
        raise typer.BadParameter(
            f"--query-index must be in [0, {ds.num_queries - 1}], got {query_index}"
        )

    index, _ = _build_with_progress(
        ds, "building HNSW", M=m, ef_construction=ef_construction, seed=seed
    )
    assert ds.ground_truth is not None
    trace_query(
        index,
        ds.test[query_index],
        k=k,
        ef=ef,
        exact_ids=ds.ground_truth[query_index],
        console=console,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
