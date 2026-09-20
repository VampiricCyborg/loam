"""Printing the path a single query took through the graph.

This is the module that turns HNSW from an assertion into something you can
watch. A trace shows the two phases of Algorithm 5 as distinct shapes:

* **Upper layers** run with ``ef = 1``. Each layer does a handful of greedy
  hops and hands one node down. The distance to the query should drop at every
  hop and across every layer boundary; if it does not, the descent entered a
  local minimum, which is the failure mode ``ef`` exists to fix.
* **Layer 0** runs with the full beam. It expands far more nodes, and the
  interesting number is how many it expands *after* it has already found its
  eventual best result. That tail is the price of confirming the answer, and
  it is most of what a larger ``ef`` buys.

Comparing a trace at ``ef=8`` against one at ``ef=128`` on the same query is
the fastest way to build intuition for what the parameter does.
"""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.table import Table

from .hnsw import HNSWIndex
from .stats import SearchTrace


def format_trace(
    trace: SearchTrace,
    ids: np.ndarray,
    dists: np.ndarray,
    exact_ids: np.ndarray | None = None,
    max_expansions: int = 12,
    console: Console | None = None,
) -> None:
    """Print a search trace, layer by layer."""
    console = console or Console()

    console.print()
    console.print(
        f"[bold]Query trace[/bold]  k={trace.k}  ef={trace.ef}  "
        f"entry point=node {trace.entry_point} on layer {trace.top_layer}"
    )

    for layer_trace in trace.layers:
        layer = layer_trace.layer
        greedy = layer_trace.ef == 1
        phase = "greedy descent" if greedy else "beam search"
        console.print(
            f"\n[bold cyan]layer {layer}[/bold cyan]  "
            f"({phase}, ef={layer_trace.ef}, "
            f"{len(layer_trace.expansions)} node(s) expanded)"
        )

        entry_desc = ", ".join(f"node {n} @ {d:.4f}" for n, d in layer_trace.entry_points[:3])
        if len(layer_trace.entry_points) > 3:
            entry_desc += f", +{len(layer_trace.entry_points) - 3} more"
        console.print(f"  enter at: {entry_desc}")

        table = Table(box=None, pad_edge=False, show_edge=False)
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("expanded", justify="right", width=9)
        table.add_column("distance", justify="right", width=10)
        table.add_column("new neighbors", justify="right", width=14)
        table.add_column("", width=10)

        best_so_far = float("inf")
        shown = layer_trace.expansions[:max_expansions]
        for i, (node, dist, new_neighbors) in enumerate(shown):
            improved = dist < best_so_far
            best_so_far = min(best_so_far, dist)
            table.add_row(
                str(i + 1),
                str(node),
                f"{dist:.5f}",
                str(new_neighbors),
                "[green]closer[/green]" if improved else "[dim]--[/dim]",
            )
        console.print(table)

        hidden = len(layer_trace.expansions) - len(shown)
        if hidden > 0:
            console.print(f"  [dim]... {hidden} further expansions omitted[/dim]")

        best = layer_trace.best
        if best is not None:
            if greedy:
                console.print(f"  [yellow]-> hands node {best[1]} down at distance {best[0]:.5f}")
            else:
                console.print(f"  [yellow]-> best in beam: node {best[1]} at {best[0]:.5f}")

    console.print("\n[bold]Result[/bold]")
    result = Table(box=None, pad_edge=False, show_edge=False)
    result.add_column("rank", justify="right", style="dim", width=5)
    result.add_column("node", justify="right", width=8)
    result.add_column("distance", justify="right", width=10)
    if exact_ids is not None:
        result.add_column("in exact top-k?", width=16)

    exact_set = set(exact_ids.tolist()) if exact_ids is not None else set()
    for rank, (node, dist) in enumerate(zip(ids.tolist(), dists.tolist()), start=1):
        row = [str(rank), str(node), f"{dist:.5f}"]
        if exact_ids is not None:
            row.append("[green]yes[/green]" if node in exact_set else "[red]NO[/red]")
        result.add_row(*row)
    console.print(result)

    if exact_ids is not None and len(ids):
        hits = len(set(ids.tolist()) & exact_set)
        console.print(f"\nrecall@{len(ids)} for this query: [bold]{hits / len(ids):.2f}[/bold]")

    if trace.stats is not None:
        s = trace.stats
        hops = "  ".join(f"L{layer}:{count}" for layer, count in s.hops_per_layer)
        console.print(
            f"work: [bold]{s.distance_computations}[/bold] distance computations, "
            f"[bold]{s.visited}[/bold] nodes visited, hops per layer: {hops}"
        )


def trace_query(
    index: HNSWIndex,
    query: np.ndarray,
    k: int = 10,
    ef: int | None = None,
    exact_ids: np.ndarray | None = None,
    console: Console | None = None,
) -> SearchTrace:
    """Run one query with tracing on, print the path, and return the trace."""
    ids, dists, trace = index.search_traced(query, k=k, ef=ef)
    format_trace(trace, ids, dists, exact_ids=exact_ids, console=console)
    return trace
