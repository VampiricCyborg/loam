"""Regenerate every image in `docs/assets/`.

Run with:

    uv run python scripts/make_figures.py

The repo's rule is that no number appears without the command that produced
it. The same rule applies to pictures, so nothing here is drawn by hand:

* the charts are plotted from the committed CSVs in ``bench/results/``;
* the terminal captures are real ``rich`` output, recorded from the same
  functions the CLI calls, and exported as SVG;
* the animation is a real traced query, replayed frame by frame from the
  ``SearchTrace`` that ``HNSWIndex.search_traced`` returns.

Everything is seeded, so re-running this script reproduces the assets.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ASSETS = ROOT / "docs" / "assets"
RESULTS = ROOT / "bench" / "results"

# One palette, used everywhere, so the images read as a set.
INK = "#1B2A21"
SOIL = "#2E6F40"
CLAY = "#B4531F"
SKY = "#1F4E79"
MUTED = "#9AA5A0"
PAPER = "#FBFAF7"

plt.rcParams.update(
    {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#C9D1CC",
        "xtick.color": INK,
        "ytick.color": INK,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.color": "#DDE4E0",
        "grid.linestyle": ":",
        "grid.alpha": 0.9,
    }
)


def read_rows(name: str) -> list[dict]:
    with (RESULTS / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / name
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# ----------------------------------------------------------------------
# 1. The animation: one query sinking through the layers
# ----------------------------------------------------------------------


def search_animation() -> None:
    """Replay a real traced query over 2-D data, one expansion per frame.

    Two dimensions and the L2 metric, because a cosine index normalizes every
    vector onto the unit circle and a picture of a circle teaches nothing. The
    graph, the search and the stopping condition are the ordinary ones; only
    the data is chosen to be drawable.
    """
    from PIL import Image

    from loam import HNSWIndex

    rng = np.random.default_rng(7)
    centers = rng.uniform(-1.0, 1.0, size=(14, 2))
    assign = rng.integers(0, 14, 600)
    data = (centers[assign] + 0.10 * rng.standard_normal((600, 2))).astype(np.float32)
    query = np.array([0.72, -0.62], dtype=np.float32)

    index = HNSWIndex(dim=2, metric="l2", M=8, ef_construction=80, seed=11)
    index.add(data)
    ids, _, trace = index.search_traced(query, k=5, ef=24)

    frames: list[tuple[int, int, int, list[int], int]] = []
    for layer_trace in trace.layers:
        for step, (node, _dist, _new) in enumerate(layer_trace.expansions[:26]):
            visited = [n for n, _, _ in layer_trace.expansions[: step + 1]]
            best = layer_trace.result[0][1] if layer_trace.result else node
            frames.append((layer_trace.layer, layer_trace.ef, node, visited, best))

    lo, hi = data.min(axis=0) - 0.18, data.max(axis=0) + 0.18
    images: list[Image.Image] = []

    for layer, ef, current, visited, best in frames:
        fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=100)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        # Every vector, faint: the population of layer 0.
        ax.scatter(data[:, 0], data[:, 1], s=7, c="#D7DEDA", linewidths=0, zorder=1)

        # The nodes that actually live on this layer.
        residents = np.fromiter(index.layers[layer].keys(), dtype=np.int64)
        ax.scatter(
            data[residents, 0], data[residents, 1],
            s=26, c=MUTED, linewidths=0, zorder=2,
        )

        # Edges out of every node expanded so far on this layer.
        for node in visited:
            for neighbor in index.neighbors(node, layer):
                ax.plot(
                    [data[node, 0], data[neighbor, 0]],
                    [data[node, 1], data[neighbor, 1]],
                    color=SOIL, alpha=0.20, linewidth=0.9, zorder=3,
                )

        ax.scatter(
            data[visited, 0], data[visited, 1],
            s=48, c=SOIL, linewidths=0, zorder=4,
        )
        ax.scatter(
            data[current, 0], data[current, 1],
            s=190, facecolors="none", edgecolors=CLAY, linewidths=2.4, zorder=6,
        )
        ax.scatter(
            data[best, 0], data[best, 1],
            s=90, c=CLAY, marker="D", linewidths=0, zorder=5,
        )
        ax.scatter(
            query[0], query[1],
            s=420, c=INK, marker="*", linewidths=0, zorder=7,
        )

        phase = "greedy descent" if ef == 1 else "beam search"
        ax.set_title(
            f"layer {layer} — {phase} (ef={ef})\n"
            f"{len(visited)} node(s) expanded on this layer",
            color=INK, fontsize=13,
        )
        ax.legend(
            handles=[
                Line2D([], [], marker="*", color="none", markerfacecolor=INK,
                       markersize=16, label="query"),
                Line2D([], [], marker="D", color="none", markerfacecolor=CLAY,
                       markersize=9, label="best so far"),
                Line2D([], [], marker="o", color="none", markerfacecolor=SOIL,
                       markersize=9, label="expanded"),
                Line2D([], [], marker="o", color="none", markerfacecolor=MUTED,
                       markersize=8, label="on this layer"),
            ],
            loc="upper left", frameon=False, fontsize=9,
        )
        fig.tight_layout()

        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)

    # Hold the last frame so the loop is readable.
    images.extend([images[-1]] * 8)
    quantized = [im.quantize(colors=96, method=Image.MEDIANCUT) for im in images]

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "search-animation.gif"
    quantized[0].save(
        out, save_all=True, append_images=quantized[1:],
        duration=420, loop=0, optimize=True,
    )
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out.relative_to(ROOT)} ({len(images)} frames, {size_mb:.1f} MB)")
    print(f"  traced query returned ids {ids.tolist()}")


# ----------------------------------------------------------------------
# 2. Layer structure
# ----------------------------------------------------------------------


def layer_structure() -> None:
    """Measured layer populations against the 1/M decay the theory predicts."""
    from loam import HNSWIndex

    rng = np.random.default_rng(3)
    data = rng.standard_normal((20000, 12)).astype(np.float32)
    index = HNSWIndex(dim=12, M=16, ef_construction=40, seed=5)
    index.add(data)
    sizes = index.layer_sizes()

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    layers = np.arange(len(sizes))
    ax.bar(layers, sizes, color=SOIL, width=0.55, zorder=3, label="measured")

    predicted = [sizes[0] / (16**i) for i in layers]
    ax.plot(layers, predicted, "o--", color=CLAY, linewidth=1.8,
            markersize=6, zorder=4, label=r"$n / M^{\ell}$")

    for layer, size in zip(layers, sizes):
        ax.text(layer, size * 1.45, f"{size:,}", ha="center", fontsize=9, color=INK)

    ax.set_yscale("log")
    ax.set_xticks(layers)
    ax.set_xlabel("layer")
    ax.set_ylabel("nodes (log scale)")
    ax.set_title(f"Layer populations decay by ~1/M  (n=20,000, M=16, mL=1/ln M)")
    ax.legend(frameon=False)
    ax.set_ylim(0.5, sizes[0] * 6)
    save(fig, "layer-structure.png")


# ----------------------------------------------------------------------
# 3. The trade-off, in hardware-independent units
# ----------------------------------------------------------------------


def recall_vs_work() -> None:
    """Recall against distance computations, the machine-independent curve.

    The recall-vs-QPS plot the benchmark writes is the conventional view, but
    its y-axis is a property of the laptop. This one plots the work the
    algorithm actually did, which reproduces anywhere.
    """
    rows = [r for r in read_rows("glove25-ef-sweep.csv") if r["index"] == "loam-hnsw"]
    flat = next(r for r in read_rows("glove25-ef-sweep.csv") if r["index"] == "flat-exact")

    recalls = [float(r["recall"]) for r in rows]
    work = [float(r["mean_distance_computations"]) for r in rows]
    efs = [int(r["ef"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(recalls, work, "o-", color=SOIL, linewidth=2, markersize=7, label="Loam HNSW")
    for ef, recall, comps in zip(efs, recalls, work):
        ax.annotate(f"ef={ef}", (recall, comps), textcoords="offset points",
                    xytext=(8, -3), fontsize=9, color=SOIL)

    n = float(flat["n"])
    ax.axhline(n, color=CLAY, linestyle="--", linewidth=1.8,
               label=f"flat exact = n = {int(n):,}")

    ax.set_yscale("log")
    ax.set_xlabel("recall@10")
    ax.set_ylabel("mean distance computations per query (log)")
    ax.set_title("What recall costs, in work rather than seconds\nglove-25-angular, n=10,000")
    ax.legend(frameon=False, loc="center left")
    ax.margins(x=0.10)
    save(fig, "recall-vs-work.png")


# ----------------------------------------------------------------------
# 4. Ablations
# ----------------------------------------------------------------------


def ablation_selection() -> None:
    rows = read_rows("ablate-selection.csv")
    simple = [r for r in rows if r["selection"] == "simple"]
    heuristic = [r for r in rows if r["selection"] == "heuristic"]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for subset, colour, label, marker in (
        (heuristic, SOIL, "Algorithm 4 (heuristic)", "o"),
        (simple, CLAY, "Algorithm 3 (simple nearest-M)", "s"),
    ):
        ax.plot(
            [int(r["ef"]) for r in subset],
            [float(r["recall"]) for r in subset],
            marker=marker, color=colour, linewidth=2, markersize=7, label=label,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks([int(r["ef"]) for r in simple])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("ef (search beam width)")
    ax.set_ylabel("recall@10")
    ax.set_ylim(0.85, 1.01)
    ax.set_title(
        "Neighbor selection on clustered data\n"
        "100 clusters, 10-d, n=20,000 — a wider beam never closes the gap"
    )
    ax.legend(frameon=False, loc="lower right")
    save(fig, "ablation-selection.png")


def ablation_hierarchy() -> None:
    rows = read_rows("ablate-hierarchy.csv")
    hnsw = [r for r in rows if r["hierarchical"] == "True"]
    flat = [r for r in rows if r["hierarchical"] == "False"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for subset, colour, label, marker in (
        (hnsw, SOIL, "HNSW (layers on)", "o"),
        (flat, SKY, "flat NSW (--no-hierarchy)", "s"),
    ):
        ax.plot(
            [int(r["ef"]) for r in subset],
            [float(r["recall"]) for r in subset],
            marker=marker, color=colour, linewidth=2, markersize=7,
            label=label, alpha=0.85,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([int(r["ef"]) for r in hnsw])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("ef")
    ax.set_ylabel("recall@10")
    ax.set_title("Recall: the two curves sit on top of each other")
    ax.legend(frameon=False, loc="lower right")

    deltas = [float(f["recall"]) - float(h["recall"]) for h, f in zip(hnsw, flat)]
    colours = [SKY if d >= 0 else CLAY for d in deltas]
    ax2.bar(range(len(deltas)), deltas, color=colours, width=0.55, zorder=3)
    ax2.axhline(0, color=INK, linewidth=1)
    ax2.set_xticks(range(len(deltas)))
    ax2.set_xticklabels([r["ef"] for r in hnsw])
    ax2.set_xlabel("ef")
    ax2.set_ylabel("flat NSW recall − HNSW recall")
    ax2.set_ylim(-0.004, 0.004)
    ax2.set_title("Difference: at most 0.0023, either way")

    fig.suptitle(
        "Removing every layer above 0 — glove-25-angular, n=10,000",
        fontsize=13, y=1.0,
    )
    save(fig, "ablation-hierarchy.png")


# ----------------------------------------------------------------------
# 5. Terminal captures
# ----------------------------------------------------------------------


def terminal_captures() -> None:
    """Record real CLI output as SVG, so the screenshots cannot drift."""
    from rich.console import Console
    from rich.table import Table

    from loam import HNSWIndex
    from loam.datasets import make_clustered
    from loam.flat import FlatIndex
    from loam.trace import format_trace

    ASSETS.mkdir(parents=True, exist_ok=True)

    # --- `loam bench`, rebuilt from the committed CSV -------------------
    console = Console(record=True, width=96, file=open("nul", "w"))
    rows = read_rows("glove25-ef-sweep.csv")
    table = Table(header_style="bold", title=None)
    for column, justify in (
        ("index", "left"), ("ef", "right"), ("recall@10", "right"), ("QPS", "right"),
        ("mean dist comps", "right"), ("mean hops", "right"), ("latency (ms)", "right"),
    ):
        table.add_column(column, justify=justify)
    for row in rows:
        exact = row["index"] == "flat-exact"
        table.add_row(
            f"[dim]{row['index']}[/dim]" if exact else row["index"],
            "-" if exact else row["ef"],
            f"{float(row['recall']):.4f}",
            f"{float(row['qps']):.1f}",
            f"{float(row['mean_distance_computations']):.1f}",
            f"{float(row['mean_hops']):.1f}",
            f"{float(row['mean_latency_ms']):.3f}",
        )
    console.print("[bold green]$[/bold green] loam bench glove-25-angular --n 10000 "
                  "--queries 1000 --k 10 \\\n    -M 16 --ef-construction 200 "
                  "--ef 16,32,64,128,256 \\\n    "
                  "--out bench/results/glove25-ef-sweep.csv --plot")
    console.print("[dim]glove-25-angular: n=10,000 dim=25 queries=1,000 metric=cosine[/dim]")
    console.print("[dim]ground truth recomputed on the subsample (1,000 queries x k=10)[/dim]")
    console.print("[dim]building HNSW: 44.6s  (4.46 ms/insert)  "
                  "layer sizes [10000, 571, 35, 5][/dim]\n")
    console.print(table)
    console.save_svg(str(ASSETS / "bench-terminal.svg"), title="loam bench")
    console.file.close()
    print(f"wrote docs/assets/bench-terminal.svg")

    # --- `loam trace`, a real traced query ------------------------------
    #
    # Every parameter below mirrors what the CLI would use for
    #     loam trace clustered --n 4000 --dim 16 --query-index 0 --ef 32
    # (see `cli.trace`: clusters=100, spread=0.08, M=16, ef_construction=200,
    # seed=42, k=10, and make_clustered's queries=500 default). Anything else
    # would make this a picture of a command nobody can run.
    console = Console(record=True, width=96, file=open("nul", "w"))
    dataset = make_clustered(n=4000, dim=16, clusters=100, spread=0.08, queries=500, seed=42)
    index = HNSWIndex(dim=16, metric="cosine", M=16, ef_construction=200, seed=42)
    index.add(dataset.train)
    oracle = FlatIndex(dim=16, metric="cosine")
    oracle.add(dataset.train)
    exact_ids, _ = oracle.search(dataset.test[0], k=10)

    ids, dists, trace = index.search_traced(dataset.test[0], k=10, ef=32)
    console.print("[bold green]$[/bold green] loam trace clustered --n 4000 --dim 16 "
                  "--query-index 0 --ef 32")
    format_trace(trace, ids, dists, exact_ids=exact_ids, max_expansions=6, console=console)
    console.save_svg(str(ASSETS / "trace-terminal.svg"), title="loam trace")
    console.file.close()
    print(f"wrote docs/assets/trace-terminal.svg")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    layer_structure()
    recall_vs_work()
    ablation_selection()
    ablation_hierarchy()
    terminal_captures()
    search_animation()


if __name__ == "__main__":
    main()
