# Loam

**A readable, instrumented HNSW vector index in pure Python. It is built line-by-line from the paper, checked against brute force, and reports its recall honestly.**

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Core](https://img.shields.io/badge/core-numpy%20only-informational)
![Status](https://img.shields.io/badge/status-learning%20project-orange)

Loam is a from-scratch implementation of **Hierarchical Navigable Small World (HNSW)** graphs, the approximate nearest-neighbor (ANN) algorithm behind most modern vector search (hnswlib, FAISS's `IndexHNSWFlat`, pgvector's HNSW index, Qdrant, Weaviate, Elasticsearch). Every core function maps one-to-one to an algorithm in Malkov & Yashunin's paper. Every search result can be checked against an exact brute-force oracle, and every query can report the work it did.

The name comes from soil. Loam is layered and porous, and water finds fast paths through it. HNSW works the same way: queries sink through sparse upper layers to reach a dense bottom layer quickly.

> **Rule for this repo:** every number in this README is pasted from the stdout of a command in this repo, run on a clean checkout. If a number has no command next to it, it does not belong here.

---

## Table of Contents

- [Why Loam](#why-loam)
- [What Makes Loam Different](#what-makes-loam-different)
- [Features](#features)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage Examples](#usage-examples)
- [Benchmarks](#benchmarks)
- [Testing](#testing)
- [One-Day Build Plan](#one-day-build-plan)
- [Learning Objectives](#learning-objectives)
- [Non-Goals](#non-goals)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Why Loam

Every RAG pipeline ends in a call like `collection.query(embedding, k=10)`. Behind that call sits an ANN index that makes a trade you never see. It returns *approximately* the nearest neighbors, and how approximate depends on parameters such as `M`, `ef_construction` and `ef`, which most users never touch.

This causes three practical problems:

1. **Recall degrades silently.** A too-small `ef` returns plausible but wrong neighbors. Nothing errors, and your retrieval quality just drops.
2. **The trade-off is invisible.** Production libraries expose knobs but not the work each query did (hops, visited nodes, distance computations), so you can't build intuition for why a setting helps.
3. **The algorithm is treated as magic.** HNSW is a graph built in a few hundred lines of logic. After building it once, tuning any vector database stops being guesswork.

Loam exists to make that machinery visible: small enough to read in one sitting, and measured precisely enough to trust.

---

## What Makes Loam Different

Loam is **not** faster than hnswlib or FAISS, and never will be. Those are optimized C++ with SIMD. Loam's value is in what it lets you *see and verify*:

| Capability | hnswlib / FAISS / pgvector | Loam |
|---|---|---|
| Production speed | ✅ | ❌ (pure Python + numpy) |
| Code maps 1:1 to the paper's Algorithms 1–5 | ❌ | ✅ |
| Per-query instrumentation (hops per layer, visited nodes, distance computations) | ❌ | ✅ |
| Built-in exact oracle for recall@k on every benchmark | ❌ (external) | ✅ |
| Search-path trace through each layer | ❌ | ✅ `loam trace` |
| One-flag ablation: heuristic vs simple neighbor selection | ❌ | ✅ |
| One-flag ablation: hierarchy on vs off (flat NSW) | ❌ | ✅ |

The two ablations reproduce **published claims** at laptop scale:

- **Neighbor-selection heuristic.** The HNSW paper (Fig. 7) reports that the diversity heuristic (Algorithm 4) matters most on *clustered* data, where simple nearest-M selection gets stuck at cluster boundaries. Loam ships a synthetic clustered-data generator to test this directly.
- **Is the "H" necessary?** *Down with the Hierarchy* (Munyampirwa et al., arXiv:2412.01940) reports that on high-dimensional data a flat navigable small-world graph matches HNSW's recall and latency, and attributes this to a "highway" of hub nodes. Loam's `--no-hierarchy` flag collapses every node to layer 0 so you can measure this yourself.

Loam reports what it measures, whichever way the result goes. If the results disagree with a paper, the README says so.

---

## Features

- **Exact flat index (`FlatIndex`).** A vectorized brute-force k-NN that serves as the correctness oracle for everything else.
- **HNSW index (`HNSWIndex`)** implementing the paper's algorithms under their own names:
  - `insert` → Algorithm 1 (INSERT)
  - `_search_layer` → Algorithm 2 (SEARCH-LAYER)
  - `_select_neighbors_simple` → Algorithm 3
  - `_select_neighbors_heuristic` → Algorithm 4 (with `extend_candidates` and `keep_pruned_connections` flags)
  - `knn_search` → Algorithm 5 (K-NN-SEARCH)
- **Paper-default parameters:** `M=16`, `M_max0 = 2·M`, `mL = 1/ln(M)`, `ef_construction=200`, all overridable.
- **Two metrics:** `cosine` (vectors are L2-normalized at insert, so distance = `1 − dot`) and `l2` (squared Euclidean).
- **Deterministic builds.** Pass a `seed` and the same data produces the same graph, bit for bit.
- **Per-query instrumentation.** Distance computations, nodes visited, and hops per layer, returned in a `QueryStats` object.
- **Benchmark harness (`loam bench`).** Sweeps `ef` and records recall@k against the flat oracle, single-threaded QPS, mean distance computations per query and build time. Writes a CSV and a recall-vs-QPS plot.
- **Ablation runner (`loam ablate`):** `--selection {simple,heuristic}` and `--no-hierarchy`.
- **Search tracer (`loam trace`).** Prints the path a single query takes through each layer.
- **Dataset loader** for ANN-Benchmarks HDF5 files. It can subsample the data, and it recomputes ground truth after subsampling, because the shipped ground truth covers the *full* train set. It also includes a synthetic clustered-data generator.
- **Optional reference comparison** against hnswlib, run on the same data with the same `M`, `ef_construction` and `ef`.

---

## Architecture

```mermaid
flowchart LR
    subgraph Data["Data layer"]
        D1["ANN-Benchmarks HDF5<br/>(train / test / neighbors)"]
        D2["Synthetic clustered<br/>generator"]
        D3["Subsample + recompute<br/>ground truth"]
    end

    subgraph Core["Core (numpy only)"]
        V["VectorStore<br/>preallocated float32 matrix"]
        F["FlatIndex<br/>exact oracle"]
        H["HNSWIndex<br/>Algorithms 1–5"]
        I["Instrumentation<br/>QueryStats counters"]
    end

    subgraph Tools["Tooling"]
        B["bench<br/>ef sweep → CSV + plot"]
        A["ablate<br/>selection / hierarchy"]
        T["trace<br/>per-layer search path"]
        R["reference<br/>hnswlib (optional)"]
    end

    D1 --> D3
    D2 --> D3
    D3 --> V
    V --> F
    V --> H
    H --> I
    F --> B
    H --> B
    H --> A
    H --> T
    R --> B
    B --> OUT[("bench/results/<br/>*.csv, *.png")]
```

**Design decisions**

- **Vectors live in one preallocated `float32` matrix.** Node IDs are row indices. Distances from a query to a whole neighbor list are computed in a single numpy operation (`matrix[neighbor_ids] @ q`) instead of a Python loop. This is the single most important performance decision in a pure-Python HNSW.
- **Adjacency is `layers[level][node_id] -> list[int]`**, plain Python lists. They are readable, easy to inspect, and easy to assert invariants on.
- **Candidate set `C` is a min-heap; result set `W` is a max-heap** (distances negated), using `heapq`, exactly as Algorithm 2 describes.
- **Cosine is implemented by normalizing at insert time**, so the hot path is a dot product.
- **The oracle is non-negotiable.** No recall number is reported without an exact k-NN computed by `FlatIndex` on the same data.

---

## How It Works

### The layered graph

Each inserted element gets a random maximum layer `l = ⌊−ln(U(0,1)) · mL⌋`, where `U(0,1)` is uniform and `mL = 1/ln(M)`. Few elements reach the high layers, and every element exists on layer 0. Upper layers hold long-range links, and layer 0 holds short, dense ones.

```mermaid
flowchart TB
    subgraph L2["Layer 2: sparsest, long hops"]
        A2((A)) --- E2((E))
    end
    subgraph L1["Layer 1"]
        A1((A)) --- C1((C)) --- E1((E)) --- G1((G))
    end
    subgraph L0["Layer 0: every node, max 2·M links"]
        A0((A)) --- B0((B)) --- C0((C)) --- D0((D)) --- E0((E)) --- F0((F)) --- G0((G)) --- H0((H))
    end
    A2 -.-> A1
    E2 -.-> E1
    A1 -.-> A0
    C1 -.-> C0
    E1 -.-> E0
    G1 -.-> G0
```

### Search (Algorithm 5 → Algorithm 2)

```mermaid
flowchart TD
    S["Query q"] --> EP["Start at global entry point<br/>(top layer L)"]
    EP --> G{"Current layer > 0?"}
    G -- yes --> GR["SEARCH-LAYER with ef = 1<br/>greedy walk to closest node"]
    GR --> DN["Drop one layer;<br/>closest node becomes entry point"]
    DN --> G
    G -- no --> B0["SEARCH-LAYER on layer 0<br/>with beam width ef"]
    B0 --> K["Return K nearest from W"]
```

Inside **SEARCH-LAYER** (Algorithm 2):

```mermaid
flowchart TD
    I["C = W = {entry}; visited = {entry}"] --> P{"C empty?"}
    P -- yes --> RET["return W"]
    P -- no --> X["c = pop nearest from C<br/>f = furthest in W"]
    X --> STOP{"dist(c,q) > dist(f,q)?"}
    STOP -- yes --> RET
    STOP -- no --> N["For each unvisited neighbor e of c:<br/>mark visited"]
    N --> ADD{"dist(e,q) < dist(f,q)<br/>or |W| < ef?"}
    ADD -- yes --> PUSH["push e into C and W;<br/>if |W| > ef, pop furthest from W"]
    ADD -- no --> P
    PUSH --> P
```

### Insert (Algorithm 1)

```mermaid
sequenceDiagram
    participant U as Caller
    participant H as HNSWIndex
    participant S as SEARCH-LAYER
    participant N as SELECT-NEIGHBORS
    U->>H: insert(vector)
    H->>H: draw level l = floor(-ln(U) * mL)
    loop layers L down to l+1
        H->>S: greedy search (ef = 1)
        S-->>H: closest node → next entry point
    end
    loop layers min(L, l) down to 0
        H->>S: search (ef = ef_construction)
        S-->>H: candidate set W
        H->>N: pick M neighbors from W
        N-->>H: neighbors
        H->>H: add bidirectional links
        H->>N: shrink any neighbor over M_max (M_max0 on layer 0)
    end
    H->>H: if l > L, new node becomes entry point
```

### Neighbor selection: simple vs heuristic

- **Simple (Algorithm 3)** keeps the `M` closest candidates.
- **Heuristic (Algorithm 4)** walks candidates nearest-first and keeps `e` only if `e` is closer to `q` than to every neighbor already kept. The result is spread across directions instead of bunched into one cluster, which is what keeps the graph navigable on clustered data.

### Benchmark workflow

```mermaid
flowchart LR
    DS["Dataset<br/>(HDF5 or synthetic)"] --> SUB["Subsample n train vectors"]
    SUB --> GT["FlatIndex: exact top-k<br/>for each test query"]
    SUB --> BUILD["Build HNSW<br/>(timed)"]
    BUILD --> SW["For ef in sweep:<br/>run all queries (timed)"]
    GT --> REC["recall@k = |approx ∩ exact| / k"]
    SW --> REC
    SW --> ST["QPS, mean distance comps,<br/>mean visited"]
    REC --> CSV[("results.csv")]
    ST --> CSV
    CSV --> PLOT["recall vs QPS plot"]
```

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Language | **Python 3.12+** | Fast to build in a day; readability is the product |
| Numerics | **numpy 2.x** | Vectorized distance computations; the only core dependency |
| Data | **h5py** | Reads ANN-Benchmarks HDF5 datasets |
| CLI | **typer** + **rich** | Typed commands and readable terminal tables |
| Plots | **matplotlib** | Recall-vs-QPS curves |
| Testing | **pytest** + **hypothesis** | Unit tests plus property-based graph-invariant tests |
| Packaging | **uv** (pip also works) | Reproducible environments |
| Reference (optional) | **hnswlib** | Sanity comparison on the same data and parameters |
| CI | **GitHub Actions** | Tests on every push |

> `hnswlib` 0.8.0 is published on PyPI as a source distribution only, so installing it requires a C++ compiler. It is an optional extra and nothing in Loam's core depends on it.

---

## Project Structure

```text
Loam/
├── src/loam/
│   ├── __init__.py          # exports FlatIndex, HNSWIndex, QueryStats
│   ├── distance.py          # cosine (pre-normalized) and squared-L2, batched
│   ├── store.py             # VectorStore: growable float32 matrix
│   ├── flat.py              # FlatIndex: exact brute-force oracle
│   ├── hnsw.py              # HNSWIndex: Algorithms 1–5, named after the paper
│   ├── stats.py             # QueryStats: distance comps, visited, hops per layer
│   ├── datasets.py          # HDF5 loader, subsample + ground truth, clustered generator
│   ├── bench.py             # ef sweep, recall/QPS measurement, CSV + plot
│   ├── trace.py             # per-layer search path printer
│   └── cli.py               # `loam` entrypoint: fetch, bench, ablate, trace
├── tests/
│   ├── test_flat.py         # oracle vs numpy argsort
│   ├── test_invariants.py   # hypothesis: degree bounds, layer nesting, no self-loops
│   ├── test_recall.py       # recall floors on small fixed-seed datasets
│   └── test_determinism.py  # same seed → identical graph
├── bench/results/           # generated CSVs and plots (committed for the README)
├── .github/workflows/ci.yml
├── pyproject.toml
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Getting Started

### Prerequisites

- Python **3.12 or newer** (numpy 2.x requires ≥ 3.12)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- ~130 MB of free disk space for the default dataset (`glove-25-angular`)
- *Optional:* a C++ compiler, needed only for the hnswlib reference comparison

### Installation

```bash
git clone https://github.com/VampiricCyborg/Loam.git
cd Loam
uv sync
```

With pip:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional hnswlib reference comparison:

```bash
uv sync --extra reference
```

Verify the install:

```bash
uv run pytest -q
uv run loam --help
```

### Fetch a dataset

```bash
# GloVe-25 angular from ANN-Benchmarks (127 MB, HDF5)
uv run loam fetch glove-25-angular
```

The HDF5 file contains `train`, `test`, `neighbors` and `distances` arrays. Because Loam benchmarks on a **subsample** of `train`, the shipped `neighbors` are not valid for it. Loam always recomputes ground truth with `FlatIndex` after subsampling.

---

## Usage Examples

### Python API

```python
import numpy as np
from loam import FlatIndex, HNSWIndex

rng = np.random.default_rng(0)
data = rng.standard_normal((10_000, 64), dtype=np.float32)
queries = rng.standard_normal((100, 64), dtype=np.float32)

index = HNSWIndex(dim=64, metric="cosine", M=16, ef_construction=200, seed=42)
index.add(data)                                   # ids are row positions 0..n-1

ids, dists, stats = index.search(queries[0], k=10, ef=64, return_stats=True)
print(ids)
print(stats.distance_computations, stats.visited, stats.hops_per_layer)

# Check against the exact oracle
oracle = FlatIndex(dim=64, metric="cosine")
oracle.add(data)
exact_ids, _ = oracle.search(queries[0], k=10)
recall = len(set(ids) & set(exact_ids)) / 10
print(f"recall@10 = {recall:.2f}")
```

### Benchmark: ef sweep

```bash
uv run loam bench glove-25-angular \
  --n 10000 --queries 1000 --k 10 \
  -M 16 --ef-construction 200 \
  --ef 16,32,64,128,256 \
  --out bench/results/glove25-ef-sweep.csv --plot
```

### Ablations

```bash
# Heuristic vs simple neighbor selection on clustered synthetic data
uv run loam ablate selection --dataset clustered --clusters 100 --dim 10 --n 20000

# Hierarchy on vs off (flat NSW): tests the "Hub Highway" claim
uv run loam ablate hierarchy --dataset glove-25-angular --n 10000
```

### Trace a single query

```bash
uv run loam trace glove-25-angular --n 5000 --query-index 0 --ef 32
```

Prints the entry point, each greedy hop on the upper layers, and the beam expansion on layer 0. Actual output, abridged only where the run itself prints `... N further expansions omitted`:

```text
building HNSW: 11.0s  (2.20 ms/insert)  layer sizes [5000, 283, 18, 2]

Query trace  k=10  ef=32  entry point=node 3517 on layer 3

layer 3  (greedy descent, ef=1, 2 node(s) expanded)
  enter at: node 3517 @ 1.0063
   #   expanded    distance   new neighbors
   1       3517     1.00629               1  closer
   2       4350     0.99973               0  closer
  -> hands node 4350 down at distance 0.99973

layer 2  (greedy descent, ef=1, 2 node(s) expanded)
  enter at: node 4350 @ 0.9997
   #   expanded    distance   new neighbors
   1       4350     0.99973              16  closer
   2       2516     0.47033               1  closer
  -> hands node 2516 down at distance 0.47033

layer 1  (greedy descent, ef=1, 2 node(s) expanded)
  enter at: node 2516 @ 0.4703
   #   expanded    distance   new neighbors
   1       2516     0.47033              16  closer
   2       4874     0.21436              11  closer
  -> hands node 4874 down at distance 0.21436

layer 0  (beam search, ef=32, 32 node(s) expanded)
  enter at: node 4874 @ 0.2144
   #   expanded    distance   new neighbors
   1       4874     0.21436              16  closer
   2       4833     0.20585              15  closer
   3       2412     0.16873              24  closer
   4        592     0.19179              21  --
   5       3620     0.19887              15  --
   6         64     0.20688              20  --
   ... 20 further expansions omitted
  -> best in beam: node 2412 at 0.16873

Result
 rank      node    distance  in exact top-k?
    1      2412     0.16873  yes
    2      1280     0.18833  yes
    3      3160     0.18894  yes
   ...
   10      4874     0.21436  yes

recall@10 for this query: 1.00
work: 466 distance computations, 466 nodes visited, hops per layer: L3:2  L2:2  L1:2  L0:32
```

The two phases of Algorithm 5 have visibly different shapes. The descent spends **two hops per layer** and cuts the distance from 1.006 to 0.214 — most of the progress, for almost none of the work. Layer 0 then spends **32 expansions**, and finds its eventual best answer on the *third* of them. The other 29 are the price of confirming that nothing better exists, and that is what a larger `ef` actually buys: not a better answer found sooner, but more confidence that the answer already found is right.

### Compare against hnswlib (optional)

```bash
uv run loam bench glove-25-angular --n 10000 --ef 16,32,64,128 --reference hnswlib
```

---

## Benchmarks

> Every cell below is pasted from the stdout of the command printed above it, run on a clean checkout.

**Machine for every table on this page:** `Windows 11, Intel64 Family 6 Model 154 Stepping 3, GenuineIntel, Python 3.13.1` (numpy 2.5.3, single-threaded). That string is what `loam version` prints, and every command echoes it.

### ef sweep: glove-25-angular

```bash
uv run loam bench glove-25-angular --n 10000 --queries 1000 --k 10 \
  -M 16 --ef-construction 200 --ef 16,32,64,128,256 \
  --out bench/results/glove25-ef-sweep.csv --plot
```

Build: **44.6 s** (4.46 ms/insert). Layer sizes `[10000, 571, 35, 5]` — each layer holds roughly `1/M` of the one below it, as `mL = 1/ln(M)` intends.

| ef | recall@10 | QPS (1 thread) | mean distance comps / query | mean hops | latency (ms) |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.9360 | 3315.1 | 397.2 | 23.5 | 0.302 |
| 32 | 0.9814 | 1724.9 | 618.4 | 38.9 | 0.580 |
| 64 | 0.9970 | 993.0 | 1001.6 | 70.5 | 1.007 |
| 128 | 0.9997 | 622.4 | 1623.3 | 134.3 | 1.607 |
| 256 | 1.0000 | 285.1 | 2595.1 | 262.2 | 3.507 |
| flat (exact) | 1.0000 | 6348.7 | 10000.0 | — | 0.158 |

![recall vs QPS on glove-25-angular](bench/results/glove25-ef-sweep.png)

**Read this table honestly: at n = 10,000 the exact index is the faster one.**

That is not a bug, and it is the most useful thing in this repo. The algorithmic win is real and large — at `ef=16`, HNSW answers with **397 distance computations instead of 10,000**, a 25× reduction, and still gets 93.6% of the true top-10. But Loam loses on wall clock anyway, because the two indexes pay very different prices per distance:

- `FlatIndex` computes all 10,000 distances in **one** numpy call: a single BLAS matmul over one contiguous `(10000, 25)` matrix.
- `HNSWIndex` computes its 397 distances across **23.5 separate** gather-plus-matmul calls, one per expanded node, each carrying Python interpreter overhead that dwarfs its arithmetic.

So a 25× reduction in *work* becomes a 20× increase in *time*. The crossover where HNSW wins on the clock needs either a much larger `n` (where the flat matmul stops being cheap) or an implementation where each distance costs the same — which is exactly what hnswlib's C++ buys, and exactly why Loam does not try to compete with it. Loam's job is to show you the 397, and it does.

### Ablation 1: neighbor selection, Algorithm 3 vs Algorithm 4

```bash
uv run loam ablate selection --dataset clustered --clusters 100 --dim 10 \
  --n 20000 --queries 1000 --ef 16,32,64,128 \
  --out bench/results/ablate-selection.csv
```

Build: simple **19.7 s**, heuristic **41.5 s**. Same data, same seed, same `M` and `ef_construction`; the only difference is which SELECT-NEIGHBORS runs.

| ef | simple recall@10 | heuristic recall@10 | difference | simple QPS | heuristic QPS |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.8989 | **1.0000** | +0.1011 | 4664.2 | 3735.4 |
| 32 | 0.9110 | **1.0000** | +0.0890 | 3334.5 | 2609.3 |
| 64 | 0.9303 | **1.0000** | +0.0697 | 2105.2 | 1642.6 |
| 128 | 0.9545 | **1.0000** | +0.0455 | 1544.3 | 1012.3 |

**The paper's Fig. 7 claim reproduces.** On 100 tight clusters in 10 dimensions, nearest-M selection tops out at 0.9545 even at `ef=128`, while the diversity heuristic is exact at every `ef` tested. The shape matters as much as the gap: simple selection does not merely start lower, it *converges slowly*, because widening the beam cannot help a search that has no edge out of the cluster it landed in. Diverse links create those edges; extra beam width only searches harder within the same trap.

The cost is real and visible: the heuristic build takes 2.1× as long, and its graphs are slower to query (it fills the degree budget, so each hop scores more neighbors). On clustered data that is an obvious trade to make.

### Ablation 2: is the "H" in HNSW doing anything?

```bash
uv run loam ablate hierarchy --dataset glove-25-angular --n 10000 \
  --queries 1000 --ef 16,32,64,128,256 \
  --out bench/results/ablate-hierarchy.csv
```

Build: HNSW **45.9 s**, flat NSW **23.4 s**.

| ef | HNSW recall@10 | flat NSW recall@10 | difference | HNSW QPS | flat NSW QPS |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.9360 | 0.9337 | −0.0023 | 3242.7 | 3622.3 |
| 32 | 0.9814 | 0.9816 | +0.0002 | 2087.4 | 2226.8 |
| 64 | 0.9970 | 0.9969 | −0.0001 | 1202.6 | 1279.0 |
| 128 | 0.9997 | 0.9997 | +0.0000 | 649.1 | 676.8 |
| 256 | 1.0000 | 1.0000 | +0.0000 | 362.6 | 366.3 |

**Down with the Hierarchy (arXiv:2412.01940) reproduces here too.** Deleting every layer above 0 costs at most 0.0023 recall, and the flat graph is *slightly faster at every `ef`* while building in half the time. At this scale and dimension the layers earn nothing.

Two honest caveats before anyone generalizes this:

1. **25 dimensions is not "high-dimensional"** by that paper's standard, and its argument is specifically about high-dimensional regimes. This result is consistent with the paper's claim but is not a test of its hardest case.
2. **n = 10,000 is small.** The hierarchy exists to shorten the greedy walk from a random entry point to the query's neighborhood, and that walk is short when the graph is small. The upper layers here hold 571, 35 and 5 nodes; there is not much routing for them to do.

What the trace shows is consistent with this: the descent through layers 3, 2 and 1 costs only a handful of hops, so removing it saves little and costs little.

### Methodology, and a caveat about the timings

- Recall is the mean of `|approx ∩ exact| / k` over all queries, against ground truth recomputed by `FlatIndex` on the same subsample.
- QPS is single-threaded wall-clock over all queries, after a warm-up pass. Loam does not batch queries, so QPS is the reciprocal of mean latency.
- Instrumentation is collected on the same pass that is timed; the counters are integer increments and are noise next to the numpy calls around them.

**Wall-clock on this machine is noisy, and the repo would rather say so than hide it.** Running the `ef` sweep command above three times gave build times of 194.4 s, 31.5 s and 44.6 s, and flat-index throughput of 15519, 13509 and 6349 QPS — a 2.4× spread on an identical workload, consistent with thermal and power management on a laptop. Across all three runs, **recall and mean distance computations were identical to every digit reported**, because the build is deterministic given a seed.

That contrast is the argument for instrumenting distance computations in the first place. `397.2` is a property of the algorithm and reproduces anywhere; `3315.1 QPS` is a property of this laptop on that afternoon. The tables above report one run each, so the numbers within a table are mutually consistent; treat the QPS columns as ±40% and the distance-computation columns as exact.

---

## Testing

```bash
uv run pytest -q                    # full suite
uv run pytest tests/test_invariants.py -q
```

**What the suite guarantees**

- **Oracle correctness:** `FlatIndex` matches `numpy.argsort` on random data, for both metrics.
- **Graph invariants** (property-based, via hypothesis), checked after every build:
  - no node has more than `M_max` links on layers ≥ 1, or more than `M_max0` on layer 0;
  - a node present on layer `l` is present on every layer below `l`;
  - no self-loops, and every neighbor ID refers to an existing node;
  - the entry point sits on the top layer.
- **Recall floors:** on small fixed-seed datasets, recall@10 stays above a threshold that is calibrated once from measured runs and then locked in as a regression guard.
- **Determinism:** the same data and seed produce an identical adjacency structure.

---

## One-Day Build Plan

Each milestone is done only when its acceptance check passes.

| # | Block | Deliverable | Acceptance check |
|---|---|---|---|
| 1 | Hour 0–1 | Scaffold, `distance.py`, `store.py`, `FlatIndex` | `test_flat.py` passes for cosine and L2 |
| 2 | Hour 1–4 | `HNSWIndex`: level draw, Algorithms 1, 2, 3, 5 | Recall@10 ≥ 0.9 at ef=64 on 5k random vectors (floor to calibrate) |
| 3 | Hour 4–5 | Algorithm 4 heuristic + neighbor-list shrinking | `test_invariants.py` passes under hypothesis |
| 4 | Hour 5–7 | `datasets.py` (HDF5 + subsample + ground truth + clustered), `bench.py`, CLI | `loam bench` writes CSV and plot for glove-25 |
| 5 | Hour 7–8 | `QueryStats` instrumentation + `loam ablate` | Both ablations produce a table |
| 6 | Hour 8–9 | `loam trace` | Readable per-layer path for one query |
| 7 | Hour 9–10 | README numbers, CI, LICENSE, CONTRIBUTING | CI green; every README number has its command |

**Scope cut order if behind schedule:** hnswlib reference → `trace` → hierarchy ablation. Never cut: the oracle, the invariant tests, or the honest benchmark table.

---

## Learning Objectives

After building Loam you should be able to explain:

- why exact k-NN is `O(n·d)` per query, and why that cost becomes infeasible at scale;
- how skip lists inspired HNSW's layer hierarchy, and why the level distribution is exponential;
- what `M`, `M_max0`, `ef_construction` and `ef` each control, and what each costs;
- why greedy graph search gets stuck in local minima, and how beam width (`ef`) and diverse neighbor selection avoid it;
- how to measure an approximate algorithm honestly, using an oracle, recall@k and a trade-off curve instead of a single number;
- what the "Hub Highway Hypothesis" claims and how to test it.

---

## Non-Goals

- Deletion or updates (the paper supports them; Loam skips them for scope)
- Concurrency or multi-threaded build and search
- SIMD, Cython, Numba or any compiled speedups
- Quantization (PQ, SQ) or disk-resident indexes (DiskANN-style)
- Being a production vector database

---

## Roadmap

- [ ] Persistence: save/load the index to `.npz`
- [ ] Deletion via tombstones
- [ ] 2D visualization of layers on synthetic data
- [ ] Hub analysis: measure in-degree distribution and hub traversal frequency on flat vs hierarchical graphs
- [ ] Rust port of the hot path, benchmarked against the Python original
- [ ] A recorded terminal demo of `loam trace` and `loam bench`

---

## Contributing

Contributions are welcome, especially bug reports where Loam's behavior diverges from the paper.

1. **Open an issue first** for anything beyond a small fix, so the approach can be agreed before code is written.
2. **Fork** the repo and create a branch: `git checkout -b feat/short-description` or `fix/short-description`.
3. **Keep the core dependency-free:** `src/loam/` core modules import only numpy and the standard library (the CLI, datasets and bench modules may use their listed dependencies).
4. **Add tests** for any behavior change. Algorithm changes must keep `test_invariants.py` and `test_recall.py` green.
5. **Run the suite locally:**

   ```bash
   uv run pytest -q
   ```

6. **Benchmark claims need commands.** If a PR changes performance, include the exact `loam bench` command and its before/after output in the description.
7. Use clear commit messages (for example, `fix(hnsw): shrink layer-0 neighbors with M_max0`) and open a **pull request against `main`** describing what changed and why.

---

## License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

```text
MIT License

Copyright (c) 2026 Madhav (VampiricCyborg)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## References

1. Yu. A. Malkov and D. A. Yashunin. *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI 42(4), 824–836, 2020. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320) · doi:10.1109/TPAMI.2018.2889473
2. Yu. A. Malkov, A. Ponomarenko, A. Logvinov and V. Krylov. *Approximate nearest neighbor algorithm based on navigable small world graphs.* Information Systems 45, 61–68, 2014.
3. B. Munyampirwa, V. Lakshman and B. Coleman. *Down with the Hierarchy: The 'H' in HNSW Stands for "Hubs".* [arXiv:2412.01940](https://arxiv.org/abs/2412.01940)
4. W. Pugh. *Skip Lists: A Probabilistic Alternative to Balanced Trees.* Communications of the ACM 33(6), 668–676, 1990.
5. M. Aumüller, E. Bernhardsson and A. Faithfull. *ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.* [github.com/erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks) (datasets used here; the project notes it is no longer actively maintained)
6. hnswlib: [github.com/nmslib/hnswlib](https://github.com/nmslib/hnswlib)
