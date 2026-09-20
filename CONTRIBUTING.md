# Contributing to Loam

Contributions are welcome, especially bug reports where Loam's behavior diverges from the paper.

Loam is a learning project, and its product is *readability plus honest measurement*. A change that makes the code faster but harder to read against Malkov & Yashunin's Algorithms 1–5 is usually the wrong trade here. A change that makes a number more trustworthy is almost always the right one.

## Ground rules

1. **Open an issue first** for anything beyond a small fix, so the approach can be agreed before code is written.
2. **Fork** the repo and create a branch: `git checkout -b feat/short-description` or `fix/short-description`.
3. **Keep the core dependency-free.** `distance.py`, `store.py`, `flat.py`, `hnsw.py` and `stats.py` import only numpy and the standard library. `datasets.py`, `bench.py`, `trace.py` and `cli.py` may use the project's declared dependencies.
4. **Add tests** for any behavior change. Algorithm changes must keep `tests/test_invariants.py` and `tests/test_recall.py` green.
5. **Run the suite locally:**

   ```bash
   uv run pytest -q
   ```

6. **Benchmark claims need commands.** If a PR changes performance, include the exact `loam bench` command and its before/after output in the description.
7. Use clear commit messages (for example, `fix(hnsw): shrink layer-0 neighbors with M_max0`) and open a **pull request against `main`** describing what changed and why.

## The rule about numbers

Every number in `README.md` is pasted from the stdout of a command in this repo, run on a clean checkout, and the command sits next to it. If you change a number, change its command too, and re-run it. If you cannot produce the command, the number does not go in.

This applies to negative results as well. If a measurement disagrees with a paper, the README says so and shows the measurement. Do not tune parameters until a result matches an expectation.

## Where things live

| You want to change... | Look in |
|---|---|
| A paper algorithm | `src/loam/hnsw.py` — methods are named after Algorithms 1–5 |
| Distance math | `src/loam/distance.py` |
| How vectors are stored | `src/loam/store.py` |
| The correctness oracle | `src/loam/flat.py` |
| What a query reports about itself | `src/loam/stats.py` |
| Dataset loading, subsampling, ground truth | `src/loam/datasets.py` |
| Recall/QPS measurement, CSV, plots | `src/loam/bench.py` |
| The per-layer search printout | `src/loam/trace.py` |
| CLI commands and flags | `src/loam/cli.py` |

## Performance changes

Pure-Python HNSW is dominated by interpreter overhead, not arithmetic. Two things follow.

First, **measure before you optimize**. Several "obvious" vectorizations in this codebase were tried and reverted because a numpy call on a 32-element adjacency list costs more than the Python loop it replaces. `src/loam/hnsw.py` carries comments where that happened; please add one if you find another.

Second, **compiled speedups are out of scope**. No Cython, Numba, SIMD or C extensions in the core. That is a stated non-goal: hnswlib already exists and is excellent, and Loam is not trying to compete with it.

## Adding a benchmark or ablation

An ablation changes exactly one thing and holds everything else fixed, including the seed. `loam ablate selection` and `loam ablate hierarchy` are the templates. If a new ablation needs a new build flag, it goes on `HNSWIndex.__init__` with a default that preserves current behavior, and `tests/test_determinism.py::test_ablation_flags_change_the_graph` gets a line proving the flag does something.

## Reporting a divergence from the paper

The most valuable issue you can file. Please include:

- the algorithm and step number from arXiv:1603.09320,
- what Loam does instead, with a `file.py:line` reference,
- a minimal script that shows the difference, ideally one that fails an assertion.
