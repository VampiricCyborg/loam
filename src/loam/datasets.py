"""Datasets: ANN-Benchmarks HDF5 files, subsampling, and synthetic generators.

The one thing worth reading carefully here is :func:`subsample`.

ANN-Benchmarks HDF5 files ship four arrays: ``train``, ``test``, ``neighbors``
and ``distances``. The ``neighbors`` array is the ground truth *for the full
train set*. Loam benchmarks on a subsample of ``train``, because a pure-Python
build of a million vectors is not a laptop-scale operation. The moment you take
a subsample, the shipped ground truth is wrong: it names IDs that are no longer
in the index, and the true nearest neighbor within the subsample is a different
vector entirely.

So Loam always recomputes ground truth with ``FlatIndex`` after subsampling.
Using the shipped ``neighbors`` on a subsample would silently deflate every
recall number in this repo, which is exactly the class of quiet wrongness the
project exists to avoid.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .distance import Metric
from .flat import FlatIndex

ANN_BENCHMARKS_URL = "https://ann-benchmarks.com/{name}.hdf5"

_USER_AGENT = "Mozilla/5.0 (compatible; loam/0.1; +https://github.com/VampiricCyborg/Loam)"

#: The distance metric each ANN-Benchmarks dataset is scored under.
DATASET_METRICS: dict[str, Metric] = {
    "glove-25-angular": "cosine",
    "glove-50-angular": "cosine",
    "glove-100-angular": "cosine",
    "glove-200-angular": "cosine",
    "nytimes-256-angular": "cosine",
    "sift-128-euclidean": "l2",
    "fashion-mnist-784-euclidean": "l2",
    "mnist-784-euclidean": "l2",
    "gist-960-euclidean": "l2",
    "lastfm-64-dot": "cosine",
}

DEFAULT_DATA_DIR = Path("data")


@dataclass
class Dataset:
    """Train vectors, query vectors, and ground truth that matches them."""

    name: str
    train: np.ndarray
    test: np.ndarray
    metric: Metric
    ground_truth: np.ndarray | None = None
    k: int = 0

    @property
    def n(self) -> int:
        return self.train.shape[0]

    @property
    def dim(self) -> int:
        return self.train.shape[1]

    @property
    def num_queries(self) -> int:
        return self.test.shape[0]

    def compute_ground_truth(self, k: int) -> np.ndarray:
        """Exact top-``k`` for every query, via the brute-force oracle."""
        oracle = FlatIndex(dim=self.dim, metric=self.metric)
        oracle.add(self.train)
        self.ground_truth, _ = oracle.search_batch(self.test, k=k)
        self.k = k
        return self.ground_truth

    def describe(self) -> str:
        return (
            f"{self.name}: n={self.n:,} dim={self.dim} "
            f"queries={self.num_queries:,} metric={self.metric}"
        )


# ----------------------------------------------------------------------
# ANN-Benchmarks HDF5
# ----------------------------------------------------------------------


def dataset_path(name: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> Path:
    return Path(data_dir) / f"{name}.hdf5"


def fetch(
    name: str,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download an ANN-Benchmarks HDF5 dataset, unless it is already here."""
    path = dataset_path(name, data_dir)
    if path.exists() and not force:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = ANN_BENCHMARKS_URL.format(name=name)
    tmp = path.with_suffix(".hdf5.part")

    # ann-benchmarks.com answers 403 to urllib's default User-Agent, so send a
    # browser-ish one. urlretrieve cannot set headers, hence urlopen by hand.
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if progress and total:
                pct = 100.0 * done / total
                print(
                    f"\r  {name}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)",
                    end="",
                    flush=True,
                )
    if progress:
        print()
    tmp.replace(path)
    return path


def load_hdf5(
    name: str,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    download: bool = True,
) -> Dataset:
    """Load an ANN-Benchmarks dataset, fetching it first if needed.

    The shipped ``neighbors`` array is deliberately *not* read: it is only
    valid for the full train set, and Loam subsamples. See the module docstring.
    """
    import h5py  # imported lazily so the core stays numpy-only

    path = dataset_path(name, data_dir)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found; run `loam fetch {name}` first")
        fetch(name, data_dir)

    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        metric = DATASET_METRICS.get(name)
        if metric is None:
            attr = f.attrs.get("distance", "angular")
            metric = "cosine" if str(attr) in ("angular", "dot") else "l2"

    return Dataset(name=name, train=train, test=test, metric=metric)


# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------


def make_clustered(
    n: int,
    dim: int,
    clusters: int = 100,
    spread: float = 0.08,
    queries: int = 500,
    metric: Metric = "cosine",
    seed: int = 0,
) -> Dataset:
    """Tight Gaussian blobs around random centers.

    This is the regime where Algorithm 4 is supposed to matter. With simple
    nearest-M selection, a node's links all land inside its own blob, because
    its own blob contains the ``M`` nearest points by a wide margin. The graph
    then has few edges crossing between blobs, and greedy search that starts in
    the wrong blob has no way out. The paper's Fig. 7 is measured on exactly
    this kind of data.

    Queries are drawn from the same mixture, so they sit inside clusters rather
    than in empty space.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    assignment = rng.integers(0, clusters, size=n)
    train = centers[assignment] + spread * rng.standard_normal((n, dim)).astype(np.float32)

    q_assignment = rng.integers(0, clusters, size=queries)
    test = centers[q_assignment] + spread * rng.standard_normal((queries, dim)).astype(np.float32)

    return Dataset(
        name=f"clustered-{clusters}c-{dim}d",
        train=train.astype(np.float32),
        test=test.astype(np.float32),
        metric=metric,
    )


def make_random(
    n: int,
    dim: int,
    queries: int = 500,
    metric: Metric = "cosine",
    seed: int = 0,
) -> Dataset:
    """Uniform Gaussian data: the easy, unclustered baseline."""
    rng = np.random.default_rng(seed)
    return Dataset(
        name=f"random-{dim}d",
        train=rng.standard_normal((n, dim)).astype(np.float32),
        test=rng.standard_normal((queries, dim)).astype(np.float32),
        metric=metric,
    )


# ----------------------------------------------------------------------
# Subsampling
# ----------------------------------------------------------------------


def subsample(
    dataset: Dataset,
    n: int | None = None,
    queries: int | None = None,
    seed: int = 0,
) -> Dataset:
    """Take ``n`` train vectors and ``queries`` test vectors.

    Ground truth is *not* carried over. It is recomputed by the caller (or by
    :meth:`Dataset.compute_ground_truth`) against the subsample, because any
    ground truth computed on a different set of vectors is wrong for this one.
    """
    rng = np.random.default_rng(seed)
    train = dataset.train
    test = dataset.test

    if n is not None and n < train.shape[0]:
        # Sorted so the subsample keeps the source file's row order, which
        # makes a given (dataset, n, seed) triple reproducible and inspectable.
        idx = np.sort(rng.choice(train.shape[0], size=n, replace=False))
        train = train[idx]
    if queries is not None and queries < test.shape[0]:
        idx = np.sort(rng.choice(test.shape[0], size=queries, replace=False))
        test = test[idx]

    return Dataset(
        name=dataset.name,
        train=np.ascontiguousarray(train),
        test=np.ascontiguousarray(test),
        metric=dataset.metric,
    )


def load(
    spec: str,
    n: int | None = None,
    queries: int | None = None,
    dim: int = 32,
    clusters: int = 100,
    spread: float = 0.08,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    seed: int = 0,
    download: bool = True,
) -> Dataset:
    """Resolve a dataset name from the CLI into a ready-to-use ``Dataset``.

    ``clustered`` and ``random`` are generated at the requested size; anything
    else is treated as an ANN-Benchmarks dataset name and subsampled.
    """
    if spec == "clustered":
        return make_clustered(
            n=n or 20_000,
            dim=dim,
            clusters=clusters,
            spread=spread,
            queries=queries or 500,
            seed=seed,
        )
    if spec == "random":
        return make_random(n=n or 20_000, dim=dim, queries=queries or 500, seed=seed)

    dataset = load_hdf5(spec, data_dir=data_dir, download=download)
    return subsample(dataset, n=n, queries=queries, seed=seed)
