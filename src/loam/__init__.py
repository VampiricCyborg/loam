"""Loam: a readable, instrumented HNSW vector index in pure Python.

The public surface is small on purpose:

``FlatIndex``
    Exact brute-force k-NN. The oracle every recall number is checked against.
``HNSWIndex``
    The approximate index, implementing the paper's Algorithms 1 to 5.
``QueryStats``
    The work a single query did: distance computations, nodes visited, hops
    per layer.
"""

from .distance import Metric
from .flat import FlatIndex
from .hnsw import HNSWIndex
from .stats import LayerTrace, QueryStats, SearchTrace
from .store import VectorStore

__version__ = "0.1.0"

__all__ = [
    "FlatIndex",
    "HNSWIndex",
    "LayerTrace",
    "Metric",
    "QueryStats",
    "SearchTrace",
    "VectorStore",
    "__version__",
]
