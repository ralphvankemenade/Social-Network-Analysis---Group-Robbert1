"""Graph construction and basic analysis.

This package contains functions to build NetworkX graphs from adjacency
matrices and compute basic statistics.  These are low‑level operations
used by the analytics layer.
"""

from . import build_graph  # noqa: F401
from . import stats  # noqa: F401
from . import layouts  # noqa: F401