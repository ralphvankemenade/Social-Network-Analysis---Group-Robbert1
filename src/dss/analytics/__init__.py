"""Analytic functions for the DSS.

Each submodule in this package implements a specific kind of analysis:

* `centrality` – compute various centrality measures and aggregate them.
* `roles` – build role similarity matrices and cluster nodes into roles.
* `communities` – detect communities and compute modularity.
* `robustness` – perform perturbation tests and calculate ARI/NMI.
* `kemeny` – compute the Kemeny constant and study its sensitivity.
* `arrest_optimization` – solve the balanced cut problem for arrest assignment.
"""

from . import centrality  # noqa: F401
from . import roles  # noqa: F401
from . import communities  # noqa: F401
from . import robustness  # noqa: F401
from . import kemeny  # noqa: F401
from . import arrest_optimization  # noqa: F401