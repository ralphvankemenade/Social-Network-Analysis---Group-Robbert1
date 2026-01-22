"""Central configuration for the DSS.

This module stores default values and constants used throughout the
application.  Centralising configuration makes it easier to adjust
parameters (such as default numbers of top nodes to show or perturbation
runs) without touching the analytic logic itself.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Defaults:
    """Container for default configuration values."""

    # Number of top nodes to highlight by default in centrality analyses
    top_n: int = 5

    # Alpha parameter for regret weighting in the arrest optimisation
    alpha: float = 1.0

    # Beta parameter for penalty weighting in the arrest optimisation
    beta: float = 1.0

    # Fraction of edges to remove in perturbation tests
    perturbation_fraction: float = 0.05

    # Number of perturbation runs
    perturbation_runs: int = 50

    # Default layout for network plots
    layout: str = "spring"

    # Spacing multiplier for network layouts (higher = more separation)
    layout_spacing: float = 4.0
    
    # Random seed for reproducibility
    seed: int = 55


DEFAULTS = Defaults()


if __name__ == "__main__":
    # Example usage: print the defaults
    import pprint

    pprint.pprint(DEFAULTS)
