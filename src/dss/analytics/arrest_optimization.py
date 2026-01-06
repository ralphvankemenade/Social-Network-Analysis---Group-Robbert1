"""Arrest assignment via integer linear programming and heuristics."""

from typing import Dict, Any, List, Optional, Tuple
import math
import networkx as nx
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatusOptimal, PULP_CBC_CMD, value

from ..types import ArrestAssignmentResult
from ..logging_config import get_logger

logger = get_logger(__name__)


def _compute_edge_weights(
    G: nx.Graph,
    communities: Dict[Any, int],
    centrality: Optional[pd.Series] = None,
    alpha: float = 1.0,
) -> Dict[Tuple[Any, Any], float]:
    """Compute weights for each edge based on community and centrality.

    Parameters
    ----------
    G: networkx.Graph
        The graph whose edges are weighted.
    communities: dict
        Mapping from node to community index.
    centrality: pandas.Series, optional
        Centrality scores for each node; if provided, weights are
        increased proportionally to the sum of centralities of the
        incident nodes.
    alpha: float, optional
        Strength of the regret term; controls the penalty for splitting
        community edges and for high‑centrality nodes across departments.
    """
    weights: Dict[Tuple[Any, Any], float] = {}
    # Scale centrality to [0, 1] if provided
    if centrality is not None:
        max_c = centrality.max()
        min_c = centrality.min()
        denom = max_c - min_c if max_c != min_c else 1.0
        scaled_c = {n: (centrality[n] - min_c) / denom for n in G.nodes()}
    else:
        scaled_c = {n: 0.0 for n in G.nodes()}
    for u, v in G.edges():
        w = 1.0
        # Increase weight if nodes are in same community
        if communities.get(u) == communities.get(v):
            w += alpha
        # Increase further by centrality contributions
        if centrality is not None:
            w += alpha * (scaled_c[u] + scaled_c[v])
        weights[(u, v)] = w
    return weights


def _solve_ilp(
    G: nx.Graph,
    weights: Dict[Tuple[Any, Any], float],
    capacity: int,
) -> Optional[ArrestAssignmentResult]:
    """Solve the balanced cut problem as an ILP.

    Returns an `ArrestAssignmentResult` if solved optimally; otherwise
    returns None.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    # Define ILP
    prob = LpProblem("arrest_assignment", LpMinimize)
    x = LpVariable.dicts("x", nodes, lowBound=0, upBound=1, cat="Binary")
    y = LpVariable.dicts("y", list(G.edges()), lowBound=0, upBound=1, cat="Binary")
    # Objective: sum w_ij y_ij
    prob += lpSum(weights[(u, v)] * y[(u, v)] for (u, v) in G.edges())
    # Capacity constraints: size of Dept1 <= capacity and Dept2 <= capacity
    prob += lpSum(x[node] for node in nodes) <= capacity
    prob += lpSum((1 - x[node]) for node in nodes) <= capacity
    # Linking constraints: y_ij >= x_i - x_j and >= x_j - x_i
    for (u, v) in G.edges():
        prob += y[(u, v)] >= x[u] - x[v]
        prob += y[(u, v)] >= x[v] - x[u]
    # Solve
    try:
        solver = PULP_CBC_CMD(msg=False)
        prob.solve(solver)
        if prob.status != LpStatusOptimal:
            logger.warning("ILP did not find an optimal solution.")
            return None
        # Extract assignment
        assignment = {node: int(x[node].value()) for node in nodes}
        cut_edges = sum(1 for (u, v) in G.edges() if assignment[u] != assignment[v])
        obj_val = value(prob.objective)
        risk_edges = [(u, v) for (u, v) in G.edges() if assignment[u] != assignment[v]]
        return ArrestAssignmentResult(
            assignment=assignment,
            objective=obj_val,
            cut_edges=cut_edges,
            effective_arrests=float(n - cut_edges),
            risk_edges=risk_edges,
        )
    except Exception as e:
        logger.warning(f"ILP solver error: {e}")
        return None


def _heuristic_assignment(
    G: nx.Graph,
    communities: Dict[Any, int],
    centrality: Optional[pd.Series],
    capacity: int,
) -> ArrestAssignmentResult:
    """Heuristic assignment when ILP fails.

    A simple heuristic assigns entire communities to departments until
    capacity is reached.  If a community does not fit entirely, its
    members are distributed by descending centrality.  This is a
    greedy approximation to the balanced cut problem.
    """
    nodes = list(G.nodes())
    assignment: Dict[Any, int] = {}
    dept_counts = {0: 0, 1: 0}
    # Group nodes by community
    community_nodes: Dict[int, List[Any]] = {}
    for node in nodes:
        community_nodes.setdefault(communities.get(node, -1), []).append(node)
    # Sort communities by size descending
    comms_sorted = sorted(community_nodes.items(), key=lambda x: len(x[1]), reverse=True)
    for comm, members in comms_sorted:
        # Determine which department has more capacity left
        dept = 0 if dept_counts[0] <= dept_counts[1] else 1
        space = capacity - dept_counts[dept]
        if space <= 0:
            # No space left; assign to other department if possible
            dept = 1 - dept
            space = capacity - dept_counts[dept]
        if len(members) <= space:
            for m in members:
                assignment[m] = dept
            dept_counts[dept] += len(members)
        else:
            # Assign highest centrality nodes first
            if centrality is not None:
                sorted_members = sorted(members, key=lambda n: centrality.get(n, 0.0), reverse=True)
            else:
                sorted_members = members
            for m in sorted_members:
                if dept_counts[dept] < capacity:
                    assignment[m] = dept
                    dept_counts[dept] += 1
                else:
                    assignment[m] = 1 - dept
                    dept_counts[1 - dept] += 1
    # Compute metrics
    cut_edges = sum(1 for (u, v) in G.edges() if assignment[u] != assignment[v])
    risk_edges = [(u, v) for (u, v) in G.edges() if assignment[u] != assignment[v]]
    n = len(nodes)
    return ArrestAssignmentResult(
        assignment=assignment,
        objective=float(cut_edges),
        cut_edges=cut_edges,
        effective_arrests=float(n - cut_edges),
        risk_edges=risk_edges,
    )


def arrest_assignment(
    G: nx.Graph,
    communities: Dict[Any, int],
    centrality: Optional[pd.Series] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> ArrestAssignmentResult:
    """Assign nodes to two departments subject to capacity and regret.

    Parameters
    ----------
    G: networkx.Graph
        The graph representing the network.
    communities: dict
        Community labels for each node; used to weight edges.
    centrality: pandas.Series, optional
        Centrality scores used to penalise splitting high‑centrality nodes.
    alpha: float, optional
        Strength of the regret for splitting within‑community edges and
        high‑centrality nodes.
    beta: float, optional
        Penalty for cross‑department edges when computing effective arrests.

    Returns
    -------
    ArrestAssignmentResult
        The department assignment, objective value, number of cut edges and
        estimated effective arrests.
    """
    n = G.number_of_nodes()
    capacity = math.ceil(n / 2)
    weights = _compute_edge_weights(G, communities, centrality, alpha)
    # Try exact ILP solution
    result = _solve_ilp(G, weights, capacity)
    if result is not None:
        # Adjust effective arrests using beta
        result.effective_arrests = float(n - beta * result.cut_edges)
        return result
    # Fallback to heuristic
    logger.warning("Falling back to heuristic arrest assignment.")
    result = _heuristic_assignment(G, communities, centrality, capacity)
    result.effective_arrests = float(n - beta * result.cut_edges)
    return result


if __name__ == "__main__":
    import networkx as nx
    G = nx.karate_club_graph()
    # Use Louvain communities as example
    from .communities import compute_communities
    comm_result = compute_communities(G, method="louvain")
    centralities = pd.Series(dict(nx.degree_centrality(G)))
    result = arrest_assignment(G, comm_result.labels, centralities, alpha=1.0, beta=1.0)
    print(result)