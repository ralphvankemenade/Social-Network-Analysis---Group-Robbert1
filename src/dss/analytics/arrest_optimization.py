"""Arrest assignment via integer linear programming and heuristics."""

from typing import Dict, Any, List, Optional, Tuple
import math
import networkx as nx
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatusOptimal, PULP_CBC_CMD, value,LpStatus

from ..types import ArrestAssignmentResult
from ..logging_config import get_logger

from collections import Counter

logger = get_logger(__name__)


def _compute_edge_weights(
    G: nx.Graph,
    communities: Dict[Any, int],
    centrality: Optional[pd.Series] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
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
    # if centrality is not None:
    max_c = centrality.max()
    min_c = centrality.min()
    denom = max_c - min_c if max_c != min_c else 1.0
    scaled_c = {n: (centrality[n] - min_c) / denom for n in G.nodes()}
    # else:
    #     scaled_c = {n: 0.0 for n in G.nodes()}

    for u, v in G.edges():
        w = 1.0
        # Increase weight if nodes are in same community
        if communities.get(u) == communities.get(v):
            w += alpha
        # Increase further by centrality contributions
        # if centrality is not None:
        # w += alpha * (scaled_c[u] + scaled_c[v])
        w += beta * (scaled_c[u] + scaled_c[v])
        weights[(u, v)] = w
    return weights

def compute_effective_arrests(n: int, risk_edges: List[Tuple[Any, Any]], weights: Dict[Tuple[Any, Any], float]) -> float:
    """Compute the estimated effective arrests based on split edges and their weights."""
    if not risk_edges:
        return float(n)
    max_weight = max(weights.values())
    reduction = sum(weights[edge] / max_weight for edge in risk_edges)
    return max(0.0, n - reduction)

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
        print("LP status:", LpStatus[prob.status])
        if prob.status != LpStatusOptimal:
            logger.warning("ILP did not find an optimal solution.")
            return None
        # Extract assignment
        assignment = {node: int(x[node].value()) for node in nodes}
        cut_edges = sum(1 for (u, v) in G.edges() if assignment[u] != assignment[v])
        obj_val = value(prob.objective)
        risk_edges = [(u, v) for (u, v) in G.edges() if assignment[u] != assignment[v]]
        effective_arrests = compute_effective_arrests(n, risk_edges, weights)
        return ArrestAssignmentResult(
            assignment=assignment,
            objective=obj_val,
            cut_edges=cut_edges,
            # effective_arrests=float(n - cut_edges),
            effective_arrests = effective_arrests,
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
    weights: Dict[Tuple[Any, Any], float],
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
            # if centrality is not None:
            sorted_members = sorted(members, key=lambda n: centrality.get(n, 0.0), reverse=True)
            # else:
            #     sorted_members = members
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
    effective_arrests = compute_effective_arrests(n, risk_edges, weights)
    return ArrestAssignmentResult(
        assignment=assignment,
        objective=float(cut_edges),
        cut_edges=cut_edges,
        # effective_arrests=float(n - cut_edges),
        effective_arrests = effective_arrests,
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
    weights = _compute_edge_weights(G, communities, centrality, alpha,beta)
    # Try exact ILP solution
    result = _solve_ilp(G, weights, capacity)
    if result is not None:
    #     # Adjust effective arrests using beta
    #     # result.effective_arrests = float(n - beta * result.cut_edges)
       
        return result
    # Fallback to heuristic
    logger.warning("Falling back to heuristic arrest assignment.")
    result = _heuristic_assignment(G, communities, centrality, capacity,weights)
    # result.effective_arrests = float(n - beta * result.cut_edges)
    return result

    
def compute_arrest_order(
    G: nx.Graph,
    assignment: Dict,
    centrality: pd.Series,
    risk_edges,
    gamma: float = 1.0,
) -> pd.DataFrame:
    """
    Compute a recommended arrest order based on centrality and risky edges.

    Parameters
    ----------
    G : networkx.Graph
        The network graph.
    assignment : dict
        Mapping node -> department (0 or 1).
    centrality : pandas.Series
        Centrality scores per node.
    risk_edges : list of tuples
        List of risky (cross-department) edges.
    gamma : float, optional
        Penalty weight for risky edges (default = 1.0).

    Returns
    -------
    pandas.DataFrame
        Table with arrest order and underlying scores.
    """

    # Count risky edges per node
    risk_counts = Counter()
    for u, v in risk_edges:
        risk_counts[u] += 1
        risk_counts[v] += 1

    # Normalize centrality to [0, 1]
    c_min, c_max = centrality.min(), centrality.max()
    denom = c_max - c_min if c_max != c_min else 1.0
    centrality_norm = (centrality - c_min) / denom

    # Compute priority score
    priority_score = {
        node: centrality_norm.get(node, 0.0) - gamma * risk_counts.get(node, 0)
        for node in G.nodes()
    }

    # Build arrest order table
    df = pd.DataFrame({
        "Node": list(G.nodes()),
        "Dept.": [assignment[node] + 1 for node in G.nodes()],
        "Centrality (norm.)": [centrality_norm.get(node, 0.0) for node in G.nodes()],
        "Risky edges": [risk_counts.get(node, 0) for node in G.nodes()],
        "Priority score": [priority_score[node] for node in G.nodes()],
    })

    df = df.sort_values("Priority score", ascending=False).reset_index(drop=True)
    df["Arrest order"] = df.index + 1

    return df



def simulate_sequential_arrests(
    G: nx.Graph,
    arrest_order_df: pd.DataFrame,
    risk_edges: List[Tuple],
) -> pd.DataFrame:
    """
    Simulate a sequential arrest process with information leakage.

    After arresting a member, all members connected via risky edges
    become unavailable for arrest.

    Parameters
    ----------
    G : networkx.Graph
        The network graph.
    arrest_order_df : pandas.DataFrame
        Table sorted by arrest priority (must contain 'Node').
    risk_edges : list of tuples
        Risky cross-department edges.

    Returns
    -------
    pandas.DataFrame
        Arrest simulation results with feasibility status.
    """

    # Build adjacency list for risky edges
    tipped_by = {}
    for u, v in risk_edges:
        tipped_by.setdefault(u, set()).add(v)
        tipped_by.setdefault(v, set()).add(u)

    arrested = set()
    unavailable = set()
    rows = []

    for _, row in arrest_order_df.iterrows():
        node = row["Node"]

        if node in unavailable:
            status = "Tipped"
        else:
            # Arrest succeeds
            status = "Arrested"
            arrested.add(node)

            # Tip connected nodes
            for tipped in tipped_by.get(node, []):
                if tipped not in arrested:
                    unavailable.add(tipped)

        rows.append({
            **row.to_dict(),
            "Status": status,
            "Tipped members": ", ".join(map(str, tipped_by.get(node, []))) if status == "Arrested" else ""
        })

    result_df = pd.DataFrame(rows)
    result_df["Arrest step"] = range(1, len(result_df) + 1)

    return result_df



if __name__ == "__main__":
    import networkx as nx
    G = nx.karate_club_graph()
    # Use Louvain communities as example
    from .communities import compute_communities
    comm_result = compute_communities(G, method="louvain")
    centralities = pd.Series(dict(nx.degree_centrality(G)))
    result = arrest_assignment(G, comm_result.labels, centralities, alpha=1.0, beta=1.0)
    print(result)