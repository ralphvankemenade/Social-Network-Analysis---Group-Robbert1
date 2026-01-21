"""Role similarity and clustering based on Cooper & Barahona (2010).

This module constructs structural signatures for each node in a network
and computes pairwise similarities.  It then clusters nodes into roles
based on these similarities using either spectral or hierarchical
clustering.  A summary of each role cluster is provided in terms of
basic centrality statistics.
"""

# typing.List is needed for type annotations of lists of labels returned by clustering
from typing import Dict, Any, Optional, Tuple, Callable, List
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import time



from ..types import RoleResult
from .centrality import compute_centralities
from ..logging_config import get_logger

from graphrole.roles.extract import RoleExtractor

logger = get_logger(__name__)


def _k_hop_signature(G: nx.Graph, k: int = 3) -> np.ndarray:
    """Compute k‑hop degree distribution signatures for each node.

    For each node, a vector of length `k` is created where the `i`‑th
    element counts the number of neighbours at distance exactly `i+1`.

    Parameters
    ----------
    G: networkx.Graph
        The graph to analyse.
    k: int, optional
        The maximum hop distance to include; defaults to 3.

    Returns
    -------
    numpy.ndarray
        An array of shape (n_nodes, k) containing the signatures.
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    index_map = {node: idx for idx, node in enumerate(nodes)}
    signatures = np.zeros((n, k), dtype=float)
    # Precompute shortest path lengths
    lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff=k))
    for node, dist_dict in lengths.items():
        row = index_map[node]
        for target, dist in dist_dict.items():
            if dist == 0 or dist > k:
                continue
            signatures[row, dist - 1] += 1
    return signatures


def _random_walk_profiles(G: nx.Graph, t: int = 3) -> np.ndarray:
    """Compute random‑walk probability profiles after `t` steps for each node.

    The transition matrix `P` is built by normalising rows of the adjacency
    matrix.  The `t`‑step transition probabilities are obtained from `P^t`.

    Parameters
    ----------
    G: networkx.Graph
        The graph to analyse.
    t: int, optional
        Number of steps for the random walk; defaults to 3.

    Returns
    -------
    numpy.ndarray
        An array of shape (n_nodes, n_nodes) where each row `i` contains the
        probability distribution of being at each node after `t` steps starting
        from node `i`.
    """
    # Build transition matrix
    nodes = list(G.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    P = np.zeros((n, n), dtype=float)
    for i, u in enumerate(nodes):
        deg = G.degree(u)
        if deg > 0:
            for v in G.neighbors(u):
                j = idx_map[v]
                P[i, j] = 1.0 / deg
    # Compute P^t
    Pt = np.linalg.matrix_power(P, t)
    return Pt


def _compute_similarity_matrix(
    features: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute a similarity matrix from feature vectors.

    Parameters
    ----------
    features: ndarray
        Array of shape (n_samples, n_features) containing feature vectors.
    metric: str, optional
        Similarity measure: "cosine" or "correlation".

    Returns
    -------
    ndarray
        A square matrix of pairwise similarities in the range [0, 1].
    """
    # Compute pairwise distances; convert to similarity
    if metric == "cosine":
        distances = pairwise_distances(features, metric="cosine")
    elif metric == "correlation":
        distances = pairwise_distances(features, metric="correlation")
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")
    # Similarity = 1 - distance, but clip to [0, 1]
    sim = 1.0 - distances
    sim = np.clip(sim, 0.0, 1.0)
    return sim


def _cluster_similarity_matrix(
    similarity: np.ndarray,
    n_clusters: Optional[int] = None,
    method: str = "spectral",
) -> List[int]:
    """Cluster a similarity matrix and return labels.

    Parameters
    ----------
    similarity: ndarray
        Square similarity matrix.
    n_clusters: int, optional
        Desired number of clusters.  If None, defaults to the square root
        of the number of nodes rounded up.
    method: str, optional
        "spectral" for SpectralClustering or "hierarchical" for
        AgglomerativeClustering.

    Returns
    -------
    list of int
        Cluster labels for each sample.
    """
    n = similarity.shape[0]
    if n_clusters is None:
        n_clusters = max(2, int(np.ceil(np.sqrt(n))))
    if method == "spectral":
        # Spectral clustering expects an affinity (similarity) matrix
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=0,
        )
        labels = sc.fit_predict(similarity)
    elif method == "hierarchical":
        try:
            hc = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="precomputed",
                linkage="average",
            )
        except TypeError:
            hc = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                linkage="average",
            )
        labels = hc.fit_predict(1.0 - similarity)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    return labels.tolist()


def compute_cooperbarahona(
    G: nx.Graph,
    signature: str = "k-hop",
    k: int = 3,
    t: int = 3,
    similarity_metric: str = "cosine",
    clustering_method: str = "spectral",
    n_clusters: Optional[int] = None,
    centrality_table: Optional[pd.DataFrame] = None,
) -> RoleResult:
    """Compute role similarity, cluster roles and summarise clusters.

    Parameters
    ----------
    G: networkx.Graph
        Graph on which to compute roles.
    signature: str, optional
        Type of structural signature: "k-hop" or "random-walk".
    k: int, optional
        Maximum hop for k-hop signatures; ignored for random-walk.
    t: int, optional
        Number of steps for random-walk profiles; ignored for k-hop.
    similarity_metric: str, optional
        Similarity metric: "cosine" or "correlation".
    clustering_method: str, optional
        Clustering algorithm: "spectral" or "hierarchical".
    n_clusters: int, optional
        Desired number of clusters; if None, uses sqrt heuristic.
    centrality_table: pandas.DataFrame, optional
        Precomputed centralities; if provided, cluster summaries will
        include mean centralities per role.

    Returns
    -------
    RoleResult
        A container with the similarity matrix, cluster labels and summary.
    """
    nodes = list(G.nodes())
    if signature == "k-hop":
        features = _k_hop_signature(G, k)
    elif signature == "random-walk":
        features = _random_walk_profiles(G, t)
    else:
        raise ValueError(f"Unsupported signature type: {signature}")
    similarity = _compute_similarity_matrix(features, metric=similarity_metric)
    labels_list = _cluster_similarity_matrix(similarity, n_clusters, method=clustering_method)
    # Build labels dictionary keyed by node
    labels = {node: labels_list[i] for i, node in enumerate(nodes)}
    print('labels:', labels)
    # Compute summary statistics per role
    if centrality_table is None:
        centrality_table = compute_centralities(G)
    df = centrality_table.copy()
    df["role"] = [labels[node] for node in nodes]
    summary = df.groupby("role").mean()
    summary["size"] = df.groupby("role").size()
    # Create RoleResult
    return RoleResult(similarity_matrix=similarity, labels=labels, summary=summary)

def createadjmatrix(Ni,Nj,R):
    m = len(Ni)
    n = len(Nj)
    tempm1 = np.zeros((m,n))
    tempm2 = np.zeros((m,m))
    tempm3 = np.zeros((n,n))

    for i in range(len(Ni)):
        for j in range(len(Nj)):
            tempm1[i,j] = R[Ni[i],Nj[j]]

    tempfinalm = np.concatenate((np.concatenate((tempm2,tempm1.T)),np.concatenate((tempm1,tempm3))),axis=1)
    return tempfinalm

def rolesim_calc(G, beta=0.10,maxiter=100):
    start = time.perf_counter()
    n = G.number_of_nodes()
    convtol = beta/100

    #Initialize scores
    scores0 = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if G.degree[i]==G.degree[j]:
                scores0[i,j] = 1
                
    scores = {}
    scores[0] = scores0
    for k in range(1,maxiter):
        stemp = np.zeros((n,n))
        sprev = scores[k-1]
        for i in range(n):
            for j in range(n):
                Ni = list(G[i])
                Nj = list(G[j])
                Nil = len(Ni)
                Njl = len(Nj)
                Ml = np.min([Nil,Njl])
                adjmtemp = createadjmatrix(Ni,Nj,sprev)
                Gtemp = nx.from_numpy_array(adjmtemp)
                M = list(nx.max_weight_matching(Gtemp))
                '''
                if Ml != len(M):
                    print(i,Ni,Nil)
                    print(j,Nj,Njl)
                    print(Ml,len(M))
                    print(M)    
                '''    
                temp = 0
                for edge in M:
                    temp += adjmtemp[edge[0],edge[1]]
                stemp[i,j] = (1-beta)*(temp/(Nil+Njl-Ml))+beta
        scores[k] = stemp
        tempm = np.abs(stemp-sprev)

        if all(x < convtol for x in tempm.flatten()):
            print(k,'Stop, convergence reached')
            break
        if k % 5 == 0:
            print(f'{k} iterations computed, not yet converged')
            end = time.perf_counter()
            print(k,end-start)
            print(np.max(tempm.flatten()))
        
    return scores[(len(scores)-1)]

def rolesim_star_calc(G, beta=0.10,lambd=0.70,maxiter=100):
    start = time.perf_counter()
    n = G.number_of_nodes()
    convtol = beta/100

    #Initialize scores
    scores0 = np.ones((n,n))
    for i in range(n):
        for j in range(n):
            if G.degree[i]==G.degree[j]:
                scores0[i,j] = 1
                
    scores = {}
    scores[0] = scores0
    for k in range(1,maxiter+1):
        stemp = np.zeros((n,n))
        sprev = scores[k-1]
        for i in range(n):
            for j in range(n):
                Ni = list(G[i])
                Nj = list(G[j])
                Nil = len(Ni)
                Njl = len(Nj)
                Ml = np.min([Nil,Njl])
                adjmtemp = createadjmatrix(Ni,Nj,sprev)
                Gtemp = nx.from_numpy_array(adjmtemp)
                M = list(nx.max_weight_matching(Gtemp))
                '''
                if Ml != len(M):
                    print(i,Ni,Nil)
                    print(j,Nj,Njl)
                    print(Ml,len(M))
                    print(M)    
                '''    
                temp = 0
                #For all edges inside the matching
                for edge in M:
                    temp += adjmtemp[edge[0],edge[1]]
                    
                #For all edges (pairs) not in the matching
                temp2 = 0

                M2 = [x[::-1] for x in M]
                Mtot = M + M2
                for edge in Gtemp.edges()-Mtot:
                    if (Nil*Njl-Ml)==0:
                        print(i,j,Ni,Nj)
                        print(Gtemp.edges())
                        print(M)
                        print(Gtemp.edges()-M)
                        print(edge)
                        
                    temp2 += adjmtemp[edge[0],edge[1]]

                '''
                if (Nil*Njl-Ml)==0:
                    print(Nil,Njl,Ml)
                    print((Nil*Njl-Ml))
                    print(temp2)
                '''
                if (Nil*Njl-Ml)==0:
                    stemp[i,j] = (1-beta)*(lambd*(temp/(Nil+Njl-Ml))+(1-lambd)*(1))+beta
                #elif i==j:
                #    stemp[i,j] = (1-beta)*(lambd*(temp/(Nil+Njl-Ml))+(1-lambd)*(1))+beta
                else:
                    stemp[i,j] = (1-beta)*(lambd*(temp/(Nil+Njl-Ml))+(1-lambd)*(temp2/(Nil*Njl-Ml)))+beta
        scores[k] = stemp
        tempm = np.abs(stemp-sprev)

        if all(x < convtol for x in tempm.flatten()):
            print(k,'Stop, convergence reached')
            break
        if k % 5 == 0:
            print(f'{k} iterations computed, not yet converged')
            end = time.perf_counter()
            print(k,end-start)
            print(np.max(tempm.flatten()))
        
    return scores[(len(scores)-1)]

def compute_rolesim(G,beta,maxiter,clustering_method,n_roles,centrality_table):
    nodes = list(G.nodes())
    similarity = rolesim_calc(G,beta,maxiter)
    labels_list = _cluster_similarity_matrix(similarity, n_roles, method=clustering_method)
    # Build labels dictionary keyed by node
    labels = {node: labels_list[i] for i, node in enumerate(nodes)}
    # Compute summary statistics per role
    if centrality_table is None:
        centrality_table = compute_centralities(G)
    df = centrality_table.copy()
    df["role"] = [labels[node] for node in nodes]
    summary = df.groupby("role").mean()
    summary["size"] = df.groupby("role").size()
    # Create RoleResult
    return RoleResult(similarity_matrix=similarity, labels=labels, summary=summary)

def compute_rolesim_star(G,beta,lambd,maxiter,clustering_method,n_roles,centrality_table):
    nodes = list(G.nodes())
    similarity = rolesim_star_calc(G,beta,lambd,maxiter)
    labels_list = _cluster_similarity_matrix(similarity, n_roles, method=clustering_method)
    # Build labels dictionary keyed by node
    labels = {node: labels_list[i] for i, node in enumerate(nodes)}
    # Compute summary statistics per role
    if centrality_table is None:
        centrality_table = compute_centralities(G)
    df = centrality_table.copy()
    df["role"] = [labels[node] for node in nodes]
    summary = df.groupby("role").mean()
    summary["size"] = df.groupby("role").size()
    # Create RoleResult
    return RoleResult(similarity_matrix=similarity, labels=labels, summary=summary)

def transform_roles(roles,n_roles):
    if n_roles == None:
        n_roles = 0
        for i in range(100):
            temp = False
            for key in roles.keys():
                if roles[key] == f'role_{i}':
                    if i+1 > n_roles:
                        n_roles = i+1
                        temp = True

    newroles = {}
    for key in roles.keys():
        for i in range(n_roles):
            if roles[key] == f'role_{i}':
                newroles[key] = i
    return newroles

def compute_rolx(G,n_roles,centrality_table):
    nodes = list(G.nodes())
    role_extractor = RoleExtractor(n_roles=n_roles)
    role_extractor.extract_role_factors(centrality_table)
    labels = transform_roles(role_extractor.roles,n_roles)
    # Compute summary statistics per role
    if centrality_table is None:
        centrality_table = compute_centralities(G)
    df = centrality_table.copy()
    df["role"] = [labels[node] for node in nodes]
    summary = df.groupby("role").mean()
    summary["size"] = df.groupby("role").size()
    # Create RoleResult
    return RoleResult(similarity_matrix=None, labels=labels, summary=summary)

def leaderranking(df):
    sizes = df['size']
    df = df.drop('size',axis=1)
    maxima = df.max()
    points = [10,10,10,10,10,10]
    scores = np.zeros(df.shape)
    for i in range(len(scores[0,:])):
        for j in range(len(scores[:,0])):
            scores[j,i] = points[i]*(df.iloc[j,i]/maxima.iloc[i])
    return pd.DataFrame(scores.T).mean()

def compute_roles(G,method,centralities,info):
    if method == "Cooper and Barahona":
        return compute_cooperbarahona(G,info["signature"],info["k"],info["t"],
                                      info["similarity_metric"],info["clustering_method"],
                                      info["n_roles"],centralities)
    elif method == "RoleSim":
        return compute_rolesim(G,info["beta"],info["maxiter"],info["clustering_method"],
                               info["n_roles"],centralities)
    elif method == "RoleSim*":
        return compute_rolesim_star(G,info["beta"],info["lambd"],info["maxiter"],info["clustering_method"],
                               info["n_roles"],centralities)
    elif method == "RolX":
        return compute_rolx(G,info["n_roles"],centralities)
    else:
        return KeyError('Not possible')
 

if __name__ == "__main__":
    import networkx as nx
    G = nx.karate_club_graph()
    result = compute_roles(G, signature="k-hop", k=2, similarity_metric="cosine", clustering_method="spectral", n_clusters=4)
    print("Role labels:", result.labels)
    print("Summary:")
    print(result.summary)
