import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from collections import deque
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)

def _hierachical_tree_clustering(corr_matrix: np.ndarray) -> None:
    corr_dist_matrix = (0.5 * (1 - corr_matrix))**0.5
    #np.fill_diagonal(corr_dist_matrix, 0)
    linkage_matrix = linkage(corr_dist_matrix, 'single', 'euclidean')

    return linkage_matrix


def _seriate_matrix(linkage_matrix: np.ndarray) -> None:
    sorted_elems = []
    size = int(linkage_matrix[-1, 3])

    stack = deque([int(linkage_matrix[-1, 1]), int(linkage_matrix[-1, 0])])
    while stack:
        curr_node = stack.pop()

        loc = curr_node - size

        if loc < 0:
            sorted_elems.append(curr_node)
        else:
            stack.append(int(linkage_matrix[loc, 1]))
            stack.append(int(linkage_matrix[loc, 0]))
    
    return sorted_elems

def _cluster_variance(cluster: list, cov_matrix: np.ndarray) -> float:
    cov_slice = cov_matrix[cluster, :][:, cluster]

    ivp = 1/np.diag(cov_slice)
    w = ivp / ivp.sum()

    cluster_var = np.dot(np.dot(w.T, cov_slice), w)

    return cluster_var


def _recursive_bisection(sorted_elems: list, cov_matrix: np.ndarray) -> None:
    weights = []

    elems_stack = deque(
        [sorted_elems[len(sorted_elems)//2:len(sorted_elems)], 
        sorted_elems[0:len(sorted_elems)//2]]
    )

    weights_stack = deque([1])

    while elems_stack:
        left_node = elems_stack.pop()
        right_node = elems_stack.pop()

        curr_weight = weights_stack.pop()

        left_cluster_var = _cluster_variance(left_node, cov_matrix)
        right_cluster_var = _cluster_variance(right_node, cov_matrix)

        alpha = 1 - left_cluster_var/(left_cluster_var + right_cluster_var)

        if len(right_node) > 1:
            elems_stack.append(right_node[len(right_node)//2:len(right_node)])
            elems_stack.append(right_node[0:len(right_node)//2])
            weights_stack.append(curr_weight * (1 - alpha))
        
        if len(left_node) > 1:
            elems_stack.append(left_node[len(left_node)//2:len(left_node)])
            elems_stack.append(left_node[0:len(left_node)//2])
            weights_stack.append(curr_weight * (alpha))

        if len(left_node) == 1:
            weights.append(curr_weight * (alpha))
        
        if len(right_node) == 1:
            weights.append(curr_weight * (1 - alpha))

    return weights

def hierarchical_risk_parity(x: pd.DataFrame):
    cov_matrix = x.cov().to_numpy()
    corr_matrix = x.corr().to_numpy()

    linkage_matrix = _hierachical_tree_clustering(corr_matrix=corr_matrix)
    sorted_elems = _seriate_matrix(linkage_matrix=linkage_matrix)
    weights_allocation = _recursive_bisection(sorted_elems=sorted_elems, cov_matrix=cov_matrix)

    final_result = {x.columns[i]: round(j, 5) for i, j in zip(sorted_elems, weights_allocation)}

    return dict(sorted(final_result.items()))
