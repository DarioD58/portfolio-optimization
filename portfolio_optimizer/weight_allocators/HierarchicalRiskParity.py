import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from collections import deque
import seaborn as sns
import matplotlib.pyplot as plt
from portfolio_optimizer.weight_allocators.SimpleAllocator import SimpleAllocator
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)


class HierarchicalRiskParity(SimpleAllocator):
    def __init__(self) -> None:
        self.tag = f"HRP"

    def _hierachical_tree_clustering(self) -> None:
        self.corr_dist_matrix = (0.5 * (1 - self.corr_matrix))**0.5
        #np.fill_diagonal(self.corr_dist_matrix, 0)
        linkage_matrix = linkage(self.corr_dist_matrix, 'single', 'euclidean')

        self.linkage_matrix = linkage_matrix

    
    def _seriate_matrix(self) -> None:
        sorted_elems = []
        size = int(self.linkage_matrix[-1, 3])

        stack = deque([int(self.linkage_matrix[-1, 1]), int(self.linkage_matrix[-1, 0])])
        while stack:
            curr_node = stack.pop()

            loc = curr_node - size

            if loc < 0:
                sorted_elems.append(curr_node)
            else:
                stack.append(int(self.linkage_matrix[loc, 1]))
                stack.append(int(self.linkage_matrix[loc, 0]))
        
        self.sorted_elems = sorted_elems

    def _cluster_variance(self, cluster: list) -> float:
        cov_slice = self.cov_matrix[cluster, :][:, cluster]

        ivp = 1/np.diag(cov_slice)
        w = ivp / ivp.sum()

        cluster_var = np.dot(np.dot(w.T, cov_slice), w)

        return cluster_var


    def _recursive_bisection(self) -> None:
        weights = []

        elems_stack = deque(
            [self.sorted_elems[len(self.sorted_elems)//2:len(self.sorted_elems)], 
            self.sorted_elems[0:len(self.sorted_elems)//2]]
        )

        weights_stack = deque([1])

        while elems_stack:
            left_node = elems_stack.pop()
            right_node = elems_stack.pop()

            curr_weight = weights_stack.pop()

            left_cluster_var = self._cluster_variance(left_node)
            right_cluster_var = self._cluster_variance(right_node)

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

        self.weights_allocation = weights

    def fit(self, x: pd.DataFrame):
        self.cov_matrix = x.cov().to_numpy()
        self.corr_matrix = x.corr().to_numpy()

        self._hierachical_tree_clustering()
        self._seriate_matrix()
        self._recursive_bisection()

        final_result = {x.columns[i]: round(j, 5) for i, j in zip(self.sorted_elems, self.weights_allocation)}

        return dict(sorted(final_result.items()))

        