
from fastcluster import linkage
from matplotlib import pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform, pdist


def seriation(Z, N, cur_index):
    """
    It computes the order implied by a hierarchical tree (dendrogram)
    :{"param_1 Z": a hierarchical tree (dendrogram)
    "param_2 N": the number of points given to the clustering process
     "param_3 cur_index": the position in the tree for the recursive traversal
    }
    :return: order implied by the hierarchical tree Z
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return seriation(Z, N, left) + seriation(Z, N, right)


def compute_serial_matrix(dist_matrix, method="ward"):
    """
    It transforms a distance matrix into a sorted distance matrix according to the order implied by the
    hierarchical tree (dendrogram)
    :{"param_1": dist_matrix: input a distance matrix to get a sorted one,
    "param_2": method: method = ["ward","single","average","complete"]
    }
    :{"return_1": seriated_dist: input dist_mat, but with re-ordered rows and columns according to the seriation,
    i.e. the order implied by the hierarchical tree
    "return_2": res_order: is the order implied by the hierarchical tree
    "return_3": res_linkage: is the hierarchical tree (dendrogram)
    }
    """
    N = len(dist_matrix)
    flat_dist_matrix = squareform(dist_matrix)
    res_linkage = linkage(flat_dist_matrix, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_matrix[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]
    return seriated_dist, res_order, res_linkage


def plot_heatmap(data_pca, data_tr):
    length = len(data_tr.data_clustering)
    dist_matrix = squareform(pdist(data_tr.data_clustering))
    plt.pcolormesh(dist_matrix, cmap="rainbow")
    plt.title("Distance matrix")
    plt.colorbar()
    plt.xlim([0, length])
    plt.ylim([0, length])
    plt.show()

    methods = ["ward", "single", "average", "complete"]
    for method in methods:
        print("Method:\t", method)
        ordered_dist_matrix, res_order, res_linkage = compute_serial_matrix(dist_matrix, method)
        plt.pcolormesh(ordered_dist_matrix, cmap="rainbow")
        plt.colorbar()
        plt.xlim([0, len(dist_matrix)])
        plt.ylim([0, len(dist_matrix)])
        plt.title(method)
        plt.show()


