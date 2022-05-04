import numpy as np
from fastcluster import linkage
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans


def seriation(Z, N, cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return seriation(Z, N, left) + seriation(Z, N, right)


def compute_serial_matrix(dist_matrix, method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
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
    # print("data_clustering:", data_tr.data_clustering)
    length = len(data_tr.data_clustering)
    dist_matrix = squareform(pdist(data_tr.data_clustering))
    plt.pcolormesh(dist_matrix)
    plt.colorbar()
    plt.xlim([0, length])
    plt.ylim([0, length])
    plt.show()
    # print("The distance matrix:", dist_matrix)
    label = KMeans(data_pca)
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(data_pca[label == i, 0], data_pca[label == i, 1], label=i)
        plt.legend()
        plt.show()
    # print(len(dist_matrix))
    methods = ["ward", "single", "average", "complete"]
    for method in methods:
        print("Method:\t", method)
        ordered_dist_matrix, res_order, res_linkage = compute_serial_matrix(dist_matrix, method)
        plt.pcolormesh(ordered_dist_matrix)
        plt.colorbar()
        plt.xlim([0, len(dist_matrix)])
        plt.ylim([0, len(dist_matrix)])
        plt.show()
        # print(squareform(dist_matrix))