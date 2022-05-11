
from matplotlib import pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform, pdist
import seaborn as sns
import pandas as pd


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


def compute_serial_matrix(country_distance, method=""):
    """
    It transforms a distance matrix into a sorted distance matrix according to the order implied by the
    hierarchical tree (dendrogram)
    :{"param_1": dist_matrix: input a distance matrix to get a sorted one by method
    "param_2": method: method = ["ward","single","average","complete"]
    }
    :{"return_1": seriated_dist: input country dist_mat, but with re-ordered rows and columns according to the seriation
    i.e. the order implied by the hierarchical tree
    "return_2": res_order: is the order implied by the hierarchical tree
    "return_3": res_linkage: is the hierarchical tree (dendrogram)
    }
    """
    N = len(country_distance)
    flat_dist_matrix = squareform(country_distance)
    res_linkage = sch.linkage(flat_dist_matrix, method=method)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1, m=N)
    seriated_dist[a, b] = country_distance[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]
    return seriated_dist, res_order, res_linkage


def plot_heatmap(data_tr):
    country_names = data_tr.country_names
    data = pd.DataFrame(data_tr.data_clustering)
    data.index = country_names
    country_distance = squareform(pdist(data))
    columns = data.index
    rows = data.index
    dist = pd.DataFrame(country_distance, index=rows, columns=columns)

    plt.figure(figsize=(12, 10))
    plt.xticks(ticks=np.arange(len(country_names)), labels=country_names, rotation=90)
    plt.yticks(ticks=np.arange(len(country_names)), labels=country_names, rotation=0)
    plt.title("Countries by unordered distances")
    plt.xlabel("Europe countries")
    plt.ylabel("Europe countries")
    az = plt.imshow(country_distance, cmap="rainbow", interpolation="nearest", vmin=0, vmax=0.45)
    plt.colorbar(az)

    country_distance = squareform(pdist(data))
    plt.figure(figsize=(14, 8))
    sns.heatmap(dist, annot=True, cmap="YlGnBu")
    plt.title("Heatmap Distance between countries")
    plt.show()

    #  Dendrogram
    plt.figure(figsize=(24, 12))
    sch.dendrogram(sch.linkage(country_distance), labels=country_names)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Countries listed')
    plt.ylabel('Dissimilarity between countries')

    methods = ["complete", "single", "ward", "average"]
    for method in methods:
        print("Method:\t", method)

        ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(country_distance, method)
        plt.figure(figsize=(16, 10))
        plt.xticks(ticks=res_order, labels=country_names, rotation=45)
        plt.yticks(ticks=res_order, labels=country_names, rotation=0)
        plt.title("Countries by distance")
        plt.xlabel("European countries")
        plt.ylabel("European countries")
        plt.title('Ordering method: {}'.format(method))
        ax = plt.imshow(ordered_dist_mat, cmap="rainbow", interpolation="nearest", vmin=0, vmax=0.4)
        plt.colorbar(ax)

        #  Heatmap
        #ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(country_distance, method)
        #sns.heatmap(ordered_dist_mat, annot=True, cmap="YlGnBu", xticklabels=True, yticklabels=True)
        #plt.figure(figsize=(14, 8))
        #plt.xticks(ticks=res_order, labels=country_names, rotation=90)
        #plt.yticks(ticks=res_order, labels=country_names, rotation=0)
        #plt.title('Heatmap Ordering method {}'.format(method))
        #plt.ylabel('Countries')
        #plt.xlabel('Countries')
        #plt.show()

        #  Dendrogram
        plt.figure(figsize=(25, 12))
        sch.dendrogram(sch.linkage(ordered_dist_mat), labels=country_names)
        plt.title('Hierarchical Clustering Dendrogram: {}'.format(method))
        plt.xlabel('Countries listed')
        plt.ylabel('Dissimilarity between countries')

        # correlation of the countries
        plt.figure(figsize=(16, 12))
        sns.heatmap(np.corrcoef(country_distance), annot=True, cmap="YlGnBu")
        plt.xticks(ticks=res_order, labels=country_names, rotation=90)
        plt.yticks(ticks=res_order, labels=country_names, rotation=0)
        plt.title('Heatmap Correlation between countries {}'.format(method))
        plt.show()














