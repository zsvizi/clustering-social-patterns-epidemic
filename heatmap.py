
from fastcluster import linkage
from matplotlib import pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import manhattan_distances
from scipy.spatial.distance import squareform
import seaborn as sns
import pandas as pd


class Hierarchical:
    def __init__(self, data_clustering, data_transformer, country_names):
        self.data_tr = data_transformer
        self.data_clustering = data_clustering
        self.country_names = country_names
        self.data = None

    def get_euclidean_distance(self) -> np.array:
        """
        Calculates euclidean distance of a 39 * 136 matrix and returns 39*39 distance matrix
        :return matrix: square distance matrix with zero diagonals
        """
        # convert the data into dataframe
        data = pd.DataFrame(self.data_tr.data_clustering)
        # replace the indexes of the data with the country names
        data.index = self.country_names
        # rename the columns and rows of the euclidean_distance with country names and return a matrix distance
        columns = data.index
        rows = data.index
        #  calculate the euclidean distance
        pairwise_euc = data.apply(lambda row: [np.linalg.norm(row.values - data.loc[[i], :].values, 2)
                                               for i in data.index.values], axis=1)
        # Reformatting the above into readable format
        euc_distance = pd.DataFrame(
            data=pairwise_euc.values.tolist(),
            columns=data.index.tolist(),  # convert pandas DataFrame Column to List
            index=data.index.tolist())  # function return a list of the values.
        pd.DataFrame(euc_distance, index=rows, columns=columns)  # rename rows and columns
        return euc_distance

    def get_manhattan_distance(self):
        """
        Calculates Manhattan distance of a 39 * 136 matrix and returns 39*39 distance matrix
        :return matrix: square distance matrix with zero diagonals
        """
        self.data = pd.DataFrame(self.data_tr.data_clustering)
        country_names = self.country_names
        self.data.index = self.country_names
        manhattan = self.data.apply(lambda row: [np.sum(abs(row.values - self.data.loc[[i], :].values))
                                                 for i in self.data.index.values], axis=1)  # along columns
        # lambda function in python is a small anonymous function that can take any number of arguments and
        # execute an expression.
        # Reformatting the above into readable format
        manhattan_distance = pd.DataFrame(
            data=manhattan.values.tolist(),
            columns=self.data.index.tolist(),
            index=self.data.index.tolist())  # function return a list of the values
        pd.DataFrame(manhattan_distance, index=country_names, columns=country_names)
        return manhattan_distance

    def seriation(self, z, n, cur_index):
        """
              It computes the order implied by a hierarchical tree (dendrogram)
              :{"param_1 z": a hierarchical tree (dendrogram)
              "param_2 n": the number of points given to the clustering process
              "param_3 cur_index": the position in the tree for the recursive traversal
              }
              :return: order implied by the hierarchical tree z
              """
        if cur_index < n:
            return [cur_index]
        else:
            left = int(z[cur_index - n, 0])
            right = int(z[cur_index - n, 1])
            return (self.seriation(z, n, left) +
                    self.seriation(z, n, right))

    def compute_serial_matrix(self, manhattan_distance, method="ward"):
        """
                It transforms a distance matrix into a sorted distance matrix according to the order implied by the
                hierarchical tree (dendrogram)
                 :{"param_1": dist_matrix: input a distance matrix to get a sorted one by method
                "param_2": method: method = ["ward","single","average","complete"]
                 }
                 :{"return_1": seriated_dist: input country dist_mat, but with re-ordered rows and columns
                 according to the seriation
                 i.e. the order implied by the hierarchical tree
                "return_2": res_order: is the order implied by the hierarchical tree
                "return_3": res_linkage: is the hierarchical tree (dendrogram)
                }
                """
        n = len(manhattan_distance)
        flat_dist_matrix = squareform(manhattan_distance)
        res_linkage = linkage(flat_dist_matrix, method=method)
        res_order = self.seriation(res_linkage, n, n + n - 2)
        seriated_dist = np.zeros((n, n))
        a, b = np.triu_indices(n, k=1)
        seriated_dist[a, b] = manhattan_distance[[res_order[i] for i in a], [res_order[j] for j in b]]
        seriated_dist[b, a] = seriated_dist[a, b]
        return seriated_dist, res_order, res_linkage

    def plot_distances(self):
        manhattan_distance = self.get_manhattan_distance()
        self.country_names = self.data_tr.country_names
        plt.figure(figsize=(20, 15))
        plt.xticks(ticks=np.arange(len(self.country_names)),
                   labels=self.country_names,
                   rotation=90)
        plt.yticks(ticks=np.arange(len(self.country_names)),
                   labels=self.country_names,
                   rotation=0)
        plt.title("Distances between countries based on social contacts")
        az = plt.imshow(manhattan_distance, cmap='nipy_spectral',
                        interpolation="nearest",
                        vmin=0, vmax=2.2)
        plt.colorbar(az,
                     ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4,
                            1.6, 1.8, 2.0, 2.2])

    def plot_dendrogram(self):
        manhattan_distance = self.get_manhattan_distance()
        country_names = self.data_tr.country_names
        plt.figure(figsize=(26, 15))
        sch.dendrogram(sch.linkage(manhattan_distance.to_numpy()),
                       color_threshold=1.0,
                       leaf_rotation=90,
                       leaf_font_size=12,
                       show_leaf_counts=True,
                       labels=country_names,
                       above_threshold_color='blue',
                       orientation="top",
                       distance_sort='descending')
        plt.axhline(y=1.0, c='red', lw=1, linestyle="dashed")
        plt.title('Hierarchical Clustering Dendrogram')
        plt.ylabel('Social contact Distance between countries')

    def plot_ordered_distance(self):
        country_names = self.data_tr.country_names
        manhattan_distancess = manhattan_distances(self.data_tr.data_clustering)
        #manhattan_distance = self.get_manhattan_distance()
        methods = ["single", "complete", "average", "ward"]
        for method in methods:
            print("Method:\t", method)
            ordered_dist_mat, res_order, res_linkage = self.compute_serial_matrix(manhattan_distancess, method)
            plt.figure(figsize=(18, 15))
            plt.title("Distances between countries based on social contacts: method: {}".format(method))
            az = plt.imshow(ordered_dist_mat, cmap='rainbow',
                            alpha=.9, interpolation="nearest", vmin=0, vmax=2.2)
            plt.xticks(ticks=res_order, labels=country_names, rotation=90)
            plt.yticks(ticks=res_order, labels=country_names, rotation=0)
            plt.colorbar(az,
                         ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2])

            #  Dendrogram

            plt.figure(figsize=(25, 10))
            sch.dendrogram(sch.linkage(ordered_dist_mat),
                           color_threshold=0.8,
                           leaf_rotation=90,
                           leaf_font_size=12,
                           show_leaf_counts=True,
                           labels=country_names,
                           above_threshold_color='blue',
                           orientation="top",
                           distance_sort='descending')

            plt.title('Hierarchical Clustering Dendrogram: {}'.format(method))
            plt.ylabel('Social contact Distance between countries')
            plt.axhline(y=1.0, c='red', lw=1, linestyle="dashed")

    def plot_correlation(self):
        country_distance = self.get_manhattan_distance()
        country_names = self.data_tr.country_names
        plt.figure(figsize=(16, 12), frameon=False)
        sns.heatmap(np.corrcoef(country_distance), annot=True, cmap="nipy_spectral", linewidths=0.5)
        res_order = self.compute_serial_matrix(country_distance, method="complete")
        plt.xticks(ticks=res_order, labels=country_names, rotation=90)
        plt.yticks(ticks=res_order, labels=country_names, rotation=0)
        plt.title('Correlation between countries social contact distance')
        plt.show()
