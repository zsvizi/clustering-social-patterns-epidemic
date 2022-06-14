
from scipy.cluster.hierarchy import fcluster
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import manhattan_distances
import seaborn as sns


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

    def compute_serial_matrix(self, manhattan_distance):  # method="complete"
        """
                It transforms a distance matrix into a sorted distance matrix according to the order implied by the
                hierarchical tree (dendrogram)
                 :{"param_1": country_distance: input a distance matrix to get a sorted one by method
                "param_2": method: method = ["ward","single","average","complete"]
                 }
                 :{"return_1": seriated_dist: input country distance, but with re-ordered rows and columns
                 according to the seriation
                 i.e. the order implied by the hierarchical tree
                "return_2": res_order: is the order implied by the hierarchical tree
                "return_3": res_linkage: is the hierarchical tree (dendrogram)
                }
                """
        self.get_manhattan_distance()
        data = pd.DataFrame(self.data_tr.data_clustering)
        data.index = self.country_names
        n = len(manhattan_distance)   # 39 countries
        res_linkage = sch.linkage(manhattan_distance, method="complete")

        order = fcluster(res_linkage, manhattan_distance.max(), criterion='distance')

        seriated_dist = np.zeros((n, n))  # Return a new array of given shape and type, filled with zeros.
        a, b = np.triu_indices(n, k=1)  # Return the indices for the upper-triangle of an (n, m) array

        # reorder the elements using permute so that the sum of sequential pairwise distance is minimal
        seriated_dist[a, b] = manhattan_distance[[order[i] for i in a], [order[j] for j in b]]
        seriated_dist[b, a] = seriated_dist[a, b]  # symmetric matrix
        return seriated_dist, order, res_linkage  # seriated_dist is 39*39, res_order is 39, res_linage provides
    # clusters merged, distance, and frequency, that is 39*4

    def plot_distances(self):
        manhattan_distance = self.get_manhattan_distance()
        self.country_names = self.data_tr.country_names
        plt.figure(figsize=(36, 28))
        plt.xticks(ticks=np.arange(len(self.country_names)),
                   labels=self.country_names,
                   rotation=90, fontsize=24)
        plt.yticks(ticks=np.arange(len(self.country_names)),
                   labels=self.country_names,
                   rotation=0, fontsize=24)
        plt.title("Measure of closeness  between countries before reordering",
                  fontsize=42, fontweight="bold")
        az = plt.imshow(manhattan_distance, cmap='nipy_spectral',
                        interpolation="nearest",
                        vmin=0, vmax=2.2)
        plt.colorbar(az,
                     ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4,
                            1.6, 1.8, 2.0, 2.2])

    def plot_dendrogram(self):
        manhattan_distance = self.get_manhattan_distance()
        country_names = self.data_tr.country_names
        plt.figure(figsize=(33, 31))
        sch.dendrogram(sch.linkage(manhattan_distance.to_numpy(), method="complete"),
                       get_leaves=True,
                       leaf_rotation=0,
                       leaf_font_size=32,
                       show_leaf_counts=True,
                       labels=country_names,
                       above_threshold_color='blue',
                       orientation="right",
                       distance_sort=True)
        plt.title('Hierarchical Clustering Dendrogram before reordering', fontsize=44, fontweight="bold")
        plt.xlabel('Distance between Clusters', fontsize=42)

    def plot_ordered_distance(self):
        manhattan_distance = manhattan_distances(self.data_tr.data_clustering)  # get pairwise manhattan distance

        # convert the data into dataframe
        # replace the indexes of the distance with the country names
        # rename the columns and rows of the distance with country names and return a matrix distance
        dt = pd.DataFrame(manhattan_distance, index=self.country_names, columns=self.country_names)

        # Return a copy of the manhattan distance collapsed into one dimension.
        distances = manhattan_distance[np.triu_indices(np.shape(manhattan_distance)[0], k=1)].flatten()

        #  Perform hierarchical clustering using complete method.
        res = sch.linkage(distances, method="complete")

        #  flattens the dendrogram, obtaining as a result an assignation of the original data points to single clusters.
        order = fcluster(res, 0.2 * manhattan_distance.max(), criterion='distance')

        # Perform an indirect sort along the along first axis
        columns = [dt.columns.tolist()[i] for i in list((np.argsort(order)))]

        # Place columns(sorted countries) in the both axes
        dt = dt.reindex(columns, axis='index')
        dt = dt.reindex(columns, axis='columns')

        plt.figure(figsize=(35, 28))

        plt.title("Measure of closeness between countries",
                  fontsize=43,
                  fontweight="bold")
        az = plt.imshow(dt, cmap='nipy_spectral',
                        alpha=.9, interpolation="nearest", vmin=0, vmax=2.2)
        plt.xticks(ticks=np.arange(len(columns)),
                   labels=columns,
                   rotation=90, fontsize=24)
        plt.yticks(ticks=np.arange(len(columns)),
                   labels=columns,
                   rotation=0, fontsize=24)
        plt.colorbar(az,
                     ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2])

        #  Original uncolored Dendrogram
        plt.figure(figsize=(45, 30))
        sch.dendrogram(res,
                       color_threshold='default',
                       leaf_rotation=90,
                       leaf_font_size=26,
                       labels=np.array(columns),
                       orientation="top",
                       show_leaf_counts=True,
                       distance_sort=True)
        plt.title('Cluster Analysis with no threshold', fontsize=50, fontweight="bold")
        plt.ylabel('Distance between Clusters', fontsize=45)

        #  Colored Dendrogram based on threshold (7 clusters)
        # the longest  vertical distance without any horizontal line  passing   through  it is selected and a
        #  horizontal   line is drawn   through it.

        plt.figure(figsize=(40, 30), dpi=80)
        sch.dendrogram(res,
                       color_threshold=1.25,  # sets the color of the links above the color_threshold
                       leaf_rotation=90,
                       leaf_font_size=26,  # the size based on the number of nodes in the dendrogram.
                       show_leaf_counts=True,
                       labels=np.array(columns),
                       above_threshold_color='grey',
                       orientation="top",
                       get_leaves=True,
                       distance_sort=True)
        plt.title('Cluster Analysis with a threshold', fontsize=44, fontweight="bold")
        plt.ylabel('Distance between Clusters', fontsize=42)
        line = 1.25
        plt.axhline(y=line, c='green', lw=2, linestyle='--')

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
