import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances


class Hierarchical:
    def __init__(self, data_transformer, country_names, img_prefix,
                 dist: str = "euclidean"):
        self.data_tr = data_transformer
        self.country_names = country_names
        self.img_prefix = img_prefix
        if dist == "euclidean":
            self.get_distance_matrix = self.get_euclidean_distance
        elif dist == "manhattan":
            self.get_distance_matrix = self.get_manhattan_distance

        os.makedirs("../plots", exist_ok=True)

    def plot_ordered_distance(self, threshold: float):
        # calculate ordered distance matrix
        columns, dt, res = self.calculate_ordered_distance_matrix(threshold=threshold)

        # plot ordered distance matrix
        self.plot_ordered_distance_matrix(columns=columns, dt=dt)

        #  Original uncolored Dendrogram
        self.plot_dendrogram(res=res)

        #  Colored Dendrogram based on threshold (4 clusters)
        # cutting the dendrogram where the gap between two successive merges is at the largest.
        #  horizontal   line is drawn   through it.
        self.plot_dendrogram_with_threshold(res=res, threshold=threshold)

    def plot_distances(self):
        distance, _ = self.get_distance_matrix()
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
        az = plt.imshow(distance, cmap='rainbow',
                        interpolation="nearest",
                        vmin=0)
        plt.colorbar(az)
        plt.savefig("../plots/" + self.img_prefix + "_" + "distances.pdf")

    def calculate_ordered_distance_matrix(self, threshold, verbose: bool = True):
        dt, distance = self.get_distance_matrix()
        # Return a copy of the distance collapsed into one dimension.
        distances = distance[np.triu_indices(np.shape(distance)[0], k=1)].flatten()
        #  Perform hierarchical clustering using complete method.
        res = sch.linkage(distances, method="complete")
        #  flattens the dendrogram, obtaining as a result an assignation of the original data points to single clusters.
        order = sch.fcluster(res, threshold, criterion='distance')
        if verbose:
            for x in np.unique(order):
                print("cluster " + str(x) + ":", dt.columns[order == x])
        # Perform an indirect sort along the along first axis
        columns = [dt.columns.tolist()[i] for i in list((np.argsort(order)))]
        # Place columns(sorted countries) in the both axes
        dt = dt.reindex(columns, axis='index')
        dt = dt.reindex(columns, axis='columns')
        return columns, dt, res

    def plot_ordered_distance_matrix(self, columns, dt):
        plt.figure(figsize=(45, 35), dpi=300)
        az = plt.imshow(dt, cmap='rainbow',
                        alpha=.9, interpolation="nearest")
        plt.xticks(ticks=np.arange(len(columns)),
                   labels=columns,
                   rotation=90, fontsize=32)
        plt.yticks(ticks=np.arange(len(columns)),
                   labels=columns,
                   rotation=0, fontsize=45)
        cbar = plt.colorbar(az)
        tick_font_size = 115
        cbar.ax.tick_params(labelsize=tick_font_size)

        plt.savefig("../plots/" + self.img_prefix + "_" + "ordered_distance_1.pdf")

    def plot_dendrogram(self, res):
        fig, axes = plt.subplots(1, 1, figsize=(35, 25), dpi=150)
        sch.dendrogram(res,
                       leaf_rotation=90,
                       leaf_font_size=25,
                       labels=self.country_names,
                       orientation="top",
                       show_leaf_counts=True,
                       distance_sort=True)
        axes.tick_params(axis='both', which='major', labelsize=26)
        plt.title('Cluster Analysis without threshold', fontsize=50, fontweight="bold")
        plt.ylabel('Distance between Clusters', fontsize=45)
        plt.tight_layout()
        plt.savefig("../plots/" + self.img_prefix + "_" + "ordered_distance_2.pdf")

    def plot_dendrogram_with_threshold(self, res, threshold):
        fig, axes = plt.subplots(1, 1, figsize=(35, 24), dpi=300)
        sch.dendrogram(res,
                       color_threshold=threshold,  # sets the color of the links above the color_threshold
                       leaf_rotation=90,
                       leaf_font_size=24,  # the size based on the number of nodes in the dendrogram.
                       show_leaf_counts=True,
                       labels=self.country_names,
                       above_threshold_color='blue',
                       ax=axes,
                       orientation="top",
                       get_leaves=True,
                       distance_sort=True)
        plt.title('Cluster Analysis with a threshold', fontsize=49, fontweight="bold")
        plt.ylabel('Distance between Clusters', fontsize=30)
        plt.tight_layout()
        axes.tick_params(axis='both', which='major', labelsize=26)
        plt.savefig("../plots/" + self.img_prefix + "_" + "ordered_distance_3.pdf")

    def get_manhattan_distance(self):
        """
        Calculates Manhattan distance of a 39 * 136 matrix and returns 39*39 distance matrix
        :return matrix: square distance matrix with zero diagonals
        """
        manhattan_distance = manhattan_distances(self.data_tr.data_clustering)  # get pairwise manhattan distance
        # manhattan_distance = manhattan_distances(self.data_dpca.pca_reduced)
        # convert the data into dataframe
        # replace the indexes of the distance with the country names
        # rename the columns and rows of the distance with country names and return a matrix distance
        dt = pd.DataFrame(manhattan_distance,
                          index=self.country_names, columns=self.country_names)
        return dt, manhattan_distance

    def get_euclidean_distance(self) -> np.array:
        """
        Calculates euclidean distance of a 39 * 136 matrix and returns 39*39 distance matrix
        :return matrix: square distance matrix with zero diagonals
        """
        # convert the data into dataframe
        euc_distance = euclidean_distances(self.data_tr.data_clustering)
        dt = pd.DataFrame(euc_distance,
                          index=self.country_names, columns=self.country_names)  # rename rows and columns
        return dt, euc_distance

    def heatmap_ten_countries(self):
        #  plot the heatmap for the first 10 countries
        plt.figure(figsize=(12, 10))
        distance, _ = self.get_distance_matrix()
        heatmap = distance.iloc[0: 10:, 0: 10]
        sns.heatmap(heatmap, annot=True, cmap="rainbow", vmin=0)
        plt.savefig("../plots/" + self.img_prefix + "_" + "heatmap.pdf")

    def hungary_contacts(self):
        # home contact
        plt.imshow(self.data_tr.data_all_dict['Hungary']['contact_home'], cmap='jet', vmin=0, vmax=4, alpha=.9,
                   interpolation="nearest")
        ticks = np.arange(0, 16, 2)
        plt.xticks(ticks, fontsize=24)
        plt.yticks(ticks, fontsize=24)
        plt.savefig("../plots/" + "hungary_home.pdf")
        plt.show()

        # school contact
        plt.imshow(self.data_tr.data_all_dict['Hungary']['contact_school'], cmap='jet', vmin=0, vmax=4, alpha=.9,
                   interpolation="nearest")
        ticks = np.arange(0, 16, 2)
        plt.xticks(ticks, fontsize=24)
        plt.yticks(ticks, fontsize=24)
        plt.savefig("../plots/" + "hungary_school.pdf")
        plt.show()

        # work contact
        plt.imshow(self.data_tr.data_all_dict['Hungary']['contact_work'], cmap='jet', vmin=0, vmax=4, alpha=.9,
                   interpolation="nearest")
        ticks = np.arange(0, 16, 2)
        plt.xticks(ticks, fontsize=24)
        plt.yticks(ticks, fontsize=24)
        plt.savefig("../plots/" + "hungary_work.pdf")
        plt.show()

        # other contact
        plt.imshow(self.data_tr.data_all_dict['Hungary']['contact_other'], cmap='jet', vmin=0, vmax=4, alpha=.9,
                   interpolation="nearest")
        ticks = np.arange(0, 16, 2)
        plt.xticks(ticks, fontsize=24)
        plt.yticks(ticks, fontsize=24)
        plt.savefig("../plots/" + "hungary_other.pdf")
        plt.show()

        # all contact matrix with the column bar
        full = plt.imshow(self.data_tr.data_all_dict['Hungary']['contact_full'],
                          cmap='jet', vmin=0, vmax=4, alpha=.9, interpolation="nearest")
        cbar = plt.colorbar(full)
        tick_font_size = 40
        cbar.ax.tick_params(labelsize=tick_font_size)
        plt.xticks(ticks, fontsize=24)
        plt.yticks(ticks, fontsize=24)
        plt.savefig("../plots/" + "hungary_full.pdf")
        plt.show()

    def country_contacts(self):
        # contact matrix Armenia
        plt.imshow(self.data_tr.data_matrix['Armenia'], cmap='jet', vmin=0, vmax=0.2,
                   alpha=.9, interpolation="nearest")
        ticks = np.arange(0, 16, 2)
        plt.xticks(ticks, fontsize=20)
        plt.yticks(ticks, fontsize=20)
        plt.savefig("../plots/" + "Armenia.pdf")
        plt.show()

        # contact matrix Belgium
        plt.imshow(self.data_tr.data_matrix['Belgium'], cmap='jet', vmin=0, vmax=0.2,
                   alpha=.9, interpolation="nearest")
        ticks = np.arange(0, 16, 2)
        plt.xticks(ticks, fontsize=20)
        plt.yticks(ticks, fontsize=20)
        plt.savefig("../plots/" + "Belgium.pdf")
        plt.show()

        # contact matrix Estonia
        plt.imshow(self.data_tr.data_matrix['Estonia'], cmap='jet', vmin=0, vmax=0.2,
                   alpha=.9, interpolation="nearest")
        ticks = np.arange(0, 16, 2)
        plt.xticks(ticks, fontsize=20)
        plt.yticks(ticks, fontsize=20)
        plt.savefig("../plots/" + "Estonia.pdf")
        plt.show()

        # contact matrix Italy
        plt.imshow(self.data_tr.data_matrix['Italy'], cmap='jet', vmin=0, vmax=0.2, alpha=.9,
                   interpolation="nearest")
        ticks = np.arange(0, 16, 2)
        plt.xticks(ticks, fontsize=20)
        plt.yticks(ticks, fontsize=20)
        plt.savefig("../plots/" + "Italia.pdf")
        plt.show()

        # execute Netherlands contact matrix with the column bar
        Net = plt.imshow(self.data_tr.data_matrix['Netherlands'],
                         cmap='jet', vmin=0, vmax=0.2, alpha=.9, interpolation="nearest")
        ticks = np.arange(0, 16, 2)
        plt.xticks(ticks, fontsize=20)
        plt.yticks(ticks, fontsize=20)
        cbar = plt.colorbar(Net)
        tick_font_size = 25
        cbar.ax.tick_params(labelsize=tick_font_size)
        plt.savefig("../plots/" + "Netherlands.pdf")
        plt.show()












