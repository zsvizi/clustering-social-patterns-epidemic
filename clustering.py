
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from heatmap import Hierarchical
from dataloader import DataLoader
from plotter import Plotter
import pandas as pd
from simulation import Simulation


class DataTransformer:
    def __init__(self):
        self.data = DataLoader()

        self.susc = 1.0
        self.base_r0 = 2.2

        self.upper_tri_indexes = np.triu_indices(16)
        self.country_names = list(self.data.age_data.keys())

        self.data_all_dict = dict()
        self.data_mtx_dict = dict()
        self.data_clustering = []

        self.get_data_for_clustering()

    def get_data_for_clustering(self):
        for country in self.country_names:
            age_vector = self.data.age_data[country]["age"].reshape((-1, 1))
            contact_matrix = self.data.contact_data[country]["home"] + \
                self.data.contact_data[country]["work"] + \
                self.data.contact_data[country]["school"] + \
                self.data.contact_data[country]["other"]
            contact_home = self.data.contact_data[country]["home"]
            contact_school = self.data.contact_data[country]["school"]
            contact_work = self.data.contact_data[country]["work"]
            contact_other = self.data.contact_data[country]["other"]

            susceptibility = np.array([1.0] * 16)
            susceptibility[:4] = self.susc
            simulation = Simulation(data=self.data, base_r0=self.base_r0,
                                    contact_matrix=contact_matrix,
                                    contact_home=contact_home,
                                    age_vector=age_vector,
                                    susceptibility=susceptibility)
            self.data_all_dict.update(
                {country: {"beta": simulation.beta,
                           "age_vector": age_vector,
                           "contact_full": contact_matrix,
                           "contact_home": contact_home,
                           "contact_school": contact_school,
                           "contact_work": contact_work,
                           "contact_other": contact_other
                           }
                 })
            self.data_mtx_dict.update(
                {country: {"full": simulation.beta * contact_matrix[self.upper_tri_indexes],
                           "home": simulation.beta * contact_home[self.upper_tri_indexes],
                           "school": simulation.beta * contact_school[self.upper_tri_indexes],
                           "work": simulation.beta * contact_work[self.upper_tri_indexes],
                           "other": simulation.beta * contact_other[self.upper_tri_indexes]
                           }
                 })
            self.data_clustering.append(
                simulation.beta * contact_matrix[self.upper_tri_indexes])
        self.data_clustering = np.array(self.data_clustering)


class Clustering:
    def __init__(self, data):
        self.data = data
        self.n_cl = 3

        self.k_means_pred = None
        self.centroids = None
        self.closest_points = None
        self.closest_point_idx = None

    def run_clustering(self):
        k_means = KMeans(n_clusters=self.n_cl, random_state=1)
        k_means.fit(self.data)
        self.k_means_pred = k_means.predict(self.data)
        self.centroids = k_means.cluster_centers_

    def get_closest_points(self):
        self.closest_point_idx = (-1) * np.ones(self.n_cl).astype(int)
        for c_idx, centroid in enumerate(self.centroids):
            min_dist = None
            for idx, point in enumerate(self.data):
                if self.k_means_pred[idx] == c_idx:
                    dist = np.sum((point - centroid) ** 2)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        self.closest_point_idx[c_idx] = idx
        self.closest_points = self.data[np.array(self.closest_point_idx).astype(int), :2]


def main():
    # Create data for clustering
    data_tr = DataTransformer()

    # Reduce dimensionality
    pca = PCA(n_components=12)
    pca.fit(data_tr.data_clustering)
    data_pca = pca.transform(data_tr.data_clustering)
    print("Explained variance ratios:", pca.explained_variance_ratio_,
          "->", sum(pca.explained_variance_ratio_))
    # Execute heatmap
    distance = Hierarchical(data_clustering=data_tr, data_transformer=data_tr, country_names=data_tr.country_names)
    print("Euclidean distance:", pd.DataFrame.round(distance.get_euclidean_distance(), 3))
    print("Manhattan distance:", pd.DataFrame.round(distance.get_manhattan_distance(), 3))
    distance.plot_distances()
    distance.plot_dendrogram()
    distance.plot_ordered_distance()

    # Execute clustering
    clust = Clustering(data=data_pca)
    clust.run_clustering()
    clust.get_closest_points()

    # Plot results for analysis
    plotter = Plotter(clustering=clust,
                      data_transformer=data_tr, country_names=data_tr)
    plotter.plot_clustering()
    centroids_orig = pca.inverse_transform(clust.centroids)
    plotter.plot_heatmap_centroid(centroids=centroids_orig)
    plotter.plot_heatmap_closest()

    # List cluster members
    for cluster in range(clust.n_cl):
        print("Cluster", cluster, "(" + plotter.colors[cluster] + ")", ":",
              {data_tr.country_names[idx]: data_tr.data_all_dict[data_tr.country_names[idx]]["beta"]
               for idx, x in enumerate(clust.k_means_pred) if x == cluster})


if __name__ == "__main__":
    main()
