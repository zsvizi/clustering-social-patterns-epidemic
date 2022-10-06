import numpy as np
from sklearn.cluster import KMeans


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
