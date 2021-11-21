import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, clustering, data_transformer):
        self.colors = ['r', 'g', 'b']
        self.clust = clustering
        self.data_tr = data_transformer

    def plot_clustering(self):
        color_vector = [self.colors[idx] for idx in self.clust.k_means_pred]
        plt.scatter(self.clust.data[:, 0], self.clust.data[:, 1], c=color_vector)
        plt.scatter(self.clust.centroids[:, 0], self.clust.centroids[:, 1],
                    marker='*', c=self.colors[:self.clust.n_cl], s=180)
        plt.scatter(self.clust.closest_points[:, 0], self.clust.closest_points[:, 1],
                    c=self.colors[:self.clust.n_cl], s=120)

        for c_idx, (idx, point) in enumerate(zip(self.clust.closest_point_idx, self.clust.closest_points)):
            plt.annotate(self.data_tr.country_names[idx][:3], point)
            print("The country nearest to the centroid", c_idx, ":", self.data_tr.country_names[idx])
        plt.show()

    def plot_heatmap_centroid(self, centroids):
        for idx, centroid in enumerate(centroids):
            new_contact_mtx = np.zeros((16, 16))
            new_contact_mtx[self.data_tr.upper_tri_indexes] = centroid
            new_2 = new_contact_mtx.T
            new_2[self.data_tr.upper_tri_indexes] = centroid

            plot_contact_matrix(contact_matrix=new_2)
            plt.title("Centroid" + str(idx))
            plt.show()

    def plot_heatmap_closest(self):
        for idx, closest_idx in enumerate(self.clust.closest_point_idx):
            new_contact_mtx = self.data_tr.data_all_dict[self.data_tr.country_names[closest_idx]]["contact_home"]
            plot_contact_matrix(contact_matrix=new_contact_mtx)
            plt.title(self.data_tr.country_names[closest_idx])
            plt.show()


def plot_contact_matrix(contact_matrix):
    param_list = range(0, 16, 1)
    corr = pd.DataFrame(contact_matrix, columns=param_list, index=param_list)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    ax = sns.heatmap(corr, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.yticks(rotation=0)
