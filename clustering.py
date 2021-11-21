import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from dataloader import DataLoader
from simulation import Simulation


def main():
    data = DataLoader()

    susc = 1.0
    base_r0 = 2.2

    data_all_dict = dict()
    data_mtx_dict = dict()
    data_clustering = []

    upper_tri_indexes = np.triu_indices(16)
    country_names = list(data.age_data.keys())
    for country in country_names:
        age_vector = data.age_data[country]["age"].reshape((-1, 1))
        contact_matrix = data.contact_data[country]["home"] + data.contact_data[country]["work"] + \
            data.contact_data[country]["school"] + data.contact_data[country]["other"]
        contact_home = data.contact_data[country]["home"]

        susceptibility = np.array([1.0] * 16)
        susceptibility[:4] = susc
        simulation = Simulation(data=data, base_r0=base_r0,
                                contact_matrix=contact_matrix,
                                contact_home=contact_home,
                                age_vector=age_vector,
                                susceptibility=susceptibility)
        data_all_dict.update({country: {"beta": simulation.beta,
                                        "age_vector": age_vector,
                                        "contact_full": contact_matrix,
                                        "contact_home": contact_home
                                        }
                              })
        data_mtx_dict.update({country: {"full": simulation.beta * contact_matrix[upper_tri_indexes],
                                        "home": simulation.beta * contact_home[upper_tri_indexes]
                                        }
                              })
        data_clustering.append(simulation.beta * contact_matrix[upper_tri_indexes])
    data_clustering = np.array(data_clustering)

    colors = ['r', 'g', 'b']

    pca = PCA(n_components=12)
    pca.fit(data_clustering)
    data_pca = pca.transform(data_clustering)

    print("Explained variance ratios:", pca.explained_variance_ratio_, "->", sum(pca.explained_variance_ratio_))

    n_cl = 3
    k_means = KMeans(n_clusters=n_cl, random_state=1)
    k_means.fit(data_pca)
    k_means_pred = k_means.predict(data_pca)
    centroids = k_means.cluster_centers_

    closest_point_idx = (-1) * np.ones(n_cl).astype(int)
    for c_idx, centroid in enumerate(centroids):
        min_dist = None
        for idx, point in enumerate(data_pca):
            if k_means_pred[idx] == c_idx:
                dist = np.sum((point - centroid) ** 2)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    closest_point_idx[c_idx] = idx
    closest_points = data_pca[np.array(closest_point_idx).astype(int), :2]

    color_vector = [colors[idx] for idx in k_means_pred]
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=color_vector)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c=colors[:n_cl], s=180)
    plt.scatter(closest_points[:, 0], closest_points[:, 1], c=colors[:n_cl], s=120)

    centroid_idx = 0
    for idx, point in zip(closest_point_idx, closest_points):
        plt.annotate(country_names[idx][:3], point)
        print("The country nearest to the centroid", centroid_idx, ":", country_names[idx])
        centroid_idx += 1
    plt.show()

    centroid_orig = pca.inverse_transform(centroids)

    for idx, centroid in enumerate(centroid_orig):
        new_contact_mtx = np.zeros((16, 16))
        new_contact_mtx[upper_tri_indexes] = centroid
        new_2 = new_contact_mtx.T
        new_2[upper_tri_indexes] = centroid

        param_list = range(0, 16, 1)
        corr = pd.DataFrame(new_2, columns=param_list, index=param_list)
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        ax = sns.heatmap(corr, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.yticks(rotation=0)
        plt.title("Centroid" + str(idx))
        plt.show()

    for cluster in range(n_cl):
        print("Cluster", cluster, "(" + colors[cluster] + ")", ":",
              {country_names[idx]: data_all_dict[country_names[idx]]["beta"]
               for idx, x in enumerate(k_means_pred) if x == cluster})

    for idx, closest_idx in enumerate(closest_point_idx):
        new_contact_mtx = data_all_dict[country_names[closest_idx]]["contact_home"]

        param_list = range(0, 16, 1)
        corr = pd.DataFrame(new_contact_mtx, columns=param_list, index=param_list)
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        ax = sns.heatmap(corr, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.yticks(rotation=0)
        plt.title(country_names[closest_idx])
        plt.show()


if __name__ == "__main__":
    main()
