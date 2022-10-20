import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from data_transformer import DataTransformer
from d2pca import D2PCA
from hierarchical import Hierarchical


class Analysis:
    def __init__(self, data_tr: DataTransformer, img_prefix, threshold,
                 n_components: int = 4,
                 dim_red: str = None, distance: str = "euclidean"):
        self.data_tr = data_tr
        self.dim_red = dim_red
        self.img_prefix = img_prefix
        self.threshold = threshold
        self.distance = distance
        self.n_components = n_components
        if dim_red is not None:
            self.apply_pca()

    def run(self):
        hierarchical = Hierarchical(data_transformer=self.data_tr,
                                    country_names=self.data_tr.country_names,
                                    img_prefix=self.img_prefix,
                                    dist=self.distance)
        hierarchical.run(threshold=self.threshold)

    def apply_pca(self):
        if self.dim_red == "PCA":
            pca = PCA(n_components=self.n_components)
            pca.fit(self.data_tr.data_cm_1dpca)
            data_pca = pca.transform(self.data_tr.data_cm_1dpca)
            print("Explained variance ratios:",
                  pca.explained_variance_ratio_,
                  "->", sum(pca.explained_variance_ratio_))
        elif self.dim_red == "2DPCA":
            data_dpca = D2PCA(country_names=self.data_tr.country_names, data_tr=self.data_tr)
            data_dpca.apply_dpca()
            data_pca = data_dpca.pca_reduced
        else:
            raise Exception("Provide a type for dimensionality reduction.")
        self.data_tr.data_cm_pca = data_pca


def hungary_contacts(data_tr):
    for typ in ["contact_home", "contact_school", "contact_work", "contact_other", "contact_full"]:
        # home contact
        img = plt.imshow(data_tr.data_all_dict['Hungary'][typ],
                         cmap='jet', vmin=0, vmax=4, alpha=.9, interpolation="nearest")
        ticks = np.arange(0, 16, 2)
        if typ == 'contact_full':
            cbar = plt.colorbar(img)
            tick_font_size = 40
            cbar.ax.tick_params(labelsize=tick_font_size)
        plt.xticks(ticks, fontsize=24)
        plt.yticks(ticks, fontsize=24)
        plt.savefig("../plots/" + "hungary_" + typ.split("contact_")[1] + ".pdf")


def country_contacts(data_tr):
    for country in ["Armenia", "Belgium", "Estonia"]:
        # contact matrix Armenia
        matrix_to_plot = data_tr.data_all_dict[country]["contact_full"] * \
            data_tr.data_all_dict[country]["beta"]
        img = plt.imshow(matrix_to_plot,
                         cmap='jet', vmin=0, vmax=0.2,
                         alpha=.9, interpolation="nearest")
        ticks = np.arange(0, 16, 2)
        plt.xticks(ticks, fontsize=20)
        plt.yticks(ticks, fontsize=20)
        if country == "Estonia":
            cbar = plt.colorbar(img)
            tick_font_size = 25
            cbar.ax.tick_params(labelsize=tick_font_size)
        plt.savefig("../plots/" + country + ".pdf")


def main():
    do_clustering_pca = True
    do_clustering_dpca = True

    # Create data for clustering
    susc = 1.0
    base_r0 = 2.2
    data_tr = DataTransformer(susc=susc, base_r0=base_r0)

    # Create plots for the paper
    hungary_contacts(data_tr=data_tr)
    country_contacts(data_tr=data_tr)

    # Do analysis of the pca
    if do_clustering_pca:
        n_components = 4
        # do analysis for reduced data
        Analysis(data_tr=data_tr, dim_red="PCA", n_components=4,
                 img_prefix="pca_" + str(n_components), threshold=0.25).run()

    # do analysis of 2dpca
    if do_clustering_dpca:
        Analysis(data_tr=data_tr, dim_red="2DPCA",
                 img_prefix="dpca_", threshold=5.5).run()


if __name__ == "__main__":
    main()
