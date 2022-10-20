from sklearn.decomposition import PCA

from data_transformer import DataTransformer
from d2pca import D2PCA
from hierarchical import Hierarchical


class Analysis:
    def __init__(self, data_tr, img_prefix, threshold,
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
        hierarchical.plot_distances()
        # hierarchical.heatmap_ten_countries()
        hierarchical.plot_ordered_distance(threshold=self.threshold)
        # hierarchical.hungary_contacts()
        # hierarchical.country_contacts()

    def apply_pca(self):
        if self.dim_red == "PCA":
            pca = PCA(n_components=self.n_components)
            pca.fit(self.data_tr.data_clustering)
            data_pca = pca.transform(self.data_tr.data_clustering)
            print("Explained variance ratios:",
                  pca.explained_variance_ratio_,
                  "->", sum(pca.explained_variance_ratio_))
        elif self.dim_red == "2DPCA":
            data_dpca = D2PCA(country_names=self.data_tr.country_names, data_tr=self.data_tr)
            data_dpca.apply_dpca()
            data_pca = data_dpca.pca_reduced
        else:
            raise Exception("Provide a type for dimensionality reduction.")
        self.data_tr.data_clustering = data_pca


def main():
    do_clustering_pca = True
    do_clustering_dpca = True

    # Create data for clustering
    data_tr = DataTransformer()

    # do analysis for original data
    Analysis(data_tr=data_tr,
             img_prefix="original", threshold=0.23).run()

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
