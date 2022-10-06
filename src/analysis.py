from sklearn.decomposition import PCA

from data_transformer import DataTransformer
from dimensionality import D2PCA
from hierarchical import Hierarchical


class Analysis:
    def __init__(self, data_tr, data_dpca, img_prefix, threshold, distance: str = "euclidean"):
        self.data_tr = data_tr
        self.data_dpca = data_dpca
        self.img_prefix = img_prefix
        self.threshold = threshold
        self.distance = distance

    def run(self):
        hierarchical = Hierarchical(data_transformer=self.data_tr, dimensionality=self.data_dpca,
                                    country_names=self.data_tr.country_names,
                                    img_prefix=self.img_prefix,
                                    dist=self.distance)
        hierarchical.plot_distances()
        # hierarchical.heatmap_ten_countries()
        hierarchical.plot_ordered_distance(threshold=self.threshold)

    @staticmethod
    def apply_pca(data_tr, n_components):
        pca = PCA(n_components=n_components)
        pca.fit(data_tr.data_clustering)
        data_pca = pca.transform(data_tr.data_clustering)
        print("Explained variance ratios:",
              pca.explained_variance_ratio_,
              "->", sum(pca.explained_variance_ratio_))
        data_tr.data_clustering = data_pca


def main():

    do_clustering_pca = False
    do_clustering_dpca = True

    # Create data for clustering
    data_tr = DataTransformer()
    data_dpca = D2PCA(country_names=data_tr.country_names, data_tr=data_tr,
                      data_contact_matrix=data_tr.data_contact_matrix,
                      contact_matrix_transposed=data_tr.contact_matrix_transposed)
    # data_dpca.apply_dpca()
    # execute class 2D2PCA
    # print(data_dpca.pca_reduced.shape)

    # do analysis for original data
    Analysis(data_tr=data_tr, data_dpca=data_dpca, img_prefix="original", threshold=0.23).run()

    # Do analysis of the pca
    if do_clustering_pca:
        n_components = 4
        Analysis.apply_pca(data_tr=data_tr, n_components=n_components)
        # do analysis for reduced data
        Analysis(data_tr=data_tr, data_dpca=data_dpca, img_prefix="pca_" + str(n_components),
                 threshold=0.25).run()

    # do analysis of 2dpca
    if do_clustering_dpca:
        D2PCA.apply_dpca(self=data_dpca)
        Analysis(data_tr=data_tr, data_dpca=data_dpca, img_prefix="dpca_",
                 threshold=5.5).run()


if __name__ == "__main__":
    main()

