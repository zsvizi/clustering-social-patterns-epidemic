from sklearn.decomposition import PCA
from data_transformer import DataTransformer
from hierarchical import Hierarchical
from data_transformer import DataTransformer
from hierarchical import Hierarchical
from dpca import DPCA
from dpca import Dimension


class Analysis:
    def __init__(self, data_tr, data_dpca, data_dpca2, img_prefix, threshold, distance: str = "euclidean"):
        self.data_tr = data_tr
        self.data_dpca = data_dpca
        self.data_dpca2 = data_dpca2
        self.img_prefix = img_prefix
        self.threshold = threshold
        self.distance = distance

    def run(self):
        hierarchical = Hierarchical(data_transformer=self.data_tr,
                                    country_names=self.data_tr.country_names,
                                    img_prefix=self.img_prefix,
                                    dist=self.distance)
        hierarchical.plot_distances()
        # distance.heatmap_ten_countries()
        hierarchical.plot_ordered_distance(threshold=self.threshold)

    @staticmethod
    def apply_pca(data_dpca2, n_components):
        pca = PCA(n_components=n_components)
        pca.fit(data_dpca2.reduced_matrix)
        data_pca = pca.transform(data_dpca2.reduced_matrix)
        print("Explained variance ratios:",
              pca.explained_variance_ratio_,
              "->", sum(pca.explained_variance_ratio_))
        data_dpca2.reduced_matrix = data_pca


def main():

    do_clustering_pca = True

    # Create data for clustering
    data_tr = DataTransformer()
    data_dpca = DPCA(country_names=data_tr.country_names, data_tr=data_tr,
                     data_contact_matrix=data_tr.data_contact_matrix,
                     data_contact_hmatrix=data_tr.data_contact_hmatrix, flatten_matrix=data_tr.flatten_matrix)
    data_dpca2 = Dimension(data_tr=data_tr, country_names=data_tr.country_names,
                           data_matrix=data_tr.data_matrix)

    # execute class 2D2PCA
    # print(data_dpca.pca_reduced)

    # execute class Dimension
    # print(data_dpca2.matrix_reduced)  # 2*2 matrix for each country with the countries
    # print(data_dpca2.reduced_matrix)  # 39*4 matrix

    # do analysis for original data
    Analysis(data_tr=data_tr, data_dpca=data_dpca, data_dpca2=data_dpca2, img_prefix="original", threshold=0.22).run()
    if do_clustering_pca:
        n_components = 3
        Analysis.apply_pca(data_dpca2=data_dpca2, n_components=n_components)
        # do analysis for reduced data
        Analysis(data_tr=data_tr, data_dpca=data_dpca, data_dpca2=data_dpca2, img_prefix="pca_" + str(n_components),
                 threshold=0.23).run()


if __name__ == "__main__":
    main()

