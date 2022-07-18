from sklearn.decomposition import PCA

from data_transformer import DataTransformer
from hierarchical import Hierarchical


class Analysis:
    def __init__(self, data_tr, img_prefix, threshold, distance: str = "manhattan"):
        self.data_tr = data_tr
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
    def apply_pca(data_tr, n_components):
        pca = PCA(n_components=n_components)
        pca.fit(data_tr.data_clustering)
        data_pca = pca.transform(data_tr.data_clustering)
        print("Explained variance ratios:",
              pca.explained_variance_ratio_,
              "->", sum(pca.explained_variance_ratio_))
        data_tr.data_clustering = data_pca


def main():
    do_clustering_pca = True

    # Create data for clustering
    data_tr = DataTransformer()

    # do analysis for original data
    Analysis(data_tr=data_tr, img_prefix="original", threshold=1.5).run()
    if do_clustering_pca:
        n_components = 5
        Analysis.apply_pca(data_tr=data_tr, n_components=n_components)
        # do analysis for reduced data
        Analysis(data_tr=data_tr, img_prefix="pca_" + str(n_components), threshold=0.45).run()


if __name__ == "__main__":
    main()
