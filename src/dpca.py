
import numpy as np
import pandas as pd

from src.dataloader import DataLoader
from src.data_transformer import DataTransformer
from src.simulation import Simulation
from sklearn.decomposition import PCA
from sklearn import preprocessing


class DPCA:
    def __init__(self, data_tr, country_names, data_contact_matrix, data_contact_hmatrix, flatten_matrix):
        self.data = DataLoader()
        self.country_names = country_names
        self.data_tr = data_tr

        self.data_contact_matrix = data_contact_matrix
        self.data_contact_hmatrix = data_contact_hmatrix
        self.flatten_matrix = flatten_matrix
        self.matrix_reduced = []
        self.apply_dpca(data_tr=data_tr)

    def apply_dpca(self, data_tr):

        # working with 39 * 256 matrix
        df = pd.DataFrame(self.flatten_matrix, index=self.country_names)
        scale = preprocessing.scale(self.flatten_matrix)
        pca = PCA(n_components=2).fit_transform(scale)

        # working with 624 * 16 matrix (v-stack)
        # normalize data
        data_scaled = preprocessing.scale(self.data_contact_matrix)
        pca1 = PCA(n_components=2)
        pca1.fit(data_scaled)
        pca_data = pca1.transform(data_scaled)

        # create a DataFrame  that will execute principal component of all the 624 dimension matrix
        pc = pd.DataFrame(data=pca_data,
                          columns=['PC1', 'PC2'])  # 624 * 2

        print("Explained variance ratios:", pca1.explained_variance_ratio_,
              "->", sum(pca1.explained_variance_ratio_), "principal components:",
              pca1.components_,  # (2, 16)
              pca1.n_components_)  # 2

        # Split concatenated array into 39 sub-arrays of equal size i.e. 39 countries.
        split = np.array_split(pc, 39)
        self.matrix_reduced = split

    # working with 16 * 624 matrix (h-stack)
        data_scaled2 = preprocessing.scale(self.data_contact_hmatrix)
        pca2 = PCA(n_components=2)
        pca2.fit(data_scaled2)
        pca2_data = pca2.transform(data_scaled2)

        # create a DataFrame that will execute principal component of the 16 * 624 dimension matrix
        pc2 = pd.DataFrame(data=pca2_data,
                           columns=['pc 1', 'pc 2'])

        print("Explained variance ratios 2:", pca2.explained_variance_ratio_,
              "->", sum(pca2.explained_variance_ratio_), "principal components 2:",
              pca2.components_, pca2.n_components_)




