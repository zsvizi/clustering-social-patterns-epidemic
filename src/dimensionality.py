import numpy as np
import pandas as pd

from src.dataloader import DataLoader
from sklearn.decomposition import PCA
from sklearn import preprocessing


class D2PCA:
    """
        This is (2D)^2 PCA class that applies both row-row and column-column directions to perform dimension reduction.
        input: 39 countries each 16 * 16 matrix concatenated row wise and column wise
        output: 39 countries each 2 * 2 matrix, and 39 * 4 (2 * 2 flatten matrix)
    """
    def __init__(self, data_tr, country_names, data_contact_matrix, contact_matrix_transposed):
        self.proj_matrix_2 = []
        self.data_split = []
        self.proj_matrix_1 = []
        self.data = DataLoader()
        self.country_names = country_names
        self.data_tr = data_tr

        self.data_contact_matrix = data_contact_matrix
        self.contact_matrix_transposed = contact_matrix_transposed
        self.pca_reduced = []
        self.apply_dpca()

    def row_direction_pca(self):

        # working with 624 * 16 matrix (v-stack)
        # center the data
        centered_data = self.data_contact_matrix - np.mean(self.data_contact_matrix, axis=0)

        # normalize data
        data_scaled = preprocessing.scale(centered_data)
        pca1 = PCA(n_components=2)
        pca1.fit(data_scaled)
        pca_data = pca1.transform(data_scaled)

        # create a DataFrame  that will execute principal component of all the 624 dimension matrix
        pc = pd.DataFrame(data=pca_data,
                          columns=['PC1', 'PC2'])  # 624 * 2

        print("Explained variance ratios:", pca1.explained_variance_ratio_,
              "->", sum(pca1.explained_variance_ratio_), "Eigenvectors:",
              pca1.components_,  # (2, 16)
              "Singular values:", pca1.singular_values_)  # 2 leading eigenvalues

        # Projection matrix for row direction matrix
        proj_matrix_1 = pca1.components_.T  # 16 * 2 projection matrix 1

        # Now split concatenated original data into 39 sub-arrays of equal size i.e. 39 countries.
        split = np.array_split(data_scaled, 39)

        # convert split data to a numpy array
        data_split = np.array(split)   # 39 * (16 * 16) row direction matrix for each country

        return proj_matrix_1, data_split

    def alternative_direction_pca(self):
        # again,  centering and scaling the transposed data
        mean = np.mean(self.contact_matrix_transposed, axis=0)
        centered_data2 = self.contact_matrix_transposed - mean

        # normalize data
        data_scaled2 = preprocessing.scale(centered_data2)
        pca2 = PCA(n_components=2)
        pca2.fit(data_scaled2)
        pca_data2 = pca2.transform(data_scaled2)

        # create a DataFrame  that will execute principal component of all the 624 dimension matrix
        pc2 = pd.DataFrame(data=pca_data2,
                           columns=['PC 1', 'PC 2'])  # 624 * 2

        print("Explained variance ratios 2:", pca2.explained_variance_ratio_,
              "->", sum(pca2.explained_variance_ratio_), "Eigenvectors 2:",
              pca2.components_,  # (2, 16)
              "Singular values 2:", pca2.singular_values_)  # 2 leading eigenvalues
        # print("PC 2", pc2)

        # Projection matrix for column direction matrix
        proj_matrix_2 = pca2.components_.T  # 16 * 2 projection matrix 2
        return proj_matrix_2

    def apply_dpca(self):
        self.proj_matrix_1, self.data_split = self.row_direction_pca()
        self.proj_matrix_2 = self.alternative_direction_pca()

        # Now apply (2D)^2 PCA simultaneously using projection matrix 1 and 2
        matrix = self.proj_matrix_1.T @ self.data_split @ self.proj_matrix_2

        # Now reshape the matrix to get desired 39 * 4
        self.pca_reduced = matrix.reshape((39, 4))





















