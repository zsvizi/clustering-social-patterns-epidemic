import numpy as np

from sklearn.decomposition import PCA
from sklearn import preprocessing


class D2PCA:
    """
        This is (2D)^2 PCA class that applies both row-row and column-column directions to perform dimension reduction.
        input: 39 countries each 16 * 16 matrix concatenated row wise and column wise
        output: 39 countries each 2 * 2 matrix, and 39 * 4 (2 * 2 flatten matrix)
    """
    def __init__(self, data_tr, country_names):
        self.country_names = country_names
        self.data_tr = data_tr

        self.data_contact_matrix = data_tr.data_contact_matrix
        self.contact_matrix_transposed = data_tr.contact_matrix_transposed

        self.data_split = []
        self.proj_matrix_2 = []
        self.proj_matrix_1 = []
        self.pca_reduced = []

    @staticmethod
    def preprocess_data(data):
        # center the data
        centered_data = data - np.mean(data, axis=0)
        # normalize data
        data_scaled = preprocessing.scale(centered_data)
        return data_scaled

    def column_pca(self, col_dim: int = 2):
        data_scaled = self.preprocess_data(data=self.data_contact_matrix)
        pca_1 = PCA(n_components=col_dim)
        pca_1.fit(data_scaled)

        print("Explained variance ratios:", pca_1.explained_variance_ratio_,
              "->", sum(pca_1.explained_variance_ratio_), "Eigenvectors:",
              pca_1.components_,  # (col_dim, 16)
              "Singular values:", pca_1.singular_values_)  # col_dim leading eigenvalues

        # Projection matrix for row direction matrix
        proj_matrix_1 = pca_1.components_.T  # 16 * col_dim projection matrix 1

        return proj_matrix_1

    def row_pca(self, row_dim: int = 2):
        data_scaled_2 = self.preprocess_data(data=self.contact_matrix_transposed)

        pca_2 = PCA(n_components=row_dim)
        pca_2.fit(data_scaled_2)

        print("Explained variance ratios 2:", pca_2.explained_variance_ratio_,
              "->", sum(pca_2.explained_variance_ratio_), "Eigenvectors 2:",
              pca_2.components_,  # (row_dim, 16)
              "Singular values 2:", pca_2.singular_values_)  # row_dim leading eigenvalues
        # print("PC 2", pc2)

        # Projection matrix for column direction matrix
        proj_matrix_2 = pca_2.components_.T  # 16 * row_dim projection matrix 2
        return proj_matrix_2

    def apply_dpca(self):
        # Now split concatenated original data into 39 sub-arrays of equal size i.e. 39 countries.
        data_scaled = self.preprocess_data(data=self.data_contact_matrix)
        split = np.array_split(data_scaled, 39)
        self.data_split = np.array(split)
        # Get projection matrix for column direction
        self.proj_matrix_1 = self.column_pca()
        # Get projection matrix for row direction
        self.proj_matrix_2 = self.row_pca()

        # Now apply (2D)^2 PCA simultaneously using projection matrix 1 and 2
        matrix = self.proj_matrix_1.T @ self.data_split @ self.proj_matrix_2

        # Now reshape the matrix to get desired 39 * 4
        self.pca_reduced = matrix.reshape((39, 4))
