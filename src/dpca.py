
import numpy as np
import pandas as pd

from src.dataloader import DataLoader
from src.data_transformer import DataTransformer
from src.simulation import Simulation
from sklearn.decomposition import PCA
from sklearn import preprocessing


class DPCA:
    """
        This is the classical PCA class which uses PCA and SVD approach.
        input: 39 countries each 16 * 16 matrix concatenated row wise and column wise
        output: 39 countries each 2 * 2 matrix, and 39 * 4 (2 * 2 flatten matrix)
    """
    def __init__(self, data_tr, country_names, data_contact_matrix, data_contact_hmatrix, flatten_matrix):
        self.data = DataLoader()
        self.country_names = country_names
        self.data_tr = data_tr

        self.data_contact_matrix = data_contact_matrix
        self.data_contact_hmatrix = data_contact_hmatrix
        self.flatten_matrix = flatten_matrix
        self.pca_reduced = []
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
              pca1.singular_values_)  # 2

        # Split concatenated array into 39 sub-arrays of equal size i.e. 39 countries.
        split = np.array_split(pc, 39)

        # convert split to a numpy array
        arr_t = np.array(split)

        # Now we reshape the array
        result = arr_t.reshape((39, 2, 16))
        # self.matrix_reduced = split

        # working with 16 * 624 matrix (h-stack)
        data_scaled2 = preprocessing.scale(self.data_contact_hmatrix)
        pca2 = PCA(n_components=2)
        pca2.fit(data_scaled2)
        pca2_data = pca2.transform(data_scaled2)

        # create a DataFrame that will execute principal component of the 16 * 624 dimension matrix
        pc2 = pd.DataFrame(data=pca2_data,
                           columns=['pc 1', 'pc 2'])  # 2 * 216

        print("Explained variance ratios 2:", pca2.explained_variance_ratio_,
              "->", sum(pca2.explained_variance_ratio_), "principal components 2:",
              pca2.components_,  # 2 * 624
              pca2.singular_values_)  # 2

        # using pc2 as the projection matrix, we can create 2 * 2 matrix for each country
        matrix = np.dot(result, pc2)  # 2 * 2 matrices for each country

        # let's flatten the matrix
        s = matrix.flatten()

        # Now reshape the matrix to get desired 39 * 4
        self.pca_reduced = s.reshape((39, 4))


class Dimension:
    """
    This is the (2D)^2 PCA class which simultaneously considers both 2DPCA(only works in the row direction) and
    alternative 2DPCA (works on the column direction).
    input: 39 countries each 16 * 16 matrix
    output: 39 countries each 2 * 2 matrix, and 39 * 4 (2 * 2 flatten matrix)
    """
    def __init__(self, data_tr, country_names, data_matrix):
        self.data = DataLoader()
        self.country_names = country_names
        self.data_tr = data_tr

        self.data_matrix = data_matrix
        self.matrix_reduced = dict()
        self.reduced_matrix = []
        self.get_dpca(data_tr=data_tr)

    def get_dpca(self, data_tr):
        for country in self.country_names:
            # create the scaled full contact matrix for all the 39 countries
            # remove the countries keys from the dictionary
            # provides a numpy array without countries for easy manipulation
            data = data_tr.data_matrix.pop(country)  # 16*16 matrix for each country

            # for PCA, we need a centered data
            centered_data = data - np.mean(data, axis=0)  # compute mean along the rows

            # 2DPCA works in the row direction i.e. Y = AX
            # Let's calculate the row-row covariance matrix in the row-row direction
            row_cov = np.cov(centered_data)  # symmetric

            # get the eigenvectors and eigenvalues
            row_eigen_values, row_eigen_vectors = np.linalg.eigh(row_cov)

            # sort the eigenvalues in descending order
            sorted_index = np.argsort(row_eigen_values)[::-1]
            row_sorted_eigenvalues = row_eigen_values[sorted_index]

            # sort the eigenvectors as well
            row_sorted_eigenvectors = row_eigen_vectors[:, sorted_index]

            # Now we reduce the dimension to 2
            row_subset_eigenvector = row_sorted_eigenvectors[:, 0:2]

            # Now transform the data
            row_matrix = np.dot(centered_data, row_subset_eigenvector)  # 16*2 for each country

            # alternative_two_DPCA
            # works in the column direction
            col_cov = np.cov(centered_data, rowvar=False)  # column direction
            col_eigen_values, col_eigen_vectors = np.linalg.eigh(col_cov)

            # get the eigenvalues and the eigen vectors
            # sort again the eigenvalues in descending order
            col_sorted_index = np.argsort(col_eigen_values)[::-1]
            col_sorted_eigenvalues = col_eigen_values[col_sorted_index]

            # sort the eigenvectors as well
            col_sorted_eigenvectors = col_eigen_vectors[:, col_sorted_index]

            # Now we reduce the dimension to 2
            col_subset_eigenvector = col_sorted_eigenvectors[:, 0:2]

            # Now transform the data, B = Z^T * A
            B = np.dot(col_subset_eigenvector.transpose(), centered_data)  # 2*16 for each country

            # Now combine matrix Y and matrix B to get 2*2 matrix (2D^2 PCA)
            matrix = np.dot(col_subset_eigenvector.transpose(), row_matrix)  # 2*2 matrix for each country

            d = dict(enumerate(np.array(matrix), 1))  # Use flatten and then create the dictionary with the help of
            # enumerate starting from 1
            # update the countries
            self.matrix_reduced.update(
                {country: d})
            reduced_contact_matrix = matrix
            self.reduced_matrix.append(matrix.flatten())
        self.reduced_matrix = np.array(self.reduced_matrix)  # 39*4 matrix
        self.matrix_reduced = self.matrix_reduced   # 2*2 matrix for each country with the countries
        self.data_matrix = data_tr.data_matrix
        # print(self.matrix_reduced)













