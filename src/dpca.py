
import numpy as np

from src.dataloader import DataLoader
from src.data_transformer import DataTransformer
from src.simulation import Simulation


class DPCA:
    def __init__(self, data_tr, country_names, data_contact_matrix):
        self.data = DataLoader()
        self.country_names = country_names
        self.data_tr = data_tr

        self.data_contact_matrix = data_contact_matrix
        self.matrix_reduced = dict()
        self.reduced_contact_matrix = []
        self.apply_dpca(data_tr=data_tr)

    def apply_dpca(self, data_tr):
        for country in self.country_names:
            # create the scaled full contact matrix for all the 39 countries
            # remove the countries keys from the dictionary
            # provides a numpy array without countries for easy manipulation
            data_matrix = data_tr.data_contact_matrix.pop(country)  # 16*16 matrix for each country

            # for PCA, we need a centered data
            centered_data = data_matrix - np.mean(data_matrix, axis=0)  # compute mean along the rows

            # 2DPCA works in the row direction i.e. Y = AX
            # Let's calculate the row-row covariance matrix in the row-row direction
            row_cov = np.cov(centered_data)  # symmetric

            # get the eigenvectors and eigenvalues
            row_eigen_values, row_eigen_vectors = np.linalg.eigh(row_cov)

            # sort the eigenvalues in descending order i.e. from largest to smallest
            sorted_index = np.argsort(row_eigen_values)[::-1]  # Perform an indirect sort along the given axis
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
            self.reduced_contact_matrix.append(matrix.flatten())
        self.reduced_contact_matrix = np.array(self.reduced_contact_matrix)  # 39*4 matrix
        self.matrix_reduced = self.matrix_reduced   # 2*2 matrix for each country with the countries
        self.data_contact_matrix = data_tr.data_contact_matrix
        # print(self.matrix_reduced)
