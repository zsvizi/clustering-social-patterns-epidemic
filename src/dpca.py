
import numpy as np
import pandas as pd

from src.dataloader import DataLoader
from src.data_transformer import DataTransformer
from src.simulation import Simulation
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import preprocessing
from scipy.linalg import svd


class DPCA:
    """
        This is the classical PCA class which uses PCA, SVD, and covariance approach.
        input: 39 countries each 16 * 16 matrix concatenated row wise and column wise
        output: 39 countries each 2 * 2 matrix, and 39 * 4 (2 * 2 flatten matrix)
    """
    def __init__(self, data_tr, country_names, data_contact_matrix):
        self.data = DataLoader()
        self.country_names = country_names
        self.data_tr = data_tr

        self.data_contact_matrix = data_contact_matrix
        self.pca_reduced = []
        self.apply_dpca()

    def apply_dpca(self):

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
              "->", sum(pca1.explained_variance_ratio_), "Eigenvectors:",
              pca1.components_,  # (2, 16)
              "Singular values:", pca1.singular_values_)  # 2 leading eigenvalues
        print("PC", pc)

        # Split concatenated array into 39 sub-arrays of equal size i.e. 39 countries.
        split = np.array_split(pc, 39)

        # convert split to a numpy array
        arr_t = np.array(split)

        # Now we reshape the array
        result = arr_t.reshape((39, 2, 16))

        # using the principal components as the projection matrix, we can create 2 * 2 matrix for each country
        matrix = np.dot(result, pca1.components_.transpose())  # 2 * 2 matrices for each country

        # let's flatten the matrix
        s = matrix.flatten()

        # Now reshape the matrix to get desired 39 * 4
        self.pca_reduced = s.reshape((39, 4))

    def reduce_dimension_svd(self):
        # Let's scale the data
        scaled_data = preprocessing.scale(self.data_contact_matrix)

        # compute svd of the matrix
        u_, s_, vt = svd(scaled_data, full_matrices=False)

        # rows of vT provides direction of the maximum variance since s_ is in descending order
        # let's consider first two directions
        eigenvectors = vt[0:2]

        # principal components can be computed conveniently using:
        pcs = u_ @ np.diag(s_)

        # pcs2 = scaled_data @ vt[0:2].T
        # print("pcs2", pcs2)

        # We want to retain only 2 principal components
        print(("SVD PCS:", pcs[:, 0:2][0:5]))  # just 5 rows
        print("Eigenvectors:", eigenvectors)
        print("SVD Singular values:", s_[0:2])  # first 2 values

        # alternatively this svd process can be done easily using truncated svd
        SVD = TruncatedSVD(n_components=2, random_state=2)
        SVD.fit_transform(scaled_data)
        # print("SVD Explained variance ratios:", SVD.explained_variance_ratio_,
        #      "->", sum(SVD.explained_variance_ratio_), "SVD Eigenvectors:",
        #      SVD.components_,  # (2, 16)
        #      "SVD Singular values:", SVD.singular_values_)  # 2

    def covariance_dimension_reduction(self):
        # let's center the data around the mean
        data_centered = self.data_contact_matrix - np.mean(self.data_contact_matrix, axis=0)  # mean along the rows

        # scaling the data rather than centering it
        scale = preprocessing.scale(self.data_contact_matrix)  # 624 * 16

        # Let's calculate the col-col covariance matrix in the row-row direction
        cov = np.cov(scale, rowvar=False)  # 16 * 16, symmetric and positive definite

        # get the eigenvectors and eigenvalues
        eigen_values, eigen_vectors = np.linalg.eigh(cov)

        # sort the eigenvalues in descending order
        sorted_index = np.argsort(eigen_values)[::-1]  # negate an array, the lowest elements become the highest
        sorted_eigenvalues = eigen_values[sorted_index]

        # sort the eigenvectors as well
        sorted_eigenvectors = eigen_vectors[:, sorted_index]

        # Now we reduce the dimension to 2 i.e. directions with more variance
        subset_eigenvector = sorted_eigenvectors[:, 0:2]  # 16 * 2

        # Now we project our data onto this principal axis
        dim = np.dot(scale, subset_eigenvector)  # 624 * 2
        print("d", dim[0:5])











