import numpy as np

from dataloader import DataLoader
from simulation import Simulation


class DataTransformer:
    def __init__(self, susc: float = 1.0, base_r0: float = 2.2):
        self.susc = susc
        self.base_r0 = base_r0

        self.data = DataLoader()
        self.upper_tri_indexes = np.triu_indices(16)
        self.country_names = list(self.data.age_data.keys())

        self.data_all_dict = dict()
        self.data_cm_d2pca_col = []
        self.data_cm_d2pca_row = []
        self.data_cm_1dpca = []
        self.data_cm_pca = []

        self.get_data_for_clustering()

    def get_data_for_clustering(self):
        for country in self.country_names:
            age_vector = self.data.age_data[country]["age"].reshape((-1, 1))
            contact_home = self.data.contact_data[country]["home"]
            contact_school = self.data.contact_data[country]["school"]
            contact_work = self.data.contact_data[country]["work"]
            contact_other = self.data.contact_data[country]["other"]
            contact_matrix = contact_home + contact_school + contact_work + contact_other

            susceptibility = np.array([1.0] * 16)
            susceptibility[:4] = self.susc
            simulation = Simulation(data=self.data, base_r0=self.base_r0,
                                    contact_matrix=contact_matrix,
                                    contact_home=contact_home,
                                    age_vector=age_vector,
                                    susceptibility=susceptibility)
            # Create dictionary with all necessary data
            self.data_all_dict.update(
                {country: {"beta": simulation.beta,
                           "age_vector": age_vector,
                           "contact_full": contact_matrix,
                           "contact_home": contact_home,
                           "contact_school": contact_school,
                           "contact_work": contact_work,
                           "contact_other": contact_other
                           }
                 })
            # Create separated data structure for (2D)^2 PCA
            self.data_cm_d2pca_col.append(
                simulation.beta * contact_matrix)
            self.data_cm_d2pca_row.append(
                simulation.beta * contact_matrix.T
            )
            # Create separated data structure for 1D PCA
            self.data_cm_1dpca.append(
                simulation.beta * contact_matrix[self.upper_tri_indexes])
        self.data_cm_1dpca = np.array(self.data_cm_1dpca)
        # Final shape of the np.ndarrays: (624, 16)
        self.data_cm_d2pca_col = np.vstack(self.data_cm_d2pca_col)
        self.data_cm_d2pca_row = np.vstack(self.data_cm_d2pca_row)
