import numpy as np

from dataloader import DataLoader
from simulation import Simulation


class DataTransformer:
    def __init__(self):
        self.data = DataLoader()

        self.susc = 1.0
        self.base_r0 = 2.2

        self.upper_tri_indexes = np.triu_indices(16)
        self.country_names = list(self.data.age_data.keys())

        self.data_all_dict = dict()
        self.data_mtx_dict = dict()
        self.data_clustering = []

        self.get_data_for_clustering()

    def get_data_for_clustering(self):
        for country in self.country_names:
            age_vector = self.data.age_data[country]["age"].reshape((-1, 1))
            contact_matrix = self.data.contact_data[country]["home"] + \
                self.data.contact_data[country]["work"] + \
                self.data.contact_data[country]["school"] + \
                self.data.contact_data[country]["other"]
            contact_home = self.data.contact_data[country]["home"]
            contact_school = self.data.contact_data[country]["school"]
            contact_work = self.data.contact_data[country]["work"]
            contact_other = self.data.contact_data[country]["other"]

            susceptibility = np.array([1.0] * 16)
            susceptibility[:4] = self.susc
            simulation = Simulation(data=self.data, base_r0=self.base_r0,
                                    contact_matrix=contact_matrix,
                                    contact_home=contact_home,
                                    age_vector=age_vector,
                                    susceptibility=susceptibility)
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
            self.data_mtx_dict.update(
                {country: {"full": simulation.beta * contact_matrix[self.upper_tri_indexes],
                           "home": simulation.beta * contact_home[self.upper_tri_indexes],
                           "school": simulation.beta * contact_school[self.upper_tri_indexes],
                           "work": simulation.beta * contact_work[self.upper_tri_indexes],
                           "other": simulation.beta * contact_other[self.upper_tri_indexes]
                           }
                 })
            self.data_clustering.append(
                simulation.beta * contact_matrix[self.upper_tri_indexes])
        self.data_clustering = np.array(self.data_clustering)
