import numpy as np
from tqdm import tqdm
from model import RostHungaryModel
from r0 import R0Generator


def get_contact_matrix_from_upper_triu(rvector, number_of_age_groups, age_vector):
    upper_tri_indexes = np.triu_indices(number_of_age_groups)
    new_contact_mtx = np.zeros((number_of_age_groups, number_of_age_groups))
    new_contact_mtx[upper_tri_indexes] = rvector
    new_2 = new_contact_mtx.T
    new_2[upper_tri_indexes] = rvector
    vector = np.array(new_2 / age_vector)
    return vector


class Simulation:

    def __init__(self, data, contact_matrix, contact_home, age_vector, susceptibility, base_r0):
        # Set parameters
        self.data = data
        self.contact_matrix = contact_matrix
        self.contact_home = contact_home
        self.age_vector = age_vector

        self.params = self.data.model_parameters_data
        self.params.update({"susc": susceptibility})

        r0generator = R0Generator(param=self.params)
        self.beta = base_r0 / r0generator.get_eig_val(self.contact_matrix)
        self.params.update({"beta": self.beta})

        # Initial values for model
        self.iv = {"l1": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "l2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "ip": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "ia1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "ia2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "ia3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "is1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "is2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "is3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "ih": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "ic": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "icr": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "d": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "r": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "c": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        self.iv.update({"s": self.data.global_unit_set["age"] - self.iv["l1"]})

        self.t = np.linspace(0, 600, 600)
        self.output = []

        self.run_plotting = True

    def sim_run(self, lhs_table):
        results = list(tqdm(map(self.solve_model, lhs_table), total=lhs_table.shape[0]))
        results = np.array(results)
        results = results[results[:, 137].argsort()]
        self.output = np.array(results)

    def solve_model(self, rvector):

        vector = get_contact_matrix_from_upper_triu(rvector=rvector,
                                                    number_of_age_groups=16,
                                                    age_vector=self.age_vector)

        model = RostHungaryModel(init_values=self.iv, contact_matrix=np.reshape(vector, (16, 16)),
                                 parameters=self.params)

        output = np.append(rvector, [0, model.r0])
        # output = np.append(output, list(model.get_final_size_distribution()))
        output = np.append(output, [0] * 16)

        return list(output)

    def solve_model_plot(self, rvector):
        vector = get_contact_matrix_from_upper_triu(rvector=rvector,
                                                    number_of_age_groups=16,
                                                    age_vector=self.age_vector)

        model = RostHungaryModel(init_values=self.iv, contact_matrix=np.reshape(vector, (16, 16)),
                                 parameters=self.params, to_solve=True)
        return model.solution
