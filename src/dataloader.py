import copy
import json
import os

import numpy as np
import xlrd


class DataLoader:
    """
    The DataLoader object manages getting all data necessary for simulation

    Member variables:
        *._data_file     Data file paths
        *._data          Loaded data
    """
    def __init__(self):
        """
        Constructor that defines file paths and loads all data
        """
        self._age_data_file = "../data/age_data.xlsx"
        self.contact_types = np.array(["home", "school", "work", "other"])
        self._contact_data_file = ["../data/contact_" + c_type + ".xls" for c_type in self.contact_types]
        self._model_parameters_data_file = "../data/model_parameters.json"
        # Get values for data members
        self._get_data()

    def _get_data(self):
        """
        Main function for data loading - defines all data members
        :return: None
        """
        self._get_age_data()
        self._get_contact_mtx()
        self._get_model_parameters_data()

    def _get_age_data(self):
        """
        Creates age specific attribute and fills with loaded data
        age_data: dict
          e.g.
          {"unit_1": {"pop": pop1, "age": np.array([age_pop_11, age_pop_12, ..., age_pop_1K])},
           "unit_2": {"pop": pop2, "age": np.array([age_pop_21, age_pop_22, ..., age_pop_2K])},
           ...
           "unit_N": {"pop": popN, "age": np.array([age_pop_N1, age_pop_N2, ..., age_pop_NK])}
          }
        :return: None
        """
        wb = xlrd.open_workbook(self._age_data_file)
        sheet = wb.sheet_by_index(0)
        datalist = [sheet.row_values(i) for i in range(1, sheet.nrows)]
        wb.unload_sheet(0)

        output = dict()
        population_global = 0
        age_distribution_global = None

        for row in datalist:
            # Store data for units
            key = row[0]
            # Assumes that the data file contains age distribution from second column until the end
            value = np.array(row[1:]).astype(int)
            output.update({key: {"pop": np.sum(value), "age": value}})
            # Store data for the whole country
            population_global += output[key]["pop"]
            if age_distribution_global is None:
                age_distribution_global = copy.deepcopy(output[key]["age"])
            else:
                age_distribution_global += output[key]["age"]

        self.global_unit_set = {"pop": population_global, "age": age_distribution_global}
        self.age_data = output

    def _get_contact_mtx(self):
        """
        Creates contact specific attribute and fills with loaded data
        contact_data: dict
          e.g.
          {"c_unit_1": contact_mtx_1, "c_unit_2": contact_mtx_2, ..., "c_unit_U": contact_mtx_U}
          where contact_mtx_i has the following structure:
             {"home":   np.array(globals.age, globals.age),
              "work":   np.array(globals.age, globals.age),
              "school": np.array(globals.age, globals.age),
              "other":  np.array(globals.age, globals.age)
             }
        :return: None
        """
        wb = xlrd.open_workbook(self._contact_data_file[0], on_demand=True)
        output = {key: dict() for key in wb.sheet_names()}
        for (path, contact_type) in zip(self._contact_data_file, self.contact_types):
            wb = xlrd.open_workbook(path)
            for country in wb.sheet_names():
                sheet = wb.sheet_by_name(country)
                datalist = np.array([sheet.row_values(i) for i in range(0, sheet.nrows)])
                wb.unload_sheet(country)
                datalist = self._transform_matrix(datalist, country)
                output[country].update({contact_type: datalist})
        self.contact_data = output

    def _get_model_parameters_data(self):
        """
        Creates attribute related to model parameters and fills with loaded data
        model_parameters_data: dict
          e.g.
          {"param_1": float,
           "param_2": np.array(globals.age),
           ...
           "param_P": float
          }
        :return: None
        """
        # Load model parameters
        with open(self._model_parameters_data_file) as f:
            parameters = json.load(f)
        self.model_parameters_data = dict()
        for param in parameters.keys():
            param_value = parameters[param]["value"]
            if isinstance(param_value, list):
                self.model_parameters_data.update({param: np.array(param_value)})
            else:
                self.model_parameters_data.update({param: param_value})

    def _transform_matrix(self, matrix: np.ndarray, country: str):
        """
        Transforms input contact matrix using age distribution of the global unit (e.g. Hungary)
        :param matrix: ndarray, e.g. np.array(globals.age, globals.age)
        :return: ndarray, has the same size as input matrix
        """
        age_distribution = self.age_data[country]["age"].reshape((-1, 1))
        matrix_1 = matrix * age_distribution
        output = (matrix_1 + matrix_1.T) / (2 * age_distribution)  # we symmetrize the matrix by the formula
        return output


