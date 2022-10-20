# Clustering of countries based on the associated social contact patterns in epidemiological modelling

This repository was developed for paper Korir, E.K. and Vizi, Z., Clustering of countries based on the 
associated social contact patterns in epidemiological modelling.

## Install
This project is developed and tested with Python 3.8+. To install project dependencies, execute the following steps:
1) Install *virtualenv*: `pip3 install virtualenv`
2) Create a virtual environment (named *venv*) in the project directory: `python3 -m venv venv`
3) Activate *venv*: `source venv/bin/activate`
4) Install dependencies from `requirements.txt`: `pip install -r requirements.txt`

## Data

- contact matrices for countries can be found in `./data/contact_{x}.xls` file, 
where _x_ can be 'home', 'school', 'work', 'other'
- population vector for the European countries are located in `./data/age_data.xlsx`
- epidemic model parameters are listed in `./data/model_parameters.json`

## Framework
A summary about the steps of the procedure:
![alt text](https://drive.google.com/uc?export=download&id=1NiFSmvhrkwG6QVQjm-aQbX1u0BHtvUh_)

## Simulation
Code for running the framework is available in folder `./src`. 
Here you find
- the class for running the complete pipeline (`./src/analysis.py`)
- the data loading functionalities (`./src/dataloader.py`)
- the classes related to epidemic modelling (`./src/model.py`, `./src/r0.py` and `./src/simulation.py`)
- the standardization module (`./src/data_transformer.py`)
- the class for dimensionality reduction (`./src/d2pca.py`)
- the module for executing clustering (`./src/hierarchical.py`)

## Plots
If you run the code, the plots will be generated in the in-use created folder `./plots`.
