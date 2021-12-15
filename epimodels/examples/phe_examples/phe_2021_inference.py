# Load necessary libraries
import os
import numpy as np
import pandas as pd
from scipy.stats import gamma
import epimodels as em
from iteration_utilities import deepflatten

# Populate the model
total_days = 496
regions = ['EE', 'London', 'Mid', 'NE', 'NW', 'SE', 'SW']
age_groups = ['0-1', '1-5', '5-15', '15-25', '25-45', '45-65', '65-75', '75+']

weeks = list(range(1, int(np.ceil(total_days/7))+1))
matrices_region = []

# Initial state of the system
for w in weeks:
    weeks_matrices_region = []
    for r in regions:
        path = os.path.join(
            '../../data/final_contact_matrices/{}_W{}.csv'.format(r, w))
        region_data_matrix = pd.read_csv(path, header=None, dtype=np.float64)
        # region_data_matrix_var.iloc[:, 5] = region_data_matrix_var.iloc[:, 5]
        # * 2
        regional = em.RegionMatrix(r, age_groups, region_data_matrix)
        weeks_matrices_region.append(regional)

    matrices_region.append(weeks_matrices_region)

contacts = em.ContactMatrix(
    age_groups, np.ones((len(age_groups), len(age_groups))))
matrices_contact = [contacts]

# Matrices contact
time_changes_contact = [1]
time_changes_region = np.arange(1, total_days+1, 7).tolist()

# Instantiate model
model = em.PheSEIRModel()

# Set the region names, contact and regional data of the model
model.set_regions(regions)
model.read_contact_data(matrices_contact, time_changes_contact)
model.read_regional_data(matrices_region, time_changes_region)

# Initial number of susceptibles
susceptibles = [
    [68124, 299908, 773741, 668994, 1554740, 1632059, 660187, 578319],
    [117840, 488164, 1140597, 1033029, 3050671, 2050173, 586472, 495043],
    [116401, 508081, 1321675, 1319046, 2689334, 2765974, 1106091, 943363],
    [85845, 374034, 978659, 1005275, 2036049, 2128261, 857595, 707190],
    [81258, 348379, 894662, 871907, 1864807, 1905072, 750263, 624848],
    [95825, 424854, 1141632, 1044242, 2257437, 2424929, 946459, 844757],
    [53565, 237359, 641486, 635602, 1304264, 1499291, 668999, 584130]]

# infectives1 = (10 * np.ones((len(regions), len(age_groups)))).tolist()
# infectives1 = (1 * np.ones((len(regions), len(age_groups)))).tolist()
infectives1 = [
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0]]
infectives2 = np.zeros((len(regions), len(age_groups))).tolist()

dI = 4
dL = 4

# Initial R number by region
# psis = gamma.rvs(31.36, scale=1/224, size=len(regions))
psis = (31.36/224)*np.ones(len(regions))
initial_r = np.multiply(
    dI*psis,
    np.divide(np.square((dL/2)*psis+1), 1 - 1/np.square((dI/2)*psis+1)))

# List of times at which we wish to evaluate the states of the compartments of
# the model
times = np.arange(1, total_days+1, 1).tolist()

# Simulate for all regions
output_scipy_solver = []

for r, reg in enumerate(regions):
    # List of common initial conditions and parameters that characterise the
    # fixed and variable model
    parameters = [
        initial_r, r+1, susceptibles,
        np.zeros((len(regions), len(age_groups))).tolist(),
        np.zeros((len(regions), len(age_groups))).tolist(),
        infectives1, infectives2,
        np.zeros((len(regions), len(age_groups))).tolist(),
        np.ones((len(regions), len(times))).tolist(), dL, dI, 0.5]

    # Simulate using the ODE solver from scipy
    scipy_method = 'RK45'
    parameters.append(scipy_method)

    output_scipy_solver.append(
        model.simulate(list(deepflatten(parameters, ignore=str)), times))

# Set information
fatality_ratio = (1/100 * np.array(
    [0.0016, 0.0016, 0.0043, 0.019, 0.08975, 0.815, 3.1, 6.05])).tolist()
time_to_death = [0.5] * len(times)
niu = float(gamma.rvs(1, scale=1/0.2, size=1))

tests = [np.array([[1000] * len(age_groups)] * len(times))] * len(regions)
sens = 0.7
spec = 0.95

# Read in death and positive data from external files
deaths_data = []
positives_data = []

for region in regions:
    deaths_data.append(
        np.loadtxt('inference_data/{}_Deaths.csv'.format(region),
                   dtype=int, delimiter=','))
    positives_data.append(
        np.loadtxt('inference_data/{}_Positives.csv'.format(region),
                   dtype=int, delimiter=','))

# Initialise inference for the model
phe_inference = em.PheSEIRInfer(model)

# Add death and tests data to the inference structure
phe_inference.read_deaths_data(deaths_data, fatality_ratio, time_to_death)
phe_inference.read_serology_data(tests, positives_data, sens, spec)

# Run inference structure
phe_inference.inference_problem_setup(times)
