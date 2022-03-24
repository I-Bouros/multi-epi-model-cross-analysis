#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#

import unittest

import os
import numpy as np
import pandas as pd
import numpy.testing as npt
from scipy.stats import gamma
from iteration_utilities import deepflatten

import epimodels as em


#
# Testing Model Class
#

class TestPHEModel(em.PheSEIRModel):
    """
    """
    def __init__(self):
        # Populate the model
        regions = ['SW']
        age_groups = [
            '0-1', '1-5']

        matrices_region = []

        # Initial state of the system
        weeks_matrices_region = []
        for r in regions:
            path = os.path.join('../../data/final_contact_matrices/BASE.csv')
            region_data_matrix = pd.read_csv(
                path, header=None, dtype=np.float64)
            regional = em.RegionMatrix(r, age_groups, region_data_matrix)
            weeks_matrices_region.append(regional)

        matrices_region.append(weeks_matrices_region)

        contacts = em.ContactMatrix(
            age_groups, np.ones((len(age_groups), len(age_groups))))
        matrices_contact = [contacts]

        # Matrices contact
        time_changes_contact = [1]
        time_changes_region = [1]

        # Instantiate model
        self._model = em.PheSEIRModel()

        # Set the region names, age groups, contact and regional data of the
        # model
        self._model.set_regions(regions)
        self._model.set_age_groups(age_groups)
        self._model.read_contact_data(matrices_contact, time_changes_contact)
        self._model.read_regional_data(matrices_region, time_changes_region)

    def set_initial_conditions(self, total_days):
        # Initial number of susceptibles
        susceptibles = [[53565, 237359]]

        # Initial number of infectives
        ICs_multiplier = 30
        infectives1 = [ICs_multiplier] * self._model._num_ages

        infectives2 = np.zeros(
            (len(self._model.regions), self._model._num_ages)).tolist()

        dI = 4
        dL = 4

        # Initial R number by region - use mean value from prior for psi
        psis = (31.36/224)*np.ones(len(self._model.regions))
        initial_r = np.multiply(
            dI*psis,
            np.divide(np.square((dL/2)*psis+1), 1-1/np.square((dI/2)*psis+1)))

        # List of times at which we wish to evaluate the states of the
        # compartments of the model
        times = np.arange(1, total_days+1, 1).tolist()

        # Temporal and regional fluctuation matrix in transmissibility
        betas = np.ones((len(self._model.regions), len(times))).tolist()

        # List of common initial conditions and parameters that characterise
        # the model
        parameters = [
            initial_r, 1, susceptibles,
            np.zeros(
                (len(self._model.regions), self._model._num_ages)).tolist(),
            np.zeros(
                (len(self._model.regions), self._model._num_ages)).tolist(),
            infectives1, infectives2,
            np.zeros(
                (len(self._model.regions), self._model._num_ages)).tolist(),
            betas, dL, dI, 0.5]

        # Simulate using the ODE solver from scipy
        scipy_method = 'RK45'
        parameters.append(scipy_method)

        return parameters, times


class TestPHELogLik(unittest.TestCase):
    """
    Test the 'PheSEIRModel' class.
    """
    def test__init__(self):
        model = TestPHEModel()
        em.inference.PHELogLik()
