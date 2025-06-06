#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#

import unittest

import numpy as np
from scipy.stats import gamma

import epimodels as em


#
# Toy Model Class
#

class TestPHEModel(em.PheSEIRModel):
    """
    Toy PHE model class used for testing.
    """
    def __init__(self):
        # Instantiate model
        super(TestPHEModel, self).__init__()

        # Populate the model
        regions = ['SW']
        age_groups = [
            '65-75', '75+']

        matrices_region = []

        # Initial state of the system
        weeks_matrices_region = []
        for r in regions:
            region_data_matrix = np.array([[0.5025, 0.1977], [0.1514, 0.7383]])
            regional = em.RegionMatrix(r, age_groups, region_data_matrix)
            weeks_matrices_region.append(regional)

        matrices_region.append(weeks_matrices_region)

        contacts = em.ContactMatrix(
            age_groups, np.ones((len(age_groups), len(age_groups))))
        matrices_contact = [contacts]

        # Matrices contact
        time_changes_contact = [1]
        time_changes_region = [1]

        # Set the region names, age groups, contact and regional data of the
        # model
        self.set_regions(regions)
        self.set_age_groups(age_groups)
        self.read_contact_data(matrices_contact, time_changes_contact)
        self.read_regional_data(matrices_region, time_changes_region)

    def set_initial_conditions(self, total_days):
        # Initial number of susceptibles
        susceptibles = [[668999, 584130]]

        # Initial number of infectives
        ICs_multiplier = 30
        infectives1 = (ICs_multiplier * np.ones(
            (len(self.regions), self._num_ages))).tolist()

        infectives2 = np.zeros(
            (len(self.regions), self._num_ages)).tolist()

        dI = 4
        dL = 4

        # Initial R number by region
        initial_r = [0.5]

        # List of times at which we wish to evaluate the states of the
        # compartments of the model
        times = np.arange(1, total_days+1, 1).tolist()

        # Temporal and regional fluctuation matrix in transmissibility
        betas = np.ones((len(self.regions), len(times))).tolist()

        # List of common initial conditions and parameters that characterise
        # the model
        parameters = [
            initial_r, 1, susceptibles,
            np.zeros(
                (len(self.regions), self._num_ages)).tolist(),
            np.zeros(
                (len(self.regions), self._num_ages)).tolist(),
            infectives1, infectives2,
            np.zeros(
                (len(self.regions), self._num_ages)).tolist(),
            betas, dL, dI, 0.5]

        # Simulate using the ODE solver from scipy
        scipy_method = 'RK45'
        parameters.append(scipy_method)


#
# Toy Death Data Class
#

class TestDeathData(object):
    """
    Toy Death Data class used for testing.
    """
    def __init__(self, total_days):
        # Toy values for data structures about death
        self.deaths = [4 * np.ones((total_days, 2), dtype=int)]

        td_mean = 15.0
        td_var = 12.1**2
        theta = td_var / td_mean
        k = td_mean / theta
        self.time_to_death = \
            gamma(k, scale=theta).pdf(np.arange(1, 31)).tolist()
        self.time_to_death.extend([0.0] * (total_days-30))

        self.deaths_times = np.arange(1, total_days+1, 1).tolist()
        self.fatality_ratio = (1/100 * np.array([3.1, 6.05])).tolist()

    def __call__(self):
        return (
            self.deaths, self.time_to_death, self.deaths_times,
            self.fatality_ratio)


#
# Toy Serology Data Class
#

class TestSerologyData(object):
    """
    Toy Serology Data class used for testing.
    """
    def __init__(self, total_days):
        # Toy values for data structures about serology
        self.tests_data = [100 * np.ones((total_days, 2), dtype=int)]
        self.positives_data = [10 * np.ones((total_days, 2), dtype=int)]
        self.serology_times = np.arange(1, total_days+1, 1).tolist()
        self.sens = 0.7
        self.spec = 0.95

    def __call__(self):
        return (
            self.tests_data, self.positives_data, self.serology_times,
            self.sens, self.spec)


#
# Test PHE Log-Likelihood Class
#

class TestPHELogLik(unittest.TestCase):
    """
    Test the 'PheSEIRModel' class.
    """
    def test__call__(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model, death and serology data
        model = TestPHEModel()
        model.set_initial_conditions(len(times))
        deaths, time_to_death, deaths_times, fatality_ratio = \
            TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()

        # Set toy model initial conditions
        susceptibles_data = [[668999, 584130]]
        infectives_data = [[30, 30]]

        # Set log-likelihood object
        log_lik = em.inference.PHELogLik(
            model, susceptibles_data, infectives_data, times,
            deaths, time_to_death, deaths_times, fatality_ratio,
            tests_data, positives_data, serology_times, sens, spec,
            wd=1, wp=1)

        self.assertIsInstance(log_lik([3, 1, 0.1]), (int, float))
        self.assertEqual(log_lik([3, 1, 0.1]) < 0, True)

    def test_n_parameters(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model, death and serology data
        model = TestPHEModel()
        model.set_initial_conditions(len(times))
        deaths, time_to_death, deaths_times, fatality_ratio = \
            TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()

        # Set toy model initial conditions
        susceptibles_data = [[668999, 584130]]
        infectives_data = [[30, 30]]

        # Set log-likelihood object
        log_lik = em.inference.PHELogLik(
            model, susceptibles_data, infectives_data, times,
            deaths, time_to_death, deaths_times, fatality_ratio,
            tests_data, positives_data, serology_times, sens, spec,
            wd=1, wp=1)

        self.assertEqual(log_lik.n_parameters(), 3)


#
# Test PHE Log-Prior Class
#

class TestPHELogPrior(unittest.TestCase):
    """
    Test the 'PHELogPrior' class.
    """
    def test__call__(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model
        model = TestPHEModel()
        model.set_initial_conditions(len(times))

        # Set log-prior object
        log_prior = em.inference.PHELogPrior(model, times)

        self.assertIsInstance(log_prior([3, 1, 0.1]), (int, float))
        self.assertEqual(log_prior([3, 1, 0.1]) < 0, True)

    def test_n_parameters(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model
        model = TestPHEModel()
        model.set_initial_conditions(len(times))

        # Set log-prior object
        log_prior = em.inference.PHELogPrior(model, times)

        self.assertEqual(log_prior.n_parameters(), 3)


#
# Test PHE Inference and Optimisation Class
#

class TestPheSEIRInfer(unittest.TestCase):
    """
    Test the 'PheSEIRInfer' class.
    """
    def test__init__(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model
        model = TestPHEModel()
        model.set_initial_conditions(len(times))

        # Set up PHE Inference class
        inference = em.inference.PheSEIRInfer(model)

        self.assertIsInstance(inference._model, em.PheSEIRModel)

        with self.assertRaises(TypeError):
            em.inference.PheSEIRInfer(0)

    def test_read_data(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model, death and serology data
        model = TestPHEModel()
        model.set_initial_conditions(len(times))
        deaths, time_to_death, deaths_times, fatality_ratio = \
            TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()

        # Set toy model initial conditions
        susceptibles_data = [[668999, 584130]]
        infectives_data = [[30, 30]]

        # Set up PHE Inference class
        inference = em.inference.PheSEIRInfer(model)

        # Test read_model_data
        inference.read_model_data(susceptibles_data, infectives_data)

        self.assertEqual(
            np.asarray(inference._susceptibles_data).shape, (1, 2))
        self.assertEqual(
            np.asarray(inference._infectives_data).shape, (1, 2))

        self.assertEqual(inference._susceptibles_data, susceptibles_data)
        self.assertEqual(inference._infectives_data, infectives_data)

        # Test read_serology_data
        inference.read_serology_data(
            tests_data, positives_data, serology_times, sens, spec)

        self.assertEqual(
            np.asarray(inference._total_tests).shape, (1, len(times), 2))
        self.assertEqual(
            np.asarray(inference._positive_tests).shape, (1, len(times), 2))

        self.assertEqual(inference._total_tests, tests_data)
        self.assertEqual(inference._positive_tests, positives_data)
        self.assertEqual(inference._serology_times, times)
        self.assertEqual(inference._sens, 0.7)
        self.assertEqual(inference._spec, 0.95)

        # Test read_deaths_data
        inference.read_deaths_data(
            deaths, deaths_times, time_to_death, fatality_ratio)

        self.assertEqual(
            np.asarray(inference._deaths).shape, (1, len(times), 2))

        self.assertEqual(inference._deaths, deaths)
        self.assertEqual(inference._deaths_times, times)
        self.assertEqual(inference._time_to_death, time_to_death)
        self.assertEqual(inference._fatality_ratio, fatality_ratio)

    def test_return_loglikelihood(self):
        # Set times for inference
        times = np.arange(1, 60, 1).tolist()

        # Set toy model, death and serology data
        model = TestPHEModel()
        model.set_initial_conditions(len(times))
        deaths, time_to_death, deaths_times, fatality_ratio = \
            TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()

        # Set toy model initial conditions
        susceptibles_data = [[668999, 584130]]
        infectives_data = [[30, 30]]

        # Set up PHE Inference class
        inference = em.inference.PheSEIRInfer(model)

        # Add model, death and tests data to the inference structure
        inference.read_model_data(susceptibles_data, infectives_data)
        inference.read_serology_data(
            tests_data, positives_data, serology_times, sens, spec)
        inference.read_deaths_data(
            deaths, deaths_times, time_to_death, fatality_ratio)

        # Compute the log likelihood at chosen point in the parameter space
        log_lik = inference.return_loglikelihood(times, [3, 1, 1, 1, 0.1])

        self.assertIsInstance(log_lik, (int, float))
        self.assertEqual(log_lik < 0, True)

    def test_optimisation_problem_setup(self):
        # Set times for optimisation
        times = np.arange(1, 60, 1).tolist()

        # Set toy model, death and serology data
        model = TestPHEModel()
        model.set_initial_conditions(len(times))
        deaths, time_to_death, deaths_times, fatality_ratio = \
            TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()

        # Set toy model initial conditions
        susceptibles_data = [[668999, 584130]]
        infectives_data = [[30, 30]]

        # Set up PHE Inference class for optimisation
        optimisation = em.inference.PheSEIRInfer(model)

        # Add model, death and tests data to the optimisation structure
        optimisation.read_model_data(susceptibles_data, infectives_data)
        optimisation.read_serology_data(
            tests_data, positives_data, serology_times, sens, spec)
        optimisation.read_deaths_data(
            deaths, deaths_times, time_to_death, fatality_ratio)

        # Set up and run the optimisation problem
        found, log_post_value = optimisation.optimisation_problem_setup(times)

        self.assertEqual(len(found), 5)
        self.assertIsInstance(log_post_value, (int, float))
        self.assertEqual(log_post_value < 0, True)

    def test_inference_problem_setup(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model, death and serology data
        model = TestPHEModel()
        model.set_initial_conditions(len(times))
        deaths, time_to_death, deaths_times, fatality_ratio = \
            TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()

        # Set toy model initial conditions
        susceptibles_data = [[668999, 584130]]
        infectives_data = [[30, 30]]

        # Set up PHE Inference class
        inference = em.inference.PheSEIRInfer(model)

        # Add model, death and tests data to the inference structure
        inference.read_model_data(susceptibles_data, infectives_data)
        inference.read_serology_data(
            tests_data, positives_data, serology_times, sens, spec)
        inference.read_deaths_data(
            deaths, deaths_times, time_to_death, fatality_ratio)

        # Set up and run the inference problem
        samples = inference.inference_problem_setup(times, num_iter=600)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (600, 3))
