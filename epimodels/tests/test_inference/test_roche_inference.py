#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#

import unittest

import numpy as np
import numpy.testing as npt
from scipy.stats import gamma

import epimodels as em


#
# Toy Model Class
#

class TestRocheModel(em.RocheSEIRModel):
    """
    Toy Roche model class used for testing.
    """
    def __init__(self):
        # Instantiate model
        super(TestRocheModel, self).__init__()

        # Populate the model
        regions = ['SW']
        age_groups = ['65-75', '75+']

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

        # NPIs data
        max_levels_npi = [3, 3, 2, 4, 2, 3, 2, 4, 2]
        targeted_npi = [True, True, True, True, True, True, True, False, True]
        general_npi = [
            [True, False, True, True, False, False, False, False, False],
            [True, False, True, True, True, True, False, False, False]]
        time_changes_flag = [1, 12]

        reg_levels_npi = [
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]]]
        time_changes_npi = [1, 14]

        # Set the region names, age groups, contact and regional data of the
        # model
        self.set_regions(regions)
        self.set_age_groups(age_groups)
        self.read_contact_data(matrices_contact, time_changes_contact)
        self.read_regional_data(matrices_region, time_changes_region)
        self.read_npis_data(max_levels_npi, targeted_npi, general_npi,
                            reg_levels_npi, time_changes_npi,
                            time_changes_flag)

    def set_initial_conditions(self):
        # Initial number of susceptibles
        susceptibles = [[668999, 584130]]

        # Initial number of infectives
        ICs_multiplier = 30
        infectives_pre = (ICs_multiplier * np.ones(
            (len(self.regions), self._num_ages))).tolist()

        infectives_pre_ss = (ICs_multiplier * np.ones(
            (len(self.regions), self._num_ages))).tolist()

        # Average times in compartments
        k = 3.43
        kS = 2.57
        kQ = 1
        kR = 9 * np.ones(self._num_ages)
        kRI = 10 * np.ones(self._num_ages)

        # Proportion of asymptomatic, super-spreader and dead cases
        Pa = 0.658 * np.ones(self._num_ages)
        Pss = 0.0955
        Pd = 0.05 * np.ones(self._num_ages)

        # Transmission parameters
        beta_min = 0.228
        beta_max = 0.927
        bss = 3.11
        gamma = 1
        s50 = 35.3

        # List of common initial conditions and parameters that characterise
        # the model
        parameters = [
            1, susceptibles,
            np.zeros(
                (len(self.regions), self._num_ages)).tolist(),
            infectives_pre,
            np.zeros(
                (len(self.regions), self._num_ages)).tolist(),
            np.zeros(
                (len(self.regions), self._num_ages)).tolist(),
            infectives_pre_ss,
            np.zeros(
                (len(self.regions), self._num_ages)).tolist(),
            np.zeros(
                (len(self.regions), self._num_ages)).tolist(),
            np.zeros(
                (len(self.regions), self._num_ages)).tolist(),
            np.zeros(
                (len(self.regions), self._num_ages)).tolist(),
            np.zeros(
                (len(self.regions), self._num_ages)).tolist(),
            np.zeros(
                (len(self.regions), self._num_ages)).tolist(),
            k, kS, kQ, kR, kRI, Pa, Pss, Pd,
            beta_min, beta_max, bss, gamma, s50]

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
# Toy NPIs Data Class
#

class TestNPIsData(object):
    """
    Toy NPIs Data class used for testing.
    """
    def __init__(self, num_reg):
        # Toy values for data structures about NPIs
        self.max_levels_npi = [3, 3, 2, 4, 2, 3, 2, 4, 2]
        self.targeted_npi = [
            True, True, True, True, True, True, True, False, True]
        self.general_npi = [
            [True, False, True, True, False, False, False, False, False]]
        self.time_changes_flag = [1]

        self.time_changes_npi = [1]
        self.reg_levels_npi = np.tile(
            np.array([3, 3, 2, 4, 2, 3, 2, 4, 2]),
            (num_reg, len(self.time_changes_npi), 1)).tolist()

    def __call__(self):
        return (
            self.max_levels_npi, self.targeted_npi, self.general_npi,
            self.reg_levels_npi, self.time_changes_npi, self.time_changes_flag)


#
# Test Roche Log-Likelihood Class
#

class TestRocheLogLik(unittest.TestCase):
    """
    Test the 'RocheSEIRModel' class.
    """
    def test__call__(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model, death, serology and NPIs data
        model = TestRocheModel()
        model.set_initial_conditions()
        deaths, time_to_death, deaths_times, fatality_ratio = \
            TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()
        max_levels_npi, targeted_npi, general_npi, reg_levels_npi, \
            time_changes_npi, time_changes_flag = TestNPIsData(
                len(model.regions))()

        # Set toy model initial conditions
        susceptibles_data = [[668999, 584130]]
        infectives_data = [[30, 30]]

        # Set log-likelihood object
        log_lik = em.inference.RocheLogLik(
            model, susceptibles_data, infectives_data, times,
            deaths, time_to_death, deaths_times, fatality_ratio,
            tests_data, positives_data, serology_times, sens, spec,
            max_levels_npi, targeted_npi, general_npi, reg_levels_npi,
            time_changes_npi, time_changes_flag, wd=1, wp=1)

        self.assertIsInstance(log_lik([0.5, 3, 3, 3, 3, 50]), (int, float))
        self.assertEqual(log_lik([0.5, 3, 3, 3, 3, 50]) < 0, True)

    def test_n_parameters(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model, death, serology and NPIs data
        model = TestRocheModel()
        model.set_initial_conditions()
        deaths, time_to_death, deaths_times, fatality_ratio = \
            TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()
        max_levels_npi, targeted_npi, general_npi, reg_levels_npi, \
            time_changes_npi, time_changes_flag = TestNPIsData(
                len(model.regions))()

        # Set toy model initial conditions
        susceptibles_data = [[668999, 584130]]
        infectives_data = [[30, 30]]

        # Set log-likelihood object
        log_lik = em.inference.RocheLogLik(
            model, susceptibles_data, infectives_data, times,
            deaths, time_to_death, deaths_times, fatality_ratio,
            tests_data, positives_data, serology_times, sens, spec,
            max_levels_npi, targeted_npi, general_npi, reg_levels_npi,
            time_changes_npi, time_changes_flag, wd=1, wp=1)

        self.assertEqual(log_lik.n_parameters(), 6)


#
# Test Roche Log-Prior Class
#

class TestRocheLogPrior(unittest.TestCase):
    """
    Test the 'RocheLogPrior' class.
    """
    def test__call__(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model
        model = TestRocheModel()
        model.set_initial_conditions()

        # Set log-prior object
        log_prior = em.inference.RocheLogPrior(model, times)

        self.assertIsInstance(log_prior([0.5, 3, 3, 3, 3, 50]), (int, float))
        self.assertEqual(log_prior([0.5, 3, 3, 3, 3, 50]) < 0, True)

    def test_n_parameters(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model
        model = TestRocheModel()
        model.set_initial_conditions()

        # Set log-prior object
        log_prior = em.inference.RocheLogPrior(model, times)

        self.assertEqual(log_prior.n_parameters(), 6)


#
# Test Roche Inference and Optimisation Class
#

class TestRocheSEIRInfer(unittest.TestCase):
    """
    Test the 'RocheSEIRInfer' class.
    """
    def test__init__(self):
        # Set toy model
        model = TestRocheModel()
        model.set_initial_conditions()

        # Set up Roche Inference class
        inference = em.inference.RocheSEIRInfer(model)

        self.assertIsInstance(inference._model, em.RocheSEIRModel)

        with self.assertRaises(TypeError):
            em.inference.RocheSEIRInfer(0)

    def test_read_data(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model, death, serology and NPIs data
        model = TestRocheModel()
        model.set_initial_conditions()
        deaths, time_to_death, deaths_times, fatality_ratio = \
            TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()
        max_levels_npi, targeted_npi, general_npi, reg_levels_npi, \
            time_changes_npi, time_changes_flag = TestNPIsData(
                len(model.regions))()

        # Set toy model initial conditions
        susceptibles_data = [[668999, 584130]]
        infectives_data = [[30, 30]]

        # Set up Roche Inference class
        inference = em.inference.RocheSEIRInfer(model)

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

        # Test read_npis_data
        inference.read_npis_data(
            max_levels_npi, targeted_npi, general_npi, reg_levels_npi,
            time_changes_npi, time_changes_flag
        )

        self.assertEqual(
            np.asarray(inference._deaths).shape, (1, len(times), 2))

        self.assertEqual(inference._max_levels_npi, max_levels_npi)
        self.assertEqual(inference._targeted_npi, targeted_npi)
        self.assertEqual(inference._general_npi, general_npi)
        npt.assert_array_equal(inference._reg_levels_npi, reg_levels_npi)
        self.assertEqual(inference._time_changes_npi, time_changes_npi)
        self.assertEqual(inference._time_changes_flag, time_changes_flag)

    def test_return_loglikelihood(self):
        # Set times for inference
        times = np.arange(1, 60, 1).tolist()

        # Set toy model, death, serology and NPIs data
        model = TestRocheModel()
        model.set_initial_conditions()
        deaths, time_to_death, deaths_times, fatality_ratio = \
            TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()
        max_levels_npi, targeted_npi, general_npi, reg_levels_npi, \
            time_changes_npi, time_changes_flag = TestNPIsData(
                len(model.regions))()

        # Set toy model initial conditions
        susceptibles_data = [[668999, 584130]]
        infectives_data = [[30, 30]]

        # Set up Roche Inference class
        inference = em.inference.RocheSEIRInfer(model)

        # Add model, death and tests data to the inference structure
        inference.read_model_data(susceptibles_data, infectives_data)
        inference.read_serology_data(
            tests_data, positives_data, serology_times, sens, spec)
        inference.read_deaths_data(
            deaths, deaths_times, time_to_death, fatality_ratio)
        inference.read_npis_data(
            max_levels_npi, targeted_npi, general_npi, reg_levels_npi,
            time_changes_npi, time_changes_flag
        )

        # Compute the log likelihood at chosen point in the parameter space
        log_lik = inference.return_loglikelihood(times, [0.5, 3, 3, 3, 3, 50])

        self.assertIsInstance(log_lik, (int, float))
        self.assertEqual(log_lik < 0, True)

    def test_optimisation_problem_setup(self):
        # Set times for optimisation
        times = np.arange(1, 60, 1).tolist()

        # Set toy model, death, serology and NPIs data
        model = TestRocheModel()
        model.set_initial_conditions()
        deaths, time_to_death, deaths_times, fatality_ratio = \
            TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()
        max_levels_npi, targeted_npi, general_npi, reg_levels_npi, \
            time_changes_npi, time_changes_flag = TestNPIsData(
                len(model.regions))()

        # Set toy model initial conditions
        susceptibles_data = [[668999, 584130]]
        infectives_data = [[30, 30]]

        # Set up Roche Inference class for optimisation
        optimisation = em.inference.RocheSEIRInfer(model)

        # Add model, death, tests and NPIs data to the optimisation structure
        optimisation.read_model_data(susceptibles_data, infectives_data)
        optimisation.read_serology_data(
            tests_data, positives_data, serology_times, sens, spec)
        optimisation.read_deaths_data(
            deaths, deaths_times, time_to_death, fatality_ratio)
        optimisation.read_npis_data(
            max_levels_npi, targeted_npi, general_npi, reg_levels_npi,
            time_changes_npi, time_changes_flag
        )

        # Set up and run the optimisation problem
        found, log_post_value = optimisation.optimisation_problem_setup(times)

        self.assertEqual(len(found), 6)
        self.assertIsInstance(log_post_value, (int, float))
        self.assertEqual(log_post_value < 0, True)

    def test_inference_problem_setup(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model, death, serology and NPIs data
        model = TestRocheModel()
        model.set_initial_conditions()
        deaths, time_to_death, deaths_times, fatality_ratio = \
            TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()
        max_levels_npi, targeted_npi, general_npi, reg_levels_npi, \
            time_changes_npi, time_changes_flag = TestNPIsData(
                len(model.regions))()

        # Set toy model initial conditions
        susceptibles_data = [[668999, 584130]]
        infectives_data = [[30, 30]]

        # Set up Roche Inference class
        inference = em.inference.RocheSEIRInfer(model)

        # Add model, death, tests and NPIs data to the inference structure
        inference.read_model_data(susceptibles_data, infectives_data)
        inference.read_serology_data(
            tests_data, positives_data, serology_times, sens, spec)
        inference.read_deaths_data(
            deaths, deaths_times, time_to_death, fatality_ratio)
        inference.read_npis_data(
            max_levels_npi, targeted_npi, general_npi, reg_levels_npi,
            time_changes_npi, time_changes_flag
        )

        # Set up and run the inference problem
        samples = inference.inference_problem_setup(times, num_iter=600)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (600, 6))
