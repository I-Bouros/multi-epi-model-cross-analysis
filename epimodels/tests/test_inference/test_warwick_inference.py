#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#

import unittest

import os
import numpy as np
import numpy.testing as npt
import pandas as pd
import scipy.stats

import epimodels as em


#
# Change number of age groups function
#

def update_age_groups(population, parameter_vector):
    """
    """
    new_vector = np.empty(8)

    ind_old = [
        np.array([0]),
        np.array([0]),
        np.array(range(1, 3)),
        np.array(range(3, 5)),
        np.array(range(5, 9)),
        np.array(range(9, 13)),
        np.array(range(13, 15)),
        np.array(range(15, 21))]

    for _ in range(8):
        new_vector[_] = np.average(
            np.array(parameter_vector)[ind_old[_][:, None]],
            weights=population[ind_old[_][:, None]])

    return new_vector


#
# Toy Model Class
#

class TestWarwickModel(em.WarwickSEIRModel):
    """
    Toy Warwick model class used for testing.
    """
    def __init__(self):
        # Instantiate model
        super(TestWarwickModel, self).__init__()

        # Populate the model
        regions = ['SW']
        age_groups = [
            '0-1', '1-5', '5-15', '15-25', '25-45', '45-65', '65-75', '75+']

        house_matrices_region = []
        school_matrices_region = []
        work_matrices_region = []
        other_matrices_region = []

        # Initial state of the system
        house_weeks_matrices_region = []
        school_weeks_matrices_region = []
        work_weeks_matrices_region = []
        other_weeks_matrices_region = []

        for r in regions:
            path = os.path.join(
                os.path.dirname(__file__),
                '../../data/final_contact_matrices/house_BASE.csv')
            house_region_data_matrix = pd.read_csv(
                path, header=None, dtype=np.float64)
            house_regional = em.RegionMatrix(
                r, age_groups, house_region_data_matrix)
            house_weeks_matrices_region.append(house_regional)

            path2 = os.path.join(
                os.path.dirname(__file__),
                '../../data/final_contact_matrices/school_BASE.csv')
            school_region_data_matrix = pd.read_csv(
                path2, header=None, dtype=np.float64)
            school_regional = em.RegionMatrix(
                r, age_groups, school_region_data_matrix)
            school_weeks_matrices_region.append(school_regional)

            path3 = os.path.join(
                os.path.dirname(__file__),
                '../../data/final_contact_matrices/work_BASE.csv')
            work_region_data_matrix = pd.read_csv(
                path3, header=None, dtype=np.float64)
            work_regional = em.RegionMatrix(
                r, age_groups, work_region_data_matrix)
            work_weeks_matrices_region.append(work_regional)

            path4 = os.path.join(
                os.path.dirname(__file__),
                '../../data/final_contact_matrices/other_BASE.csv')
            other_region_data_matrix = pd.read_csv(
                path4, header=None, dtype=np.float64)
            other_regional = em.RegionMatrix(
                r, age_groups, other_region_data_matrix)
            other_weeks_matrices_region.append(other_regional)

        house_matrices_region.append(house_weeks_matrices_region)
        school_matrices_region.append(school_weeks_matrices_region)
        work_matrices_region.append(work_weeks_matrices_region)
        other_matrices_region.append(other_weeks_matrices_region)

        contacts = em.ContactMatrix(
            age_groups, np.ones((len(age_groups), len(age_groups))))
        house_matrices_contact = [contacts]
        school_matrices_contact = [contacts]
        work_matrices_contact = [contacts]
        other_matrices_contact = [contacts]

        # Matrices contact
        time_changes_contact = [1]
        time_changes_region = [1]

        # Set the region names, age groups, contact and regional data of the
        # model
        self.set_regions(regions)
        self.set_age_groups(age_groups)
        self.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        self.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

    def set_initial_conditions(self):
        # Initial number of susceptibles
        susceptibles = [np.loadtxt(os.path.join(
            os.path.dirname(__file__),
            '../../data/england_population/England_population.csv'),
            dtype=int, delimiter=',').tolist()[-1]]

        # Initial number of infectives
        ICs_multiplier = 40
        detected_f = (ICs_multiplier * np.ones(
            (len(self.regions), self._num_ages))).tolist()

        # Regional household quarantine proportions
        h = [0.8] * len(self.regions)

        # Disease-specific parameters
        tau = 0.4
        d = 0.4 * np.ones(self._num_ages)

        # Transmission parameters
        epsilon = 0.1895
        gamma = 0.083
        sigma = 0.5 * np.ones(self._num_ages)

        # List of common initial conditions and parameters that characterise
        # the model
        parameters = [
            1, susceptibles,
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
            detected_f,
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
            np.zeros(
                (len(self.regions), self._num_ages)).tolist(),
            np.zeros(
                (len(self.regions), self._num_ages)).tolist(),
            sigma, tau, epsilon, gamma, d, h]

        # Simulate using the ODE solver from scipy
        scipy_method = 'RK45'
        parameters.append(scipy_method)

    def set_social_distancing_parameters(self):
        self.social_distancing_param = em.WarwickSocDistParameters(self)()


#
# Toy Extended Population Data Class
#

class TestExtendedPopData(object):
    """
    Toy Extended number of age groups population Data class used for testing.
    """
    def __init__(self):
        # Toy values for data structures about NPIs
        self.extended_susceptibles = np.loadtxt(
            os.path.join(
                os.path.dirname(__file__),
                '../../data/england_population/UK_PP.csv'),
            delimiter=',').astype(int)
        self.extended_infectives_prop = np.loadtxt(
            os.path.join(
                os.path.dirname(__file__),
                '../../data/england_population/UK_Ages.csv'),
            delimiter=',')

    def __call__(self):
        return (
            self.extended_susceptibles, self.extended_infectives_prop)


#
# Toy Extended Contact Data Class
#

class TestExtendedContactData(object):
    """
    Toy Extended number of age groups contact matrices Data class used for
    testing.
    """
    def __init__(self):
        # Toy values for data structures about extended contact matrices
        self.extended_house_cont_mat = np.loadtxt(
            os.path.join(
                os.path.dirname(__file__),
                '../../data/england_population/UK_from_toH.csv'),
            delimiter=',')
        self.extended_school_cont_mat = np.loadtxt(
            os.path.join(
                os.path.dirname(__file__),
                '../../data/england_population/UK_from_toS.csv'),
            delimiter=',')
        self.extended_work_cont_mat = np.loadtxt(
            os.path.join(
                os.path.dirname(__file__),
                '../../data/england_population/UK_from_toW.csv'),
            delimiter=',')
        self.extended_other_cont_mat = np.loadtxt(
            os.path.join(
                os.path.dirname(__file__),
                '../../data/england_population/UK_from_toO.csv'),
            delimiter=',')

    def __call__(self):
        return (
            self.extended_house_cont_mat, self.extended_school_cont_mat,
            self.extended_work_cont_mat, self.extended_other_cont_mat)


#
# Toy Death Data Class
#

class TestDeathData(object):
    """
    Toy Death Data class used for testing.
    """
    def __init__(self, total_days):
        # Toy values for data structures about death
        self.deaths = [4 * np.ones((total_days, 8), dtype=int)]
        self.deaths_times = np.arange(1, total_days+1, 1).tolist()

    def __call__(self):
        return (self.deaths, self.deaths_times)


#
# Toy Serology Data Class
#

class TestSerologyData(object):
    """
    Toy Serology Data class used for testing.
    """
    def __init__(self, total_days):
        # Toy values for data structures about serology
        self.tests_data = [100 * np.ones((total_days, 8), dtype=int)]
        self.positives_data = [10 * np.ones((total_days, 8), dtype=int)]
        self.serology_times = np.arange(1, total_days+1, 1).tolist()
        self.sens = 0.7
        self.spec = 0.95

    def __call__(self):
        return (
            self.tests_data, self.positives_data, self.serology_times,
            self.sens, self.spec)


#
# Toy Delay Data Class
#

class TestDelayData(object):
    """
    Toy hospitalisation and death Data class used for testing.
    """
    def __init__(self, extended_susceptibles):
        # Toy values for data structures about NPIs
        path = os.path.join(os.path.dirname(__file__), '../../data/')

        RF_df = pd.read_csv(
            os.path.join(path, 'risks_death/Risks_United Kingdom.csv'),
            dtype=np.float64)

        param_df = pd.read_csv(
                os.path.join(path, 'global_parameters/parameters.csv'),
                dtype=np.float64)

        self.pDtoH = update_age_groups(
            extended_susceptibles,
            RF_df['hospitalisation_risk'].tolist())
        self.pHtoDeath = update_age_groups(
            extended_susceptibles,
            RF_df['death_risk'].tolist())

        th_mean = param_df['hosp_lag'].tolist()[0]+0.00001
        th_var = 12.1**2
        theta = th_var / th_mean
        k = th_mean / theta
        self.dDtoH = scipy.stats.gamma(k, scale=theta).pdf(
            np.arange(1, 31)).tolist()

        td_mean = param_df['death_lag'].tolist()[0]
        td_var = 12.1**2
        theta = td_var / td_mean
        k = td_mean / theta
        self.dHtoDeath = scipy.stats.gamma(k, scale=theta).pdf(
            np.arange(1, 31)).tolist()

    def __call__(self):
        return (
            self.pDtoH, self.dDtoH, self.pHtoDeath, self.dHtoDeath)


#
# Test Warwick Log-Likelihood Class
#

class TestWarwickLogLik(unittest.TestCase):
    """
    Test the 'WarwickSEIRModel' class.
    """
    def test__call__(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model, death, serology, delay, extended population
        # and extended contact data
        model = TestWarwickModel()
        model.set_initial_conditions()
        model.set_social_distancing_parameters()
        extended_susceptibles, extended_infectives_prop = \
            TestExtendedPopData()()
        extended_house_cont_mat, extended_school_cont_mat, \
            extended_work_cont_mat, extended_other_cont_mat = \
            TestExtendedContactData()()
        deaths, deaths_times = TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()
        pDtoH, dDtoH, pHtoDeath, dHtoDeath = TestDelayData(
            extended_susceptibles)()

        # Set toy model initial conditions
        susceptibles_data = [np.loadtxt(os.path.join(
            os.path.dirname(__file__),
            '../../data/england_population/England_population.csv'),
            dtype=int, delimiter=',').tolist()[-1]]
        infectives_data = (40 * np.ones(
            (len(model.regions), len(model.age_groups)))).tolist()

        # Set log-likelihood object
        log_lik = em.inference.WarwickLogLik(
            model, extended_susceptibles,
            extended_infectives_prop, extended_house_cont_mat,
            extended_school_cont_mat, extended_work_cont_mat,
            extended_other_cont_mat,
            pDtoH, dDtoH, pHtoDeath, dHtoDeath,
            susceptibles_data, infectives_data, times,
            deaths, deaths_times,
            tests_data, positives_data, serology_times,
            sens, spec, wd=1, wp=1)

        self.assertIsInstance(log_lik([0.9, 0.1, 10]), (int, float))
        self.assertEqual(log_lik([0.9, 0.1, 10]) < 0, True)

    def test_n_parameters(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model, death, serology, delay, extended population
        # and extended contact data
        model = TestWarwickModel()
        model.set_initial_conditions()
        model.set_social_distancing_parameters()
        extended_susceptibles, extended_infectives_prop = \
            TestExtendedPopData()()
        extended_house_cont_mat, extended_school_cont_mat, \
            extended_work_cont_mat, extended_other_cont_mat = \
            TestExtendedContactData()()
        deaths, deaths_times = TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()
        pDtoH, dDtoH, pHtoDeath, dHtoDeath = TestDelayData(
            extended_susceptibles)()

        # Set toy model initial conditions
        susceptibles_data = [np.loadtxt(os.path.join(
            os.path.dirname(__file__),
            '../../data/england_population/England_population.csv'),
            dtype=int, delimiter=',').tolist()[-1]]
        infectives_data = (40 * np.ones(
            (len(model.regions), len(model.age_groups)))).tolist()

        # Set log-likelihood object
        log_lik = em.inference.WarwickLogLik(
            model, extended_susceptibles,
            extended_infectives_prop, extended_house_cont_mat,
            extended_school_cont_mat, extended_work_cont_mat,
            extended_other_cont_mat,
            pDtoH, dDtoH, pHtoDeath, dHtoDeath,
            susceptibles_data, infectives_data, times,
            deaths, deaths_times,
            tests_data, positives_data, serology_times,
            sens, spec, wd=1, wp=1)

        self.assertEqual(log_lik.n_parameters(), 2)


#
# Test Warwick Log-Prior Class
#

class TestWarwickLogPrior(unittest.TestCase):
    """
    Test the 'WarwickLogPrior' class.
    """
    def test__call__(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model
        model = TestWarwickModel()
        model.set_initial_conditions()
        model.set_social_distancing_parameters()

        # Set log-prior object
        log_prior = em.inference.WarwickLogPrior(model, times)

        self.assertIsInstance(log_prior([0.9, 0.1, 10]), (int, float))
        self.assertEqual(log_prior([0.9, 0.1, 10]) < 0, True)

    def test_n_parameters(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model
        model = TestWarwickModel()
        model.set_initial_conditions()
        model.set_social_distancing_parameters()

        # Set log-prior object
        log_prior = em.inference.WarwickLogPrior(model, times)

        self.assertEqual(log_prior.n_parameters(), 2)


#
# Test Warwick Inference and Optimisation Class
#

class TestWarwickSEIRInfer(unittest.TestCase):
    """
    Test the 'WarwickSEIRInfer' class.
    """
    def test__init__(self):
        # Set toy model
        model = TestWarwickModel()
        model.set_initial_conditions()
        model.set_social_distancing_parameters()

        # Set up Warwick Inference class
        inference = em.inference.WarwickSEIRInfer(model)

        self.assertIsInstance(inference._model, em.WarwickSEIRModel)

        with self.assertRaises(TypeError):
            em.inference.WarwickSEIRInfer(0)

    def test_read_data(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model, death, serology, delay, extended population
        # and extended contact data
        model = TestWarwickModel()
        model.set_initial_conditions()
        model.set_social_distancing_parameters()
        extended_susceptibles, extended_infectives_prop = \
            TestExtendedPopData()()
        extended_house_cont_mat, extended_school_cont_mat, \
            extended_work_cont_mat, extended_other_cont_mat = \
            TestExtendedContactData()()
        deaths, deaths_times = TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()
        pDtoH, dDtoH, pHtoDeath, dHtoDeath = TestDelayData(
            extended_susceptibles)()

        # Set toy model initial conditions
        susceptibles_data = [np.loadtxt(os.path.join(
            os.path.dirname(__file__),
            '../../data/england_population/England_population.csv'),
            dtype=int, delimiter=',').tolist()[-1]]
        infectives_data = (40 * np.ones(
            (len(model.regions), len(model.age_groups)))).tolist()

        # Set up Warwick Inference class
        inference = em.inference.WarwickSEIRInfer(model)

        # Test read_model_data
        inference.read_model_data(susceptibles_data, infectives_data)

        self.assertEqual(
            np.asarray(inference._susceptibles_data).shape, (1, 8))
        self.assertEqual(
            np.asarray(inference._infectives_data).shape, (1, 8))

        self.assertEqual(inference._susceptibles_data, susceptibles_data)
        self.assertEqual(inference._infectives_data, infectives_data)

        # Test read_extended_population_structure
        inference.read_extended_population_structure(
            extended_susceptibles, extended_infectives_prop)

        self.assertEqual(
            np.asarray(inference._extended_susceptibles).shape, (21,))
        self.assertEqual(
            np.asarray(inference._extended_infectives_prop).shape, (21,))

        npt.assert_array_equal(
            inference._extended_susceptibles, extended_susceptibles)
        npt.assert_array_equal(
            inference._extended_infectives_prop, extended_infectives_prop)

        # Test read_extended_contact_matrices
        inference.read_extended_contact_matrices(
            extended_house_cont_mat, extended_school_cont_mat,
            extended_work_cont_mat, extended_other_cont_mat)

        self.assertEqual(
            np.asarray(inference._extended_house_cont_mat).shape, (21, 21))
        self.assertEqual(
            np.asarray(inference._extended_school_cont_mat).shape, (21, 21))
        self.assertEqual(
            np.asarray(inference._extended_work_cont_mat).shape, (21, 21))
        self.assertEqual(
            np.asarray(inference._extended_other_cont_mat).shape, (21, 21))

        npt.assert_array_equal(
            inference._extended_house_cont_mat, extended_house_cont_mat)
        npt.assert_array_equal(
            inference._extended_school_cont_mat, extended_school_cont_mat)
        npt.assert_array_equal(
            inference._extended_work_cont_mat, extended_work_cont_mat)
        npt.assert_array_equal(
            inference._extended_other_cont_mat, extended_other_cont_mat)

        # Test read_serology_data
        inference.read_serology_data(
            tests_data, positives_data, serology_times, sens, spec)

        self.assertEqual(
            np.asarray(inference._total_tests).shape, (1, len(times), 8))
        self.assertEqual(
            np.asarray(inference._positive_tests).shape, (1, len(times), 8))

        self.assertEqual(inference._total_tests, tests_data)
        self.assertEqual(inference._positive_tests, positives_data)
        self.assertEqual(inference._serology_times, times)
        self.assertEqual(inference._sens, 0.7)
        self.assertEqual(inference._spec, 0.95)

        # Test read_delay_data
        inference.read_delay_data(pDtoH, dDtoH, pHtoDeath, dHtoDeath)

        self.assertEqual(
            np.asarray(inference._pDtoH).shape, (8, ))
        self.assertEqual(
            np.asarray(inference._dDtoH).shape, (30,))
        self.assertEqual(
            np.asarray(inference._pHtoDeath).shape, (8,))
        self.assertEqual(
            np.asarray(inference._dHtoDeath).shape, (30,))

        npt.assert_array_equal(inference._pDtoH, pDtoH)
        self.assertEqual(inference._dDtoH, dDtoH)
        npt.assert_array_equal(inference._pHtoDeath, pHtoDeath)
        self.assertEqual(inference._dHtoDeath, dHtoDeath)

        # Test read_deaths_data
        inference.read_deaths_data(deaths, deaths_times)

        self.assertEqual(
            np.asarray(inference._deaths).shape, (1, len(times), 8))

        self.assertEqual(inference._deaths, deaths)
        self.assertEqual(inference._deaths_times, times)

    def test_return_loglikelihood(self):
        # Set times for inference
        times = np.arange(1, 60, 1).tolist()

        # Set toy model, death, serology, delay, extended population
        # and extended contact data
        model = TestWarwickModel()
        model.set_initial_conditions()
        model.set_social_distancing_parameters()
        extended_susceptibles, extended_infectives_prop = \
            TestExtendedPopData()()
        extended_house_cont_mat, extended_school_cont_mat, \
            extended_work_cont_mat, extended_other_cont_mat = \
            TestExtendedContactData()()
        deaths, deaths_times = TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()
        pDtoH, dDtoH, pHtoDeath, dHtoDeath = TestDelayData(
            extended_susceptibles)()

        # Set toy model initial conditions
        susceptibles_data = [np.loadtxt(os.path.join(
            os.path.dirname(__file__),
            '../../data/england_population/England_population.csv'),
            dtype=int, delimiter=',').tolist()[-1]]
        infectives_data = (40 * np.ones(
            (len(model.regions), len(model.age_groups)))).tolist()

        # Set up Warwick Inference class
        inference = em.inference.WarwickSEIRInfer(model)

        # Add model, death, serology, delay, extended population
        # and extended contact data to the inference structure
        inference.read_model_data(susceptibles_data, infectives_data)
        inference.read_extended_population_structure(
            extended_susceptibles, extended_infectives_prop)
        inference.read_extended_contact_matrices(
            extended_house_cont_mat, extended_school_cont_mat,
            extended_work_cont_mat, extended_other_cont_mat)
        inference.read_deaths_data(deaths, deaths_times)
        inference.read_delay_data(pDtoH, dDtoH, pHtoDeath, dHtoDeath)
        inference.read_serology_data(
            tests_data, positives_data, serology_times, sens, spec)

        # Compute the log likelihood at chosen point in the parameter space
        log_lik = inference.return_loglikelihood(times, [0.9, 0.1, 10])

        self.assertIsInstance(log_lik, (int, float))
        self.assertEqual(log_lik < 0, True)

    def test_optimisation_problem_setup(self):
        # Set times for optimisation
        times = np.arange(1, 60, 1).tolist()

        # Set toy model, death, serology, delay, extended population
        # and extended contact data
        model = TestWarwickModel()
        model.set_initial_conditions()
        model.set_social_distancing_parameters()
        extended_susceptibles, extended_infectives_prop = \
            TestExtendedPopData()()
        extended_house_cont_mat, extended_school_cont_mat, \
            extended_work_cont_mat, extended_other_cont_mat = \
            TestExtendedContactData()()
        deaths, deaths_times = TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()
        pDtoH, dDtoH, pHtoDeath, dHtoDeath = TestDelayData(
            extended_susceptibles)()

        # Set toy model initial conditions
        susceptibles_data = [np.loadtxt(os.path.join(
            os.path.dirname(__file__),
            '../../data/england_population/England_population.csv'),
            dtype=int, delimiter=',').tolist()[-1]]
        infectives_data = (40 * np.ones(
            (len(model.regions), len(model.age_groups)))).tolist()

        # Set up Warwick Inference class for optimisation
        optimisation = em.inference.WarwickSEIRInfer(model)

        # Add model, death, serology, delay, extended population
        # and extended contact data to the optimisation structure
        optimisation.read_model_data(susceptibles_data, infectives_data)
        optimisation.read_extended_population_structure(
            extended_susceptibles, extended_infectives_prop)
        optimisation.read_extended_contact_matrices(
            extended_house_cont_mat, extended_school_cont_mat,
            extended_work_cont_mat, extended_other_cont_mat)
        optimisation.read_deaths_data(deaths, deaths_times)
        optimisation.read_delay_data(pDtoH, dDtoH, pHtoDeath, dHtoDeath)
        optimisation.read_serology_data(
            tests_data, positives_data, serology_times, sens, spec)

        # Set up and run the optimisation problem
        found, log_post_value = optimisation.optimisation_problem_setup(times)

        self.assertEqual(len(found), 2)
        self.assertIsInstance(log_post_value, (int, float))
        self.assertEqual(log_post_value < 0, True)

    def test_inference_problem_setup(self):
        # Set times for inference
        times = np.arange(1, 50, 1).tolist()

        # Set toy model, death, serology, delay, extended population
        # and extended contact data
        model = TestWarwickModel()
        model.set_initial_conditions()
        model.set_social_distancing_parameters()
        extended_susceptibles, extended_infectives_prop = \
            TestExtendedPopData()()
        extended_house_cont_mat, extended_school_cont_mat, \
            extended_work_cont_mat, extended_other_cont_mat = \
            TestExtendedContactData()()
        deaths, deaths_times = TestDeathData(len(times))()
        tests_data, positives_data, serology_times, sens, spec = \
            TestSerologyData(len(times))()
        pDtoH, dDtoH, pHtoDeath, dHtoDeath = TestDelayData(
            extended_susceptibles)()

        # Set toy model initial conditions
        susceptibles_data = [np.loadtxt(os.path.join(
            os.path.dirname(__file__),
            '../../data/england_population/England_population.csv'),
            dtype=int, delimiter=',').tolist()[-1]]
        infectives_data = (40 * np.ones(
            (len(model.regions), len(model.age_groups)))).tolist()

        # Set up Warwick Inference class
        inference = em.inference.WarwickSEIRInfer(model)

        # Add model, death, serology, delay, extended population
        # and extended contact data to the inference structure
        inference.read_model_data(susceptibles_data, infectives_data)
        inference.read_extended_population_structure(
            extended_susceptibles, extended_infectives_prop)
        inference.read_extended_contact_matrices(
            extended_house_cont_mat, extended_school_cont_mat,
            extended_work_cont_mat, extended_other_cont_mat)
        inference.read_deaths_data(deaths, deaths_times)
        inference.read_delay_data(pDtoH, dDtoH, pHtoDeath, dHtoDeath)
        inference.read_serology_data(
            tests_data, positives_data, serology_times, sens, spec)

        # Set up and run the inference problem
        samples = inference.inference_problem_setup(times, num_iter=600)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (600, 2))
