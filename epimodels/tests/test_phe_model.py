#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#

import unittest

import numpy as np
import numpy.testing as npt

import epimodels as em


class TestPheSEIRModel(unittest.TestCase):
    """
    Test the 'PheSEIRModel' class.
    """
    def test__init__(self):
        model = em.PheSEIRModel()

        self.assertEqual(
            model._output_names,
            ['S', 'E1', 'E2', 'I1', 'I2', 'R', 'Incidence'])
        self.assertEqual(
            model._parameter_names,
            ['S0', 'E10', 'E20', 'I10', 'I20', 'R0', 'beta', 'kappa', 'gamma'])
        self.assertEqual(model._n_outputs, 7)
        self.assertEqual(model._n_parameters, 9)

    def test_n_outputs(self):
        model = em.PheSEIRModel()
        self.assertEqual(model.n_outputs(), 7)

    def test_n_parameters(self):
        model = em.PheSEIRModel()
        self.assertEqual(model.n_parameters(), 9)

    def test_output_names(self):
        model = em.PheSEIRModel()
        self.assertEqual(
            model.output_names(),
            ['S', 'E1', 'E2', 'I1', 'I2', 'R', 'Incidence'])

    def test_parameter_names(self):
        model = em.PheSEIRModel()
        self.assertEqual(
            model.parameter_names(),
            ['S0', 'E10', 'E20', 'I10', 'I20', 'R0', 'beta', 'kappa', 'gamma'])

    def test_set_regions(self):
        model = em.PheSEIRModel()
        regions = ['London', 'Cornwall']
        model.set_regions(regions)

        self.assertEqual(
            model.region_names(),
            ['London', 'Cornwall'])

    def test_set_outputs(self):
        model = em.PheSEIRModel()
        outputs = ['S', 'I1', 'I2', 'Incidence']
        model.set_outputs(outputs)

        with self.assertRaises(ValueError):
            outputs1 = ['S', 'E', 'I1', 'I2', 'Incidence']
            model.set_outputs(outputs1)

    def test_simulate(self):
        model = em.PheSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        contact_data_matrix_0 = np.array([[1, 0], [0, 3]])
        contact_data_matrix_1 = np.array([[10, 5.2], [0, 3]])

        region_data_matrix_0_0 = np.array([[0.5, 0], [0, 6]])
        region_data_matrix_0_1 = np.array([[1, 10], [1, 0]])
        region_data_matrix_1_0 = np.array([[0.5, 1.2], [0.29, 6]])
        region_data_matrix_1_1 = np.array([[0.85, 1], [0.9, 6]])

        susceptibles = [[5, 6], [7, 8]]
        dI = 4

        contacts_0 = em.ContactMatrix(age_groups, contact_data_matrix_0)
        contacts_1 = em.ContactMatrix(age_groups, contact_data_matrix_1)
        regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0_0)
        regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_0_1)
        regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_1_0)
        regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1_1)

        # Matrices contact
        matrices_contact = [contacts_0, contacts_1]
        time_changes_contact = [1, 3]
        matrices_region = [
            [regional_0_0, regional_0_1],
            [regional_1_0, regional_1_1]]
        time_changes_region = [1, 2]

        initial_r = [0.5, 1]

        parameters = [
            1, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
            [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
            [[1]*2, [1]*2], 4, dI, 0.5]

        times = [1, 2]

        output_my_solver = model.simulate(
            parameters, times, matrices_contact, time_changes_contact,
            regions, initial_r, matrices_region, time_changes_region,
            method='my-solver')

        npt.assert_almost_equal(
            output_my_solver,
            np.array([
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]), decimal=3)

        output_scipy_solver = model.simulate(
            parameters, times, matrices_contact, time_changes_contact,
            regions, initial_r, matrices_region, time_changes_region,
            method='RK45')

        npt.assert_almost_equal(
            output_scipy_solver,
            np.array([
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]), decimal=3)

        with self.assertRaises(TypeError):
            model.simulate(
                parameters,
                0,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(TypeError):
            model.simulate(
                parameters,
                ['1', 2],
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(ValueError):
            model.simulate(
                parameters,
                [0, 1],
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(TypeError):
            model.simulate(
                'parameters',
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(ValueError):
            model.simulate(
                [0],
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(TypeError):
            parameters1 = [
                0.5, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[1]*2, [1]*2], 4, dI, 0.005]

            model.simulate(
                parameters1,
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(ValueError):
            parameters1 = [
                0, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[1]*2, [1]*2], 4, dI, 0.005]

            model.simulate(
                parameters1,
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(ValueError):
            parameters1 = [
                3, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[1]*2, [1]*2], 4, dI, 0.005]

            model.simulate(
                parameters1,
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(ValueError):
            susceptibles1 = [5, 6]

            parameters1 = [
                1, susceptibles1, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[1]*2, [1]*2], 4, dI, 0.005]

            model.simulate(
                parameters1,
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[1]*2, [1]*2], 4, dI, 0.5]

            model.simulate(
                parameters1,
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0], [0, 0]],
                [[1]*2, [1]*2], 4, dI, 0.005]

            model.simulate(
                parameters1,
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[[1]*2, [1]*2], [[1]*2, [1]*2]], 4, dI, 0.005]

            model.simulate(
                parameters1,
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[1]*2, [1]*2, [1]*2], 4, dI, 0.005]

            model.simulate(
                parameters1,
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[1]*4, [1]*4], 4, dI, 0.005]

            model.simulate(
                parameters1,
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[1]*2, [1]*2], '4', dI, 0.005]

            model.simulate(
                parameters1,
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[1]*2, [1]*2], -1, dI, 0.005]

            model.simulate(
                parameters1,
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[1]*2, [1]*2], 4, '4', 0.005]

            model.simulate(
                parameters1,
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[1]*2, [1]*2], 4, 0, 0.005]

            model.simulate(
                parameters1,
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[1]*2, [1]*2], 4, dI, '0.005']

            model.simulate(
                parameters1,
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                [[1]*2, [1]*2], 4, dI, 0]

            model.simulate(
                parameters1,
                times,
                matrices_contact,
                time_changes_contact,
                regions,
                initial_r,
                matrices_region,
                time_changes_region,
                method='my-solver')

    def test_new_infections(self):
        model = em.PheSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        contact_data_matrix_0 = np.array([[1, 0], [0, 3]])
        contact_data_matrix_1 = np.array([[10, 5.2], [0, 3]])

        region_data_matrix_0_0 = np.array([[0.5, 0], [0, 6]])
        region_data_matrix_0_1 = np.array([[1, 10], [1, 0]])
        region_data_matrix_1_0 = np.array([[0.5, 1.2], [0.29, 6]])
        region_data_matrix_1_1 = np.array([[0.85, 1], [0.9, 6]])

        susceptibles = [[5, 6], [7, 8]]
        dI = 4

        contacts_0 = em.ContactMatrix(age_groups, contact_data_matrix_0)
        contacts_1 = em.ContactMatrix(age_groups, contact_data_matrix_1)
        regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0_0)
        regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_0_1)
        regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_1_0)
        regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1_1)

        # Matrices contact
        matrices_contact = [contacts_0, contacts_1]
        time_changes_contact = [1, 3]
        matrices_region = [
            [regional_0_0, regional_0_1],
            [regional_1_0, regional_1_1]]
        time_changes_region = [1, 2]

        initial_r = [0.5, 1]

        parameters = [
            1, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
            [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
            [[1]*2, [1]*2], 4, dI, 0.5]

        times = [1, 2]

        output = model.simulate(
            parameters, times, matrices_contact, time_changes_contact,
            regions, initial_r, matrices_region, time_changes_region,
            method='my-solver')

        npt.assert_array_equal(
            model.new_infections(output),
            np.array([[0, 0], [0, 0]]))

        with self.assertRaises(ValueError):
            output1 = np.array([5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            model.new_infections(output1)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0]])
            model.new_infections(output1)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            model.new_infections(output1)

        with self.assertRaises(TypeError):
            output1 = np.array([
                ['5', 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, '0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            model.new_infections(output1)

    def test_loglik_deaths(self):
        model = em.PheSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        contact_data_matrix_0 = np.array([[1, 0], [0, 3]])
        contact_data_matrix_1 = np.array([[10, 5.2], [0, 3]])

        region_data_matrix_0_0 = np.array([[0.5, 0], [0, 6]])
        region_data_matrix_0_1 = np.array([[1, 10], [1, 0]])
        region_data_matrix_1_0 = np.array([[0.5, 1.2], [0.29, 6]])
        region_data_matrix_1_1 = np.array([[0.85, 1], [0.9, 6]])

        susceptibles = [[5, 6], [7, 8]]
        dI = 4

        contacts_0 = em.ContactMatrix(age_groups, contact_data_matrix_0)
        contacts_1 = em.ContactMatrix(age_groups, contact_data_matrix_1)
        regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0_0)
        regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_0_1)
        regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_1_0)
        regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1_1)

        # Matrices contact
        matrices_contact = [contacts_0, contacts_1]
        time_changes_contact = [1, 3]
        matrices_region = [
            [regional_0_0, regional_0_1],
            [regional_1_0, regional_1_1]]
        time_changes_region = [1, 2]

        initial_r = [0.5, 1]

        parameters = [
            1, susceptibles, [[0, 0], [0, 0]], [[0.1, 0.2], [0, 0]],
            [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
            [[1]*2, [1]*2], 4, dI, 0.5]

        times = [1, 2]

        output = model.simulate(
            parameters, times, matrices_contact, time_changes_contact,
            regions, initial_r, matrices_region, time_changes_region,
            method='my-solver')

        obs_death = [10, 12]
        fatality_ratio = [0.1, 0.5]
        time_to_death = [0.5, 0.5]

        self.assertEqual(
            model.loglik_deaths(
                obs_death, output, fatality_ratio,
                time_to_death, 0.5, 1).shape,
            (len(age_groups),))

        with self.assertRaises(TypeError):
            model.loglik_deaths(
                obs_death, output, fatality_ratio,
                time_to_death, '0.5', 1)

        with self.assertRaises(ValueError):
            model.loglik_deaths(
                obs_death, output, fatality_ratio,
                time_to_death, -2, 1)

        with self.assertRaises(ValueError):
            model.loglik_deaths(
                obs_death, output, fatality_ratio,
                time_to_death, 0.5, -1)

        with self.assertRaises(TypeError):
            model.loglik_deaths(
                obs_death, output, fatality_ratio,
                time_to_death, 0.5, '1')

        with self.assertRaises(ValueError):
            model.loglik_deaths(
                obs_death, output, fatality_ratio,
                time_to_death, 0.5, -1)

        with self.assertRaises(ValueError):
            model.loglik_deaths(
                obs_death, output, fatality_ratio,
                time_to_death, 0.5, 2)

        with self.assertRaises(ValueError):
            model.loglik_deaths(
                0, output, fatality_ratio,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            obs_death1 = np.array([5, 6, 0, 0])

            model.loglik_deaths(
                obs_death1, output, fatality_ratio,
                time_to_death, 0.5, 1)

        with self.assertRaises(TypeError):
            obs_death1 = np.array(['5', 6])

            model.loglik_deaths(
                obs_death1, output, fatality_ratio,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            obs_death1 = np.array([5, -1])

            model.loglik_deaths(
                obs_death1, output, fatality_ratio,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            output1 = np.array([5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            model.loglik_deaths(
                obs_death, output1, fatality_ratio,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0]])

            model.loglik_deaths(
                obs_death, output1, fatality_ratio,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

            model.loglik_deaths(
                obs_death, output1, fatality_ratio,
                time_to_death, 0.5, 1)

        with self.assertRaises(TypeError):
            output1 = np.array([
                ['5', 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, '0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

            model.loglik_deaths(
                obs_death, output1, fatality_ratio,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            fatality_ratio1 = 0

            model.loglik_deaths(
                obs_death, output, fatality_ratio1,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            fatality_ratio1 = np.array([0.1, 0.5, 0.1])

            model.loglik_deaths(
                obs_death, output, fatality_ratio1,
                time_to_death, 0.5, 1)

        with self.assertRaises(TypeError):
            fatality_ratio1 = np.array([0.1, '0.5'])

            model.loglik_deaths(
                obs_death, output, fatality_ratio1,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            fatality_ratio1 = np.array([-0.1, 0.5])

            model.loglik_deaths(
                obs_death, output, fatality_ratio1,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            fatality_ratio1 = np.array([0.1, 1.5])

            model.loglik_deaths(
                obs_death, output, fatality_ratio1,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            time_to_death1 = np.array([[0.5], [0.5]])

            model.loglik_deaths(
                obs_death, output, fatality_ratio,
                time_to_death1, 0.5, 1)

        with self.assertRaises(ValueError):
            time_to_death1 = np.array([0.5, 0.5, 0.15])

            model.loglik_deaths(
                obs_death, output, fatality_ratio,
                time_to_death1, 0.5, 1)

        with self.assertRaises(TypeError):
            time_to_death1 = np.array(['0.1', 0.5])

            model.loglik_deaths(
                obs_death, output, fatality_ratio,
                time_to_death1, 0.5, 1)

        with self.assertRaises(ValueError):
            time_to_death1 = np.array([-0.1, 0.5])

            model.loglik_deaths(
                obs_death, output, fatality_ratio,
                time_to_death1, 0.5, 1)

        with self.assertRaises(ValueError):
            time_to_death1 = np.array([0.5, 1.1])

            model.loglik_deaths(
                obs_death, output, fatality_ratio,
                time_to_death1, 0.5, 1)

    def test_samples_deaths(self):
        model = em.PheSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        contact_data_matrix_0 = np.array([[1, 0], [0, 3]])
        contact_data_matrix_1 = np.array([[10, 5.2], [0, 3]])

        region_data_matrix_0_0 = np.array([[0.5, 0], [0, 6]])
        region_data_matrix_0_1 = np.array([[1, 10], [1, 0]])
        region_data_matrix_1_0 = np.array([[0.5, 1.2], [0.29, 6]])
        region_data_matrix_1_1 = np.array([[0.85, 1], [0.9, 6]])

        susceptibles = [[5, 6], [7, 8]]
        dI = 4

        contacts_0 = em.ContactMatrix(age_groups, contact_data_matrix_0)
        contacts_1 = em.ContactMatrix(age_groups, contact_data_matrix_1)
        regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0_0)
        regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_0_1)
        regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_1_0)
        regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1_1)

        # Matrices contact
        matrices_contact = [contacts_0, contacts_1]
        time_changes_contact = [1, 3]
        matrices_region = [
            [regional_0_0, regional_0_1],
            [regional_1_0, regional_1_1]]
        time_changes_region = [1, 2]

        initial_r = [0.5, 1]

        parameters = [
            1, susceptibles, [[0, 0], [0, 0]], [[0.1, 0.2], [0, 0]],
            [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
            [[1]*2, [1]*2], 4, dI, 0.5]

        times = [1, 2]

        output = model.simulate(
            parameters, times, matrices_contact, time_changes_contact,
            regions, initial_r, matrices_region, time_changes_region,
            method='my-solver')

        fatality_ratio = [0.1, 0.5]
        time_to_death = [0.5, 0.5]

        self.assertEqual(
            model.samples_deaths(
                output, fatality_ratio,
                time_to_death, 0.5, 1).shape,
            (len(age_groups),))

        with self.assertRaises(TypeError):
            model.samples_deaths(
                output, fatality_ratio,
                time_to_death, '0.5', 1)

        with self.assertRaises(ValueError):
            model.samples_deaths(
                output, fatality_ratio,
                time_to_death, -2, 1)

        with self.assertRaises(ValueError):
            model.samples_deaths(
                output, fatality_ratio,
                time_to_death, 0.5, -1)

        with self.assertRaises(TypeError):
            model.samples_deaths(
                output, fatality_ratio,
                time_to_death, 0.5, '1')

        with self.assertRaises(ValueError):
            model.samples_deaths(
                output, fatality_ratio,
                time_to_death, 0.5, -1)

        with self.assertRaises(ValueError):
            model.samples_deaths(
                output, fatality_ratio,
                time_to_death, 0.5, 2)

        with self.assertRaises(ValueError):
            output1 = np.array([5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            model.samples_deaths(
                output1, fatality_ratio,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0]])

            model.samples_deaths(
                output1, fatality_ratio,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

            model.samples_deaths(
                output1, fatality_ratio,
                time_to_death, 0.5, 1)

        with self.assertRaises(TypeError):
            output1 = np.array([
                ['5', 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, '0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

            model.samples_deaths(
                output1, fatality_ratio,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            fatality_ratio1 = 0

            model.samples_deaths(
                output, fatality_ratio1,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            fatality_ratio1 = np.array([0.1, 0.5, 0.1])

            model.samples_deaths(
                output, fatality_ratio1,
                time_to_death, 0.5, 1)

        with self.assertRaises(TypeError):
            fatality_ratio1 = np.array([0.1, '0.5'])

            model.samples_deaths(
                output, fatality_ratio1,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            fatality_ratio1 = np.array([-0.1, 0.5])

            model.samples_deaths(
                output, fatality_ratio1,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            fatality_ratio1 = np.array([0.1, 1.5])

            model.samples_deaths(
                output, fatality_ratio1,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            time_to_death1 = np.array([[0.5], [0.5]])

            model.samples_deaths(
                output, fatality_ratio,
                time_to_death1, 0.5, 1)

        with self.assertRaises(ValueError):
            time_to_death1 = np.array([0.5, 0.5, 0.15])

            model.samples_deaths(
                output, fatality_ratio,
                time_to_death1, 0.5, 1)

        with self.assertRaises(TypeError):
            time_to_death1 = np.array(['0.1', 0.5])

            model.samples_deaths(
                output, fatality_ratio,
                time_to_death1, 0.5, 1)

        with self.assertRaises(ValueError):
            time_to_death1 = np.array([-0.1, 0.5])

            model.samples_deaths(
                output, fatality_ratio,
                time_to_death1, 0.5, 1)

        with self.assertRaises(ValueError):
            time_to_death1 = np.array([0.5, 1.1])

            model.samples_deaths(
                output, fatality_ratio,
                time_to_death1, 0.5, 1)

    def test_loglik_positive_tests(self):
        model = em.PheSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        contact_data_matrix_0 = np.array([[1, 0], [0, 3]])
        contact_data_matrix_1 = np.array([[10, 5.2], [0, 3]])

        region_data_matrix_0_0 = np.array([[0.5, 0], [0, 6]])
        region_data_matrix_0_1 = np.array([[1, 10], [1, 0]])
        region_data_matrix_1_0 = np.array([[0.5, 1.2], [0.29, 6]])
        region_data_matrix_1_1 = np.array([[0.85, 1], [0.9, 6]])

        susceptibles = [[5, 6], [7, 8]]
        dI = 4

        contacts_0 = em.ContactMatrix(age_groups, contact_data_matrix_0)
        contacts_1 = em.ContactMatrix(age_groups, contact_data_matrix_1)
        regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0_0)
        regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_0_1)
        regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_1_0)
        regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1_1)

        # Matrices contact
        matrices_contact = [contacts_0, contacts_1]
        time_changes_contact = [1, 3]
        matrices_region = [
            [regional_0_0, regional_0_1],
            [regional_1_0, regional_1_1]]
        time_changes_region = [1, 2]

        initial_r = [0.5, 1]

        parameters = [
            1, susceptibles, [[0, 0], [0, 0]], [[0.1, 0.2], [0, 0]],
            [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
            [[1]*2, [1]*2], 4, dI, 0.5]

        times = [1, 2]

        output = model.simulate(
            parameters, times, matrices_contact, time_changes_contact,
            regions, initial_r, matrices_region, time_changes_region,
            method='my-solver')

        obs_pos = [10, 12]
        tests = [20, 30]
        sens = 0.9
        spec = 0.1

        self.assertEqual(
            model.loglik_positive_tests(
                obs_pos, output, tests, sens, spec, 0).shape,
            (len(age_groups),))

        with self.assertRaises(TypeError):
            model.loglik_positive_tests(
                obs_pos, output, tests, sens, spec, '1')

        with self.assertRaises(ValueError):
            model.loglik_positive_tests(
                obs_pos, output, tests, sens, spec, -1)

        with self.assertRaises(ValueError):
            model.loglik_positive_tests(
                obs_pos, output, tests, sens, spec, 3)

        with self.assertRaises(ValueError):
            model.loglik_positive_tests(
                0, output, tests, sens, spec, 0)

        with self.assertRaises(ValueError):
            obs_pos1 = np.array([5, 6, 0, 0])

            model.loglik_positive_tests(
                obs_pos1, output, tests, sens, spec, 0)

        with self.assertRaises(TypeError):
            obs_pos1 = np.array(['5', 6])

            model.loglik_positive_tests(
                obs_pos1, output, tests, sens, spec, 0)

        with self.assertRaises(ValueError):
            obs_pos1 = np.array([5, -1])

            model.loglik_positive_tests(
                obs_pos1, output, tests, sens, spec, 0)

        with self.assertRaises(ValueError):
            output1 = np.array([5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            model.loglik_positive_tests(
                obs_pos, output1, tests, sens, spec, 0)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0]])

            model.loglik_positive_tests(
                obs_pos, output1, tests, sens, spec, 0)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

            model.loglik_positive_tests(
                obs_pos, output1, tests, sens, spec, 0)

        with self.assertRaises(TypeError):
            output1 = np.array([
                ['5', 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, '0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

            model.loglik_positive_tests(
                obs_pos, output1, tests, sens, spec, 0)

        with self.assertRaises(ValueError):
            tests1 = 100

            model.loglik_positive_tests(
                obs_pos, output, tests1, sens, spec, 0)

        with self.assertRaises(ValueError):
            tests1 = np.array([20, 30, 1])

            model.loglik_positive_tests(
                obs_pos, output, tests1, sens, spec, 0)

        with self.assertRaises(TypeError):
            tests1 = np.array([20, '30'])

            model.loglik_positive_tests(
                obs_pos, output, tests1, sens, spec, 0)

        with self.assertRaises(ValueError):
            tests1 = np.array([-1, 50])

            model.loglik_positive_tests(
                obs_pos, output, tests1, sens, spec, 0)

        with self.assertRaises(ValueError):
            tests1 = np.array([2, 50])

            model.loglik_positive_tests(
                obs_pos, output, tests1, sens, spec, 0)

        with self.assertRaises(TypeError):
            model.loglik_positive_tests(
                obs_pos, output, tests, '0.9', spec, 0)

        with self.assertRaises(ValueError):
            model.loglik_positive_tests(
                obs_pos, output, tests, -0.2, spec, 0)

        with self.assertRaises(ValueError):
            model.loglik_positive_tests(
                obs_pos, output, tests, 1.2, spec, 0)

        with self.assertRaises(TypeError):
            model.loglik_positive_tests(
                obs_pos, output, tests, sens, '0.1', 0)

        with self.assertRaises(ValueError):
            model.loglik_positive_tests(
                obs_pos, output, tests, sens, -0.1, 0)

        with self.assertRaises(ValueError):
            model.loglik_positive_tests(
                obs_pos, output, tests, sens, 1.2, 0)

    def test_samples_positive_tests(self):
        model = em.PheSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        contact_data_matrix_0 = np.array([[1, 0], [0, 3]])
        contact_data_matrix_1 = np.array([[10, 5.2], [0, 3]])

        region_data_matrix_0_0 = np.array([[0.5, 0], [0, 6]])
        region_data_matrix_0_1 = np.array([[1, 10], [1, 0]])
        region_data_matrix_1_0 = np.array([[0.5, 1.2], [0.29, 6]])
        region_data_matrix_1_1 = np.array([[0.85, 1], [0.9, 6]])

        susceptibles = [[5, 6], [7, 8]]
        dI = 4

        contacts_0 = em.ContactMatrix(age_groups, contact_data_matrix_0)
        contacts_1 = em.ContactMatrix(age_groups, contact_data_matrix_1)
        regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0_0)
        regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_0_1)
        regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_1_0)
        regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1_1)

        # Matrices contact
        matrices_contact = [contacts_0, contacts_1]
        time_changes_contact = [1, 3]
        matrices_region = [
            [regional_0_0, regional_0_1],
            [regional_1_0, regional_1_1]]
        time_changes_region = [1, 2]

        initial_r = [0.5, 1]

        parameters = [
            1, susceptibles, [[0, 0], [0, 0]], [[0.1, 0.2], [0, 0]],
            [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
            [[1]*2, [1]*2], 4, dI, 0.5]

        times = [1, 2]

        output = model.simulate(
            parameters, times, matrices_contact, time_changes_contact,
            regions, initial_r, matrices_region, time_changes_region,
            method='my-solver')

        tests = [20, 30]
        sens = 0.9
        spec = 0.1

        self.assertEqual(
            model.samples_positive_tests(
                output, tests, sens, spec, 0).shape,
            (len(age_groups),))

        with self.assertRaises(TypeError):
            model.samples_positive_tests(
                output, tests, sens, spec, '1')

        with self.assertRaises(ValueError):
            model.samples_positive_tests(
                output, tests, sens, spec, -1)

        with self.assertRaises(ValueError):
            model.samples_positive_tests(
                output, tests, sens, spec, 3)

        with self.assertRaises(ValueError):
            output1 = np.array([5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            model.samples_positive_tests(
                output1, tests, sens, spec, 0)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0]])

            model.samples_positive_tests(
                output1, tests, sens, spec, 0)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

            model.samples_positive_tests(
                output1, tests, sens, spec, 0)

        with self.assertRaises(TypeError):
            output1 = np.array([
                ['5', 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, '0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

            model.samples_positive_tests(
                output1, tests, sens, spec, 0)

        with self.assertRaises(ValueError):
            tests1 = 100

            model.samples_positive_tests(
                output, tests1, sens, spec, 0)

        with self.assertRaises(ValueError):
            tests1 = np.array([20, 30, 1])

            model.samples_positive_tests(
                output, tests1, sens, spec, 0)

        with self.assertRaises(TypeError):
            tests1 = np.array([20, '30'])

            model.samples_positive_tests(
                output, tests1, sens, spec, 0)

        with self.assertRaises(ValueError):
            tests1 = np.array([-1, 50])

            model.samples_positive_tests(
                output, tests1, sens, spec, 0)

        with self.assertRaises(TypeError):
            model.samples_positive_tests(
                output, tests, '0.9', spec, 0)

        with self.assertRaises(ValueError):
            model.samples_positive_tests(
                output, tests, -0.2, spec, 0)

        with self.assertRaises(ValueError):
            model.samples_positive_tests(
                output, tests, 1.2, spec, 0)

        with self.assertRaises(TypeError):
            model.samples_positive_tests(
                output, tests, sens, '0.1', 0)

        with self.assertRaises(ValueError):
            model.samples_positive_tests(
                output, tests, sens, -0.1, 0)

        with self.assertRaises(ValueError):
            model.samples_positive_tests(
                output, tests, sens, 1.2, 0)
