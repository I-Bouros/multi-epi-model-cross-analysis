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

    def test_set_age_groups(self):
        model = em.PheSEIRModel()
        age_groups = ['0-10', '10-20']
        model.set_age_groups(age_groups)

        self.assertEqual(
            model.age_groups_names(),
            ['0-10', '10-20'])

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

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.PheRegParameters(
            model=model,
            initial_r=[0.5, 1],
            region_index=2,
            betas=[[1]*2, [1]*2],
            times=[1, 2]
        )

        # Set ICs parameters
        ICs = em.PheICs(
            model=model,
            susceptibles_IC=[[5, 6], [7, 8]],
            exposed1_IC=[[0, 0], [0, 0]],
            exposed2_IC=[[0, 0], [0, 0]],
            infectives1_IC=[[0, 0], [0, 0]],
            infectives2_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.PheDiseaseParameters(
            model=model,
            dL=4,
            dI=4
        )

        # Set other simulation parameters
        simulation_parameters = em.PheSimParameters(
            model=model,
            delta_t=0.5,
            method='RK45'
        )

        # Set all parameters in the controller
        parameters = em.PheParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            simulation_parameters=simulation_parameters
        )

        output_my_solver = model.simulate(parameters)

        npt.assert_almost_equal(
            output_my_solver,
            np.array([
                [7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]), decimal=3)

        simulation_parameters.method = 'my-solver'

        # Set all parameters in the controller
        parameters = em.PheParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            simulation_parameters=simulation_parameters
        )

        output_scipy_solver = model.simulate(parameters)

        npt.assert_almost_equal(
            output_scipy_solver,
            np.array([
                [7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]), decimal=3)

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

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.PheRegParameters(
            model=model,
            initial_r=[0.5, 1],
            region_index=2,
            betas=[[1]*2, [1]*2],
            times=[1, 2]
        )

        # Set ICs parameters
        ICs = em.PheICs(
            model=model,
            susceptibles_IC=[[5, 6], [7, 8]],
            exposed1_IC=[[0, 0], [0, 0]],
            exposed2_IC=[[0, 0], [0, 0]],
            infectives1_IC=[[0, 0], [0, 0]],
            infectives2_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.PheDiseaseParameters(
            model=model,
            dL=4,
            dI=4
        )

        # Set other simulation parameters
        simulation_parameters = em.PheSimParameters(
            model=model,
            delta_t=0.5,
            method='my-solver'
        )

        # Set all parameters in the controller
        parameters = em.PheParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            simulation_parameters=simulation_parameters
        )

        output = model.simulate(parameters)

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

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.PheRegParameters(
            model=model,
            initial_r=[0.5, 1],
            region_index=2,
            betas=[[1]*2, [1]*2],
            times=[1, 2]
        )

        # Set ICs parameters
        ICs = em.PheICs(
            model=model,
            susceptibles_IC=[[5, 6], [7, 8]],
            exposed1_IC=[[0, 0], [0, 0]],
            exposed2_IC=[[0, 0], [0, 0]],
            infectives1_IC=[[0, 0], [0, 0]],
            infectives2_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.PheDiseaseParameters(
            model=model,
            dL=4,
            dI=4
        )

        # Set other simulation parameters
        simulation_parameters = em.PheSimParameters(
            model=model,
            delta_t=0.5,
            method='my-solver'
        )

        # Set all parameters in the controller
        parameters = em.PheParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            simulation_parameters=simulation_parameters
        )

        output = model.simulate(parameters)

        new_infections = model.new_infections(output)

        obs_death = [10, 12]
        fatality_ratio = [0.1, 0.5]
        time_to_death = [0.5, 0.5]

        self.assertEqual(
            model.loglik_deaths(
                obs_death, new_infections, fatality_ratio,
                time_to_death, 0.5, 1).shape,
            (len(age_groups),))

        with self.assertRaises(ValueError):
            model.loglik_deaths(
                obs_death, new_infections, fatality_ratio,
                time_to_death, 0.5, -1)

        with self.assertRaises(TypeError):
            model.loglik_deaths(
                obs_death, new_infections, fatality_ratio,
                time_to_death, 0.5, '1')

        with self.assertRaises(ValueError):
            model.loglik_deaths(
                obs_death, new_infections, fatality_ratio,
                time_to_death, 0.5, 2)

        with self.assertRaises(ValueError):
            model.loglik_deaths(
                0, new_infections, fatality_ratio,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            obs_death1 = np.array([5, 6, 0, 0])

            model.loglik_deaths(
                obs_death1, new_infections, fatality_ratio,
                time_to_death, 0.5, 1)

        with self.assertRaises(TypeError):
            obs_death1 = np.array(['5', 6])

            model.loglik_deaths(
                obs_death1, new_infections, fatality_ratio,
                time_to_death, 0.5, 1)

        with self.assertRaises(ValueError):
            obs_death1 = np.array([5, -1])

            model.loglik_deaths(
                obs_death1, new_infections, fatality_ratio,
                time_to_death, 0.5, 1)

    def test_check_death_format(self):
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

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.PheRegParameters(
            model=model,
            initial_r=[0.5, 1],
            region_index=2,
            betas=[[1]*2, [1]*2],
            times=[1, 2]
        )

        # Set ICs parameters
        ICs = em.PheICs(
            model=model,
            susceptibles_IC=[[5, 6], [7, 8]],
            exposed1_IC=[[0, 0], [0, 0]],
            exposed2_IC=[[0, 0], [0, 0]],
            infectives1_IC=[[0, 0], [0, 0]],
            infectives2_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.PheDiseaseParameters(
            model=model,
            dL=4,
            dI=4
        )

        # Set other simulation parameters
        simulation_parameters = em.PheSimParameters(
            model=model,
            delta_t=0.5,
            method='my-solver'
        )

        # Set all parameters in the controller
        parameters = em.PheParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            simulation_parameters=simulation_parameters
        )

        output = model.simulate(parameters)

        new_infections = model.new_infections(output)

        fatality_ratio = [0.1, 0.5]
        time_to_death = [0.5, 0.5]

        with self.assertRaises(TypeError):
            model.check_death_format(
                new_infections, fatality_ratio, time_to_death, '0.5')

        with self.assertRaises(ValueError):
            model.check_death_format(
                new_infections, fatality_ratio, time_to_death, -2)

        with self.assertRaises(ValueError):
            new_infections1 = \
                np.array([5, 6])

            model.check_death_format(
                new_infections1, fatality_ratio, time_to_death, 0.5)

        with self.assertRaises(ValueError):
            new_infections1 = np.array([
                [5, 6, 0, 0],
                [5, 6, 0, 0]])

            model.check_death_format(
                new_infections1, fatality_ratio, time_to_death, 0.5)

        with self.assertRaises(ValueError):
            new_infections1 = np.array([
                [5, 6], [5, 6], [5, 6]])

            model.check_death_format(
                new_infections1, fatality_ratio, time_to_death, 0.5)

        with self.assertRaises(TypeError):
            new_infections1 = np.array([
                ['5', 6],
                [5, '0']])

            model.check_death_format(
                new_infections1, fatality_ratio, time_to_death, 0.5)

        with self.assertRaises(ValueError):
            fatality_ratio1 = 0

            model.check_death_format(
                new_infections, fatality_ratio1, time_to_death, 0.5)

        with self.assertRaises(ValueError):
            fatality_ratio1 = np.array([0.1, 0.5, 0.1])

            model.check_death_format(
                new_infections, fatality_ratio1, time_to_death, 0.5)

        with self.assertRaises(TypeError):
            fatality_ratio1 = np.array([0.1, '0.5'])

            model.check_death_format(
                new_infections, fatality_ratio1, time_to_death, 0.5)

        with self.assertRaises(ValueError):
            fatality_ratio1 = np.array([-0.1, 0.5])

            model.check_death_format(
                new_infections, fatality_ratio1, time_to_death, 0.5)

        with self.assertRaises(ValueError):
            fatality_ratio1 = np.array([0.1, 1.5])

            model.check_death_format(
                new_infections, fatality_ratio1, time_to_death, 0.5)

        with self.assertRaises(ValueError):
            time_to_death1 = np.array([[0.5], [0.5]])

            model.check_death_format(
                new_infections, fatality_ratio, time_to_death1, 0.5)

        with self.assertRaises(ValueError):
            time_to_death1 = np.array([0.5, 0.5, 0.15])

            model.check_death_format(
                new_infections, fatality_ratio, time_to_death1, 0.5)

        with self.assertRaises(TypeError):
            time_to_death1 = np.array(['0.1', 0.5])

            model.check_death_format(
                new_infections, fatality_ratio, time_to_death1, 0.5)

        with self.assertRaises(ValueError):
            time_to_death1 = np.array([-0.1, 0.5])

            model.check_death_format(
                new_infections, fatality_ratio, time_to_death1, 0.5)

        with self.assertRaises(ValueError):
            time_to_death1 = np.array([0.5, 1.1])

            model.check_death_format(
                new_infections, fatality_ratio, time_to_death1, 0.5)

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

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.PheRegParameters(
            model=model,
            initial_r=[0.5, 1],
            region_index=1,
            betas=[[1]*60, [1]*60],
            times=np.arange(1, 61).tolist()
        )

        # Set ICs parameters
        ICs = em.PheICs(
            model=model,
            susceptibles_IC=[[5, 6], [7, 8]],
            exposed1_IC=[[0, 0], [0, 0]],
            exposed2_IC=[[0.1, 0.2], [0, 0]],
            infectives1_IC=[[0, 0], [0, 0]],
            infectives2_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.PheDiseaseParameters(
            model=model,
            dL=4,
            dI=4
        )

        # Set other simulation parameters
        simulation_parameters = em.PheSimParameters(
            model=model,
            delta_t=0.5,
            method='my-solver'
        )

        # Set all parameters in the controller
        parameters = em.PheParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            simulation_parameters=simulation_parameters
        )

        output = model.simulate(parameters)

        new_infections = model.new_infections(output)

        fatality_ratio = [0.1, 0.5]

        td_mean = 15.0
        td_var = 12.1**2
        theta = td_var / td_mean
        k = td_mean / theta
        time_to_death = gamma(k, scale=theta).pdf(np.arange(1, 60)).tolist()

        self.assertEqual(
            model.samples_deaths(
                new_infections, fatality_ratio,
                time_to_death, 0.5, 41).shape,
            (len(age_groups),))

        self.assertEqual(
            model.samples_deaths(
                new_infections, fatality_ratio,
                time_to_death, 0.5, 1).shape,
            (len(age_groups),))

        with self.assertRaises(ValueError):
            model.samples_deaths(
                new_infections, fatality_ratio,
                time_to_death, 0.5, -1)

        with self.assertRaises(TypeError):
            model.samples_deaths(
                new_infections, fatality_ratio,
                time_to_death, 0.5, '1')

        with self.assertRaises(ValueError):
            model.samples_deaths(
                new_infections, fatality_ratio,
                time_to_death, 0.5, 62)

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

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.PheRegParameters(
            model=model,
            initial_r=[0.5, 1],
            region_index=2,
            betas=[[1]*2, [1]*2],
            times=[1, 2]
        )

        # Set ICs parameters
        ICs = em.PheICs(
            model=model,
            susceptibles_IC=[[5, 6], [7, 8]],
            exposed1_IC=[[0, 0], [0, 0]],
            exposed2_IC=[[0, 0], [0, 0]],
            infectives1_IC=[[0, 0], [0, 0]],
            infectives2_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.PheDiseaseParameters(
            model=model,
            dL=4,
            dI=4
        )

        # Set other simulation parameters
        simulation_parameters = em.PheSimParameters(
            model=model,
            delta_t=0.5,
            method='my-solver'
        )

        # Set all parameters in the controller
        parameters = em.PheParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            simulation_parameters=simulation_parameters
        )

        output = model.simulate(parameters)

        obs_pos = [10, 12]
        tests = [[20, 30], [10, 0]]
        sens = 0.9
        spec = 0.1

        self.assertEqual(
            model.loglik_positive_tests(
                obs_pos, output, tests[0], sens, spec, 0).shape,
            (len(age_groups),))

        with self.assertRaises(TypeError):
            model.loglik_positive_tests(
                obs_pos, output, tests[0], sens, spec, '1')

        with self.assertRaises(ValueError):
            model.loglik_positive_tests(
                obs_pos, output, tests[0], sens, spec, -1)

        with self.assertRaises(ValueError):
            model.loglik_positive_tests(
                obs_pos, output, tests[0], sens, spec, 3)

        with self.assertRaises(ValueError):
            model.loglik_positive_tests(
                0, output, tests[0], sens, spec, 0)

        with self.assertRaises(ValueError):
            obs_pos1 = np.array([5, 6, 0, 0])

            model.loglik_positive_tests(
                obs_pos1, output, tests[0], sens, spec, 0)

        with self.assertRaises(TypeError):
            obs_pos1 = np.array(['5', 6])

            model.loglik_positive_tests(
                obs_pos1, output, tests[0], sens, spec, 0)

        with self.assertRaises(ValueError):
            obs_pos1 = np.array([5, -1])

            model.loglik_positive_tests(
                obs_pos1, output, tests[0], sens, spec, 0)

        with self.assertRaises(ValueError):
            obs_pos1 = np.array([5, 40])

            model.loglik_positive_tests(
                obs_pos1, output, tests[0], sens, spec, 0)

    def test_check_positives_format(self):
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

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.PheRegParameters(
            model=model,
            initial_r=[0.5, 1],
            region_index=2,
            betas=[[1]*2, [1]*2],
            times=[1, 2]
        )

        # Set ICs parameters
        ICs = em.PheICs(
            model=model,
            susceptibles_IC=[[5, 6], [7, 8]],
            exposed1_IC=[[0, 0], [0, 0]],
            exposed2_IC=[[0, 0], [0, 0]],
            infectives1_IC=[[0, 0], [0, 0]],
            infectives2_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.PheDiseaseParameters(
            model=model,
            dL=4,
            dI=4
        )

        # Set other simulation parameters
        simulation_parameters = em.PheSimParameters(
            model=model,
            delta_t=0.5,
            method='my-solver'
        )

        # Set all parameters in the controller
        parameters = em.PheParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            simulation_parameters=simulation_parameters
        )

        output = model.simulate(parameters)

        tests = [[20, 30], [10, 0]]
        sens = 0.9
        spec = 0.1

        with self.assertRaises(ValueError):
            output1 = np.array([5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            model.check_positives_format(
                output1, tests, sens, spec)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0]])

            model.check_positives_format(
                output1, tests, sens, spec)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

            model.check_positives_format(
                output1, tests, sens, spec)

        with self.assertRaises(TypeError):
            output1 = np.array([
                ['5', 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, '0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

            model.check_positives_format(
                output1, tests, sens, spec)

        with self.assertRaises(ValueError):
            tests1 = 100

            model.check_positives_format(
                output, tests1, sens, spec)

        with self.assertRaises(ValueError):
            tests1 = np.array([2, 50])

            model.check_positives_format(
                output, tests1, sens, spec)

        with self.assertRaises(ValueError):
            tests1 = np.array([[20, 30, 1], [10, 0, 0]])

            model.check_positives_format(
                output, tests1, sens, spec)

        with self.assertRaises(TypeError):
            tests1 = np.array([[20, '30'], [10, 0]])

            model.check_positives_format(
                output, tests1, sens, spec)

        with self.assertRaises(ValueError):
            tests1 = np.array([[-1, 50], [10, 0]])

            model.check_positives_format(
                output, tests1, sens, spec)

        with self.assertRaises(TypeError):
            model.check_positives_format(
                output, tests, '0.9', spec)

        with self.assertRaises(ValueError):
            model.check_positives_format(
                output, tests, -0.2, spec)

        with self.assertRaises(ValueError):
            model.check_positives_format(
                output, tests, 1.2, spec)

        with self.assertRaises(TypeError):
            model.check_positives_format(
                output, tests, sens, '0.1')

        with self.assertRaises(ValueError):
            model.check_positives_format(
                output, tests, sens, -0.1)

        with self.assertRaises(ValueError):
            model.check_positives_format(
                output, tests, sens, 1.2)

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

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.PheRegParameters(
            model=model,
            initial_r=[0.5, 1],
            region_index=2,
            betas=[[1]*2, [1]*2],
            times=[1, 2]
        )

        # Set ICs parameters
        ICs = em.PheICs(
            model=model,
            susceptibles_IC=[[5, 6], [7, 8]],
            exposed1_IC=[[0, 0], [0, 0]],
            exposed2_IC=[[0, 0], [0, 0]],
            infectives1_IC=[[0, 0], [0, 0]],
            infectives2_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.PheDiseaseParameters(
            model=model,
            dL=4,
            dI=4
        )

        # Set other simulation parameters
        simulation_parameters = em.PheSimParameters(
            model=model,
            delta_t=0.5,
            method='my-solver'
        )

        # Set all parameters in the controller
        parameters = em.PheParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            simulation_parameters=simulation_parameters
        )

        output = model.simulate(parameters)

        tests = [[20, 30], [10, 0]]
        sens = 0.9
        spec = 0.1

        self.assertEqual(
            model.samples_positive_tests(
                output, tests[0], sens, spec, 0).shape,
            (len(age_groups),))

        with self.assertRaises(TypeError):
            model.samples_positive_tests(
                output, tests[0], sens, spec, '1')

        with self.assertRaises(ValueError):
            model.samples_positive_tests(
                output, tests[0], sens, spec, -1)

        with self.assertRaises(ValueError):
            model.samples_positive_tests(
                output, tests[0], sens, spec, 3)
