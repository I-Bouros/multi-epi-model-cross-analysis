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


class TestRocheSEIRModel(unittest.TestCase):
    """
    Test the 'RocheSEIRModel' class.
    """
    def test__init__(self):
        model = em.RocheSEIRModel()

        self.assertEqual(
            model._output_names, [
                'S', 'E', 'Ia', 'Iaa', 'Is', 'Ias', 'Iaas', 'Iss', 'Iq', 'R',
                'Ra', 'D', 'Incidence'])
        self.assertEqual(
            model._parameter_names, [
                'S0', 'E0', 'Ia0', 'Iaa0', 'Is0', 'Ias0', 'Iaas0', 'Iss0',
                'Iq0', 'R0', 'Ra0', 'D0', 'k', 'kS', 'kQ', 'kR', 'kRI', 'Pa',
                'Pss', 'Pd', 'beta_min', 'beta_max', 'bss', 'gamma', 's50'])
        self.assertEqual(model._n_outputs, 13)
        self.assertEqual(model._n_parameters, 25)

    def test_n_outputs(self):
        model = em.RocheSEIRModel()
        self.assertEqual(model.n_outputs(), 13)

    def test_n_parameters(self):
        model = em.RocheSEIRModel()
        self.assertEqual(model.n_parameters(), 25)

    def test_output_names(self):
        model = em.RocheSEIRModel()
        self.assertEqual(
            model.output_names(),
            ['S', 'E', 'Ia', 'Iaa', 'Is', 'Ias', 'Iaas', 'Iss', 'Iq', 'R',
             'Ra', 'D', 'Incidence'])

    def test_parameter_names(self):
        model = em.RocheSEIRModel()
        self.assertEqual(
            model.parameter_names(),
            ['S0', 'E0', 'Ia0', 'Iaa0', 'Is0', 'Ias0', 'Iaas0', 'Iss0',
             'Iq0', 'R0', 'Ra0', 'D0', 'k', 'kS', 'kQ', 'kR', 'kRI', 'Pa',
             'Pss', 'Pd', 'beta_min', 'beta_max', 'bss', 'gamma', 's50'])

    def test_set_regions(self):
        model = em.RocheSEIRModel()
        regions = ['London', 'Cornwall']
        model.set_regions(regions)

        self.assertEqual(
            model.region_names(),
            ['London', 'Cornwall'])

    def test_set_age_groups(self):
        model = em.RocheSEIRModel()
        age_groups = ['0-10', '10-20']
        model.set_age_groups(age_groups)

        self.assertEqual(
            model.age_groups_names(),
            ['0-10', '10-20'])

    def test_set_outputs(self):
        model = em.RocheSEIRModel()
        outputs = ['S', 'Ia', 'Ias', 'Incidence']
        model.set_outputs(outputs)

        with self.assertRaises(ValueError):
            outputs1 = ['S', 'E', 'Ia', 'I2', 'Incidence']
            model.set_outputs(outputs1)

    def test_read_npis_data(self):
        model = em.RocheSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # NPIs data
        max_levels_npi = [3, 3, 2, 4, 2, 3, 2, 4, 2]
        targeted_npi = [True, True, True, True, True, True, True, False, True]
        general_npi = [
            True, False, True, True, False, False, False, False, False]
        reg_levels_npi = [
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]]]
        time_changes_npi = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_npis_data(
            max_levels_npi, targeted_npi, general_npi, reg_levels_npi,
            time_changes_npi)

        self.assertEqual(model.max_levels_npi, [
            3, 3, 2, 4, 2, 3, 2, 4, 2])
        self.assertEqual(model.targeted_npi, [
            True, True, True, True, True, True, True, False, True])
        self.assertEqual(model.general_npi, [
            True, False, True, True, False, False, False, False, False])
        npt.assert_array_equal(np.asarray(model.reg_levels_npi), np.array([
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]]]))

        with self.assertRaises(TypeError):
            model.read_npis_data(
                0, targeted_npi, general_npi, reg_levels_npi,
                time_changes_npi)

        with self.assertRaises(TypeError):
            max_levels_npi1 = [3, 3, 2, 4, 2.0, 3, 2, 4, 2]

            model.read_npis_data(
                max_levels_npi1, targeted_npi, general_npi, reg_levels_npi,
                time_changes_npi)

        with self.assertRaises(ValueError):
            max_levels_npi1 = [3, 3, 2, 4, 2, 3, 2, 0, 2]

            model.read_npis_data(
                max_levels_npi1, targeted_npi, general_npi, reg_levels_npi,
                time_changes_npi)

        with self.assertRaises(TypeError):
            model.read_npis_data(
                max_levels_npi, '0', general_npi, reg_levels_npi,
                time_changes_npi)

        with self.assertRaises(ValueError):
            targeted_npi1 = [True, True, True, True, True, True, True]

            model.read_npis_data(
                max_levels_npi, targeted_npi1, general_npi, reg_levels_npi,
                time_changes_npi)

        with self.assertRaises(TypeError):
            targeted_npi1 = [
                True, True, True, True, True, 1, True, False, True]

            model.read_npis_data(
                max_levels_npi, targeted_npi1, general_npi, reg_levels_npi,
                time_changes_npi)

        with self.assertRaises(TypeError):
            model.read_npis_data(
                max_levels_npi, targeted_npi, '0', reg_levels_npi,
                time_changes_npi)

        with self.assertRaises(ValueError):
            general_npi1 = [True, True, True, True, True, True, True]

            model.read_npis_data(
                max_levels_npi, targeted_npi, general_npi1, reg_levels_npi,
                time_changes_npi)

        with self.assertRaises(TypeError):
            general_npi1 = [
                True, True, True, True, True, 1, True, False, True]

            model.read_npis_data(
                max_levels_npi, targeted_npi, general_npi1, reg_levels_npi,
                time_changes_npi)

        with self.assertRaises(ValueError):
            general_npi1 = [
                True, True, True, True, True, True, True, True, True]

            model.read_npis_data(
                max_levels_npi, targeted_npi, general_npi1, reg_levels_npi,
                time_changes_npi)

        with self.assertRaises(TypeError):
            model.read_npis_data(
                max_levels_npi, targeted_npi, general_npi, 0,
                time_changes_npi)

        with self.assertRaises(ValueError):
            reg_levels_npi1 = [
                [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]],
                [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]],
                [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]]]

            model.read_npis_data(
                max_levels_npi, targeted_npi, general_npi, reg_levels_npi1,
                time_changes_npi)

        with self.assertRaises(TypeError):
            reg_levels_npi1 = [
                [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]],
                0]

            model.read_npis_data(
                max_levels_npi, targeted_npi, general_npi, reg_levels_npi1,
                time_changes_npi)

        with self.assertRaises(ValueError):
            reg_levels_npi1 = [
                [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]],
                [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [3, 3, 2, 4, 2, 3, 2, 4, 2]]]

            model.read_npis_data(
                max_levels_npi, targeted_npi, general_npi, reg_levels_npi1,
                time_changes_npi)

        with self.assertRaises(ValueError):
            reg_levels_npi1 = [
                [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]],
                [[0, 0, 0, 0, 0, 0], [3, 3, 2, 2, 4, 2]]]

            model.read_npis_data(
                max_levels_npi, targeted_npi, general_npi, reg_levels_npi1,
                time_changes_npi)

        with self.assertRaises(TypeError):
            reg_levels_npi1 = [
                [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]],
                [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4.0, 2, 3, 2, 4, 2]]]

            model.read_npis_data(
                max_levels_npi, targeted_npi, general_npi, reg_levels_npi1,
                time_changes_npi)

        with self.assertRaises(ValueError):
            reg_levels_npi1 = [
                [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]],
                [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, -1, 2, 3, 2, 4, 2]]]

            model.read_npis_data(
                max_levels_npi, targeted_npi, general_npi, reg_levels_npi1,
                time_changes_npi)

        with self.assertRaises(ValueError):
            reg_levels_npi1 = [
                [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]],
                [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 6, 2, 3, 2, 4, 2]]]

            model.read_npis_data(
                max_levels_npi, targeted_npi, general_npi, reg_levels_npi1,
                time_changes_npi)

        with self.assertRaises(TypeError):
            model.read_npis_data(
                max_levels_npi, targeted_npi, general_npi, reg_levels_npi,
                0)

        with self.assertRaises(TypeError):
            model.read_npis_data(
                max_levels_npi, targeted_npi, general_npi, reg_levels_npi,
                [1, '14'])

        with self.assertRaises(ValueError):
            model.read_npis_data(
                max_levels_npi, targeted_npi, general_npi, reg_levels_npi,
                [-1, 14])

    def test_simulate(self):
        model = em.RocheSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        contact_data_matrix_0 = np.array([[10, 5.2], [0, 3]])
        contact_data_matrix_1 = np.array([[1, 0], [0, 3]])

        region_data_matrix_0_0 = np.array([[1, 10], [1, 6]])
        region_data_matrix_0_1 = np.array([[0.5, 3], [0.3, 3]])
        region_data_matrix_1_0 = np.array([[0.85, 1], [0.9, 6]])
        region_data_matrix_1_1 = np.array([[0.5, 0.2], [0.29, 4.6]])

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
        time_changes_contact = [1, 14]
        matrices_region = [
            [regional_0_0, regional_0_1],
            [regional_1_0, regional_1_1]]
        time_changes_region = [1, 14]

        # NPIs data
        max_levels_npi = [3, 3, 2, 4, 2, 3, 2, 4, 2]
        targeted_npi = [True, True, True, True, True, True, True, False, True]
        general_npi = [
            True, False, True, True, False, False, False, False, False]
        reg_levels_npi = [
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]]]
        time_changes_npi = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)
        model.read_npis_data(
            max_levels_npi, targeted_npi, general_npi, reg_levels_npi,
            time_changes_npi)

        # Set ICs parameters
        ICs = em.RocheICs(
            model=model,
            susceptibles_IC=[[5, 6], [7, 4]],
            exposed_IC=[[0, 0], [0, 0]],
            infectives_pre_IC=[[0, 0], [0, 0]],
            infectives_asym_IC=[[0, 0], [0, 0]],
            infectives_sym_IC=[[0, 0], [0, 0]],
            infectives_pre_ss_IC=[[0, 0], [0, 0]],
            infectives_asym_ss_IC=[[0, 0], [0, 0]],
            infectives_sym_ss_IC=[[0, 0], [0, 0]],
            infectives_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]],
            recovered_asym_IC=[[0, 0], [0, 0]],
            dead_IC=[[0, 0], [0, 0]]
        )

        # Set average times in compartments
        compartment_times = em.RocheCompartmentTimes(
            model=model,
            k=3.43,
            kS=2.57,
            kQ=1,
            kR=9 * np.ones(len(model.age_groups)),
            kRI=10
        )

        # Set proportion of asymptomatic, super-spreader and dead cases
        proportion_parameters = em.RocheProportions(
            model=model,
            Pa=0.658 * np.ones(len(age_groups)),
            Pss=0.0955,
            Pd=0.05
        )

        # Set transmission parameters
        transmission_parameters = em.RocheTransmission(
            model=model,
            beta_min=0.228,
            beta_max=0.927,
            bss=3.11,
            gamma=1,
            s50=35.3
        )

        # Set other simulation parameters
        simulation_parameters = em.RocheSimParameters(
            model=model,
            region_index=2,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.RocheParametersController(
            model=model,
            ICs=ICs,
            compartment_times=compartment_times,
            proportion_parameters=proportion_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        output_scipy_solver = model.simulate(parameters)

        output = [7, 4]
        output.extend([0] * 24)

        npt.assert_almost_equal(
            output_scipy_solver,
            np.array([
                output,
                output
            ]), decimal=3)
