#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#

import unittest

import numpy as np
import numpy.testing as npt
# from scipy.stats import gamma
from iteration_utilities import deepflatten

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

        # Initial number of susceptibles
        susceptibles = [[5, 6], [7, 4]]

        # Initial number of infectives
        infectives_pre = [[0, 0], [0, 0]]
        infectives_pre_ss = [[0, 0], [0, 0]]
        infectives_asym = [[0, 0], [0, 0]]
        infectives_asym_ss = [[0, 0], [0, 0]]
        infectives_sym = [[0, 0], [0, 0]]
        infectives_sym_ss = [[0, 0], [0, 0]]

        # Average times in compartments
        k = 3.43
        kS = 2.57
        kQ = 1
        kR = 9 * np.ones(len(age_groups))
        kRI = 10 * np.ones(len(age_groups))

        # Proportion of asymptomatic, super-spreader and dead cases
        Pa = 0.658 * np.ones(len(age_groups))
        Pss = 0.0955
        Pd = 0.05 * np.ones(len(age_groups))

        # Transmission parameters
        beta_min = 0.228,
        beta_max = 0.927
        bss = 3.11
        gamma = 1
        s50 = 35.3

        parameters = [
            2, susceptibles, [[0, 0], [0, 0]], infectives_pre,
            infectives_asym, infectives_sym, infectives_pre_ss,
            infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
            [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
            k, kS, kQ, kR, kRI, Pa, Pss, Pd,
            beta_min, beta_max, bss, gamma, s50, 'RK45']

        # List of times at which we wish to evaluate the states of the
        # compartments of the model
        times = [1, 2]

        output_scipy_solver = model.simulate(
            list(deepflatten(parameters, ignore=str)), times)

        output = [7, 4]
        output.extend([0] * 24)

        npt.assert_almost_equal(
            output_scipy_solver,
            np.array([
                output,
                output
            ]), decimal=3)

        with self.assertRaises(TypeError):
            model.simulate(list(deepflatten(parameters, ignore=str)), '0')

        with self.assertRaises(TypeError):
            model.simulate(list(deepflatten(parameters, ignore=str)), ['1', 2])

        with self.assertRaises(ValueError):
            model.simulate(list(deepflatten(parameters, ignore=str)), [0, 1])

        with self.assertRaises(TypeError):
            model.simulate('parameters', times)

        with self.assertRaises(ValueError):
            model.simulate([0], times)

        with self.assertRaises(TypeError):
            parameters1 = [
                0.5, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(
                list(deepflatten(parameters1, ignore=(str, float))),
                times)

        with self.assertRaises(ValueError):
            parameters1 = [
                0, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                3, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            susceptibles1 = [5, 6]

            parameters1 = [
                1, susceptibles1, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0, 0], [0, 0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                '4', kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                -1, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, '4', kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, -1, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, '4', kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, -1, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, ['0', 1], kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, [-1, 3], kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, [3, 3, 3], kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, ['3', 0], Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, [3, -1], Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, [3], Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, ['0', 0], Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, [0.2, -1], Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, [1.5, 0.5], Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, [0.5, 0.5, 0.5], Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, '0', Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, -1, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, 1.5, Pd,
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, ['0', '0'],
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, [0.2, -1],
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, [1.5, 0.5],
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, [0.5],
                beta_min, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                '0', beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                -1, beta_max, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, '1', bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, -1, bss, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, '0', gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, -2, gamma, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, '0', s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, -1, s50, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, '0', 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, -1, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, 150, 'RK45']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(TypeError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 3]

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)

        with self.assertRaises(ValueError):
            parameters1 = [
                1, susceptibles, [[0, 0], [0, 0]], infectives_pre,
                infectives_asym, infectives_sym, infectives_pre_ss,
                infectives_asym_ss, infectives_sym_ss, [[0, 0], [0, 0]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                k, kS, kQ, kR, kRI, Pa, Pss, Pd,
                beta_min, beta_max, bss, gamma, s50, 'my-solver2']

            model.simulate(list(deepflatten(parameters1, ignore=str)), times)
