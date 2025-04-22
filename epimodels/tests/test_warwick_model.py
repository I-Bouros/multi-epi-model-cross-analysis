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


class TestWarwickSEIRModel(unittest.TestCase):
    """
    Test the 'WarwickSEIRModel' class.
    """
    def test__init__(self):
        model = em.WarwickSEIRModel()

        self.assertEqual(
            model._output_names, [
                'S', 'E1f', 'E1sd', 'E1su', 'E1q', 'E2f', 'E2sd', 'E2su',
                'E2q', 'E3f', 'E3sd', 'E3su', 'E3q', 'If', 'Isd', 'Isu', 'Iqf',
                'Iqs', 'Af', 'As', 'Aq', 'R', 'Incidence'])
        self.assertEqual(
            model._parameter_names, [
                'S0', 'E1f0', 'E1sd0', 'E1su0', 'E1q0', 'E2f0', 'E2sd0',
                'E2su0', 'E2q0', 'E3f0', 'E3sd0', 'E3su0', 'E3q0', 'If0',
                'Isd0', 'Isu0', 'Iqf0', 'Iqs0', 'Af0', 'As0', 'Aq0', 'R0',
                'sig', 'tau', 'eps', 'gamma', 'd', 'H'])
        self.assertEqual(model._n_outputs, 23)
        self.assertEqual(model._n_parameters, 28)

    def test_n_outputs(self):
        model = em.WarwickSEIRModel()
        self.assertEqual(model.n_outputs(), 23)

    def test_n_parameters(self):
        model = em.WarwickSEIRModel()
        self.assertEqual(model.n_parameters(), 28)

    def test_output_names(self):
        model = em.WarwickSEIRModel()
        self.assertEqual(
            model.output_names(),
            ['S', 'E1f', 'E1sd', 'E1su', 'E1q', 'E2f', 'E2sd', 'E2su',
             'E2q', 'E3f', 'E3sd', 'E3su', 'E3q', 'If', 'Isd', 'Isu', 'Iqf',
             'Iqs', 'Af', 'As', 'Aq', 'R', 'Incidence'])

    def test_parameter_names(self):
        model = em.WarwickSEIRModel()
        self.assertEqual(
            model.parameter_names(),
            ['S0', 'E1f0', 'E1sd0', 'E1su0', 'E1q0', 'E2f0', 'E2sd0', 'E2su0',
             'E2q0', 'E3f0', 'E3sd0', 'E3su0', 'E3q0', 'If0', 'Isd0', 'Isu0',
             'Iqf0', 'Iqs0', 'Af0', 'As0', 'Aq0', 'R0', 'sig', 'tau', 'eps',
             'gamma', 'd', 'H'])

    def test_set_regions(self):
        model = em.WarwickSEIRModel()
        regions = ['London', 'Cornwall']
        model.set_regions(regions)

        self.assertEqual(
            model.region_names(),
            ['London', 'Cornwall'])

    def test_set_age_groups(self):
        model = em.WarwickSEIRModel()
        age_groups = ['0-10', '10-20']
        model.set_age_groups(age_groups)

        self.assertEqual(
            model.age_groups_names(),
            ['0-10', '10-20'])

    def test_set_outputs(self):
        model = em.WarwickSEIRModel()
        outputs = ['S', 'If', 'Isd', 'Incidence']
        model.set_outputs(outputs)

        with self.assertRaises(ValueError):
            outputs1 = ['S', 'E', 'If', 'Isd', 'Incidence']
            model.set_outputs(outputs1)

    def test_read_contact_data(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)

        with self.assertRaises(ValueError):
            school_contacts_01 = em.ContactMatrix(
                ['0-10'], 0.3 * np.array([[10]]))

            school_matrices_contact1 = [
                school_contacts_01, school_contacts_1]

            model.read_contact_data(
                house_matrices_contact, school_matrices_contact1,
                work_matrices_contact, other_matrices_contact,
                time_changes_contact)

        with self.assertRaises(ValueError):
            work_contacts_01 = em.ContactMatrix(
                ['0-10'], 0.3 * np.array([[10]]))

            work_matrices_contact1 = [
                work_contacts_01, work_contacts_1]

            model.read_contact_data(
                house_matrices_contact, school_matrices_contact,
                work_matrices_contact1, other_matrices_contact,
                time_changes_contact)

        with self.assertRaises(ValueError):
            other_contacts_01 = em.ContactMatrix(
                ['0-10'], 0.3 * np.array([[10]]))

            other_matrices_contact1 = [
                other_contacts_01, other_contacts_1]

            model.read_contact_data(
                house_matrices_contact, school_matrices_contact,
                work_matrices_contact, other_matrices_contact1,
                time_changes_contact)

    def test_read_regional_data(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        with self.assertRaises(ValueError):
            school_regional_0_01 = em.RegionMatrix(
                regions[0], ['0-10'], 0.3 * np.array([[10]]))

            school_matrices_region1 = [
                [school_regional_0_01, school_regional_0_1],
                [school_regional_1_0, school_regional_1_1]]

            model.read_regional_data(
                house_matrices_region, school_matrices_region1,
                work_matrices_region, other_matrices_region,
                time_changes_region)

        with self.assertRaises(ValueError):
            school_regional_0_01 = em.RegionMatrix(
                regions[1], age_groups, school_region_data_matrix_0_0)

            school_matrices_region1 = [
                [school_regional_0_01, school_regional_0_1],
                [school_regional_1_0, school_regional_1_1]]

            model.read_regional_data(
                house_matrices_region, school_matrices_region1,
                work_matrices_region, other_matrices_region,
                time_changes_region)

        with self.assertRaises(ValueError):
            work_regional_0_01 = em.RegionMatrix(
                regions[0], ['0-10'], 0.3 * np.array([[10]]))

            work_matrices_region1 = [
                [work_regional_0_01, work_regional_0_1],
                [work_regional_1_0, work_regional_1_1]]

            model.read_regional_data(
                house_matrices_region, school_matrices_region,
                work_matrices_region1, other_matrices_region,
                time_changes_region)

        with self.assertRaises(ValueError):
            work_regional_0_01 = em.RegionMatrix(
                regions[1], age_groups, work_region_data_matrix_0_0)

            work_matrices_region1 = [
                [work_regional_0_01, work_regional_0_1],
                [work_regional_1_0, work_regional_1_1]]

            model.read_regional_data(
                house_matrices_region, school_matrices_region,
                work_matrices_region1, other_matrices_region,
                time_changes_region)

        with self.assertRaises(ValueError):
            other_regional_0_01 = em.RegionMatrix(
                regions[0], ['0-10'], 0.3 * np.array([[10]]))

            other_matrices_region1 = [
                [other_regional_0_01, other_regional_0_1],
                [other_regional_1_0, other_regional_1_1]]

            model.read_regional_data(
                house_matrices_region, school_matrices_region,
                work_matrices_region, other_matrices_region1,
                time_changes_region)

        with self.assertRaises(ValueError):
            other_regional_0_01 = em.RegionMatrix(
                regions[1], age_groups, other_region_data_matrix_0_0)

            other_matrices_region1 = [
                [other_regional_0_01, other_regional_0_1],
                [other_regional_1_0, other_regional_1_1]]

            model.read_regional_data(
                house_matrices_region, school_matrices_region,
                work_matrices_region, other_matrices_region1,
                time_changes_region)

    def test_simulate(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        output_scipy_solver = model.simulate(parameters)

        output = [7, 4]
        output.extend([0] * 44)

        npt.assert_almost_equal(
            output_scipy_solver,
            np.array([
                output,
                output
            ]), decimal=3)

    def test_new_infections(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        output = model.simulate(parameters)

        npt.assert_array_equal(
            model.new_infections(output),
            np.array([[0, 0], [0, 0]]))

        with self.assertRaises(ValueError):
            output1 = [5, 6]
            output1.extend([0] * 44)
            output1 = np.array(output1)
            model.new_infections(output1)

        with self.assertRaises(ValueError):
            output1 = [5, 6]
            output1.extend([0] * 42)
            output1 = np.array([output1, output1])
            model.new_infections(output1)

        with self.assertRaises(ValueError):
            output1 = [5, 6]
            output1.extend([0] * 44)
            output1 = np.array([output1, output1, output1])
            model.new_infections(output1)

        with self.assertRaises(TypeError):
            output1 = ['5', 6]
            output1.extend([0] * 44)
            output2 = [5, 6, '0']
            output2.extend([0] * 43)
            output1 = np.array([output1, output2])
            model.new_infections(output1)

    def test_new_hospitalisations(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        new_infections = model.new_infections(
            model.simulate(parameters))

        pDtoH = np.ones(len(age_groups))
        dDtoH = np.ones(30)

        npt.assert_array_equal(
            model.new_hospitalisations(new_infections, pDtoH, dDtoH),
            np.array([[0, 0], [0, 0]]))

        with self.assertRaises(ValueError):
            new_infections1 = np.array([0, 0])

            model.check_new_hospitalisation_format(
                new_infections1, pDtoH, dDtoH)

        with self.assertRaises(ValueError):
            new_infections1 = np.array([[0, 0], [0, 0], [0, 0]])

            model.check_new_hospitalisation_format(
                new_infections1, pDtoH, dDtoH)

        with self.assertRaises(ValueError):
            new_infections1 = np.array([[0, 0, 0], [0, 0, 0]])

            model.check_new_hospitalisation_format(
                new_infections1, pDtoH, dDtoH)

        with self.assertRaises(TypeError):
            new_infections1 = np.array([[0, '0'], [0, 0]])

            model.check_new_hospitalisation_format(
                new_infections1, pDtoH, dDtoH)

        with self.assertRaises(ValueError):
            pDtoH1 = np.ones((30, 2))

            model.check_new_hospitalisation_format(
                new_infections, pDtoH1, dDtoH)

        with self.assertRaises(ValueError):
            pDtoH1 = [1, 1, 1]

            model.check_new_hospitalisation_format(
                new_infections, pDtoH1, dDtoH)

        with self.assertRaises(TypeError):
            pDtoH1 = [1, '1']

            model.check_new_hospitalisation_format(
                new_infections, pDtoH1, dDtoH)

        with self.assertRaises(ValueError):
            pDtoH1 = [-0.2, 0.5]

            model.check_new_hospitalisation_format(
                new_infections, pDtoH1, dDtoH)

        with self.assertRaises(ValueError):
            pDtoH1 = [0.5, 1.5]

            model.check_new_hospitalisation_format(
                new_infections, pDtoH1, dDtoH)

        with self.assertRaises(ValueError):
            dDtoH1 = {'0.9': 0}

            model.check_new_hospitalisation_format(
                new_infections, pDtoH, dDtoH1)

        with self.assertRaises(ValueError):
            dDtoH1 = [1, 1, 1]

            model.check_new_hospitalisation_format(
                new_infections, pDtoH, dDtoH1)

        with self.assertRaises(TypeError):
            dDtoH1 = [1, '1'] * 25

            model.check_new_hospitalisation_format(
                new_infections, pDtoH, dDtoH1)

        with self.assertRaises(ValueError):
            dDtoH1 = [0.02] * 16 + [0.4] + [0.02] * 15

            model.check_new_hospitalisation_format(
                new_infections, pDtoH, dDtoH1)

        with self.assertRaises(ValueError):
            dDtoH1 = [-0.5] + [0.05] * 30

            model.check_new_hospitalisation_format(
                new_infections, pDtoH, dDtoH1)

        with self.assertRaises(ValueError):
            dDtoH1 = [0.02] * 30 + [1.1] + [-0.7]

            model.check_new_hospitalisation_format(
                new_infections, pDtoH, dDtoH1)

    def test_new_icu(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=np.arange(1, 34).tolist()
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        new_infections = model.new_infections(
            model.simulate(parameters))

        pDtoI = np.ones(len(age_groups))
        dDtoI = np.ones(30)

        npt.assert_array_equal(
            model.new_icu(new_infections, pDtoI, dDtoI),
            np.zeros((33, 2)))

        # Shorter simulation times
        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        new_infections = model.new_infections(
            model.simulate(parameters))

        pDtoI = np.ones(len(age_groups))
        dDtoI = np.ones(30)

        npt.assert_array_equal(
            model.new_icu(new_infections, pDtoI, dDtoI),
            np.array([[0, 0], [0, 0]]))

        with self.assertRaises(ValueError):
            new_infections1 = np.array([0, 0])

            model.check_new_icu_format(
                new_infections1, pDtoI, dDtoI)

        with self.assertRaises(ValueError):
            new_infections1 = np.array([[0, 0], [0, 0], [0, 0]])

            model.check_new_icu_format(
                new_infections1, pDtoI, dDtoI)

        with self.assertRaises(ValueError):
            new_infections1 = np.array([[0, 0, 0], [0, 0, 0]])

            model.check_new_icu_format(
                new_infections1, pDtoI, dDtoI)

        with self.assertRaises(TypeError):
            new_infections1 = np.array([[0, '0'], [0, 0]])

            model.check_new_icu_format(
                new_infections1, pDtoI, dDtoI)

        with self.assertRaises(ValueError):
            pDtoI1 = {'0.9': 0}

            model.check_new_icu_format(
                new_infections, pDtoI1, dDtoI)

        with self.assertRaises(ValueError):
            pDtoI1 = [1, 1, 1]

            model.check_new_icu_format(
                new_infections, pDtoI1, dDtoI)

        with self.assertRaises(TypeError):
            pDtoI1 = [1, '1']

            model.check_new_icu_format(
                new_infections, pDtoI1, dDtoI)

        with self.assertRaises(ValueError):
            pDtoI1 = [-0.2, 0.5]

            model.check_new_icu_format(
                new_infections, pDtoI1, dDtoI)

        with self.assertRaises(ValueError):
            pDtoI1 = [0.5, 1.5]

            model.check_new_icu_format(
                new_infections, pDtoI1, dDtoI)

        with self.assertRaises(ValueError):
            dDtoI1 = {'0.9': 0}

            model.check_new_icu_format(
                new_infections, pDtoI, dDtoI1)

        with self.assertRaises(ValueError):
            dDtoI1 = [1, 1, 1]

            model.check_new_icu_format(
                new_infections, pDtoI, dDtoI1)

        with self.assertRaises(TypeError):
            dDtoI1 = [1, '1'] * 25

            model.check_new_icu_format(
                new_infections, pDtoI, dDtoI1)

        with self.assertRaises(ValueError):
            dDtoI1 = [0.02] * 16 + [0.4] + [0.02] * 15

            model.check_new_icu_format(
                new_infections, pDtoI, dDtoI1)

        with self.assertRaises(ValueError):
            dDtoI1 = [-0.5] + [0.05] * 30

            model.check_new_icu_format(
                new_infections, pDtoI, dDtoI1)

        with self.assertRaises(ValueError):
            dDtoI1 = [0.02] * 30 + [1.1] + [-0.7]

            model.check_new_icu_format(
                new_infections, pDtoI, dDtoI1)

    def test_new_deaths(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        pDtoH = [1, 1]
        dDtoH = [1] * 30

        pHtoDeath = [1, 1]
        dHtoDeath = [1] * 30

        new_infections = model.new_infections(
            model.simulate(parameters))

        new_hospitalisations = model.new_hospitalisations(
            new_infections, pDtoH, dDtoH)

        npt.assert_array_equal(
            model.new_deaths(
                new_hospitalisations, pHtoDeath, dHtoDeath),
            np.array([[0, 0], [0, 0]]))

        with self.assertRaises(ValueError):
            new_hospitalisations1 = np.array([0, 0])

            model.check_new_deaths_format(
                new_hospitalisations1, pHtoDeath, dHtoDeath)

        with self.assertRaises(ValueError):
            new_hospitalisations1 = np.array([[0, 0], [0, 0], [0, 0]])

            model.check_new_deaths_format(
                new_hospitalisations1, pHtoDeath, dHtoDeath)

        with self.assertRaises(ValueError):
            new_hospitalisations1 = np.array([[0, 0, 0], [0, 0, 0]])

            model.check_new_deaths_format(
                new_hospitalisations1, pHtoDeath, dHtoDeath)

        with self.assertRaises(TypeError):
            new_hospitalisations1 = np.array([[0, 0], ['0', 0]])

            model.check_new_deaths_format(
                new_hospitalisations1, pHtoDeath, dHtoDeath)

        with self.assertRaises(ValueError):
            pHtoDeath1 = {'0.9': 0}

            model.check_new_deaths_format(
                new_hospitalisations, pHtoDeath1, dHtoDeath)

        with self.assertRaises(ValueError):
            pHtoDeath1 = [1, 1, 1]

            model.check_new_deaths_format(
                new_hospitalisations, pHtoDeath1, dHtoDeath)

        with self.assertRaises(TypeError):
            pHtoDeath1 = [1, '1']

            model.check_new_deaths_format(
                new_hospitalisations, pHtoDeath1, dHtoDeath)

        with self.assertRaises(ValueError):
            pHtoDeath1 = [-0.2, 0.5]

            model.check_new_deaths_format(
                new_hospitalisations, pHtoDeath1, dHtoDeath)

        with self.assertRaises(ValueError):
            pHtoDeath1 = [0.5, 1.5]

            model.check_new_deaths_format(
                new_hospitalisations, pHtoDeath1, dHtoDeath)

        with self.assertRaises(ValueError):
            dHtoDeath1 = {'0.9': 0}

            model.check_new_deaths_format(
                new_hospitalisations, pHtoDeath, dHtoDeath1)

        with self.assertRaises(ValueError):
            dHtoDeath1 = [1, 1, 1]

            model.check_new_deaths_format(
                new_hospitalisations, pHtoDeath, dHtoDeath1)

        with self.assertRaises(TypeError):
            dHtoDeath1 = [1, '1'] * 17

            model.check_new_deaths_format(
                new_hospitalisations, pHtoDeath, dHtoDeath1)

        with self.assertRaises(ValueError):
            dHtoDeath1 = [0.02]*16 + [0.4] + [0.02] * 15

            model.check_new_deaths_format(
                new_hospitalisations, pHtoDeath, dHtoDeath1)

        with self.assertRaises(ValueError):
            dHtoDeath1 = [-0.5] + [0.05] * 30

            model.check_new_deaths_format(
                new_hospitalisations, pHtoDeath, dHtoDeath1)

        with self.assertRaises(ValueError):
            dHtoDeath1 = [0.02] * 30 + [1.1] + [-0.7]

            model.check_new_deaths_format(
                new_hospitalisations, pHtoDeath, dHtoDeath1)

    def test_new_hospital_beds(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=np.arange(1, 34).tolist()
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        pDtoH = [1, 1]
        dDtoH = [1] * 30

        pDtoI = [1, 1]
        dDtoI = [1] * 30

        tH = 1/30 * np.ones(30)
        tItoH = 1/30 * np.ones(30)

        new_infections = model.new_infections(
            model.simulate(parameters))

        new_hospitalisations = model.new_hospitalisations(
            new_infections, pDtoH, dDtoH)

        new_icu = model.new_icu(new_infections, pDtoI, dDtoI)

        npt.assert_array_equal(
            model.new_hospital_beds(
                new_hospitalisations, new_icu, tH, tItoH),
            np.zeros((33, 2)))

        # Shorter simulation times
        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        pDtoH = [1, 1]
        dDtoH = [1] * 30

        pDtoI = [1, 1]
        dDtoI = [1] * 30

        tH = 1/30 * np.ones(30)
        tItoH = 1/30 * np.ones(30)

        new_infections = model.new_infections(
            model.simulate(parameters))

        new_hospitalisations = model.new_hospitalisations(
            new_infections, pDtoH, dDtoH)

        new_icu = model.new_icu(new_infections, pDtoI, dDtoI)

        npt.assert_array_equal(
            model.new_hospital_beds(
                new_hospitalisations, new_icu, tH, tItoH),
            np.array([[0, 0], [0, 0]]))

        with self.assertRaises(ValueError):
            new_hospitalisations1 = np.array([0, 0])

            model.check_new_hospital_beds_format(
                new_hospitalisations1, new_icu, tH, tItoH)

        with self.assertRaises(ValueError):
            new_hospitalisations1 = np.array([[0, 0], [0, 0], [0, 0]])

            model.check_new_hospital_beds_format(
                new_hospitalisations1, new_icu, tH, tItoH)

        with self.assertRaises(ValueError):
            new_hospitalisations1 = np.array([[0, 0, 0], [0, 0, 0]])

            model.check_new_hospital_beds_format(
                new_hospitalisations1, new_icu, tH, tItoH)

        with self.assertRaises(TypeError):
            new_hospitalisations1 = np.array([[0, 0], ['0', 0]])

            model.check_new_hospital_beds_format(
                new_hospitalisations1, new_icu, tH, tItoH)

        with self.assertRaises(ValueError):
            new_icu1 = np.array([0, 0])

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu1, tH, tItoH)

        with self.assertRaises(ValueError):
            new_icu1 = np.array([[0, 0], [0, 0], [0, 0]])

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu1, tH, tItoH)

        with self.assertRaises(ValueError):
            new_icu1 = np.array([[0, 0, 0], [0, 0, 0]])

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu1, tH, tItoH)

        with self.assertRaises(TypeError):
            new_icu1 = np.array([[0, 0], ['0', 0]])

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu1, tH, tItoH)

        with self.assertRaises(ValueError):
            tH1 = {'0.9': 0}

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu, tH1, tItoH)

        with self.assertRaises(ValueError):
            tH1 = [1, 1, 1]

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu, tH1, tItoH)

        with self.assertRaises(ValueError):
            tH1 = [1] * 30

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu, tH1, tItoH)

        with self.assertRaises(TypeError):
            tH1 = ['1'] * 30

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu, tH1, tItoH)

        with self.assertRaises(ValueError):
            tH1 = ([-0.02] + [0.02] * 51)

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu, tH1, tItoH)

        with self.assertRaises(ValueError):
            tH1 = [1.1, -0.1] + [0] * 50

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu, tH1, tItoH)

        with self.assertRaises(ValueError):
            tItoH1 = {'0.9': 0}

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu, tH, tItoH1)

        with self.assertRaises(ValueError):
            tItoH1 = [1, 1, 1]

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu, tH, tItoH1)

        with self.assertRaises(ValueError):
            tItoH1 = [1] * 30

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu, tH, tItoH1)

        with self.assertRaises(TypeError):
            tItoH1 = ['1'] * 30

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu, tH, tItoH1)

        with self.assertRaises(ValueError):
            tItoH1 = [-0.02] + [0.02] * 51

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu, tH, tItoH1)

        with self.assertRaises(ValueError):
            tItoH1 = [1.1, -0.1] + [0] * 50

            model.check_new_hospital_beds_format(
                new_hospitalisations, new_icu, tH, tItoH1)

    def test_new_icu_beds(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=np.arange(1, 34).tolist()
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        new_infections = model.new_infections(
            model.simulate(parameters))

        pDtoI = np.ones(len(age_groups))
        dDtoI = np.ones(30)

        new_icu = model.new_icu(new_infections, pDtoI, dDtoI)

        tI = 1/30 * np.ones(30)

        npt.assert_array_equal(
            model.new_icu_beds(
                new_icu, tI),
            np.zeros((33, 2)))

        # Shorter simulation times
        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        new_infections = model.new_infections(
            model.simulate(parameters))

        pDtoI = np.ones(len(age_groups))
        dDtoI = np.ones(30)

        tI = 1/30 * np.ones(30)

        new_icu = model.new_icu(new_infections, pDtoI, dDtoI)

        npt.assert_array_equal(
            model.new_icu_beds(
                new_icu, tI),
            np.array([[0, 0], [0, 0]]))

        with self.assertRaises(ValueError):
            new_icu1 = np.array([0, 0])

            model.check_new_icu_beds_format(new_icu1, tI)

        with self.assertRaises(ValueError):
            new_icu1 = np.array([[0, 0], [0, 0], [0, 0]])

            model.check_new_icu_beds_format(new_icu1, tI)

        with self.assertRaises(ValueError):
            new_icu1 = np.array([[0, 0, 0], [0, 0, 0]])

            model.check_new_icu_beds_format(new_icu1, tI)

        with self.assertRaises(TypeError):
            new_icu1 = np.array([[0, 0], ['0', 0]])

            model.check_new_icu_beds_format(new_icu1, tI)

        with self.assertRaises(ValueError):
            tI1 = {'0.9': 0}

            model.check_new_icu_beds_format(new_icu, tI1)

        with self.assertRaises(ValueError):
            tI1 = [1, 1, 1]

            model.check_new_icu_beds_format(new_icu, tI1)

        with self.assertRaises(ValueError):
            tI1 = [1] * 30

            model.check_new_icu_beds_format(new_icu, tI1)

        with self.assertRaises(TypeError):
            tI1 = ['1'] * 30

            model.check_new_icu_beds_format(new_icu, tI1)

        with self.assertRaises(ValueError):
            tI1 = ([-0.02] + [0.02] * 51)

            model.check_new_icu_beds_format(new_icu, tI1)

        with self.assertRaises(ValueError):
            tI1 = [1.1, -0.1] + [0] * 50

            model.check_new_icu_beds_format(new_icu, tI1)

    def test_loglik_deaths(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=np.arange(1, 11).tolist(),
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        pDtoH = [1, 1]
        dDtoH = [1] * 30

        pHtoDeath = [1, 1]
        dHtoDeath = [1] * 30

        new_infections = model.new_infections(
            model.simulate(parameters))

        new_hospitalisations = model.new_hospitalisations(
            new_infections, pDtoH, dDtoH)

        new_deaths = model.new_deaths(
            new_hospitalisations, pHtoDeath, dHtoDeath)

        obs_death = [10] * model._num_ages

        self.assertEqual(
            model.loglik_deaths(
                obs_death, new_deaths, 10**(-5), 9).shape,
            (len(age_groups),))

        with self.assertRaises(ValueError):
            model.loglik_deaths(
                obs_death, new_deaths, 10**(-5), -1)

        with self.assertRaises(TypeError):
            model.loglik_deaths(
                obs_death, new_deaths, 10**(-5), '1')

        with self.assertRaises(ValueError):
            model.loglik_deaths(
                obs_death, new_deaths, 10**(-5), 12)

        with self.assertRaises(ValueError):
            model.loglik_deaths(
                0, new_deaths, 10**(-5), 1)

        with self.assertRaises(ValueError):
            obs_death1 = np.array([5, 6, 0, 0])

            model.loglik_deaths(
                obs_death1, new_deaths, 10**(-5), 1)

        with self.assertRaises(TypeError):
            obs_death1 = np.array(['5', 6])

            model.loglik_deaths(
                obs_death1, new_deaths, 10**(-5), 1)

        with self.assertRaises(ValueError):
            obs_death1 = np.array([5, -1])

            model.loglik_deaths(
                obs_death1, new_deaths, 10**(-5), 1)

    def test_check_death_format(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        pDtoH = [1, 1]
        dDtoH = [1] * 30

        pHtoDeath = [1, 1]
        dHtoDeath = [1] * 30

        new_infections = model.new_infections(
            model.simulate(parameters))

        new_hospitalisations = model.new_hospitalisations(
            new_infections, pDtoH, dDtoH)

        new_deaths = model.new_deaths(
            new_hospitalisations, pHtoDeath, dHtoDeath)

        with self.assertRaises(TypeError):
            model.check_death_format(new_deaths, '10**(-5)')

        with self.assertRaises(ValueError):
            model.check_death_format(new_deaths, -2)

        with self.assertRaises(ValueError):
            new_deaths1 = np.array([5, 6])

            model.check_death_format(new_deaths1, 10**(-5))

        with self.assertRaises(ValueError):
            new_deaths1 = np.array([
                [5, 6, 0, 0],
                [5, 6, 0, 0]])

            model.check_death_format(new_deaths1, 10**(-5))

        with self.assertRaises(ValueError):
            new_deaths1 = np.array([
                [5, 6], [5, 6], [5, 6]])

            model.check_death_format(new_deaths1, 10**(-5))

        with self.assertRaises(TypeError):
            new_deaths1 = np.array([
                ['5', 6],
                [5, '0']])

            model.check_death_format(new_deaths1, 10**(-5))

    def test_samples_deaths(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[150, 60], [70, 40]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[10, 10], [10, 10]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=np.arange(1, 61).tolist(),
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        pDtoH = [1, 1]
        dDtoH = [1] * 30

        pHtoDeath = [1, 1]
        dHtoDeath = [1] * 30

        new_infections = model.new_infections(
            model.simulate(parameters))

        new_hospitalisations = model.new_hospitalisations(
            new_infections, pDtoH, dDtoH)

        new_deaths = model.new_deaths(
            new_hospitalisations, pHtoDeath, dHtoDeath)

        self.assertEqual(
            model.samples_deaths(new_deaths, 10**(-5), 41).shape,
            (len(age_groups),))

        self.assertEqual(
            model.samples_deaths(new_deaths, 10**(-5), 0).shape,
            (len(age_groups),))

        self.assertEqual(
            model.samples_deaths(new_deaths, 10**(-5), 1).shape,
            (len(age_groups),))

        with self.assertRaises(ValueError):
            model.samples_deaths(new_deaths, 10**(-5), -1)

        with self.assertRaises(TypeError):
            model.samples_deaths(new_deaths, 10**(-5), '1')

        with self.assertRaises(ValueError):
            model.samples_deaths(new_deaths, 10**(-5), 62)

        parameters.ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        pDtoH = [1, 1]
        dDtoH = [1] * 30

        pHtoDeath = [1, 1]
        dHtoDeath = [1] * 30

        new_infections = model.new_infections(
            model.simulate(parameters))

        new_hospitalisations = model.new_hospitalisations(
            new_infections, pDtoH, dDtoH)

        new_deaths = model.new_deaths(
            new_hospitalisations, pHtoDeath, dHtoDeath)

        self.assertEqual(
            model.samples_deaths(new_deaths, 10**(-5), 41).shape,
            (len(age_groups),))

        npt.assert_array_equal(
            model.samples_deaths(new_deaths, 10**(-5), 41),
            np.zeros(len(age_groups)))

        self.assertEqual(
            model.samples_deaths(new_deaths, 10**(-5), 1).shape,
            (len(age_groups),))

        npt.assert_array_equal(
            model.samples_deaths(new_deaths, 10**(-5), 1),
            np.zeros(len(age_groups)))

        self.assertEqual(
            model.samples_deaths(new_deaths, 10**(-5), 0).shape,
            (len(age_groups),))

        npt.assert_array_equal(
            model.samples_deaths(new_deaths, 10**(-5), 0),
            np.zeros(len(age_groups)))

    def test_loglik_positive_tests(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
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
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        output = model.simulate(parameters)

        tests = [[20, 30], [10, 0]]
        sens = 0.9
        spec = 0.1

        with self.assertRaises(ValueError):
            output1 = [5, 6]
            output1.extend([0] * 44)
            output1 = np.array(output1)

            model.check_positives_format(
                output1, tests, sens, spec)

        with self.assertRaises(ValueError):
            output1 = [5, 6]
            output1.extend([0] * 42)
            output1 = np.array([output1, output1])

            model.check_positives_format(
                output1, tests, sens, spec)

        with self.assertRaises(ValueError):
            output1 = [5, 6]
            output1.extend([0] * 44)
            output1 = np.array([output1, output1, output1])

            model.check_positives_format(
                output1, tests, sens, spec)

        with self.assertRaises(TypeError):
            output1 = ['5', 6]
            output1.extend([0] * 44)
            output2 = [5, 6, '0']
            output2.extend([0] * 43)
            output1 = np.array([output1, output2])

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
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
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

    def test_compute_transition_matrix(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        model.simulate(parameters)
        model.compute_transition_matrix()

        npt.assert_almost_equal(
            model._inv_trans_matrix[0],
            np.linalg.inv(np.block(
                [[-1.5 * np.identity(2), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2))],
                 [1.5 * np.identity(2), -1.5 * np.identity(2),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
                 [np.zeros((2, 2)), 1.5 * np.identity(2),
                  -1.5 * np.identity(2), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  -1.5 * np.identity(2), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  1.5 * np.identity(2), -1.5 * np.identity(2),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), 1.5 * np.identity(2),
                  -1.5 * np.identity(2), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  -1.5 * np.identity(2), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  1.5 * np.identity(2), -1.5 * np.identity(2),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), 1.5 * np.identity(2),
                  -1.5 * np.identity(2), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  -1.5 * np.identity(2), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  1.5 * np.identity(2), -1.5 * np.identity(2),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), 1.5 * np.identity(2),
                  -1.5 * np.identity(2), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)),
                  1.5*np.diag(0.4*np.ones(2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), -np.identity(2),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)),
                  1.5 * np.diag(0.6*np.ones(2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  -np.identity(2), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), -np.identity(2),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)),
                  1.5 * np.diag(0.4*np.ones(2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), -np.identity(2),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)),
                  1.5 * np.diag(0.6*np.ones(2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), 1.5 * np.diag(0.6*np.ones(2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), -np.identity(2), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)),
                  1.5*np.diag(0.4*np.ones(2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), -np.identity(2), np.zeros((2, 2)),
                  np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)),
                  1.5 * np.diag(0.4*np.ones(2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), -np.identity(2),
                  np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)),
                  1.5 * np.diag(0.6*np.ones(2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                  -np.identity(2)]]))
        )

    def test_compute_rt_trajectory(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        output = model.simulate(parameters)
        model.compute_transition_matrix()

        self.assertGreater(
            model.compute_rt_trajectory(output, 1),
            0)

    def test_compute_rt_trajectory_paper(self):
        model = em.WarwickSEIRModel()

        # Populate the model
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
        work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

        other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
        other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

        house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2],
                                                        [0.29, 4.6]])

        work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
        work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
        work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
        work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

        other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
        other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
        other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
        other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2],
                                                       [0.29, 4.6]])

        house_contacts_0 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_0)
        house_contacts_1 = em.ContactMatrix(
            age_groups, house_contact_data_matrix_1)
        house_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_0_0)
        house_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_0_1)
        house_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, house_region_data_matrix_1_0)
        house_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, house_region_data_matrix_1_1)

        school_contacts_0 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_0)
        school_contacts_1 = em.ContactMatrix(
            age_groups, school_contact_data_matrix_1)
        school_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_0_0)
        school_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_0_1)
        school_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, school_region_data_matrix_1_0)
        school_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, school_region_data_matrix_1_1)

        work_contacts_0 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_0)
        work_contacts_1 = em.ContactMatrix(
            age_groups, work_contact_data_matrix_1)
        work_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_0_0)
        work_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_0_1)
        work_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, work_region_data_matrix_1_0)
        work_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, work_region_data_matrix_1_1)

        other_contacts_0 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_0)
        other_contacts_1 = em.ContactMatrix(
            age_groups, other_contact_data_matrix_1)
        other_regional_0_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_0_0)
        other_regional_0_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_0_1)
        other_regional_1_0 = em.RegionMatrix(
            regions[0], age_groups, other_region_data_matrix_1_0)
        other_regional_1_1 = em.RegionMatrix(
            regions[1], age_groups, other_region_data_matrix_1_1)

        # Matrices contact
        house_matrices_contact = [house_contacts_0, house_contacts_1]
        school_matrices_contact = [school_contacts_0, school_contacts_1]
        work_matrices_contact = [work_contacts_0, work_contacts_1]
        other_matrices_contact = [other_contacts_0, other_contacts_1]
        time_changes_contact = [1, 2]

        house_matrices_region = [
            [house_regional_0_0, house_regional_0_1],
            [house_regional_1_0, house_regional_1_1]]
        school_matrices_region = [
            [school_regional_0_0, school_regional_0_1],
            [school_regional_1_0, school_regional_1_1]]
        work_matrices_region = [
            [work_regional_0_0, work_regional_0_1],
            [work_regional_1_0, work_regional_1_1]]
        other_matrices_region = [
            [other_regional_0_0, other_regional_0_1],
            [other_regional_1_0, other_regional_1_1]]
        time_changes_region = [1, 14]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(
            house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact)
        model.read_regional_data(
            house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=2,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[15, 6], [7, 4]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 0], [0, 0]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[0, 0], [0, 0]],
            detected_su_IC=[[0, 0], [0, 0]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[0, 0], [0, 0]],
            undetected_s_IC=[[0, 0], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        model.simulate(parameters)

        self.assertGreater(
            model.compute_rt_trajectory_paper(1),
            0)
