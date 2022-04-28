#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.testing as pdt

import epimodels as em

#
# Test ContactMatrix Class
#


class TestContactMatrixClass(unittest.TestCase):
    """
    Test the 'ContactMatrix' class.
    """
    def test__init__(self):
        age_groups = ['0-10', '10-25']
        data_matrix = np.array([[10, 5.2], [0, 3]])
        c = em.ContactMatrix(age_groups, data_matrix)

        self.assertEqual(c._num_a_groups, 2)
        npt.assert_array_equal(c.ages, np.array(['0-10', '10-25']))
        pdt.assert_frame_equal(
            c.contact_matrix,
            pd.DataFrame(
                data=np.array([[10, 5.2], [0, 3]]),
                index=['0-10', '10-25'],
                columns=['0-10', '10-25']))

        with self.assertRaises(ValueError):
            em.ContactMatrix('0', data_matrix)

        with self.assertRaises(TypeError):
            em.ContactMatrix([0, '1'], data_matrix)

        with self.assertRaises(TypeError):
            em.ContactMatrix(['0', 1], data_matrix)

        with self.assertRaises(ValueError):
            em.ContactMatrix(age_groups, [1])

        with self.assertRaises(ValueError):
            em.ContactMatrix(age_groups, np.array([[10, 5, 0], [0, 0, 3]]))

        with self.assertRaises(ValueError):
            em.ContactMatrix(age_groups, np.array([[10, 5], [0, 3], [0, 0]]))

        with self.assertRaises(TypeError):
            em.ContactMatrix(age_groups, np.array([[10, 5], [0, '3']]))

    def test_get_age_groups(self):
        age_groups = ['0-10', '10-25']
        data_matrix = np.array([[10, 5.2], [0, 3]])
        c = em.ContactMatrix(age_groups, data_matrix)

        self.assertEqual(
            c.get_age_groups(),
            "Polpulation is split into 2 age groups: ['0-10', '10-25'].")

    def test_change_age_groups(self):
        age_groups = ['0-10', '10-25']
        data_matrix = np.array([[10, 5.2], [0, 3]])
        c = em.ContactMatrix(age_groups, data_matrix)

        new_age_groups = ['0-15', '15-25']
        c.change_age_groups(new_age_groups)

        self.assertEqual(c._num_a_groups, 2)
        npt.assert_array_equal(c.ages, np.array(['0-15', '15-25']))
        pdt.assert_frame_equal(
            c.contact_matrix,
            pd.DataFrame(
                data=np.array([[10, 5.2], [0, 3]]),
                index=['0-15', '15-25'],
                columns=['0-15', '15-25']))

        with self.assertRaises(ValueError):
            c.change_age_groups(['0-15', '15-25', '25+'])

    def test_plot_heat_map(self):
        with patch('plotly.graph_objs.Figure.show') as show_patch:
            age_groups = ['0-10', '10-25']
            data_matrix = np.array([[10, 5.2], [0, 3]])
            c = em.ContactMatrix(age_groups, data_matrix)

            c.plot_heat_map()

        # Assert show_figure is called once
        assert show_patch.called

#
# Test RegionMatrix Class
#


class TestRegionMatrixClass(unittest.TestCase):
    """
    Test the 'RegionMatrix' class.
    """
    def test__init__(self):
        region_name = 'London'
        age_groups = ['0-10', '10-25']
        data_matrix = np.array([[10, 5.2], [0, 3]])
        r = em.RegionMatrix(region_name, age_groups, data_matrix)

        self.assertEqual(r.region, 'London')
        self.assertEqual(r.num_a_groups, 2)
        npt.assert_array_equal(r.ages, np.array(['0-10', '10-25']))
        pdt.assert_frame_equal(
            r.region_matrix,
            pd.DataFrame(
                data=np.array([[10, 5.2], [0, 3]]),
                index=['0-10', '10-25'],
                columns=['0-10', '10-25']))

        with self.assertRaises(ValueError):
            em.RegionMatrix(region_name, '0', data_matrix)

        with self.assertRaises(TypeError):
            em.RegionMatrix(region_name, [0, '1'], data_matrix)

        with self.assertRaises(TypeError):
            em.RegionMatrix(region_name, ['0', 1], data_matrix)

        with self.assertRaises(ValueError):
            em.RegionMatrix(region_name, age_groups, [1])

        with self.assertRaises(ValueError):
            em.RegionMatrix(
                region_name, age_groups, np.array([[10, 5, 0], [0, 0, 3]]))

        with self.assertRaises(ValueError):
            em.RegionMatrix(
                region_name, age_groups, np.array([[10, 5], [0, 3], [0, 0]]))

        with self.assertRaises(TypeError):
            em.RegionMatrix(
                region_name, age_groups, np.array([[10, 5], [0, '3']]))

        with self.assertRaises(TypeError):
            em.RegionMatrix([0], age_groups, data_matrix)

    def test_change_region_name(self):
        region_name = 'London'
        age_groups = ['0-10', '10-25']
        data_matrix = np.array([[10, 5.2], [0, 3]])
        r = em.RegionMatrix(region_name, age_groups, data_matrix)

        new_region_name = 'Oxford'
        r.change_region_name(new_region_name)

        self.assertEqual(r.region, 'Oxford')

        with self.assertRaises(TypeError):
            r.change_region_name(0)

    def test_change_age_groups(self):
        region_name = 'London'
        age_groups = ['0-10', '10-25']
        data_matrix = np.array([[10, 5.2], [0, 3]])
        r = em.RegionMatrix(region_name, age_groups, data_matrix)

        new_age_groups = ['0-15', '15-25']
        r.change_age_groups(new_age_groups)

        self.assertEqual(r.num_a_groups, 2)
        npt.assert_array_equal(r.ages, np.asarray(['0-15', '15-25']))
        pdt.assert_frame_equal(
            r.region_matrix,
            pd.DataFrame(
                data=np.array([[10, 5.2], [0, 3]]),
                index=['0-15', '15-25'],
                columns=['0-15', '15-25']))

        with self.assertRaises(ValueError):
            r.change_age_groups(['0-15', '15-25', '25+'])

    def test_plot_heat_map(self):
        with patch('plotly.graph_objs.Figure.show') as show_patch:
            region_name = 'London'
            age_groups = ['0-10', '10-25']
            data_matrix = np.array([[10, 5.2], [0, 3]])
            r = em.RegionMatrix(region_name, age_groups, data_matrix)

            r.plot_heat_map()

        # Assert show_figure is called once
        assert show_patch.called

#
# Test UniNextGenMatrix Class
#


class TestUniNextGenMatrixClass(unittest.TestCase):
    """
    Test the 'UniNextGenMatrix' class.
    """
    def test__init__(self):
        region_name = 'London'
        age_groups = ['0-10', '10-25']
        contact_data_matrix = np.array([[10, 5.2], [0, 3]])
        region_data_matrix = np.array([[0.5, 1.2], [0.29, 6]])
        pop_size = [18, 2]
        dI = 4

        contacts = em.ContactMatrix(age_groups, contact_data_matrix)
        regional = em.RegionMatrix(region_name, age_groups, region_data_matrix)
        next_gen = em.UniNextGenMatrix(pop_size, contacts, regional, dI)

        self.assertEqual(next_gen.region, 'London')
        npt.assert_array_equal(next_gen.ages, np.array(['0-10', '10-25']))
        npt.assert_array_equal(next_gen.susceptibles, np.array([18, 2]))
        npt.assert_array_equal(next_gen.contacts, contact_data_matrix)
        npt.assert_array_equal(next_gen.regional_suscep, region_data_matrix)
        self.assertEqual(next_gen.infection_period, 4)

        pdt.assert_frame_equal(
            next_gen.get_next_gen_matrix(),
            pd.DataFrame(
                data=np.array([[360, 449.28], [0, 144]]),
                index=['0-10', '10-25'],
                columns=['0-10', '10-25']))

        with self.assertRaises(TypeError):
            em.UniNextGenMatrix(pop_size, 0, regional, dI)

        with self.assertRaises(TypeError):
            em.UniNextGenMatrix(pop_size, contacts, 0, dI)

        with self.assertRaises(ValueError):
            new_age_groups = ['0-15', '15-25']
            regional1 = em.RegionMatrix(
                region_name, new_age_groups, region_data_matrix)
            em.UniNextGenMatrix(pop_size, contacts, regional1, dI)

        with self.assertRaises(TypeError):
            em.UniNextGenMatrix(pop_size, contacts, regional, '4')

        with self.assertRaises(ValueError):
            em.UniNextGenMatrix(pop_size, contacts, regional, 0)

        with self.assertRaises(ValueError):
            em.UniNextGenMatrix([[1], [2]], contacts, regional, dI)

        with self.assertRaises(ValueError):
            em.UniNextGenMatrix([0, 1, 1], contacts, regional, dI)

        with self.assertRaises(ValueError):
            em.UniNextGenMatrix([0, -1], contacts, regional, dI)

        with self.assertRaises(TypeError):
            em.UniNextGenMatrix([0, '1'], contacts, regional, dI)

    def test_compute_dom_eigenvalue(self):
        region_name = 'London'
        age_groups = ['0-10', '10-25']
        contact_data_matrix = np.array([[10, 0], [0, 3]])
        region_data_matrix = np.array([[0.5, 0], [0, 6]])
        pop_size = [1, 2]
        dI = 4

        contacts = em.ContactMatrix(age_groups, contact_data_matrix)
        regional = em.RegionMatrix(region_name, age_groups, region_data_matrix)
        next_gen = em.UniNextGenMatrix(pop_size, contacts, regional, dI)

        self.assertEqual(next_gen.compute_dom_eigenvalue(), 144)

#
# Test UniInfectivityMatrix Class
#


class TestMultiTimesContacts(unittest.TestCase):
    """
    Test the 'MultiTimesContacts' class.
    """
    def test__init__(self):
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

        multi_time_contacts = em.MultiTimesContacts(
            matrices_contact, time_changes_contact, regions,
            matrices_region, time_changes_region)

        npt.assert_almost_equal(
            multi_time_contacts.times_contact, np.array([1, 14]))
        npt.assert_almost_equal(
            multi_time_contacts.times_region, np.array([1, 14]))
        self.assertEqual(
            multi_time_contacts.contact_matrices, matrices_contact)
        self.assertEqual(
            multi_time_contacts.region_matrices, matrices_region)

        with self.assertRaises(ValueError):
            matrices_contact1 = [[contacts_0], [contacts_1]]

            em.MultiTimesContacts(
                matrices_contact1, time_changes_contact, regions,
                matrices_region, time_changes_region)

        with self.assertRaises(TypeError):
            matrices_contact1 = [contacts_0, '0']

            em.MultiTimesContacts(
                matrices_contact1, time_changes_contact, regions,
                matrices_region, time_changes_region)

        with self.assertRaises(ValueError):
            time_changes_contact1 = [[1], [14]]

            em.MultiTimesContacts(
                matrices_contact, time_changes_contact1, regions,
                matrices_region, time_changes_region)

        with self.assertRaises(ValueError):
            time_changes_contact1 = [1, 14, 26]

            em.MultiTimesContacts(
                matrices_contact, time_changes_contact1, regions,
                matrices_region, time_changes_region)

        with self.assertRaises(TypeError):
            time_changes_contact1 = [1.0, 14]

            em.MultiTimesContacts(
                matrices_contact, time_changes_contact1, regions,
                matrices_region, time_changes_region)

        with self.assertRaises(ValueError):
            time_changes_contact1 = [-1, 14]

            em.MultiTimesContacts(
                matrices_contact, time_changes_contact1, regions,
                matrices_region, time_changes_region)

        with self.assertRaises(ValueError):
            regions1 = [['London'], ['Cornwall']]

            em.MultiTimesContacts(
                matrices_contact, time_changes_contact, regions1,
                matrices_region, time_changes_region)

        with self.assertRaises(TypeError):
            regions1 = ['London', 0]

            em.MultiTimesContacts(
                matrices_contact, time_changes_contact, regions1,
                matrices_region, time_changes_region)

        with self.assertRaises(ValueError):
            matrices_region1 = [
                regional_0_0, regional_0_1,
                regional_1_0, regional_1_1]

            em.MultiTimesContacts(
                matrices_contact, time_changes_contact, regions,
                matrices_region1, time_changes_region)

        with self.assertRaises(ValueError):
            matrices_region1 = [
                [regional_0_0],
                [regional_1_0]]

            em.MultiTimesContacts(
                matrices_contact, time_changes_contact, regions,
                matrices_region1, time_changes_region)

        with self.assertRaises(TypeError):
            matrices_region1 = [
                [regional_0_0, 'regional_0_1'],
                [regional_1_0, regional_1_1]]

            em.MultiTimesContacts(
                matrices_contact, time_changes_contact, regions,
                matrices_region1, time_changes_region)

        with self.assertRaises(ValueError) as test_excep:
            matrices_region1 = [
                [regional_0_0, regional_0_0],
                [regional_1_0, regional_1_1]]

            em.MultiTimesContacts(
                matrices_contact, time_changes_contact, regions,
                matrices_region1, time_changes_region)

        self.assertTrue(
            'Incorrect region name' in str(test_excep.exception))

        with self.assertRaises(ValueError):
            time_changes_region1 = [[1], [14]]

            em.MultiTimesContacts(
                matrices_contact, time_changes_contact, regions,
                matrices_region, time_changes_region1)

        with self.assertRaises(ValueError):
            time_changes_region1 = [1, 14, 26]

            em.MultiTimesContacts(
                matrices_contact, time_changes_contact, regions,
                matrices_region, time_changes_region1)

        with self.assertRaises(TypeError):
            time_changes_region1 = [1, 14.0]

            em.MultiTimesContacts(
                matrices_contact, time_changes_contact, regions,
                matrices_region, time_changes_region1)

        with self.assertRaises(ValueError):
            time_changes_region1 = [-1, 14]

            em.MultiTimesContacts(
                matrices_contact, time_changes_contact, regions,
                matrices_region, time_changes_region1)

    def test_identify_current_contacts(self):
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

        multi_time_contacts = em.MultiTimesContacts(
            matrices_contact, time_changes_contact, regions,
            matrices_region, time_changes_region)

        npt.assert_almost_equal(
            multi_time_contacts.identify_current_contacts(2, 15),
            np.array([[0.5, 0], [0, 13.8]]))

#
# Test UniInfectivityMatrix Class
#


class TestUniInfectivityMatrixClass(unittest.TestCase):
    """
    Test the 'UniInfectivityMatrix' class.
    """
    def test__init__(self):
        region_name = 'London'
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        contact_data_matrix_0 = np.array([[10, 0], [0, 3]])
        region_data_matrix_0 = np.array([[0.5, 0], [0, 6]])
        init_pop_size = [1, 2]
        dI = 4

        contacts_0 = em.ContactMatrix(age_groups, contact_data_matrix_0)
        regional_0 = em.RegionMatrix(
            region_name, age_groups, region_data_matrix_0)
        next_gen_0 = em.UniNextGenMatrix(
            init_pop_size, contacts_0, regional_0, dI)

        initial_r = 0.5
        infect = em.UniInfectivityMatrix(
            initial_r,
            initial_nextgen_matrix=next_gen_0)

        self.assertEqual(infect.r0, 0.5)
        self.assertEqual(infect.r0_star, 144)

        with self.assertRaises(TypeError):
            em.UniInfectivityMatrix(
                '0',
                initial_nextgen_matrix=next_gen_0)

        with self.assertRaises(TypeError):
            em.UniInfectivityMatrix(
                initial_r,
                initial_nextgen_matrix=0)

    def test_compute_prob_infectivity_matrix(self):
        region_name = 'London'
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        contact_data_matrix_0 = np.array([[10, 0], [0, 3]])
        region_data_matrix_0 = np.array([[0.5, 0], [0, 6]])
        init_pop_size = [1, 2]
        dI = 4

        contacts_0 = em.ContactMatrix(age_groups, contact_data_matrix_0)
        regional_0 = em.RegionMatrix(
            region_name, age_groups, region_data_matrix_0)
        next_gen_0 = em.UniNextGenMatrix(
            init_pop_size, contacts_0, regional_0, dI)

        # Later time state of the system
        contact_data_matrix_1 = np.array([[10, 5.2], [0, 3]])
        region_data_matrix_1 = np.array([[0.5, 1.2], [0.29, 6]])
        current_pop_size = [18, 2]

        contacts_1 = em.ContactMatrix(age_groups, contact_data_matrix_1)
        regional_1 = em.RegionMatrix(
            region_name, age_groups, region_data_matrix_1)
        next_gen_1 = em.UniNextGenMatrix(
            current_pop_size, contacts_1, regional_1, dI)

        initial_r = 0.5
        temp_variation = 1
        infect = em.UniInfectivityMatrix(
            initial_r,
            initial_nextgen_matrix=next_gen_0)

        npt.assert_array_equal(
            infect.compute_prob_infectivity_matrix(temp_variation, next_gen_1),
            np.array([[5/288, 13/600], [0, 1/16]]))

        with self.assertRaises(TypeError):
            infect.compute_prob_infectivity_matrix('1', next_gen_1)

        with self.assertRaises(TypeError):
            infect.compute_prob_infectivity_matrix(temp_variation, 0)

    def test_compute_reproduction_number(self):
        region_name = 'London'
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        contact_data_matrix_0 = np.array([[10, 0], [0, 3]])
        region_data_matrix_0 = np.array([[0.5, 0], [0, 6]])
        init_pop_size = [1, 2]
        dI = 4

        contacts_0 = em.ContactMatrix(age_groups, contact_data_matrix_0)
        regional_0 = em.RegionMatrix(
            region_name, age_groups, region_data_matrix_0)
        next_gen_0 = em.UniNextGenMatrix(
            init_pop_size, contacts_0, regional_0, dI)

        # Later time state of the system
        contact_data_matrix_1 = np.array([[10, 5.2], [0, 3]])
        region_data_matrix_1 = np.array([[0.5, 1.2], [0.29, 6]])
        current_pop_size = [18, 2]

        contacts_1 = em.ContactMatrix(age_groups, contact_data_matrix_1)
        regional_1 = em.RegionMatrix(
            region_name, age_groups, region_data_matrix_1)
        next_gen_1 = em.UniNextGenMatrix(
            current_pop_size, contacts_1, regional_1, dI)

        initial_r = 0.5
        temp_variation = 1
        infect = em.UniInfectivityMatrix(
            initial_r,
            initial_nextgen_matrix=next_gen_0)

        self.assertEqual(
            infect.compute_reproduction_number(
                temp_variation, next_gen_1), 5/4)

        with self.assertRaises(TypeError):
            infect.compute_reproduction_number('1', next_gen_1)

        with self.assertRaises(TypeError):
            infect.compute_reproduction_number(temp_variation, 0)

#
# Test MultiTimesInfectivity Class
#


class TestMultiTimesInfectivityClass(unittest.TestCase):
    """
    Test the 'MultiTimesInfectivity' class.
    """
    def test__init__(self):
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        contact_data_matrix_0 = np.array([[10, 0], [0, 3]])
        contact_data_matrix_1 = np.array([[10, 5.2], [0, 3]])

        region_data_matrix_0_0 = np.array([[0.5, 0], [0, 6]])
        region_data_matrix_0_1 = np.array([[1, 10], [1, 0]])
        region_data_matrix_1_0 = np.array([[0.5, 1.2], [0.29, 6]])
        region_data_matrix_1_1 = np.array([[0.85, 1], [0.9, 6]])

        susceptibles = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[0, 2], [1, 1]]]
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

        m = em.MultiTimesInfectivity(
            matrices_contact,
            time_changes_contact,
            regions,
            matrices_region,
            time_changes_region,
            initial_r,
            dI,
            susceptibles[0])

        self.assertEqual(m._regions, ['London', 'Cornwall'])
        npt.assert_array_equal(m.initial_r, np.array([0.5, 1]))
        self.assertEqual(m.dI, 4)
        npt.assert_array_equal(m.times_contact, np.array([1, 3]))
        npt.assert_array_equal(m.times_region, np.array([1, 2]))
        self.assertCountEqual(m.contact_matrices, matrices_contact)
        self.assertCountEqual(m.region_matrices, matrices_region)
        self.assertEqual(len(m.initial_infec_matrices), 2)

        with self.assertRaises(ValueError):
            em.MultiTimesInfectivity(
                0,
                time_changes_contact,
                regions,
                matrices_region,
                time_changes_region,
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(TypeError):
            em.MultiTimesInfectivity(
                [contacts_0, 0],
                time_changes_contact,
                regions,
                matrices_region,
                time_changes_region,
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(ValueError):
            em.MultiTimesInfectivity(
                matrices_contact,
                1,
                regions,
                matrices_region,
                time_changes_region,
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(ValueError):
            em.MultiTimesInfectivity(
                matrices_contact,
                [1],
                regions,
                matrices_region,
                time_changes_region,
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(TypeError):
            em.MultiTimesInfectivity(
                matrices_contact,
                [1, 1.5],
                regions,
                matrices_region,
                time_changes_region,
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(ValueError):
            em.MultiTimesInfectivity(
                matrices_contact,
                [0, 1],
                regions,
                matrices_region,
                time_changes_region,
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(ValueError):
            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                'London',
                matrices_region,
                time_changes_region,
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(TypeError):
            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                [0, 'London'],
                matrices_region,
                time_changes_region,
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(ValueError):
            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                [regional_1_0, regional_1_1],
                time_changes_region,
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(ValueError):
            matrices_region_1 = [[regional_0_0], [regional_1_0]]

            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region_1,
                time_changes_region,
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(TypeError):
            matrices_region_1 = [
                [regional_0_0, 1],
                [regional_1_0, regional_1_1]
            ]

            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region_1,
                time_changes_region,
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(ValueError):
            matrices_region_1 = [
                [regional_0_0, regional_0_1],
                [
                    regional_1_0,
                    em.RegionMatrix(
                        regions[0],
                        age_groups,
                        region_data_matrix_1_1)]
            ]

            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region_1,
                time_changes_region,
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(ValueError):
            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region,
                1,
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(ValueError):
            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region,
                [1, 2, 3],
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(TypeError):
            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region,
                [1, '2'],
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(ValueError):
            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region,
                [0, 2],
                initial_r,
                dI,
                susceptibles[0])

        with self.assertRaises(ValueError):
            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region,
                time_changes_region,
                0.5,
                dI,
                susceptibles[0])

        with self.assertRaises(ValueError):
            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region,
                time_changes_region,
                [0.5],
                dI,
                susceptibles[0])

        with self.assertRaises(TypeError):
            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region,
                time_changes_region,
                [0.5, '1'],
                dI,
                susceptibles[0])

        with self.assertRaises(ValueError):
            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region,
                time_changes_region,
                [0.5, 0],
                dI,
                susceptibles[0])

        with self.assertRaises(TypeError):
            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region,
                time_changes_region,
                initial_r,
                '4',
                susceptibles[0])

        with self.assertRaises(ValueError):
            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region,
                time_changes_region,
                initial_r,
                0,
                susceptibles[0])

        with self.assertRaises(ValueError):
            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region,
                time_changes_region,
                initial_r,
                dI,
                [1])

        with self.assertRaises(ValueError):
            susceptibles_1 = [[[1], [3]], [[5], [7]], [[0], [1]]]

            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region,
                time_changes_region,
                initial_r,
                dI,
                susceptibles_1[0])

        with self.assertRaises(ValueError):
            susceptibles_1 = [[[1, 2]], [[5, 6]], [[0, 2]]]

            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region,
                time_changes_region,
                initial_r,
                dI,
                susceptibles_1[0])

        with self.assertRaises(TypeError):
            susceptibles_1 = [
                [[1, '2'], [3, 4]], [[5, 6], [7, 8]], [[0, 2], [1, 1]]]

            em.MultiTimesInfectivity(
                matrices_contact,
                time_changes_contact,
                regions,
                matrices_region,
                time_changes_region,
                initial_r,
                dI,
                susceptibles_1[0])

    def test_compute_prob_infectivity_matrix(self):
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        contact_data_matrix_0 = np.array([[10, 0], [0, 3]])
        contact_data_matrix_1 = np.array([[10, 5.2], [0, 3]])

        region_data_matrix_0_0 = np.array([[0.5, 0], [0, 6]])
        region_data_matrix_0_1 = np.array([[1, 10], [1, 0]])
        region_data_matrix_1_0 = np.array([[0.5, 1.2], [0.29, 6]])
        region_data_matrix_1_1 = np.array([[0.85, 1], [0.9, 6]])

        susceptibles = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[0, 2], [1, 1]]]
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

        m = em.MultiTimesInfectivity(
            matrices_contact,
            time_changes_contact,
            regions,
            matrices_region,
            time_changes_region,
            initial_r,
            dI,
            susceptibles[0])

        npt.assert_array_equal(
            m.compute_prob_infectivity_matrix(1, 3, susceptibles[2][0], 1),
            np.array([[5/288, 13/600], [0, 1/16]]))

        with self.assertRaises(TypeError):
            m.compute_prob_infectivity_matrix('1', 3, susceptibles[2][0], 1)

        with self.assertRaises(ValueError):
            m.compute_prob_infectivity_matrix(3, 3, susceptibles[2][0], 1)

        with self.assertRaises(ValueError):
            m.compute_prob_infectivity_matrix(0, 3, susceptibles[2][0], 1)

        with self.assertRaises(TypeError):
            m.compute_prob_infectivity_matrix(1, '3', susceptibles[2][0], 1)

        with self.assertRaises(ValueError):
            m.compute_prob_infectivity_matrix(1, 0, susceptibles[2][0], 1)

        with self.assertRaises(TypeError):
            m.compute_prob_infectivity_matrix(1, 3, susceptibles[2][0], '1')

        with self.assertRaises(ValueError):
            m.compute_prob_infectivity_matrix(1, 3, [[5, 6], [7, 8]], 1)

        with self.assertRaises(ValueError):
            m.compute_prob_infectivity_matrix(1, 3, [5, 6, 0], 1)

        with self.assertRaises(TypeError):
            m.compute_prob_infectivity_matrix(1, 3, [5, '6'], 1)

        with self.assertRaises(ValueError):
            m.compute_prob_infectivity_matrix(1, 3, [5, -6], 1)

    def test_compute_reproduction_number(self):
        regions = ['London', 'Cornwall']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        contact_data_matrix_0 = np.array([[10, 0], [0, 3]])
        contact_data_matrix_1 = np.array([[10, 5.2], [0, 3]])

        region_data_matrix_0_0 = np.array([[0.5, 0], [0, 6]])
        region_data_matrix_0_1 = np.array([[1, 10], [1, 0]])
        region_data_matrix_1_0 = np.array([[0.5, 1.2], [0.29, 6]])
        region_data_matrix_1_1 = np.array([[0.85, 1], [0.9, 6]])

        susceptibles = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[0, 2], [1, 1]]]
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

        m = em.MultiTimesInfectivity(
            matrices_contact,
            time_changes_contact,
            regions,
            matrices_region,
            time_changes_region,
            initial_r,
            dI,
            susceptibles[0])

        self.assertEqual(m.compute_reproduction_number(
            1, 3, susceptibles[2][0]), 0.5)

        with self.assertRaises(TypeError):
            m.compute_reproduction_number('1', 3, susceptibles[2][0], 1)

        with self.assertRaises(ValueError):
            m.compute_reproduction_number(3, 3, susceptibles[2][0], 1)

        with self.assertRaises(ValueError):
            m.compute_reproduction_number(0, 3, susceptibles[2][0], 1)

        with self.assertRaises(TypeError):
            m.compute_reproduction_number(1, '3', susceptibles[2][0], 1)

        with self.assertRaises(ValueError):
            m.compute_reproduction_number(1, 0, susceptibles[2][0], 1)

        with self.assertRaises(TypeError):
            m.compute_reproduction_number(1, 3, susceptibles[2][0], '1')
