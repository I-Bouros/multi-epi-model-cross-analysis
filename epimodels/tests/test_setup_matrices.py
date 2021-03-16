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


class TestContactMatrixClass(unittest.TestCase):
    """
    Test the 'ContactMatrix' class.
    """
    def test__init__(self):
        age_groups = ['0-10', '10-25']
        data_matrix = np.array([[10, 5.2], [0, 3]])
        c = em.ContactMatrix(age_groups, data_matrix)

        self.assertEqual(c._num_a_groups, 2)
        npt.assert_array_equal(c.ages, np.asarray(['0-10', '10-25']))
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
        npt.assert_array_equal(c.ages, np.asarray(['0-15', '15-25']))
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
        npt.assert_array_equal(r.ages, np.asarray(['0-10', '10-25']))
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
        npt.assert_array_equal(next_gen.ages, np.asarray(['0-10', '10-25']))
        npt.assert_array_equal(next_gen.susceptibles, np.array([18, 2]))
        npt.assert_array_equal(next_gen.contacts, contact_data_matrix)
        npt.assert_array_equal(next_gen.regional_suscep, region_data_matrix)
        self.assertEqual(next_gen.infection_period, 4)

        pdt.assert_frame_equal(
            next_gen.next_gen_matrrix,
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
