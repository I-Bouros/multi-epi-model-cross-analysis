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


class TestContactMatrixlass(unittest.TestCase):
    """
    Test the 'ContactMatrix' class.
    """
    def test__init__(self):
        age_groups = ['0-10', '10-25']
        data_matrix = np.array([[10, 5.2], [0, 3]])
        c = em.ContactMatrix(age_groups, data_matrix)

        self.assertEqual(c.num_a_groups, 2)
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
            em.ContactMatrix(age_groups, np.array([[10, 5, 0], [0, 3]],
                             dtype=object))

        with self.assertRaises(ValueError):
            em.ContactMatrix(age_groups, np.array([[10, 5], [0, 3, 0]],
                             dtype=object))

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

        self.assertEqual(c.num_a_groups, 2)
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
