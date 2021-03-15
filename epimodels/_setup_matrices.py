#
# ContactMatrix Class
#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class ContactMatrix():
    """ContactMatrix Class:
    Base class for constructing a contact matrix to be used in the
    modelling of epidemics. These matrices indicate the number
    of people a person in a given age group (i) will interact on average
    with people in a different age group (j) at a given time point (t_k).

    .. math::
        C^{t_{k}} = {C_{i,j}^{t_{k}}

    Parameters
    ----------
    age_groups
        (list of strings) List of the different age intervals according
        to which the population is split when constructing the contact
        matrix.
    data_matrix
        (numpy.array) Data array which will populate the contact matrix.
        Element (i, j) reprsents the average number of people in age group j
        a person in age group i interact with.

    """
    def __init__(self, age_groups, data_matrix):
        # Chech age_groups have correct format
        self._check_age_groups_format(age_groups)

        # Chech data_matrix has correct format
        self._check_data_matrix_format(data_matrix, age_groups)

        self.ages = age_groups
        self.num_a_groups = len(age_groups)
        self._data = np.asarray(data_matrix)
        self.contact_matrix = self._create_contact_matrix()

    def _check_data_matrix_format(self, data_matrix, age_groups):
        """
        Checks correct format of the data matrix.

        """
        if np.asarray(data_matrix).ndim != 2:
            raise ValueError(
                'Contact matrix storage format must be 2-dimensional')
        if np.asarray(data_matrix).shape[0] != len(age_groups):
            raise ValueError(
                    'Wrong number of rows for the contact matrix')
        if np.asarray(data_matrix).shape[1] != len(age_groups):
            raise ValueError(
                    'Wrong number of columns for the contact matrix')
        for r in np.asarray(data_matrix):
            for _ in r:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'Contact matrix elements must be integer or float')

    def get_age_groups(self):
        """
        Returns the number of age groups and their names.

        """
        return('Polpulation is split into {} age groups: {}.'.format(
            self.num_a_groups, self.ages))

    def _check_age_groups_format(self, age_groups):
        """
        Checks correct format of the age groups.

        """
        if np.asarray(age_groups).ndim != 1:
            raise ValueError(
                'Age groups storage format must be 1-dimensional')

        for _ in range(len(age_groups)):
            if not isinstance(age_groups[_], str):
                raise TypeError(
                    'Age groups value format must be a string')

    def change_age_groups(self, new_age_groups):
        """
        Modifies current age structure of the contact matrix.

        Parameters
        ----------
        new_age_groups
            (list of strings) List of the new age intervals according
            to which the population is split when cosntructing the contact
            matrix.

        """
        # Chech new_age_groups have correct format
        self._check_age_groups_format(new_age_groups)

        if len(new_age_groups) != self.num_a_groups:
            raise ValueError(
                'Wrong number of age group passed for the given data.')

        self.ages = new_age_groups
        self.num_a_groups = len(new_age_groups)
        self.contact_matrix = self._create_contact_matrix()

    def _create_contact_matrix(self):
        """
        Creates a pandas.Dataframe with both rows and columns named according
        to the age group structure of population
        """
        return(pd.DataFrame(
            data=self._data, index=self.ages, columns=self.ages))

    def plot_heat_map(self):
        """
        Plots a heat map of the contact matrix.

        """
        self.figure = go.Figure(data=go.Heatmap(
            x=self.contact_matrix.columns.values,
            y=self.contact_matrix.index.values,
            z=self.contact_matrix
        ))
        self.figure.update_layout(
            xaxis_title="Infectives Age",
            yaxis_title="Infected Age")
        self.figure.show()

#
# RegionMatrix Class
#


class RegionMatrix(ContactMatrix):
    """RegionMatrix Class:
    Base class for constructing a region matrix to be used in the
    modelling of epidemics. These matrices indicate the region-specific
    relative susceptibility of someone in a given age group (i) will get
    infected from somebody else in a different age group (j) at a given
    time point (t_k), assuming contact.

    .. math::
        M_{r}^{t_{k}} = {M_{r,i,j}^{t_{k}}

    Parameters
    ----------
    region_name
        (str) Name of the region to which the region matrix refers to
    age_groups
        (list of strings) List of the different age intervals according
        to which the population is split when constructing the region
        matrix.
    data_matrix
        (numpy.array) Data array which will populate the region matrix.
        Element (i, j) reprsents the relative susceptibility of someone
        in age group j to be infected by a person in age group i, if they
        come into contact.

    """
    def __init__(self, region_name, age_groups, data_matrix):
        # Chech region_name have correct format
        self._check_region_name_format(region_name)

        # Chech age_groups have correct format
        self._check_age_groups_format(age_groups)

        # Chech data_matrix has correct format
        self._check_data_matrix_format(data_matrix, age_groups)

        self.region = region_name
        self.ages = age_groups
        self.num_a_groups = len(age_groups)
        self._data = np.asarray(data_matrix)
        self.region_matrix = self._create_region_matrix()

    def _check_region_name_format(self, region_name):
        """
        Checks correct format of the region name.

        """
        if not isinstance(region_name, str):
            raise TypeError(
                'Region name associated with the matrix must be a string')

    def _check_data_matrix_format(self, data_matrix, age_groups):
        """
        Checks correct format of the data matrix.

        """
        if np.asarray(data_matrix).ndim != 2:
            raise ValueError(
                'Region matrix storage format must be 2-dimensional')
        if np.asarray(data_matrix).shape[0] != len(age_groups):
            raise ValueError(
                    'Wrong number of rows for the region matrix')
        if np.asarray(data_matrix).shape[1] != len(age_groups):
            raise ValueError(
                    'Wrong number of columns for the region matrix')
        for r in np.asarray(data_matrix):
            for _ in r:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'Region matrix elements must be integer or float')

    def _create_region_matrix(self):
        """
        Creates a pandas.Dataframe with both rows and columns named according
        to the age group structure of population
        """
        return(self._create_contact_matrix())

    def change_region_name(self, new_region_name):
        """
        Modifies current region name of the region matrix.

        Parameters
        ----------
        new_region_name
            (string) New name of the region the region-specific
            matrix is referring to.

        """
        # Chech new_age_groups have correct format
        self._check_region_name_format(new_region_name)

        self.region = new_region_name

    def change_age_groups(self, new_age_groups):
        """
        Modifies current age structure of the contact matrix.

        Parameters
        ----------
        new_age_groups
            (list of strings) List of the new age intervals according
            to which the population is split when cosntructing the contact
            matrix.

        """
        # Chech new_age_groups have correct format
        self._check_age_groups_format(new_age_groups)

        if len(new_age_groups) != self.num_a_groups:
            raise ValueError(
                'Wrong number of age group passed for the given data.')

        self.ages = new_age_groups
        self.num_a_groups = len(new_age_groups)
        self.region_matrix = self._create_region_matrix()

    def plot_heat_map(self):
        """
        Plots a heat map of the contact matrix.

        """
        self.figure = go.Figure(data=go.Heatmap(
            x=self.region_matrix.columns.values,
            y=self.region_matrix.index.values,
            z=self.region_matrix
        ))
        self.figure.update_layout(
            title_text='{}'.format(self.region),
            xaxis_title="Infectives Age",
            yaxis_title="Infected Age")
        self.figure.show()
