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
import seaborn as sns


class ContactMatrix():
    """ContactMatrix Class:
    Base class for constructing a contact matrix to be used in the
    modelling of epidemics. These matrices indicate the the number
    of people a person in a given age group (i) will interact on average
    with people in a different age group (j) at a given time point (t_k).

    .. math::
        C^{t_{k}} = {C_{i,j}^{t_{k}}

    We allow for two temporal behaviours:

    * vary over time  - `moving`;
    * remain constant over time - `frozen`.

    Parameters
    ----------
    age_groups
        (list of strings) List of the different age intervals according
        to which the population is split when cosntructing the contact
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
        if np.asarray(data_matrix).ndim != 2:
            raise ValueError(
                'Contact matrix storage format must be 2-dimensional')
        if np.asarray(data_matrix).shape[0] != np.asarray(
                data_matrix).shape[1]:
            raise ValueError(
                'Contact matrix storage format must be a square matrix')
        if np.asarray(data_matrix).shape[0] != len(age_groups):
            raise ValueError(
                    'Wrong number of rows for the contact matrix')
        if np.asarray(data_matrix).shape[1] != len(age_groups):
            raise ValueError(
                    'Wrong number of columns for the contact matrix')
        for _ in np.nditer(np.asarray(data_matrix)):
            if not isinstance(_, [int, float]):
                raise TypeError(
                    'Contact matrix elements must be integer or float')

        self.ages = age_groups
        self.num_a_groups = len(age_groups)
        self._data = np.asarray(data_matrix)
        self._create_contact_matrix()

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

        self.ages = new_age_groups
        self.num_a_groups = len(new_age_groups)
        self._create_contact_matrix()

    def _create_contact_matrix(self):
        """
        Creates a pandas.Dataframe with both rows and columns named according
        to the age group structure of population
        """
        self.contact_matrix = pd.DataFrame(
            data=self._data, index=self.ages, columns=self.ages)

    def plot_heat_map(self):
        """
        Plots a heat map of the contact matrix.

        """
        contact_matrix = self.contact_matrix.pivot(
            "infectives age", "infected age", "num contacts")
        fig = sns.heatmap(contact_matrix)
        return(fig)
