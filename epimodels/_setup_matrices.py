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
    r"""ContactMatrix Class:
    Base class for constructing a contact matrix to be used in the
    modelling of epidemics. These matrices indicate the number
    of people a person in a given age group (i) will interact on average
    with people in a different age group (j) at a given time point (t_k).

    .. math::
        C^{t_k} = \{C_{ij}^{t_k}\}

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
        self._num_a_groups = len(age_groups)
        self._data = np.asarray(data_matrix)
        self.contact_matrix = self._create_contact_matrix()

    def _check_data_matrix_format(self, data_matrix, age_groups):
        """
        Checks correct format of the data matrix.

        """
        if np.asarray(data_matrix).ndim != 2:
            raise ValueError(
                'Contact matrix storage format must be 2-dimensional.')
        if np.asarray(data_matrix).shape[0] != len(age_groups):
            raise ValueError(
                    'Wrong number of rows for the contact matrix.')
        if np.asarray(data_matrix).shape[1] != len(age_groups):
            raise ValueError(
                    'Wrong number of columns for the contact matrix.')
        for r in np.asarray(data_matrix):
            for _ in r:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'Contact matrix elements must be integer or float.')

    def get_age_groups(self):
        """
        Returns the number of age groups and their names.

        """
        return('Polpulation is split into {} age groups: {}.'.format(
            self._num_a_groups, self.ages))

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

        if len(new_age_groups) != self._num_a_groups:
            raise ValueError(
                'Wrong number of age group passed for the given data.')

        self.ages = new_age_groups
        self._num_a_groups = len(new_age_groups)
        self.contact_matrix = self._create_contact_matrix()

    def _create_contact_matrix(self):
        """
        Creates a pandas.Dataframe with both rows and columns named according
        to the age group structure of population.

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
    r"""RegionMatrix Class:
    Base class for constructing a region matrix to be used in the
    modelling of epidemics. These matrices indicate the region-specific
    relative susceptibility of someone in a given age group (i) will get
    infected from somebody else in a different age group (j) at a given
    time point (.. math:: `t_k`), assuming contact.

    .. math::
        M_{r}^{t_k} = \{M_{r, ij}^{t_k}\}

    Parameters
    ----------
    region_name
        (str) Name of the region to which the region matrix refers to.
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
                'Region name associated with the matrix must be a string.')

    def _check_data_matrix_format(self, data_matrix, age_groups):
        """
        Checks correct format of the data matrix.

        """
        if np.asarray(data_matrix).ndim != 2:
            raise ValueError(
                'Region matrix storage format must be 2-dimensional.')
        if np.asarray(data_matrix).shape[0] != len(age_groups):
            raise ValueError(
                    'Wrong number of rows for the region matrix.')
        if np.asarray(data_matrix).shape[1] != len(age_groups):
            raise ValueError(
                    'Wrong number of columns for the region matrix.')
        for r in np.asarray(data_matrix):
            for _ in r:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'Region matrix elements must be integer or float.')

    def _create_region_matrix(self):
        """
        Creates a pandas.Dataframe with both rows and columns named according
        to the age group structure of population.

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

#
# UniNextGenMatrix Class
#


class UniNextGenMatrix(object):
    r"""UniNextGenMatrix
    Class for generator matrices which are then used to determine
    the evolution of number of infectives as time goes on according
    to the following formulae - at fixed time .. math::`t_k` and
    in specific region r:

    .. math::
       :nowrap:

        \begin{eqnarray}
            \Lambda_{k, r} & = & (\Lambda_{k, r, ij}) \\
            \widetilde{C}_{r, ij}^{t_k} & = & C_{ij}^{t_k} M_{r, ij}^{t_k} \\
            \Lambda_{k, r, ij} & = & S_{r, t_k, i} \widetilde{C}_{r, ij}^{t_k}
                 d_{I}
        \end{eqnarray}

    Parameters
    ----------
    pop_size
        (list) List of number of susceptible in the population, split according
        to their corresponding age group.
    contact_matrix
        (ContactMatrix) Array which encodes the expected number of contacts in
        different age groups a person can have, dependent of which age group
        they falls into.
    region_matrix
        (RegionMatrix) Array which encodes the relative suceptibility to
        infection a person can have, depending of which age group they falls
        into, if they come into contact with people from various age groups.
    dI
        (float) Average duration of infection.

    """
    def __init__(self, pop_size, contact_matrix, region_matrix, dI):
        # Check correct format of contact and region matrices
        if not isinstance(contact_matrix, ContactMatrix):
            raise TypeError('Incorrect format for the contact matrix; \
                must be ContactMatrix.')
        if not isinstance(region_matrix, RegionMatrix):
            raise TypeError('Incorrect format for the region matrix; \
                must be ContactMatrix.')
        if not region_matrix.ages == contact_matrix.ages:
            raise ValueError('Region and Contact matrices have unmatched \
                age group structure.')

        # Check correct format of dI
        if not isinstance(dI, (int, float)):
            raise TypeError('Duration of infection must be integer or float.')
        if dI <= 0:
            raise ValueError('Duration of infection must be positive.')

        # Check correct format of susceptible compartments size
        if np.asarray(pop_size).ndim != 1:
            raise ValueError(
                'Susceptible population sizes storage format must be \
                    1-dimensional.')
        if np.asarray(pop_size).shape[0] != len(contact_matrix.ages):
            raise ValueError('Number of age groups for susceptible \
                population does not match matrices format.')
        for _ in pop_size:
            if not isinstance(_, (int, float)):
                raise TypeError('Number of susceptibles must be integer \
                    or float.')
            if _ < 0:
                raise ValueError('All susceptible population sizes must be \
                    >= 0.')

        self.region = region_matrix.region
        self.ages = region_matrix.ages
        self.susceptibles = np.asarray(pop_size)
        self.contacts = contact_matrix.contact_matrix.to_numpy()
        self.regional_suscep = region_matrix.region_matrix.to_numpy()
        self.infection_period = dI
        self.next_gen_matrix = self._compute_next_gen_matrix()
        self.unnorm_next_gen = self._compute_unnormalised_next_gen_matrix()

    def _compute_unnormalised_next_gen_matrix(self):
        """
        Computes unnormalised next genearation matrix. Element (i, j) refers
        the expected number of new infections in age group i caused by an
        infectious in age group j.

        """
        return np.multiply(self.contacts, self.regional_suscep)

    def _compute_next_gen_matrix(self):
        """
        Computes next genearation matrix. Element (i, j) refers the expected
        number of new infections in age group i caused by infectious in age
        group j.

        """
        # Computes next genearation matrix. Element (i, j) refers the expected
        # number of new infections in age group j caused by infectious in age
        # group j.
        self.C_tilde = self._compute_unnormalised_next_gen_matrix()
        self.generator = np.zeros_like(self.contacts)

        for i, row in enumerate(self.generator):
            for j, _ in enumerate(row):
                self.generator[i, j] = self.susceptibles[i] * (
                    self.C_tilde[i, j] * self.infection_period)

        return pd.DataFrame(
            data=self.generator, index=self.ages, columns=self.ages)

    def compute_dom_eigenvalue(self):
        """
        Returns the dominant (maximum) eigenvalue of the infection
        generator matrix.

        """
        return max(np.linalg.eigvals(self.generator).tolist())

#
# UniInfectivityMatrix Class
#


class UniInfectivityMatrix(object):
    r"""UniInfectivityMatrix Class:
    Base class to compute the probability of susceptible individuals in
    a given region and specified time point of getting infected as well
    as reproduction number for subsequent time points.

    Both quanities are computed using .. math::`\beta_{t_k, r}` is the further
    temporal correction term, linked to fluctuations in transmission,
    .. math::`R_{0, r}` is the initial reproduction number in region r and
    .. math::`R^{\star}_{0, r}` is the dominant eigenvalue of the initial next
    generation matrix for region r.

    Parameters
    ----------
    initial_r
        (float) Initial value of the reproduction number in the specified
        region.
    initial_nextgen_matrix
        (UniNextGenMatrix) Next generation matrix at time of beginning of
        the epidemic.

    """
    def __init__(
            self, initial_r, initial_nextgen_matrix):
        # Check correct format of inputs
        if not isinstance(initial_r, (int, float)):
            raise TypeError(
                'Initial regional R must be integer or float.')
        if not isinstance(initial_nextgen_matrix, UniNextGenMatrix):
            raise TypeError(
                'Initial next generation matrix must be a UniNextGenMatrix.')

        self.r0 = initial_r
        self.r0_star = initial_nextgen_matrix.compute_dom_eigenvalue()
        self._constant = self.r0 / self.r0_star

    def compute_prob_infectivity_matrix(
            self, temp_variation, later_nextgen_matrix):
        r"""
        Computes probability of susceptible individuals in
        a given region and specified time point of getting infected. The
        (i, j) element of the matrix refer to the probabiity of people in
        age group i to be infected by those in age group j.

        The matrix is computed using this formula:

        .. math::
            \b^{t_k}_{r, ij} = \beta_{t_k, r} R_{0, r} \frac{
                \widetilde{C}_{r, ij}^{t_k}}{R^{\star}_{0, r}}

        where .. math::`\beta_{t_k, r}` is the further temporal correction
        term, linked to fluctuations in transmission, .. math::`R_{0, r}` is
        the initial reproduction number in region r and
        .. math::`R^{\star}_{0, r}` is the dominant eigenvalue of the initial
        next generation matrix for region r.

        Parameters
        ----------
        temp_variation
            (float) Further temporal correction term, linked to fluctuations
            in transmission.
        later_nextgen_matrix
            (UniNextGenMatrix) Next generation matrix at given time during the
            the epidemic.

        """
        if not isinstance(temp_variation, (int, float)):
            raise TypeError(
                'Regional temporal correction term must be integer or float.')
        if not isinstance(later_nextgen_matrix, UniNextGenMatrix):
            raise TypeError(
                'Current next generation matrix must be a UniNextGenMatrix.')

        return later_nextgen_matrix.unnorm_next_gen * (
            self._constant * temp_variation)

    def compute_reproduction_number(
            self, temp_variation, later_nextgen_matrix):
        r"""
        Computes probability of susceptible individuals in
        a given region and specified time point of getting infected. The
        (i, j) element of the matrix refer to the probabiity of people in
        age group i to be infected by those in age group j.

        The matrix is computed using this formula:

        .. math::
            \b^{t_k}_{r, ij} = \beta_{t_k, r} R_{0, r} \frac{
                R^{\star}_{t_k, r}}{R^{\star}_{0, r}}

        where .. math::`\beta_{t_k, r}` is the further temporal correction
        term, linked to fluctuations in transmission, .. math::`R_{0, r}` is
        the initial reproduction number in region r and
        .. math::`R^{\star}_{0, r}` is the dominant eigenvalue of the initial
        next generation matrix for region r.

        The .. math::`R^{\star}_{t_k, r}` is the dominant eigenvalue of the
        current time next generation matrix for region r:

        .. math::
            \Lambda_{k, r, ij} = S_{r, t_k, i} \widetilde{C}_{r, ij}^{t_k}
                 d_{I}

        Parameters
        ----------
        temp_variation
            (float) Further temporal correction term, linked to fluctuations
            in transmission.
        later_nextgen_matrix
            (UniNextGenMatrix) Next generation matrix at given time during the
            the epidemic.

        """
        if not isinstance(temp_variation, (int, float)):
            raise TypeError(
                'Regional temporal correction term must be integer or float.')
        if not isinstance(later_nextgen_matrix, UniNextGenMatrix):
            raise TypeError(
                'Current next generation matrix must be a UniNextGenMatrix.')

        return later_nextgen_matrix.compute_dom_eigenvalue() * (
            self._constant * temp_variation)


#
# MultiTimesInfectivity Class
#


class MultiTimesInfectivity(UniInfectivityMatrix, UniNextGenMatrix):
    r"""MultiTimesInfectivity Class:
    Base class to compute the probabilities of susceptible individuals in
    a given region and specified time point of getting infected as well
    as reproduction number for subsequent time points, evaluating at multiple
    time points and in multiple regions.

    In the computation of both quanities time-dependent progressions of contact
    matrices and region matrices, accompanied by vectors of the times at which
    the changes occur.

    Parameters
    ----------
    matrices_contact
        (list of ContactMatrix) Time-dependent contact matrices used for the
        modelling.
    time_changes_contact
        (list) Times at which the next contact matrix recorded starts to be
        used. In increasing order. Start with 1.
    regions
        (list) List of region names for the region-specific relative
        susceptibility matrices.
    matrices_region
        (list of lists of RegionMatrix)) Time-dependent and region-specific
        relative susceptibility matrices used for the modelling.
    time_changes_region
        (list) Times at which the next instances of region-specific relative
        susceptibility matrices recorded start to be used. In increasing order.
        Start with 1.
    initial_r
        (list) List of initial values of the reproduction number by region.
    dI
        (float) Average duration of infection.
    susceptibles
        (numpy.array) Array of initial number of susceptibles by region and
        age-group.
    """
    def __init__(
            self, matrices_contact, time_changes_contact, regions,
            matrices_region, time_changes_region, initial_r, dI, susceptibles):
        # Check correct format of matrices_contact
        if np.asarray(matrices_contact).ndim != 1:
            raise ValueError(
                'Storage format for the multiple contact matrices \
                    must be 1-dimensional.')

        for _ in range(len(matrices_contact)):
            if not isinstance(matrices_contact[_], ContactMatrix):
                raise TypeError(
                    'Contact matrices must be in the ContactMatrix format.')

        # Check correct format of time_changes_contact
        if np.asarray(time_changes_contact).ndim != 1:
            raise ValueError(
                'Times of changes in contacts storage format must be \
                    1-dimensional')
        if len(time_changes_contact) != len(matrices_contact):
            raise ValueError(
                'Number of changing points and given contact matrices do \
                    not match.')

        for _ in range(len(time_changes_contact)):
            if not isinstance(time_changes_contact[_], int):
                raise TypeError(
                    'Times of changes in contacts must be integers.')
            if time_changes_contact[_] <= 0:
                raise ValueError('Times of changes in contacts must be \
                    positive.')

        # Check correct format of regions
        if np.asarray(regions).ndim != 1:
            raise ValueError(
                'Region names storage format must be 1-dimensional')

        for _ in range(len(regions)):
            if not isinstance(regions[_], str):
                raise TypeError(
                    'Region names value format must be a string')

        # Check correct format of matrices_region
        if np.asarray(matrices_region).ndim != 2:
            raise ValueError(
                'Storage format for the multiple regional relative \
                    susceptibility matrices must be 2-dimensional')

        for _ in range(len(matrices_region)):
            if len(regions) != len(matrices_region[_]):
                raise ValueError('Wrong number of matrices for the \
                    number of regions registered.')
            for r in range(len(matrices_region[_])):
                if not isinstance(matrices_region[_][r], RegionMatrix):
                    raise TypeError(
                        'Regional relative susceptibility matrices must \
                            be in the RegionMatrix format.')
                if matrices_region[_][r].region != regions[r]:
                    raise ValueError(
                        'Incorrect region name used for this regional relative \
                            susceptibility matrix.')

        # Check correct format of time_changes_region
        if np.asarray(time_changes_region).ndim != 1:
            raise ValueError(
                'Times of changes in regional matrices storage format must be \
                    1-dimensional.')
        if len(time_changes_region) != len(matrices_region):
            raise ValueError(
                'Number of changing points and given region-specific relative \
                    susceptibility matrices do not match.')

        for _ in range(len(time_changes_region)):
            if not isinstance(time_changes_region[_], int):
                raise TypeError(
                    'Times of changes in regional matrices must be integers.')
            if time_changes_region[_] <= 0:
                raise ValueError('Times of changes in regional matrices must be \
                    positive.')

        # Check correct format of initial_r
        if np.asarray(initial_r).ndim != 1:
            raise ValueError(
                'Storage format for the initial reproduction numbers \
                    must be 1-dimensional.')

        if len(initial_r) != len(regions):
            raise ValueError(
                'Number of initial reproduction numbers does not match \
                    that of regions.')

        for _ in range(len(initial_r)):
            if not isinstance(initial_r[_], (int, float)):
                raise TypeError(
                    'Initial reproduction numbers must be integer or float.')
            if initial_r[_] <= 0:
                raise ValueError(
                    'Initial reproduction numbers must be positive.')

        # Check correct format of dI
        if not isinstance(dI, (int, float)):
            raise TypeError('Duration of infection must be integer or float.')
        if dI <= 0:
            raise ValueError('Duration of infection must be positive.')

        # Check correct format of susceptibles
        if np.asarray(susceptibles).ndim != 2:
            raise ValueError(
                'Storage format for the numbers of susceptibles by region \
                    must be 2-dimensional.')

        if np.asarray(susceptibles).shape[0] != len(regions):
            raise ValueError(
                'Number of compartments of susceptibles by region does not match \
                    that of regions.')

        if np.asarray(susceptibles).shape[1] != len(matrices_contact[0].ages):
            raise ValueError(
                'Number of compartments of susceptibles by region does not match \
                    that of age groups.')

        for r in np.asarray(susceptibles):
            for _ in r:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'Number of susceptibles must be integer or float.')

        initial_infec_matrices = []

        for r, _ in enumerate(regions):
            initial_infec_matrices.append(UniInfectivityMatrix(
                initial_r=initial_r[r],
                initial_nextgen_matrix=UniNextGenMatrix(
                    pop_size=susceptibles[r],
                    contact_matrix=matrices_contact[0],
                    region_matrix=matrices_region[0][r],
                    dI=dI)
                ))

        self._regions = regions
        self.initial_r = np.asarray(initial_r)
        self.dI = dI
        self.times_contact = np.asarray(time_changes_contact)
        self.times_region = np.asarray(time_changes_region)
        self.contact_matrices = matrices_contact
        self.region_matrices = matrices_region
        self.initial_infec_matrices = initial_infec_matrices

    def _check_susceptible_input(
            self, susceptibles, contact_matrix):
        """
        """
        if np.asarray(susceptibles).ndim != 1:
            raise ValueError(
                'Storage format for the numbers of susceptibles for fixed region \
                    must be 1-dimensional.')

        if np.asarray(susceptibles).shape[0] != len(contact_matrix.ages):
            raise ValueError(
                'Number of compartments of susceptibles by region does not match \
                    that of age groups.')

        for _ in np.asarray(susceptibles):
            if not isinstance(_, (np.integer, np.floating)):
                raise TypeError(
                    'Number of susceptibles must be integer or float.')

    def _check_later_input(self, r, t_k, temp_variation, susceptibles):
        """
        Checks the correct format for the input of the two main methods
        for the class.

        """
        if not isinstance(r, int):
            raise TypeError(
                'Index of the region must be integer.'
                )
        if r > len(self.region_matrices):
            raise ValueError(
                'Index of the region out of bounds.'
            )
        if r <= 0:
            raise ValueError(
                'Index of the region must be >= 1.'
            )

        if not isinstance(t_k, (int, float)):
            raise TypeError(
                'Time of evaluation of next generation matrix must be integer \
                    or float.'
                )

        if t_k <= 0:
            raise ValueError(
                'Time of evaluation of next generation matrix must be >= 1.'
            )

        if not isinstance(temp_variation, (int, float)):
            raise TypeError(
                'Regional temporal correction term must be integer or float.')

    def compute_prob_infectivity_matrix(
            self, r, t_k, susceptibles, temp_variation=1):
        r"""
        Computes probability of susceptible individuals in
        a given region and specified time point of getting infected. The
        (i, j) element of the matrix refer to the probabiity of people in
        age group i to be infected by those in age group j.

        The matrix is computed using this formula:

        .. math::
            \b^{t_k}_{r, ij} = \beta_{t_k, r} R_{0, r} \frac{
                \widetilde{C}_{r, ij}^{t_k}}{R^{\star}_{0, r}}

        where .. math::`\beta_{t_k, r}` is the further temporal correction
        term, linked to fluctuations in transmission, .. math::`R_{0, r}` is
        the initial reproduction number in region r and
        .. math::`R^{\star}_{0, r}` is the dominant eigenvalue of the initial
        next generation matrix for region r.

        Parameters
        ----------
        r
            (integer) Index of the region at which the next generation matrix
            is evaluated.
        t_k
            (float) Time at which the next generation matrix is evaluated.
        temp_variation
            (float) Further temporal correction term, linked to fluctuations
            in transmission.
        susceptibles
            (numpy.array) Array of current number of susceptibles by region and
            age-group.
        """
        # Do the checks on the input
        self._check_susceptible_input(
            susceptibles, self.contact_matrices[0])
        self._check_later_input(r, t_k, temp_variation, susceptibles)

        # Identify current contact matrix
        pos = np.where(self.times_contact <= t_k)
        current_contacts = self.contact_matrices[pos[-1][-1]]

        # Identify current regional relative susceptibility matrix
        pos = np.where(self.times_region <= t_k)
        current_rel_susc = self.region_matrices[pos[-1][-1]][r-1]

        current_nextgen_matrix = UniNextGenMatrix(
                pop_size=np.asarray(susceptibles).tolist(),
                contact_matrix=current_contacts,
                region_matrix=current_rel_susc,
                dI=self.dI)

        return self.initial_infec_matrices[
            r-1].compute_prob_infectivity_matrix(
                temp_variation, current_nextgen_matrix)

    def compute_reproduction_number(
            self, r, t_k, susceptibles, temp_variation=1):
        r"""
        Computes probability of susceptible individuals in
        a given region and specified time point of getting infected. The
        (i, j) element of the matrix refer to the probabiity of people in
        age group i to be infected by those in age group j.

        The matrix is computed using this formula:

        .. math::
            \b^{t_k}_{r, ij} = \beta_{t_k, r} R_{0, r} \frac{
                R^{\star}_{t_k, r}}{R^{\star}_{0, r}}

        where .. math::`\beta_{t_k, r}` is the further temporal correction
        term, linked to fluctuations in transmission, .. math::`R_{0, r}` is
        the initial reproduction number in region r and
        .. math::`R^{\star}_{0, r}` is the dominant eigenvalue of the initial
        next generation matrix for region r.

        The .. math::`R^{\star}_{t_k, r}` is the dominant eigenvalue of the
        current time next generation matrix for region r:

        .. math::
            \Lambda_{k, r, ij} = S_{r, t_k, i} \widetilde{C}_{r, ij}^{t_k}
                 d_{I}

        Parameters
        ----------
        r
            (integer) Index of the region at which the next generation matrix
            is evaluated.
        t_k
            (float) Time at which the next generation matrix is evaluated.
        temp_variation
            (float) Further temporal correction term, linked to fluctuations
            in transmission.
        susceptibles
            (numpy.array) Array of current number of susceptibles by region and
            age-group.
        """
        # Do the checks on the input
        self._check_susceptible_input(
            susceptibles, self.contact_matrices[0])
        self._check_later_input(r, t_k, temp_variation, susceptibles)

        # Identify current contact matrix
        pos = np.where(self.times_contact <= t_k)
        current_contacts = self.contact_matrices[pos[-1][-1]]

        # Identify current regional relative susceptibility matrix
        pos = np.where(self.times_region <= t_k)
        current_rel_susc = self.region_matrices[pos[-1][-1]][r-1]

        current_nextgen_matrix = UniNextGenMatrix(
                pop_size=np.asarray(susceptibles).tolist(),
                contact_matrix=current_contacts,
                region_matrix=current_rel_susc,
                dI=self.dI)

        return self.initial_infec_matrices[r-1].compute_reproduction_number(
            temp_variation, current_nextgen_matrix)
