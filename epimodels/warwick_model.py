#
# WarwickSEIRModel Class
#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""
This script contains code for modelling the extended SEIR model
created by the University of Warwick. This is one of the official models used
by the UK government for policy making.

It uses an extended version of an SEIR model with contact and region-specific
matrices and can be used to model the effects of within-household dynamics on
the epidemic trajectory in different countries. It also differentiates between
asymptomatic and symptomatic infections.

"""

from itertools import chain

import numpy as np
import pints
from scipy.stats import nbinom, binom
from scipy.integrate import solve_ivp

import epimodels as em


class WarwickSEIRModel(pints.ForwardModel):
    r"""WarwickSEIRModel Class:
    Base class for constructing the ODE model: deterministic SEIR developed by
    University of Warwick to model the Covid-19 epidemic and the effects
    of within-household dynamics on the epidemic trajectory in different
    countries.

    The population is structured such that every individual will belong to one
    of the compartments of the extended SEIR model.

    The general SEIR Model has four compartments - susceptible individuals
    (:math:`S`), exposed but not yet infectious (:math:`E`), infectious
    (:math:`I`) and recovered (:math:`R`).

    In the Warwick-Household model framework, the exposed are split first into
    four compartments, depending on the type of infective that has infected
    them within the household, and each of them further into three sequential
    exposed compartments to illustrate different latency stages. The
    infectious compartment is split into 8 distinct ones: depending on whether
    they are symptomatic or asymptomatic infectious, and whether they are the
    first in the household to be infected, if they are quarantined, or are a
    subsequent infection. We also consider a population divided in age groups,
    as we expect people of different ages to interact differently between
    themselves and to be affected differently by the virus, i.e. have
    different susceptibilities to infection and proportions of
    asymptomatic individuals. The model structure now
    becomes, for each region:

    .. math::
       :nowrap:

        \begin{eqnarray}
            \frac{dS_i}{dt} &=& - \sigma \Big(C^N \big(\frac{S_i}{N} I^F_j +
                \frac{S_i}{N} I^{SD}_j + \frac{S_i}{N} I^{SU}_j +
                \tau(\frac{S_i}{N} A^F_j + \frac{S_i}{N} A^S_j)\big) +
                +C^H (\frac{S_i}{N} I^F_j + \frac{S_i}{N} A^F_j +
                \frac{S_i}{N} I^{QF}_j) \Big)\\
            \frac{dE^{1,F}_i}{dt} &=& \sigma C^N \big(\frac{S_i}{N} I^F_j +
                \frac{S_i}{N} I^{SD}_i + \frac{S_i}{N} I^{SU}_j +
                \tau(\frac{S_i}{N} A^F_j + \frac{S_i}{N} A^S_j)\big) -
                3 \epsilon E^{1,F}_j \\
            \frac{dE^{1,SD}_i}{dt} &=& \sigma C^H \frac{S_i}{N} I^F_i -
                3 \epsilon E^{1,SD}_i \\
            \frac{dE^{1,SU}_i}{dt} &=& \sigma C^H \frac{S_i}{N} A^F_i -
                3 \epsilon E^{1,SU}_i \\
            \frac{dE^{1,Q}_i}{dt} &=& \sigma C^H S_i I^{QF}_i -
                3 \epsilon E^{1,Q}_i \\
            \frac{dE^{2,F}_i}{dt} &=& 3 \epsilon E^{1,F}_i -
                3 \epsilon E^{2,F}_i \\
            \frac{dE^{2,SD}_i}{dt} &=& 3 \epsilon E^{1,SD}_i -
                3 \epsilon E^{2,SD}_i \\
            \frac{dE^{2,SU}_i}{dt} &=& 3 \epsilon E^{1,SU}_i -
                3 \epsilon E^{2,SU}_i \\
            \frac{dE^{2,Q}_i}{dt} &=& 3 \epsilon E^{1,Q}_i -
                3 \epsilon E^{2,Q}_i \\
            \frac{dE^{3,F}_i}{dt} &=& 3 \epsilon E^{2,F}_i -
                3 \epsilon E^{3,F}_i \\
            \frac{dE^{3,SD}_i}{dt} &=& 3 \epsilon E^{2,SD}_i -
                3 \epsilon E^{3,SD}_i \\
            \frac{dE^{3,SU}_i}{dt} &=& 3 \epsilon E^{2,SU}_i -
                3 \epsilon E^{3,SU}_i \\
            \frac{dE^{1,Q}_i}{dt} &=& 3 \epsilon E^{2,Q}_i -
                3 \epsilon E^{3,Q}_i \\
            \frac{dI^F_i}{dt} &=& 3 (1-H) \epsilon d E^{3,F}_i -
                \gamma I^F_i \\
            \frac{dI^{SD}_i}{dt} &=& 3 \epsilon d E^{3,SD}_i -
                \gamma I^{SD}_i \\
            \frac{dI^{SU}_i}{dt} &=& 3 (1-H) \epsilon d E^{3,SU}_i -
                \gamma I^{SU}_i \\
            \frac{dI^{QF}_i}{dt} &=& 3 H \epsilon d E^{3,F}_i -
                \gamma I^{QF}_i \\
            \frac{dI^{QS}_i}{dt} &=& 3 H \epsilon d E^{3,SU}_i +
                3 \epsilon d E^{3,Q}_i) - \gamma I^{QS}_i \\
            \frac{dA^F_i}{dt} &=& 3 \epsilon (1 - d) E^{3,F}_i -
                \gamma A^F_i \\
            \frac{dA^S_i}{dt} &=& 3 \epsilon (1 - d) (E^{3,SD}_i + E^{3,SU}_i)
                - \gamma A^S_i \\
            \frac{dA^Q_i}{dt} &=& 3 \epsilon (1 - d) E^{3,Q}_i -
                \gamma A^Q_i \\
            \frac{dR_i}{dt} &=& \gamma (I^F_i + I^{QF}_i + A^F_i + I^{SD}_i +
                A^S_i + I^{SU}_i + I^{QS}_i + A^Q_i)
        \end{eqnarray}

    where :math:`i` is the age group of the individual, :math:`C^H_{ij}` is
    the :math:`(i,j)` the element of the regional household contact matrix, and
    represents the expected number of contacts in age group :math:`i` within
    the household made by an individuals in age group :math:`j` on a given day.
    Similarly, :math:`C^N_{ij}` is the :math:`(i,j)` the element of the
    regional non-household contact matrix and represents the expected number of
    contacts in age group :math:`i` outside the household made by an
    individual in age group :math:`j` on a given day. :math:`N` is the total
    population size.

    Non-pharmaceutical interventions will alter the number of contacts
    individuals will have in each context per day. The correct :math:`C^H` and
    :math:`C^N` matrices are calculated each day by computing a weighted
    average between the baseline matrices for contacts within household
    (:math:`C^{H, base}`), at school (:math:`C^{S, base}`), in the workplace
    (:math:`C^{W, base}`) and in all other contexts (:math:`C^{O, base}`) and
    their full-lockdown equivalents which are given by:

    .. math::
        :nowrap:

        \begin{eqnarray}
            C^{H, lock} &=& q_H * C^{H, base} \\
            C^{S, lock} &=& q_S * C^{S, base} \\
            C^{W, lock} &=& q_W * C^{W, base} \\
            C^{O, lock} &=& q_O * C^{O, base} \\
        \end{eqnarray}

    where :math:`q_H` is the coefficient of increase in contacts within the
    household and :math:`q_S`, :math:`q_W` and :math:`q_O` are the
    decrease in contacts at school, in the workplace, and, respectively, in all
    other contexts when a full-lockdown is implemented. Therefore, the
    daily contact matrices :math:`C^H` and :math:`C^N` applied are given by:

    .. math::
        :nowrap:

        \begin{eqnarray}
            C^H &=& 1.3 * \big((1-\phi)*C^{H, base} + \phi*C^{H, lock}\big) \\
            C^S &=& \big((1-\phi)*C^{S, base} + \phi*C^{S, lock}\big) \\
            C^W &=& \big(1-\theta + \theta * (1-\phi + \phi*q_O)\big) * \big(
                (1-\phi)*C^{W, base} + \phi*C^{W, lock}\big) \\
            C^O &=& (1-\phi + \phi*q_O) * \big(
                (1-\phi)*C^{O, base} + \phi*C^{O, lock}\big) \\
        \end{eqnarray}

    where :math:`\phi` is a coefficient indicating the strength of the
    implemented interventions and :math:`\theta` is a scaling factor of any
    public-facing interactions.

    The transmission parameters are the rates with which different types of
    infectious individual infects susceptible ones. Asymptomatic infections
    are assumed to have a reduced transmission compared to their symptomatic
    counterpart. This reduction in transmission is indicated through the
    parameter :math:`\tau` in the force of infection.

    The :math:`\sigma`, :math:`H` and :math:`d` parameters represent the
    susceptibilities to the disease, the household quarantine compliance
    factor and, respectively, the proportions of people that go on to develop
    symptomatic infections. Because we expect older people to be more likely
    to be susceptible to infection and younger people to be more likely to be
    asymptomatic, we consider :math:`\sigma` and :math:`d` to be age dependent.

    The rates of progression through the different
    stages of the illness are:

        * :math:`\epsilon`: exposed to (a)symptomatic infectious status;
        * :math:`\gamma`: (a)symptomatic to recovered status.

    :math:`S(0) = S_0`, :math:`E^{1,F}(0) = E^{1,F}_0`,
    :math:`E^{1,SD}(0) = E^{1,SD}_0`, :math:`E^{1,SU}(0) = E^{1,SU}_0`,
    :math:`E^{1,Q}(0) = E^{1,Q}_0`, :math:`E^{2,F}(0) = E^{2,F}_0`,
    :math:`E^{2,SD}(0) = E^{2,SD}_0`, :math:`E^{2,SU}(0) = E^{2,SU}_0`,
    :math:`E^{2,Q}(0) = E^{2,Q}_0`, :math:`E^{3,F}(0) = E^{3,F}_0`,
    :math:`E^{3,SD}(0) = E^{3,SD}_0`, :math:`E^{3,SU}(0) = E^{3,SU}_0`,
    :math:`E^{3,Q}(0) = E^{3,Q}_0`, :math:`I^F(0) = I^F_0`,
    :math:`I^{SD}(0) = I^{SD}_0`, :math:`I^{SU}(0) = I^{SU}_0`,
    :math:`I^{QF}(0) = I^{QF}_0`, :math:`I^{QS}(0) = I^{QS}_0`,
    :math:`A^F(0) = A^F_0`, :math:`A^S(0) = A^S_0`, :math:`A^Q(0) = A^Q_0`,
    :math:`R(0) = R_0`, are also parameters of the model (evaluation at 0
    refers to the compartments' structure at initial time.

    Extends :class:`pints.ForwardModel`.

    """
    def __init__(self):
        super(WarwickSEIRModel, self).__init__()

        # Assign default values
        self._output_names = [
            'S', 'E1f', 'E1sd', 'E1su', 'E1q', 'E2f', 'E2sd', 'E2su', 'E2q',
            'E3f', 'E3sd', 'E3su', 'E3q', 'If', 'Isd', 'Isu', 'Iqf', 'Iqs',
            'Af', 'As', 'Aq', 'R', 'Incidence']
        self._parameter_names = [
            'S0', 'E1f0', 'E1sd0', 'E1su0', 'E1q0', 'E2f0', 'E2sd0', 'E2su0',
            'E2q0', 'E3f0', 'E3sd0', 'E3su0', 'E3q0', 'If0', 'Isd0', 'Isu0',
            'Iqf0', 'Iqs0', 'Af0', 'As0', 'Aq0', 'R0', 'sig', 'tau', 'eps',
            'gamma', 'd', 'H']

        # The default number of outputs is 23,
        # i.e. S, E1f, E1sd, E1su, E1q, E2f, E2sd, E2su, E2q, E3f, E3sd, E3su,
        # E3q, If, Isd, Isu, Iqf, Iqs, Af, As, Aq, R and
        # Incidence
        self._n_outputs = len(self._output_names)
        # The default number of parameters is 28,
        # i.e. 22 initial conditions and 6 parameters
        self._n_parameters = len(self._parameter_names)

        self._output_indices = np.arange(self._n_outputs)

    def n_outputs(self):
        """
        Returns the number of outputs.

        Returns
        -------
        int
            Number of outputs.

        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the number of parameters.

        Returns
        -------
        int
            Number of parameters.

        """
        return self._n_parameters

    def output_names(self):
        """
        Returns the (selected) output names.

        Returns
        -------
        list
            List of the (selected) output names.

        """
        names = [self._output_names[x] for x in self._output_indices]
        return names

    def parameter_names(self):
        """
        Returns the parameter names.

        Returns
        -------
        list
            List of the parameter names.

        """
        return self._parameter_names

    def set_regions(self, regions):
        """
        Sets region names.

        Parameters
        ----------
        regions : list
            List of region names considered by the model.

        """
        self.regions = regions

    def set_age_groups(self, age_groups):
        """
        Sets age group names and counts their number.

        Parameters
        ----------
        age_groups : list
            List of age group names considered by the model.

        """
        self.age_groups = age_groups
        self._num_ages = len(self.age_groups)

    def region_names(self):
        """
        Returns the regions names.

        Returns
        -------
        list
            List of the regions names.

        """
        return self.regions

    def age_groups_names(self):
        """
        Returns the age group names.

        Returns
        -------
        list
            List of the age group names.

        """
        return self.age_groups

    def set_outputs(self, outputs):
        """
        Checks existence of outputs and selects only those remaining.

        Parameters
        ----------
        outputs : list
            List of output names that are selected.

        """
        for output in outputs:
            if output not in self._output_names:
                raise ValueError(
                    'The output names specified must be in correct forms')

        output_indices = []
        for output_id, output in enumerate(self._output_names):
            if output in outputs:
                output_indices.append(output_id)

        # Remember outputs
        self._output_indices = output_indices
        self._n_outputs = len(outputs)

    def _compute_soc_dist_parameters(self, t):
        """
        Computes the current values of the social distancing parameters
        at a specified timepoint.

        Parameters
        ----------
        t : float
            Time point at which we compute the evaluation.

        Returns
        -------
        list of int or float
            List of values of the current social distancing parameters
            at the specified timepoint.

        """
        theta_all, phi_all, q_H_all, q_S_all, q_W_all, q_O_all, times_npis = \
            self.social_distancing_param

        # Identify the current time and region NPIs flags
        pos = np.where(np.asarray(times_npis) <= t)

        # Compute current levels of the social distancing parameters
        theta = theta_all[pos[-1][-1]]
        phi = phi_all[pos[-1][-1]]
        q_H = q_H_all[pos[-1][-1]]
        q_S = q_S_all[pos[-1][-1]]
        q_W = q_W_all[pos[-1][-1]]
        q_O = q_O_all[pos[-1][-1]]

        return theta, phi, q_H, q_S, q_W, q_O

    def _right_hand_side(self, t, r, y, c, num_a_groups):
        r"""
        Constructs the RHS of the equations of the system of ODEs for given a
        region and time point.

        Parameters
        ----------
        t : float
            Time point at which we compute the evaluation.
        r : int
            The index of the region to which the current instance of the ODEs
            system refers.
        y : numpy.array
            Array of all the compartments of the ODE system, segregated
            by age-group. It assumes y = [S, E1f, E1sd, E1su, E1q, E2f, E2sd,
            E2su, E2q, E3f, E3sd, E3su, E3q, If, Isd, Isu,
            Iqf, Iqs, Af, As, Aq, R] where each letter actually refers to all
            compartment of that type. (e.g. S refers to the compartments of
            all ages of susceptibles).
        c : list
            List of values used to compute the parameters of the ODEs
            system. It assumes c = [sig, tau, eps, gamma, d, H], where
            :math:`sig` represents the age-dependent susceptibility of
            individuals to infection, :math:`tau` is the reduction in the
            transmission rate of infection for asymptomatic individuals,
            :math:`eps` is the rate of progression to infectious disease,
            :math:`gamma` is the recovery rate, :math:`d` represents the age-
            dependent probability of displaying symptoms and :math:`H` is the
            household quarantine proportion.
        num_a_groups : int
            Number of age groups in which the population is split. It
            refers to the number of compartments of each type.

        Returns
        -------
        numpy.array
            Age-strictured matrix representation of the RHS of the ODEs system.

        """
        # Read in the number of age-groups
        a = num_a_groups

        # Split compartments into their types
        s, e1F, e1SD, e1SU, e1Q, e2F, e2SD, e2SU, e2Q, e3F, e3SD, e3SU, e3Q, \
            iF, iSD, iSU, iQF, iQS, aF, aS, aQ, _ = (
                y[:a], y[a:(2*a)], y[(2*a):(3*a)],
                y[(3*a):(4*a)], y[(4*a):(5*a)], y[(5*a):(6*a)],
                y[(6*a):(7*a)], y[(7*a):(8*a)], y[(8*a):(9*a)],
                y[(9*a):(10*a)], y[(10*a):(11*a)], y[(11*a):(12*a)],
                y[(12*a):(13*a)], y[(13*a):(14*a)], y[(14*a):(15*a)],
                y[(15*a):(16*a)], y[(16*a):(17*a)], y[(17*a):(18*a)],
                y[(18*a):(19*a)], y[(19*a):(20*a)], y[(20*a):(21*a)],
                y[(21*a):])

        # Read the social distancing parameters of the system
        theta, phi, q_H, q_S, q_W, q_O = self._compute_soc_dist_parameters(t)

        # Read parameters of the system
        sig, tau, eps, gamma, d, h_all = c[:6]

        h = h_all[self._region-1] * phi

        # Identify the appropriate contact matrix for the ODE system
        house_cont_mat = \
            self.house_contacts_timeline.identify_current_contacts(r, t)
        school_cont_mat = \
            self.school_contacts_timeline.identify_current_contacts(r, t)
        work_cont_mat = \
            self.work_contacts_timeline.identify_current_contacts(r, t)
        other_cont_mat = \
            self.other_contacts_timeline.identify_current_contacts(r, t)

        house_cont_mat = 1.3 * (1 - phi + phi * q_H) * house_cont_mat
        nonhouse_cont_mat = (1 - phi + phi * q_S) * school_cont_mat + \
            ((1 - phi + phi * q_W) * (
                1 - theta + theta * (1 - phi + phi * q_O))) * work_cont_mat + \
            ((1 - phi + phi * q_O)**2) * other_cont_mat

        # Write actual RHS
        lam_F = np.multiply(sig, np.dot(
            nonhouse_cont_mat, np.asarray(iF) + np.asarray(iSD) +
            np.asarray(iSU) + tau * np.asarray(aF) + tau * np.asarray(aS)))
        lam_F_times_s = \
            np.multiply(s, (1 / self._N[r-1]) * lam_F)

        lam_SD = np.multiply(sig, np.dot(house_cont_mat, np.asarray(iF)))
        lam_SD_times_s = \
            np.multiply(s, (1 / self._N[r-1]) * lam_SD)

        lam_SU = np.multiply(sig, tau * np.dot(house_cont_mat, np.asarray(aF)))
        lam_SU_times_s = \
            np.multiply(s, (1 / self._N[r-1]) * lam_SU)

        lam_Q = np.multiply(sig, np.dot(house_cont_mat, np.asarray(iQF)))
        lam_Q_times_s = \
            np.multiply(s, (1 / self._N[r-1]) * lam_Q)

        dydt = np.concatenate((
            -(lam_F_times_s + lam_SD_times_s + lam_SU_times_s + lam_Q_times_s),
            lam_F_times_s - 3 * eps * np.asarray(e1F),
            lam_SD_times_s - 3 * eps * np.asarray(e1SD),
            lam_SU_times_s - 3 * eps * np.asarray(e1SU),
            lam_Q_times_s - 3 * eps * np.asarray(e1Q),
            3 * eps * np.asarray(e1F) - 3 * eps * np.asarray(e2F),
            3 * eps * np.asarray(e1SD) - 3 * eps * np.asarray(e2SD),
            3 * eps * np.asarray(e1SU) - 3 * eps * np.asarray(e2SU),
            3 * eps * np.asarray(e1Q) - 3 * eps * np.asarray(e2Q),
            3 * eps * np.asarray(e2F) - 3 * eps * np.asarray(e3F),
            3 * eps * np.asarray(e2SD) - 3 * eps * np.asarray(e3SD),
            3 * eps * np.asarray(e2SU) - 3 * eps * np.asarray(e3SU),
            3 * eps * np.asarray(e2Q) - 3 * eps * np.asarray(e3Q),
            3 * eps * (1-h) * np.multiply(d, e3F) - gamma * np.asarray(iF),
            3 * eps * np.multiply(d, e3SD) - gamma * np.asarray(iSD),
            3 * eps * (1-h) * np.multiply(d, e3SU) - gamma * np.asarray(iSU),
            3 * eps * h * np.multiply(d, e3F) - gamma * np.asarray(iQF),
            3 * eps * (h * np.multiply(d, e3SU) + np.multiply(
                d, e3Q)) - gamma * np.asarray(iQS),
            3 * eps * np.multiply((1-np.asarray(d)), e3F) - gamma * np.asarray(
                aF),
            3 * eps * np.multiply(
                (1-np.asarray(d)),
                np.asarray(e3SD) + np.asarray(e3SU)) - gamma * np.asarray(aS),
            3 * eps * np.multiply((1-np.asarray(d)), e3Q) - gamma * np.asarray(
                aQ),
            gamma * (
                np.asarray(iF) + np.asarray(iQF) + np.asarray(aF) +
                np.asarray(iSD) + np.asarray(aS) + np.asarray(iSU) +
                np.asarray(iQS) + np.asarray(aQ))
            ))

        return dydt

    def _scipy_solver(self, times, num_a_groups, method):
        """
        Computes the values in each compartment of the Warwick ODEs system
        using the 'off-the-shelf' solver of the IVP from :module:`scipy`.

        Parameters
        ----------
        times : list
            List of time points at which we wish to evaluate the ODEs system.
        num_a_groups : int
            Number of age groups in which the population is split. It
            refers to the number of compartments of each type.
        method : str
            The type of solver implemented by the :meth:`scipy.solve_ivp`.

        Returns
        -------
        dict
            Solution of the ODE system at the time points provided.

        """
        # Initial conditions
        si, e1Fi, e1SDi, e1SUi, e1Qi, e2Fi, e2SDi, e2SUi, e2Qi, e3Fi, e3SDi, \
            e3SUi, e3Qi, dFi, iSDi, iSUi, iQFi, iQSi, aFi, aSi, aQi, _i = \
            np.asarray(self._y_init)[:, self._region-1]
        init_cond = list(
            chain(
                si.tolist(), e1Fi.tolist(), e1SDi.tolist(),
                e1SUi.tolist(), e1Qi.tolist(), e2Fi.tolist(),
                e2SDi.tolist(), e2SUi.tolist(), e2Qi.tolist(),
                e3Fi.tolist(), e3SDi.tolist(), e3SUi.tolist(),
                e3Qi.tolist(), dFi.tolist(), iSDi.tolist(),
                iSUi.tolist(), iQFi.tolist(), iQSi.tolist(),
                aFi.tolist(), aSi.tolist(), aQi.tolist(),
                _i.tolist()))

        # Solve the system of ODEs
        sol = solve_ivp(
            lambda t, y: self._right_hand_side(
                t, self._region, y, self._c, num_a_groups),
            [times[0], times[-1]], init_cond, method=method, t_eval=times)
        return sol

    def _split_simulate(
            self, parameters, times, method):
        r"""
        Computes the number of individuals in each compartment at the given
        time points and specified region.

        Parameters
        ----------
        parameters : list
            List of quantities that characterise the Warwick SEIR model in
            this order: index of region for which we wish to simulate,
            initial conditions matrices classifed by age (column name) and
            region (row name) for each type of compartment (s, e1F, e1SD, e1SU,
            e1Q, e2F, e2SD, e2SU, e2Q, e3F, e3SD, e3SU, e3Q, iF, iSD, iSU, iQF,
            iQS, aF, aS, aQ, _), the age-dependent
            susceptibility of individuals to infection (sig), the reduction in
            the transmission rate of infection for asymptomatic individuals
            (tau), the rate of progression to infectious disease (eps), the
            recovery rate (gamma), the age-dependent probability of displaying
            symptoms (d) and the household quarantine proportion (H).
        times : list
            List of time points at which we wish to evaluate the ODEs system.
        method : str
            The type of solver implemented by the :meth:`scipy.solve_ivp`.

        Returns
        -------
        numpy.array
            Age-structured output matrix of the simulation for the specified
            region.

        """
        # Split parameters into the features of the model
        self._region = parameters[0]
        self._y_init = parameters[1:23]
        self._N = np.sum(np.asarray(self._y_init), axis=0)
        self._c = parameters[23:29]
        self.house_contacts_timeline = em.MultiTimesContacts(
            self.house_matrices_contact,
            self.time_changes_contact,
            self.regions,
            self.house_matrices_region,
            self.time_changes_region)

        self.school_contacts_timeline = em.MultiTimesContacts(
            self.school_matrices_contact,
            self.time_changes_contact,
            self.regions,
            self.school_matrices_region,
            self.time_changes_region)

        self.work_contacts_timeline = em.MultiTimesContacts(
            self.work_matrices_contact,
            self.time_changes_contact,
            self.regions,
            self.work_matrices_region,
            self.time_changes_region)

        self.other_contacts_timeline = em.MultiTimesContacts(
            self.other_matrices_contact,
            self.time_changes_contact,
            self.regions,
            self.other_matrices_region,
            self.time_changes_region)

        self._times = np.asarray(times)

        # Simulation using the scipy solver
        sol = self._scipy_solver(times, self._num_ages, method)

        output = sol['y']

        # Age-based total infected is infectious 'i' plus recovered 'r'
        total_infected = output[
            (13*self._num_ages):(14*self._num_ages), :] + output[
            (14*self._num_ages):(15*self._num_ages), :] + output[
            (15*self._num_ages):(16*self._num_ages), :] + output[
            (16*self._num_ages):(17*self._num_ages), :] + output[
            (17*self._num_ages):(18*self._num_ages), :] + output[
            (18*self._num_ages):(19*self._num_ages), :] + output[
            (19*self._num_ages):(20*self._num_ages), :] + output[
            (20*self._num_ages):(21*self._num_ages), :] + output[
            (21*self._num_ages):(22*self._num_ages), :]

        # Number of incidences is the increase in total_infected
        # between the time points (add a 0 at the front to
        # make the length consistent with the solution
        n_incidence = np.zeros((self._num_ages, len(times)))
        n_incidence[:, 1:] = total_infected[:, 1:] - total_infected[:, :-1]

        # Append n_incidence to output
        # Output is a matrix with rows being S, Es, Is, R and Incidence
        output = np.concatenate((output, n_incidence), axis=0)

        # Get the selected outputs
        self._output_indices = np.arange(self._n_outputs)

        output_indices = []
        for i in self._output_indices:
            output_indices.extend(
                np.arange(i*self._num_ages, (i+1)*self._num_ages)
            )

        output = output[output_indices, :]

        return output.transpose()

    def read_contact_data(
            self, house_matrices_contact, school_matrices_contact,
            work_matrices_contact, other_matrices_contact,
            time_changes_contact):
        """
        Reads in the timelines of contact data used for the modelling.

        Parameters
        ----------
        house_matrices_contact : list of ContactMatrix
            List of time-dependent contact matrices used for the modelling,
            underlying household interactions.
        school_matrices_contact : list of ContactMatrix
            List of time-dependent contact matrices used for the modelling,
            underlying school interactions.
        work_matrices_contact : list of ContactMatrix
            List of time-dependent contact matrices used for the modelling,
            underlying workplace interactions.
        other_matrices_contact : list of ContactMatrix
            List of time-dependent contact matrices used for the modelling,
            underlying other non-household interactions.
        time_changes_contact : list
            List of times at which the next contact matrix recorded starts to
            be used. In increasing order.

        """
        if house_matrices_contact[0].ages != school_matrices_contact[0].ages:
            raise ValueError(
                'Contact matrices must refer to the same age groups.')
        if house_matrices_contact[0].ages != work_matrices_contact[0].ages:
            raise ValueError(
                'Contact matrices must refer to the same age groups.')
        if house_matrices_contact[0].ages != other_matrices_contact[0].ages:
            raise ValueError(
                'Contact matrices must refer to the same age groups.')

        self.house_matrices_contact = house_matrices_contact
        self.school_matrices_contact = school_matrices_contact
        self.work_matrices_contact = work_matrices_contact
        self.other_matrices_contact = other_matrices_contact
        self.time_changes_contact = time_changes_contact

    def read_regional_data(
            self, house_matrices_region, school_matrices_region,
            work_matrices_region, other_matrices_region,
            time_changes_region):
        """
        Reads in the timelines of regional data used for the modelling.

        Parameters
        ----------
        house_matrices_region : lists of RegionMatrix
            List of time-dependent and region-specific relative susceptibility
            matrices used for the modelling, underlying household interactions.
        school_matrices_region : lists of RegionMatrix
            List of time-dependent and region-specific relative susceptibility
            matrices used for the modelling, underlying school
            interactions.
        work_matrices_region : lists of RegionMatrix
            List of time-dependent and region-specific relative susceptibility
            matrices used for the modelling, underlying workplace
            interactions.
        other_matrices_region : lists of RegionMatrix
            List of time-dependent and region-specific relative susceptibility
            matrices used for the modelling, underlying other non-household
            interactions.
        time_changes_region : list
            List of times at which the next instances of region-specific
            relative susceptibility matrices recorded start to be used. In
            increasing order.

        """
        if house_matrices_region[0][0].ages != \
                school_matrices_region[0][0].ages:
            raise ValueError(
                'Regional matrices must refer to the same age groups.')
        if house_matrices_region[0][0].region != \
                school_matrices_region[0][0].region:
            raise ValueError(
                'Regional matrices must refer to the same region.')

        if house_matrices_region[0][0].ages != \
                work_matrices_region[0][0].ages:
            raise ValueError(
                'Regional matrices must refer to the same age groups.')
        if house_matrices_region[0][0].region != \
                work_matrices_region[0][0].region:
            raise ValueError(
                'Regional matrices must refer to the same region.')

        if house_matrices_region[0][0].ages != \
                other_matrices_region[0][0].ages:
            raise ValueError(
                'Regional matrices must refer to the same age groups.')
        if house_matrices_region[0][0].region != \
                other_matrices_region[0][0].region:
            raise ValueError(
                'Regional matrices must refer to the same region.')

        self.house_matrices_region = house_matrices_region
        self.school_matrices_region = school_matrices_region
        self.work_matrices_region = work_matrices_region
        self.other_matrices_region = other_matrices_region
        self.time_changes_region = time_changes_region

    def simulate(self, parameters):
        """
        Simulates the Warwick model using a
        :class:`WarwickParametersController` for the model parameters.

        Extends the :meth:`_split_simulate`. Always apply methods
        :meth:`set_regions`, :meth:`set_age_groups`, :meth:`read_contact_data`
        and :meth:`read_regional_data` before running the
        :meth:`WarwickSEIRModel.simulate`.

        Parameters
        ----------
        parameters : WarwickParametersController
            Controller class for the parameters used by the forward simulation
            of the model.

        Returns
        -------
        numpy.array
            Age-structured output matrix of the simulation for the specified
            region.

        """
        self.social_distancing_param = parameters.soc_dist_parameters()

        return self._simulate(
            parameters(), parameters.simulation_parameters.times)

    def _simulate(self, parameters, times):
        r"""
        PINTS-configured wrapper for the simulation method of the Warwick
        model.

        Extends the :meth:`_split_simulate`. Always apply methods
        :meth:`set_regions`, :meth:`set_age_groups`, :meth:`read_contact_data`
        and :meth:`read_regional_data` before running the
        :meth:`WarwickSEIRModel.simulate`.

        Parameters
        ----------
        parameters : list
            Long vector format of the quantities that characterise the Warwick
            SEIR model in this order:
            (1) index of region for which we wish to simulate,
            (2) initial conditions matrices classifed by age (column name) and
            region (row name) for each type of compartment (s, eF, eSD, eSU,
            eQ, If, Isd, Isu, Iqf, Iqs, Af, As, aQ, _),
            (3) the age-dependent susceptibility of individuals to infection
            (sig),
            (4) the reduction in the transmission rate of infection for
            asymptomatic individuals (tau),
            (5) the rate of progression to infectious disease (eps),
            (6) the recovery rate (gamma),
            (7) the age-dependent probability of displaying
            symptoms (d),
            (8) the household quarantine proportion (H) and
            (9) the type of solver implemented by the :meth:`scipy.solve_ivp`.
            Splited into the formats necessary for the :meth:`_simulate`
            method.
        times : list
            List of time points at which we wish to evaluate the ODEs
            system.

        Returns
        -------
        numpy.array
            Age-structured output matrix of the simulation for the specified
            region.

        """
        # Number of regions and age groups
        self._num_ages = self.house_matrices_contact[0]._num_a_groups

        n_ages = self._num_ages
        n_reg = len(self.regions)

        start_index = n_reg * ((len(self._output_names)-1) * n_ages) + 1

        # Separate list of parameters into the structures needed for the
        # simulation
        my_parameters = []

        # Add index of region
        my_parameters.append(parameters[0])

        # Add initial conditions for the s, e1F, e1SD, e1SU, e1Q, e2F, e2SD,
        # e2SU, e2Q, e3F, e3SD, e3SU, e3Q, iF, iSD, iSU, iQF, iQS, aF, aS, aQ
        # and r compartments
        for c in range(len(self._output_names)-1):
            initial_cond_comp = []
            for r in range(n_reg):
                ind = r * n_ages + n_reg * c * n_ages + 1
                initial_cond_comp.append(
                    parameters[ind:(ind + n_ages)])
            my_parameters.append(initial_cond_comp)

        # Add other parameters
        my_parameters.append(parameters[start_index:(start_index + n_ages)])
        my_parameters.extend(parameters[
            (start_index + n_ages):(start_index + 3 + n_ages)])
        my_parameters.append(parameters[
            (start_index + 3 + n_ages):(start_index + 3 + 2 * n_ages)])
        my_parameters.append(parameters[
            (start_index + 3 + 2 * n_ages):(
                start_index + 3 + 2 * n_ages + n_reg)])

        # Add method
        method = parameters[start_index + 3 + 2 * n_ages + n_reg]

        return self._split_simulate(my_parameters,
                                    times,
                                    method)

    def _check_output_format(self, output):
        """
        Checks correct format of the output matrix.

        Parameters
        ----------
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the WarwickSEIRModel.

        """
        if np.asarray(output).ndim != 2:
            raise ValueError(
                'Model output storage format must be 2-dimensional.')
        if np.asarray(output).shape[0] != self._times.shape[0]:
            raise ValueError(
                    'Wrong number of rows for the model output.')
        if np.asarray(output).shape[1] != 23 * self._num_ages:
            raise ValueError(
                    'Wrong number of columns for the model output.')
        for r in np.asarray(output):
            for _ in r:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'Model output elements must be integer or float.')

    def new_infections(self, output):
        """
        Computes number of new symptomatic infections at each time step in
        specified region, given the simulated timeline of susceptible number
        of individuals, for all age groups in the model.

        It uses an output of the simulation method for the WarwickSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        output : numpy.array
            Age-structured output of the simulation method for the
            WarwickSEIRModel.

        Returns
        -------
        nunmpy.array
            Age-structured matrix of the number of new symptomatic infections
            from the simulation method for the WarwickSEIRModel.

        Notes
        -----
        Always run :meth:`WarwickSEIRModel.simulate` before running this one.

        """
        # Check correct format of parameters
        self._check_output_format(output)

        # Read parameters of the system
        eps, d = self._c[2], self._c[4]

        d_infec = np.empty((self._times.shape[0], self._num_ages))

        for ind, t in enumerate(self._times.tolist()):
            # Read from output
            e3F = output[ind, :][(9*self._num_ages):(10*self._num_ages)]
            e3SD = output[ind, :][(10*self._num_ages):(11*self._num_ages)]
            e3SU = output[ind, :][(11*self._num_ages):(12*self._num_ages)]
            e3Q = output[ind, :][(12*self._num_ages):(13*self._num_ages)]

            # fraction of new infectives in delta_t time step
            d_infec[ind, :] = 3 * eps * np.multiply(d, e3F + e3SD + e3SU + e3Q)

            if np.any(d_infec[ind, :] < 0):  # pragma: no cover
                d_infec[ind, :] = np.zeros_like(d_infec[ind, :])

        return d_infec

    def _check_new_infections_format(self, new_infections):
        """
        Checks correct format of the new symptomatic infections matrix.

        Parameters
        ----------
        new_infections : numpy.array
            Age-structured matrix of the number of new symptomatic infections.

        """
        if np.asarray(new_infections).ndim != 2:
            raise ValueError(
                'Model new infections storage format must be 2-dimensional.')
        if np.asarray(new_infections).shape[0] != self._times.shape[0]:
            raise ValueError(
                    'Wrong number of rows for the model new infections.')
        if np.asarray(new_infections).shape[1] != self._num_ages:
            raise ValueError(
                    'Wrong number of columns for the model new infections.')
        for r in np.asarray(new_infections):
            for _ in r:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'Model`s new infections elements must be integer or \
                            float.')

    def new_hospitalisations(self, new_infections, pDtoH, dDtoH):
        """
        Computes number of new hospital admissions at each time step in
        specified region, given the simulated timeline of detectable
        symptomatic infected number of individuals, for all age groups
        in the model.

        It uses the array of the number of new symptomatic infections, obtained
        from an output of the simulation method for the WarwickSEIRModel,
        a distribution of the delay between onset of symptoms and
        hospitalisation, as well as the fraction of the number of symptomatic
        cases that end up hospitalised.

        Parameters
        ----------
        new_infections : numpy.array
            Age-structured array of the daily number of new symptomatic
            infections.
        pDtoH : list
            Age-dependent fractions of the number of symptomatic cases that
            end up hospitalised.
        dDtoH : list
            Distribution of the delay between onset of symptoms and
            hospitalisation. Must be normalised.

        Returns
        -------
        nunmpy.array
            Age-structured matrix of the number of new hospital admissions.

        Notes
        -----
        Always run :meth:`WarwickSEIRModel.simulate` before running this one.

        """
        n_daily_hosp = np.zeros((self._times.shape[0], self._num_ages))

        # Normalise dDtoH
        dDtoH = ((1/np.sum(dDtoH)) * np.asarray(dDtoH)).tolist()

        for ind, _ in enumerate(self._times.tolist()):
            if ind >= 30:
                n_daily_hosp[ind, :] = np.array(pDtoH) * np.sum(np.matmul(
                    np.diag(dDtoH[:30][::-1]),
                    new_infections[(ind-29):(ind+1), :]), axis=0)
            else:
                n_daily_hosp[ind, :] = np.array(pDtoH) * np.sum(np.matmul(
                    np.diag(dDtoH[:(ind+1)][::-1]),
                    new_infections[:(ind+1), :]), axis=0)

        for ind, _ in enumerate(self._times.tolist()):  # pragma: no cover
            if np.any(n_daily_hosp[ind, :] < 0):
                n_daily_hosp[ind, :] = np.zeros_like(n_daily_hosp[ind, :])

        return n_daily_hosp

    def check_new_hospitalisation_format(self, new_infections, pDtoH, dDtoH):
        """
        Checks correct format of the inputs of number of hospitalisation
        calculation.

        Parameters
        ----------
        new_infections : numpy.array
            Age-structured array of the daily number of new symptomatic
            infections.
        pDtoH : list
            Age-dependent fractions of the number of symptomatic cases that
            end up hospitalised.
        dDtoH : list
            Distribution of the delay between onset of symptoms and
            hospitalisation. Must be normalised.

        """
        self._check_new_infections_format(new_infections)

        if np.asarray(pDtoH).ndim != 1:
            raise ValueError('Fraction of the number of hospitalised \
                symptomatic cases storage format is 1-dimensional.')
        if np.asarray(pDtoH).shape[0] != self._num_ages:
            raise ValueError('Wrong number of fractions of the number of\
                hospitalised symptomatic cases .')
        for _ in pDtoH:
            if not isinstance(_, (int, float)):
                raise TypeError('Fraction of the number of hospitalised \
                    symptomatic cases must be integer or float.')
            if (_ < 0) or (_ > 1):
                raise ValueError('Fraction of the number of hospitalised \
                    symptomatic cases must be => 0 and <=1.')

        if np.asarray(dDtoH).ndim != 1:
            raise ValueError('Delays between onset of symptoms and \
                hospitalisation storage format is 1-dimensional.')
        if np.asarray(dDtoH).shape[0] < 30:
            raise ValueError('Wrong number of delays between onset of \
                symptoms and hospitalisation.')
        for _ in dDtoH:
            if not isinstance(_, (int, float)):
                raise TypeError('Delays between onset of symptoms and \
                    hospitalisation must be integer or float.')
            if (_ < 0) or (_ > 1):
                raise ValueError('Delays between onset of symptoms and \
                    hospitalisation must be => 0 and <=1.')
        if np.sum(dDtoH) != 1:
            raise ValueError('Distribution of delays between onset of\
                symptoms and hospitalisation must be normalised.')

    def new_icu(self, new_infections, pDtoI, dDtoI):
        """
        Computes number of new ICU admissions at each time step in
        specified region, given the simulated timeline of detectable
        symptomatic infected number of individuals, for all age groups
        in the model.

        It uses the array of the number of new symptomatic infections, obtained
        from an output of the simulation method for the WarwickSEIRModel,
        a distribution of the delay between onset of symptoms and
        admission to ICU, as well as the fraction of the number of symptomatic
        cases that end up in ICU.

        Parameters
        ----------
        new_infections : numpy.array
            Age-structured array of the daily number of new symptomatic
            infections.
        pDtoI : list
            Age-dependent fractions of the number of symptomatic cases that
            end up in ICU.
        dDtoI : list
            Distribution of the delay between onset of symptoms and
            admission to ICU. Must be normalised.

        Returns
        -------
        nunmpy.array
            Age-structured matrix of the number of new ICU admissions.

        Notes
        -----
        Always run :meth:`WarwickSEIRModel.simulate` before running this one.

        """
        n_daily_icu = np.zeros((self._times.shape[0], self._num_ages))

        # Normalise dDtoI
        dDtoI = ((1/np.sum(dDtoI)) * np.asarray(dDtoI)).tolist()

        for ind, _ in enumerate(self._times.tolist()):
            if ind >= 30:
                n_daily_icu[ind, :] = np.array(pDtoI) * np.sum(np.matmul(
                    np.diag(dDtoI[:30][::-1]),
                    new_infections[(ind-29):(ind+1), :]), axis=0)
            else:
                n_daily_icu[ind, :] = np.array(pDtoI) * np.sum(np.matmul(
                    np.diag(dDtoI[:(ind+1)][::-1]),
                    new_infections[:(ind+1), :]), axis=0)

        for ind, _ in enumerate(self._times.tolist()):  # pragma: no cover
            if np.any(n_daily_icu[ind, :] < 0):
                n_daily_icu[ind, :] = np.zeros_like(n_daily_icu[ind, :])

        return n_daily_icu

    def check_new_icu_format(self, new_infections, pDtoI, dDtoI):
        """
        Checks correct format of the inputs of number of ICU admissions
        calculation.

        Parameters
        ----------
        new_infections : numpy.array
            Age-structured array of the daily number of new symptomatic
            infections.
        pDtoI : list
            Age-dependent fractions of the number of symptomatic cases that
            end up in ICU.
        dDtoI : list
            Distribution of the delay between onset of symptoms and
            admission to ICU. Must be normalised.

        """
        self._check_new_infections_format(new_infections)

        if np.asarray(pDtoI).ndim != 1:
            raise ValueError('Fraction of the number of ICU admitted \
                symptomatic cases storage format is 1-dimensional.')
        if np.asarray(pDtoI).shape[0] != self._num_ages:
            raise ValueError('Wrong number of fractions of the number of\
                ICU admitted symptomatic cases .')
        for _ in pDtoI:
            if not isinstance(_, (int, float)):
                raise TypeError('Fraction of the number of ICU admitted \
                    symptomatic cases must be integer or float.')
            if (_ < 0) or (_ > 1):
                raise ValueError('Fraction of the number of ICU admitted \
                    symptomatic cases must be => 0 and <=1.')

        if np.asarray(dDtoI).ndim != 1:
            raise ValueError('Delays between onset of symptoms and \
                ICU admission storage format is 1-dimensional.')
        if np.asarray(dDtoI).shape[0] < 30:
            raise ValueError('Wrong number of delays between onset of \
                symptoms and ICU admission.')
        for _ in dDtoI:
            if not isinstance(_, (int, float)):
                raise TypeError('Delays between onset of symptoms and \
                    ICU admission must be integer or float.')
            if (_ < 0) or (_ > 1):
                raise ValueError('Delays between onset of symptoms and \
                    ICU admission must be => 0 and <=1.')
        if np.sum(dDtoI) != 1:
            raise ValueError('Distribution of delays between onset of\
                symptoms and ICU admission must be normalised.')

    def new_deaths(self, new_hospitalisation, pHtoDeath, dHtoDeath):
        """
        Computes number of new deaths at each time step in
        specified region, given the simulated timeline of hospitalised
        number of individuals, for all age groups in the model.

        It uses the array of the number of new symptomatic infections, obtained
        from an output of the simulation method for the WarwickSEIRModel,
        a distribution of the delay between onset of symptoms and
        admission to ICU, as well as the fraction of the number of hospitalised
        cases that end up dying.

        Parameters
        ----------
        new_hospitalisation : numpy.array
            Age-structured array of the daily number of new hospitalised
            cases.
        pHtoDeath : list
            Age-dependent fractions of the number of hospitalised cases that
            die.
        dHtoDeath : list
            Distribution of the delay between onset of hospitalisation and
            death. Must be normalised.

        Returns
        -------
        nunmpy.array
            Age-structured matrix of the number of new deaths from the
            simulation method for the WarwickSEIRModel.

        Notes
        -----
        Always run :meth:`WarwickSEIRModel.simulate` before running this one.

        """
        n_daily_deaths = np.zeros((self._times.shape[0], self._num_ages))

        # Normalise dHtoDeath
        dHtoDeath = ((1/np.sum(dHtoDeath)) * np.asarray(dHtoDeath)).tolist()

        for ind, _ in enumerate(self._times.tolist()):
            if ind >= 30:
                n_daily_deaths[ind, :] = np.array(pHtoDeath) * np.sum(
                    np.matmul(
                        np.diag(dHtoDeath[:30][::-1]),
                        new_hospitalisation[(ind-29):(ind+1), :]), axis=0)
            else:
                n_daily_deaths[ind, :] = np.array(pHtoDeath) * np.sum(
                    np.matmul(
                        np.diag(dHtoDeath[:(ind+1)][::-1]),
                        new_hospitalisation[:(ind+1), :]), axis=0)

        for ind, _ in enumerate(self._times.tolist()):  # pragma: no cover
            if np.any(n_daily_deaths[ind, :] < 0):
                n_daily_deaths[ind, :] = np.zeros_like(n_daily_deaths[ind, :])

        return n_daily_deaths

    def check_new_deaths_format(
            self, new_hospitalisation, pHtoDeath, dHtoDeath):
        """
        Checks correct format of the inputs of number of death
        calculation.

        Parameters
        ----------
        new_hospitalisation : numpy.array
            Age-structured array of the daily number of new hospitalised
            cases.
        pHtoDeath : list
            Age-dependent fractions of the number of hospitalised cases that
            die.
        dHtoDeath : list
            Distribution of the delay between onset of hospitalisation and
            death. Must be normalised.

        """
        self._check_new_infections_format(new_hospitalisation)

        if np.asarray(pHtoDeath).ndim != 1:
            raise ValueError('Fraction of the number of deaths \
                from hospitalised cases storage format is 1-dimensional.')
        if np.asarray(pHtoDeath).shape[0] != self._num_ages:
            raise ValueError('Wrong number of fractions of the number of\
                deaths from hospitalised cases.')
        for _ in pHtoDeath:
            if not isinstance(_, (int, float)):
                raise TypeError('Fraction of the number of deaths \
                from hospitalised cases must be integer or float.')
            if (_ < 0) or (_ > 1):
                raise ValueError('Fraction of the number of deaths \
                from hospitalised cases must be => 0 and <=1.')

        if np.asarray(dHtoDeath).ndim != 1:
            raise ValueError('Delays between hospital admission and \
                death storage format is 1-dimensional.')
        if np.asarray(dHtoDeath).shape[0] < 30:
            raise ValueError('Wrong number of delays between hospital \
                admission and death.')
        for _ in dHtoDeath:
            if not isinstance(_, (int, float)):
                raise TypeError('Delays between  hospital \
                    admission and death must be integer or float.')
            if (_ < 0) or (_ > 1):
                raise ValueError('Delays between  hospital \
                    admission and death must be => 0 and <=1.')
        if np.sum(dHtoDeath) != 1:
            raise ValueError('Distribution of delays between hospital \
                admission and death must be normalised.')

    def new_hospital_beds(self, new_hospitalisations, new_icu, tH, tItoH):
        """
        Computes number of hospital beds occupied at each time step in
        specified region, given the simulated timeline of detectable
        symptomatic infected number of individuals, for all age groups
        in the model.

        It uses the arrays of the number of new symptomatic infections
        admitted to hospital and ICU respectively, a distribution of the delay
        between onset of symptoms and hospitalisation, as well as the fraction
        of the number of symptomatic cases that end up hospitalised.

        Parameters
        ----------
        new_hospitalisations : numpy.array
            Age-structured array of the daily number of new symptomatic
            infections hospitalised.
        new_icu : numpy.array
            Age-structured array of the daily number of new symptomatic
            infections admitted to icu.
        tH : list
            Weighting distribution of the times spent in hospital by an
            admitted symptomatic case. Must be normalised.
        tItoH : list
            Weighting distribution of the times spent in icu before being
            moved to a non-icu bed by an admitted symptomatic case. Must be
            normalised.

        Returns
        -------
        nunmpy.array
            Age-structured matrix of the number of hospital beds occupied.

        Notes
        -----
        Always run :meth:`WarwickSEIRModel.simulate` before running this one.

        """
        n_hosp_occ = np.zeros((self._times.shape[0], self._num_ages))

        for ind, _ in enumerate(self._times.tolist()):
            if ind >= 30:
                n_hosp_occ[ind, :] = np.sum(np.matmul(
                    np.diag(tH[:30][::-1]),
                    new_hospitalisations[(ind-29):(ind+1), :]) +
                    np.matmul(
                    np.diag(tItoH[:30][::-1]),
                    new_icu[(ind-29):(ind+1), :]), axis=0)
            else:
                n_hosp_occ[ind, :] = np.sum(np.matmul(
                    np.diag(tH[:(ind+1)][::-1]),
                    new_hospitalisations[:(ind+1), :]) +
                    np.matmul(
                    np.diag(tItoH[:(ind+1)][::-1]),
                    new_icu[:(ind+1), :]), axis=0)

        for ind, _ in enumerate(self._times.tolist()):  # pragma: no cover
            if np.any(n_hosp_occ[ind, :] < 0):
                n_hosp_occ[ind, :] = np.zeros_like(n_hosp_occ[ind, :])

        return n_hosp_occ

    def check_new_hospital_beds_format(
            self, new_hospitalisations, new_icu, tH, tItoH):
        """
        Checks correct format of the inputs of number of hospital beds occupied
        calculation.

        Parameters
        ----------
        new_hospitalisations : numpy.array
            Age-structured array of the daily number of new symptomatic
            infections hospitalised.
        new_icu : numpy.array
            Age-structured array of the daily number of new symptomatic
            infections admitted to icu.
        tH : list
            Weighting distribution of the times spent in hospital by an
            admitted symptomatic case. Must be normalised.
        tItoH : list
            Weighting distribution of the times spent in icu before being
            moved to a non-icu bed by an admitted symptomatic case. Must be
            normalised.

        """
        self._check_new_infections_format(new_hospitalisations)
        self._check_new_infections_format(new_icu)

        if np.asarray(tH).ndim != 1:
            raise ValueError('Weighting distribution of the times spent in\
                hospital storage format is 1-dimensional.')
        if np.asarray(tH).shape[0] < 30:
            raise ValueError('Wrong number of weighting distribution of\
                the times spent in hospital.')
        for _ in tH:
            if not isinstance(_, (int, float)):
                raise TypeError('Weighting distribution of the times spent in\
                    hospital must be integer or float.')
            if (_ < 0) or (_ > 1):
                raise ValueError('Weighting distribution of the times spent in\
                    hospital must be => 0 and <=1.')
        if np.sum(tH) != 1:
            raise ValueError('Weighting distribution of the times spent in\
                hospital must be normalised.')

        if np.asarray(tItoH).ndim != 1:
            raise ValueError('Weighting distribution of the times spent in\
                icu before being moved to a non-icu bed storage format is\
                1-dimensional.')
        if np.asarray(tItoH).shape[0] < 30:
            raise ValueError('Wrong number of weighting distribution of\
                the times spent in icu before being moved to a non-icu bed.')
        for _ in tItoH:
            if not isinstance(_, (int, float)):
                raise TypeError('Weighting distribution of the times spent in\
                    icu before being moved to a non-icu bed must be integer\
                    or float.')
            if (_ < 0) or (_ > 1):
                raise ValueError('Weighting distribution of the times spent in\
                    icu before being moved to a non-icu bed must be => 0 and\
                    <=1.')
        if np.sum(tItoH) != 1:
            raise ValueError('Weighting distribution of the times spent in\
                icu before being moved to a non-icu bed must be normalised.')

    def new_icu_beds(self, new_icu, tI):
        """
        Computes number of ICU beds occupied at each time step in
        specified region, given the simulated timeline of detectable
        symptomatic infected number of individuals, for all age groups
        in the model.

        It uses the array of the number of new symptomatic infections
        admitted to ICU, as well as the weighting distribution of the times
        spent in hospital by an admitted symptomatic case.

        Parameters
        ----------
        new_icu : numpy.array
            Age-structured array of the daily number of new symptomatic
            infections admitted to icu.
        tI : list
            Weighting probability distribution of that an ICU
            admitted case is still in ICU q days later. Must be normalised.

        Returns
        -------
        nunmpy.array
            Age-structured matrix of the number of hospital beds occupied.

        Notes
        -----
        Always run :meth:`WarwickSEIRModel.simulate` before running this one.

        """
        n_daily_icu_beds = np.zeros((self._times.shape[0], self._num_ages))

        for ind, _ in enumerate(self._times.tolist()):
            if ind >= 30:
                n_daily_icu_beds[ind, :] = np.sum(np.matmul(
                    np.diag(tI[:30][::-1]),
                    new_icu[(ind-29):(ind+1), :]), axis=0)
            else:
                n_daily_icu_beds[ind, :] = np.sum(np.matmul(
                    np.diag(tI[:(ind+1)][::-1]),
                    new_icu[:(ind+1), :]), axis=0)

        for ind, _ in enumerate(self._times.tolist()):  # pragma: no cover
            if np.any(n_daily_icu_beds[ind, :] < 0):
                n_daily_icu_beds[ind, :] = np.zeros_like(n_daily_icu_beds[
                    ind, :])

        return n_daily_icu_beds

    def check_new_icu_beds_format(self, new_icu, tI):
        """
        Checks correct format of the inputs of number of ICU beds occupied
        calculation.

        Parameters
        ----------
        new_icu : numpy.array
            Age-structured array of the daily number of new symptomatic
            infections admitted to icu.
        tI : list
            Weighting probability distribution of that an ICU
            admitted case is still in ICU q days later. Must be normalised.

        """
        self._check_new_infections_format(new_icu)

        if np.asarray(tI).ndim != 1:
            raise ValueError('Weighting distribution of the times spent in\
                ICU storage format is 1-dimensional.')
        if np.asarray(tI).shape[0] < 30:
            raise ValueError('Wrong number of weighting distribution of\
                the times spent in ICU.')
        for _ in tI:
            if not isinstance(_, (int, float)):
                raise TypeError('Weighting distribution of the times spent in\
                    ICU must be integer or float.')
            if (_ < 0) or (_ > 1):
                raise ValueError('Weighting distribution of the times spent in\
                    ICU must be => 0 and <=1.')
        if np.sum(tI) != 1:
            raise ValueError('Weighting distribution of the times spent in\
                ICU must be normalised.')

    def loglik_deaths(self, obs_death, new_deaths, niu, k):
        r"""
        Computes the log-likelihood for the number of deaths at time step
        :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of deaths is assumed to be distributed according to
        a negative binomial distribution with mean :math:`\mu_{r,t_k,i}`
        and variance :math:`\mu_{r,t_k,i} (\nu + 1)`, where
        :math:`\mu_{r,t_k,i}` is the number of new deaths in specified region,
        for age group :math:`i` on day :math:`t_k`.

        It uses new_infections output of the simulation method for the
        WarwickSEIRModel, taking all the rest of the parameters necessary for
        the computation from the way its simulation has been fitted.

        Parameters
        ----------
        obs_death : list
            List of number of observed deaths by age group at time point k.
        new_deaths : numpy.array
            Age-structured matrix of the number of new deaths from the
            simulation method for the WarwickSEIRModel.
        niu : float
            Dispersion factor for the negative binomial distribution.
        k : int
            Index of day for which we intend to sample the number of deaths for
            by age group.

        Returns
        -------
        numpy.array
            Age-structured matrix of log-likelihoods for the observed number
            of deaths in specified region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`WarwickSEIRModel.new_infections` and
        :meth:`WarwickSEIRModel.check_death_format` before running this one.

        """
        self._check_time_step_format(k)

        # Check correct format for observed number of deaths
        if np.asarray(obs_death).ndim != 1:
            raise ValueError('Observed number of deaths by age category \
                storage format is 1-dimensional.')
        if np.asarray(obs_death).shape[0] != self._num_ages:
            raise ValueError('Wrong number of age groups for observed number \
                of deaths.')
        for _ in obs_death:
            if not isinstance(_, (int, np.integer)):
                raise TypeError('Observed number of deaths must be integer.')
            if _ < 0:
                raise ValueError('Observed number of deaths must be => 0.')

        if not hasattr(self, 'actual_deaths'):
            self.actual_deaths = [0] * 150
        self.actual_deaths[k] = sum(self.mean_deaths(k, new_deaths))

        # Compute mean of negative-binomial
        if k != 0:
            if np.sum(self.mean_deaths(k, new_deaths)) != 0:
                return nbinom.logpmf(
                    k=obs_death,
                    n=(1/niu) * self.mean_deaths(k, new_deaths),
                    p=1/(1+niu))
            else:
                return np.zeros(self._num_ages)
        else:
            return np.zeros(self._num_ages)

    def check_death_format(self, new_deaths, niu):
        """
        Checks correct format of the inputs of number of death calculation.

        Parameters
        ----------
        new_deaths : numpy.array
            Age-structured matrix of the number of new deaths from the
            simulation method for the WarwickSEIRModel.
        niu : float
            Dispersion factor for the negative binomial distribution.

        """
        self._check_new_infections_format(new_deaths)
        if not isinstance(niu, (int, float)):
            raise TypeError('Dispersion factor must be integer or float.')
        if niu <= 0:
            raise ValueError('Dispersion factor must be > 0.')

    def mean_deaths(self, k, new_deaths):
        """
        Computes the mean of the negative binomial distribution used to
        calculate number of deaths for specified age group.

        Parameters
        ----------
        k : int
            Index of day for which we intend to sample the number of deaths for
            by age group.
        new_deaths : numpy.array
            Age-structured matrix of the number of new deaths from the
            simulation method for the WarwickSEIRModel.

        Returns
        -------
        numpy.array
            Age-structured matrix of the expected number of deaths to be
            observed in specified region at time :math:`t_k`.

        """
        return new_deaths[k, :]

    def samples_deaths(self, new_deaths, niu, k):
        r"""
        Computes samples for the number of deaths at time step
        :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of deaths is assumed to be distributed according to
        a negative binomial distribution with mean :math:`\mu_{r,t_k,i}`
        and variance :math:`\mu_{r,t_k,i} (\nu + 1)`, where
        :math:`\mu_{r,t_k,i}` is the number of new deaths in specified region,
        for age group :math:`i` on day :math:`t_k`.

        It uses an output of the simulation method for the WarwickSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        new_deaths : numpy.array
            Age-structured matrix of the number of new deaths from the
            simulation method for the WarwickSEIRModel.
        niu : float
            Dispersion factor for the negative binomial distribution.
        k : int
            Index of day for which we intend to sample the number of deaths for
            by age group.

        Returns
        -------
        numpy.array
            Age-structured matrix of sampled number of deaths in specified
            region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`WarwickSEIRModel.new_infections` and
        :meth:`WarwickSEIRModel.check_death_format` before running this one.

        """
        self._check_time_step_format(k)

        # Compute mean of negative-binomial
        if k != 0:
            if np.sum(self.mean_deaths(k, new_deaths)) != 0:
                return nbinom.rvs(
                    n=(1/niu) * self.mean_deaths(k, new_deaths),
                    p=1/(1+niu))
            else:
                return np.zeros(self._num_ages)
        else:
            return np.zeros_like(self.mean_deaths(k, new_deaths))

    def loglik_positive_tests(self, obs_pos, output, tests, sens, spec, k):
        r"""
        Computes the log-likelihood for the number of positive tests at time
        step :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of positive tests is assumed to be distributed according to
        a binomial distribution with parameters :math:`n = n_{r,t_k,i}` and

        .. math::
            p = k_{sens} (1-\frac{S_{r,t_k,i}}{N_{r,i}}) + (
                1-k_{spec}) \frac{S_{r,t_k,i}}{N_{r,i}}

        where :math:`n_{r,t_k,i}` is the number of tests conducted for
        people in age group :math:`i` in specified region :math:`r` at time
        atep :math:`t_k`, :math:`k_{sens}` and :math:`k_{spec}` are the
        sensitivity and specificity respectively of a test, while
        is the probability of demise :math:`k-l` days after infection and
        :math:`\delta_{r,t_l,i}^{infec}` is the number of new infections
        in specified region, for age group :math:`i` on day :math:`t_l`.

        It uses an output of the simulation method for the WarwickSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        obs_pos : list
            List of number of observed positive test results by age group at
            time point k.
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the WarwickSEIRModel.
        tests : list
            List of conducted tests in specified region and at time point k
            classifed by age groups.
        sens : float or int
            Sensitivity of the test (or ratio of true positives).
        spec : float or int
            Specificity of the test (or ratio of true negatives).
        k : int
            Index of day for which we intend to sample the number of positive
            test results by age group.

        Returns
        -------
        numpy.array
            Age-structured matrix of log-likelihoods for the observed number
            of positive test results for each age group in specified region at
            time :math:`t_k`.

        Notes
        -----
        Always run :meth:`WarwickSEIRModel.simulate` and
        :meth:`WarwickSEIRModel.check_positives_format` before running this
        one.

        """
        self._check_time_step_format(k)

        # Check correct format for observed number of positive results
        if np.asarray(obs_pos).ndim != 1:
            raise ValueError('Observed number of postive tests results by age \
                category storage format is 1-dimensional.')
        if np.asarray(obs_pos).shape[0] != self._num_ages:
            raise ValueError('Wrong number of age groups for observed number \
                of postive tests results.')
        for _ in obs_pos:
            if not isinstance(_, (int, np.integer)):
                raise TypeError('Observed number of postive tests results must\
                    be integer.')
            if _ < 0:
                raise ValueError('Observed number of postive tests results \
                    must be => 0.')

        # Check correct format for number of tests based on the observed number
        # of positive results
        for i, _ in enumerate(tests):
            if _ < obs_pos[i]:
                raise ValueError('Not enough performed tests for the number \
                    of observed positives.')

        a = self._num_ages
        # Compute parameters of binomial
        suscep = output[k, :a]
        pop = 0
        for i in range(6):
            pop += output[k, (i*a):((i+1)*a)]

        return binom.logpmf(
            k=obs_pos,
            n=tests,
            p=self.mean_positives(sens, spec, suscep, pop))

    def _check_time_step_format(self, k):
        if not isinstance(k, int):
            raise TypeError('Index of time of computation of the \
                log-likelihood must be integer.')
        if k < 0:
            raise ValueError('Index of time of computation of the \
                log-likelihood must be >= 0.')
        if k >= self._times.shape[0]:
            raise ValueError('Index of time of computation of the \
                log-likelihood must be within those considered in the output.')

    def check_positives_format(self, output, tests, sens, spec):
        """
        Checks correct format of the inputs of number of positive test results
        calculation.

        Parameters
        ----------
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the WarwickSEIRModel.
        tests : list
            List of conducted tests in specified region and at time point k
            classifed by age groups.
        sens : float or int
            Sensitivity of the test (or ratio of true positives).
        spec : float or int
            Specificity of the test (or ratio of true negatives).

        """
        self._check_output_format(output)
        if np.asarray(tests).ndim != 2:
            raise ValueError('Number of tests conducted by age category \
                storage format is 2-dimensional.')
        if np.asarray(tests).shape[1] != self._num_ages:
            raise ValueError('Wrong number of age groups for observed number \
                of tests conducted.')
        for i in tests:
            for _ in i:
                if not isinstance(_, (int, np.integer)):
                    raise TypeError('Number of tests conducted must be \
                        integer.')
                if _ < 0:
                    raise ValueError('Number of tests conducted ratio must \
                        be => 0.')
        if not isinstance(sens, (int, float)):
            raise TypeError('Sensitivity must be integer or float.')
        if (sens < 0) or (sens > 1):
            raise ValueError('Sensitivity must be >= 0 and <=1.')
        if not isinstance(spec, (int, float)):
            raise TypeError('Specificity must be integer or float.')
        if (spec < 0) or (spec > 1):
            raise ValueError('Specificity must be >= 0 and >=1.')

    def mean_positives(self, sens, spec, suscep, pop):
        """
        Computes the mean of the binomial distribution used to
        calculate number of positive test results for specified age group.

        Parameters
        ----------
        sens : float or int
            Sensitivity of the test (or ratio of true positives).
        spec : float or int
            Specificity of the test (or ratio of true negatives).
        suscep : numpy.array
            Age-structured matrix of the current number of susceptibles
            in the population.
        pop : numpy.array
            Age-structured matrix of the current number of individuals
            in the population.

        Returns
        -------
        numpy.array
            Age-structured matrix of the expected number of positive test
            results to be observed in specified region at time :math:`t_k`.

        """
        return sens * (1-np.divide(suscep, pop)) + (1-spec) * np.divide(
            suscep, pop)

    def samples_positive_tests(self, output, tests, sens, spec, k):
        r"""
        Computes the samples for the number of positive tests at time
        step :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of positive tests is assumed to be distributed according to
        a binomial distribution with parameters :math:`n = n_{r,t_k,i}` and

        .. math::
            p = k_{sens} (1-\frac{S_{r,t_k,i}}{N_{r,i}}) + (
                1-k_{spec}) \frac{S_{r,t_k,i}}{N_{r,i}}

        where :math:`n_{r,t_k,i}` is the number of tests conducted for
        people in age group :math:`i` in specified region :math:`r` at time
        atep :math:`t_k`, :math:`k_{sens}` and :math:`k_{spec}` are the
        sensitivity and specificity respectively of a test, while
        is the probability of demise :math:`k-l` days after infection and
        :math:`\delta_{r,t_l,i}^{infec}` is the number of new infections
        in specified region, for age group :math:`i` on day :math:`t_l`.

        It uses an output of the simulation method for the WarwickSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the WarwickSEIRModel.
        tests : list
            List of conducted tests in specified region and at time point k
            classifed by age groups.
        sens : float or int
            Sensitivity of the test (or ratio of true positives).
        spec : float or int
            Specificity of the test (or ratio of true negatives).
        k : int
            Index of day for which we intend to sample the number of positive
            test results by age group.

        Returns
        -------
        numpy.array
            Age-structured matrix of sampled number of positive test results
            in specified region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`WarwickSEIRModel.simulate` and
        :meth:`WarwickSEIRModel.check_positives_format` before running this
        one.

        """
        self._check_time_step_format(k)

        a = self._num_ages
        # Compute parameters of binomial
        suscep = output[k, :a]
        pop = 0
        for i in range(6):
            pop += output[k, (i*a):((i+1)*a)]

        return binom.rvs(
            n=tests,
            p=self.mean_positives(sens, spec, suscep, pop))

    def compute_transition_matrix(self):
        """
        Computes the transition matrix of the Warwick-Household model.

        Returns
        -------
        numpy.array
            Transition matrix of the Warwick-Household model
            in specified region at time :math:`t_k`.

        """
        a = self._num_ages
        Zs = np.zeros((a, a))

        self._inv_trans_matrix = []

        # Read parameters of the system
        eps, gamma, d, h_all = self._c[2:6]

        # Read the social distancing parameters of the system
        for k in range(self._times[0], self._times[-1]+1):
            phi = self._compute_soc_dist_parameters(k)[1]

            h = h_all[self._region-1] * phi

            # Pre-compute block-matrices
            eps_3 = 3 * eps * np.identity(a)
            gamma_m = gamma * np.identity(a)
            d_eps_3 = 3 * eps * np.diag(d)
            one_d_eps_3 = 3 * eps * np.diag(1-np.array(d))
            H_d_eps_3 = 3 * eps * h * np.diag(d)
            one_H_d_eps_3 = 3 * eps * (1-h) * np.diag(d)

            sigma_matrix = np.block(
                [[-eps_3, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                    Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
                 [eps_3, -eps_3, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                    Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
                 [Zs, eps_3, -eps_3, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                    Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
                 [Zs, Zs, Zs, -eps_3, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                    Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
                 [Zs, Zs, Zs, eps_3, -eps_3, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                    Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
                 [Zs, Zs, Zs, Zs, eps_3, -eps_3, Zs, Zs, Zs, Zs, Zs, Zs,
                    Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
                 [Zs, Zs, Zs, Zs, Zs, Zs, -eps_3, Zs, Zs, Zs, Zs, Zs,
                    Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
                 [Zs, Zs, Zs, Zs, Zs, Zs, eps_3, -eps_3, Zs, Zs, Zs, Zs,
                    Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
                 [Zs, Zs, Zs, Zs, Zs, Zs, Zs, eps_3, -eps_3, Zs, Zs, Zs,
                    Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
                 [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, -eps_3, Zs, Zs,
                    Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
                 [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, eps_3, -eps_3, Zs,
                    Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
                 [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, eps_3, -eps_3,
                    Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
                 [Zs, Zs, one_H_d_eps_3, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                    -gamma_m, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
                 [Zs, Zs, one_d_eps_3, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                    Zs, -gamma_m, Zs, Zs, Zs, Zs, Zs, Zs],
                 [Zs, Zs, H_d_eps_3, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                    Zs, Zs, -gamma_m, Zs, Zs, Zs, Zs, Zs],
                 [Zs, Zs, Zs, Zs, Zs, d_eps_3, Zs, Zs, Zs, Zs, Zs, Zs,
                    Zs, Zs, Zs, -gamma_m, Zs, Zs, Zs, Zs],
                 [Zs, Zs, Zs, Zs, Zs, one_d_eps_3, Zs, Zs, one_d_eps_3, Zs, Zs,
                    Zs, Zs, Zs, Zs, Zs, -gamma_m, Zs, Zs, Zs],
                 [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, one_H_d_eps_3, Zs, Zs, Zs,
                    Zs, Zs, Zs, Zs, Zs, -gamma_m, Zs, Zs],
                 [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, H_d_eps_3, Zs, Zs, d_eps_3,
                    Zs, Zs, Zs, Zs, Zs, Zs, -gamma_m, Zs],
                 [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, one_d_eps_3,
                    Zs, Zs, Zs, Zs, Zs, Zs, Zs, -gamma_m]])

            self._inv_trans_matrix.append(np.linalg.inv(sigma_matrix))

    def compute_rt_trajectory(self, output, k):
        """
        Computes the time-dependent reproduction at time :math:`t_k`
        from the Warwick-Household model.

        Parameters
        ----------
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the WarwickSEIRModel.
        k : int
            Index of day for which we intend to sample the number of positive
            test results by age group.

        Returns
        -------
        float
            The reproduction number in specified region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`WarwickSEIRModel.simulate`,
        :meth:`WarwickSEIRModel.check_positives_format` and
        :meth:`WarwickSEIRModel.compute_transistion_matrix` before running this
        one.

        """
        self._check_time_step_format(k)
        r = self._region
        a = self._num_ages
        Zs = np.zeros((a, a))

        # Split compartments into their types
        suscep = output[k, :a]

        # Read parameters of the system
        sig, tau = self._c[:2]

        # Read the social distancing parameters of the system
        theta, phi, q_H, q_S, q_W, q_O = self._compute_soc_dist_parameters(k+1)

        # Identify the appropriate contact matrix for the ODE system
        house_cont_mat = \
            self.house_contacts_timeline.identify_current_contacts(r, k+1)
        school_cont_mat = \
            self.school_contacts_timeline.identify_current_contacts(r, k+1)
        work_cont_mat = \
            self.work_contacts_timeline.identify_current_contacts(r, k+1)
        other_cont_mat = \
            self.other_contacts_timeline.identify_current_contacts(r, k+1)

        house_cont_mat = 1.3 * (1 - phi + phi * q_H) * house_cont_mat
        nonhouse_cont_mat = (1 - phi + phi * q_S) * school_cont_mat + \
            ((1 - phi + phi * q_W) * (
                1 - theta + theta * (1 - phi + phi * q_O))) * work_cont_mat + \
            ((1 - phi + phi * q_O)**2) * other_cont_mat

        # Pre-compute block-matrices
        cH = np.multiply(
            suscep, (1 / self._N[r-1]) * np.multiply(sig, house_cont_mat))
        cN = np.multiply(
            suscep, (1 / self._N[r-1]) * np.multiply(sig, nonhouse_cont_mat))

        t_matrix = np.block(
            [[Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                cN, tau * cN, Zs, cN, tau * cN, cN, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                cH, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, cH, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, cH, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs],
             [Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs,
                Zs, Zs, Zs, Zs, Zs, Zs, Zs, Zs]])

        # Compute the next-generation matrix
        next_gen_matrix = - np.matmul(t_matrix, self._inv_trans_matrix[k])

        return np.max(np.absolute(np.linalg.eigvals(next_gen_matrix)))

    def compute_rt_trajectory_paper(self, k):
        """
        Computes the time-dependent reproduction at time :math:`t_k`
        from the Warwick-Household model.

        Parameters
        ----------
        k : int
            Index of day for which we intend to sample the number of positive
            test results by age group.

        Returns
        -------
        float
            The basic reproduction number in specified region at time
            :math:`t_k`.

        """
        self._check_time_step_format(k)
        r = self._region
        a = self._num_ages

        # Read the social distancing parameters of the system
        theta, phi, q_H, q_S, q_W, q_O = self._compute_soc_dist_parameters(k+1)

        house_cont_mat = \
            self.house_contacts_timeline.identify_current_contacts(r, k+1)
        school_cont_mat = \
            self.school_contacts_timeline.identify_current_contacts(r, k+1)
        work_cont_mat = \
            self.work_contacts_timeline.identify_current_contacts(r, k+1)
        other_cont_mat = \
            self.other_contacts_timeline.identify_current_contacts(r, k+1)

        house_cont_mat = 1.3 * (1 - phi + phi * q_H) * house_cont_mat
        nonhouse_cont_mat = (1 - phi + phi * q_S) * school_cont_mat + \
            ((1 - phi + phi * q_W) * (
                1 - theta + theta * (1 - phi + phi * q_O))) * work_cont_mat + \
            ((1 - phi + phi * q_O)**2) * other_cont_mat

        M_from_to = house_cont_mat + nonhouse_cont_mat

        M_from_to_HAT = np.zeros_like(M_from_to)

        # Read parameters of the system
        sigma, tau, _, __, d = self._c[:5]

        tau = tau * np.ones(a)

        for f in range(a):
            for t in range(a):
                M_from_to_HAT[f, t] = \
                    M_from_to[f, t] * d[t] * sigma[t] * (
                        1 + tau[f] * (1 - d[f]) / d[f])

        return np.max(np.absolute(np.linalg.eigvals(M_from_to_HAT)))
