#
# PheSEIRModel Class
#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""
"""

from itertools import chain

import numpy as np
from scipy.integrate import solve_ivp

import epimodels as em


class PheSEIRModel(object):
    r"""
    ODE model: deterministic SEIR
    The SEIR Model has four compartments:
    susceptible individuals (:math:`S`),
    exposed but not yet infectious (:math:`E`),
    infectious (:math:`I`) and recovered (:math:`R`):
    .. math::
        \frac{dS(t)}{dt} = -\beta S(t)I(t),
    .. math::
        \frac{dE(t)}{dt} = \beta S(t)I(t) - \kappa E(t),
    .. math::
        \frac{dI(t)}{dt} = \kappa E(t) - \gamma I(t),
    .. math::
        \frac{dR(t)}{dt} = \gamma I(t),
    where :math:`S(0) = S_0, E(0) = E_0, I(O) = I_0, R(0) = R_0`
    are also parameters of the model.
    Extends :class:`ForwardModel`.
    """

    def __init__(self):
        super(PheSEIRModel, self).__init__()

        # Assign default values
        self._output_names = ['S', 'E1', 'E2', 'I1', 'I2', 'R', 'Incidence']
        self._parameter_names = [
            'S0', 'E10', 'E20', 'I10', 'I20', 'R0', 'alpha', 'beta', 'gamma'
        ]
        # The default number of outputs is 7,
        # i.e. S, E1, E2, I1, I2, R and Incidence
        self._n_outputs = len(self._output_names)
        # The default number of outputs is 7,
        # i.e. 4 initial conditions and 3 parameters
        self._n_parameters = len(self._parameter_names)

        self._contacts = matrices_contact
        self._region = matrices_region

    def n_outputs(self):
        """
        Returns the number of output.

        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the number of parameters.

        """
        return self._n_parameters

    def output_names(self):
        """
        Returns the (selected) output names.

        """
        names = [self._output_names[x] for x in self._output_indices]
        return names

    def parameter_names(self):
        """
        Returns the parameter names.

        """
        return self._parameter_names

    def set_outputs(self, outputs):
        """
        Checks existence of outputs.

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

    def _right_hand_side(self, t, r, y, c, num_a_groups):
        """
        Constructs the derivative functions of the system of ODEs for given one
        region and one .

        Assuming y = [S, E1, E2, I1, I2, R] (the dependent variables in the
        model)
        Assuming the parameters are ordered like
        parameters = [S0, E10, E20, I10, I20, R0, beta, kappa, gamma]
        Let c = [beta, kappa, gamma]
          = [parameters[0], parameters[1], parameters[2]],
        then beta = c[0], kappa = c[1], gamma = c[2].

        """
        a = num_a_groups
        s, e1, e2, i1, i2, _ = (
            y[:a], y[a:(2*a)], y[(2*a):(3*a)],
            y[(3*a):(4*a)], y[(4*a):(5*a)], y[(5*a):])

        # print(s, e1, e2, i1, i2, _)
        beta, dL, dI = c
        kappa = 2/dL
        gamma = 2/dI
        b = self.infectivity_timeline.compute_prob_infectivity_matrix(
            r, t, s, beta)
        lam = np.empty_like(s)
        for i, l in enumerate(lam):
            prod = 0
            for j, _ in enumerate(lam):
                prod *= (1-b[i, j])**(i1[j]+i2[j])
            lam[i] = 1-prod

        lam_times_s = np.multiply(lam, np.asarray(s))
        dydt = np.concatenate((
            -lam_times_s, lam_times_s - kappa * np.asarray(e1),
            kappa * np.asarray(e1) - kappa * np.asarray(e2),
            kappa * np.asarray(e2) - gamma * np.asarray(i1),
            gamma * np.asarray(i1) - gamma * np.asarray(i2),
            gamma * np.asarray(i2)))

        return dydt

    def _my_solver(self, times, num_a_groups):
        """
        Parameters
        ----------
        times
            (list) List of time points at which
        """

        s, e1, e2, i1, i2, _ = np.asarray(self._y_init)[:, self._region-1]
        beta, dL, dI = self._c
        delta_t = self._delta_t
        kappa = delta_t * 2/dL
        gamma = delta_t * 2/dI

        solution = np.ones((len(times), num_a_groups*6))

        for ind, t in enumerate(times):
            b = self.infectivity_timeline.compute_prob_infectivity_matrix(
                self._region, t, s, beta)

            lam = np.empty_like(s)
            for i, l in enumerate(lam):
                prod = 0
                for j, _ in enumerate(lam):
                    prod *= (1-b[i, j])**(i1[j]+i2[j])
                lam[i] = 1-prod

            s_ = np.multiply(
                np.asarray(s), (np.ones_like(lam) - delta_t * lam))
            e1_ = (1 - kappa) * np.asarray(e1) + np.multiply(
                np.asarray(s), delta_t * lam)
            e2_ = (1 - kappa) * np.asarray(e2) + kappa * np.asarray(e1)
            i1_ = (1 - gamma) * np.asarray(i1) + kappa * np.asarray(e2)
            i2_ = (1 - gamma) * np.asarray(i2) + kappa * np.asarray(i1)
            r_ = kappa * np.asarray(i1)

            s, e1, e2, i1, i2, _ = (
                s_.tolist(), e1_.tolist(), e2_.tolist(),
                i1_.tolist(), i2_.tolist(), r_.tolist())

            solution[ind, :] = tuple(chain(s, e1, e2, i1, i2, _))

        return({'y': np.transpose(solution)})

    def _RK45_solver(self, times, num_a_groups, method):
        """

        Parameters
        ----------
        times
            (list) List of time points at which
        """
        # Initial conditions
        si, e1i, e2i, i1i, i2i, _i = np.asarray(
            self._y_init)[:, self._region-1]
        init_cond = list(
            chain(
                si.tolist(), e1i.tolist(), e2i.tolist(),
                i1i.tolist(), i2i.tolist(), _i.tolist()))

        # Solve the system of ODEs
        sol = solve_ivp(
            lambda t, y: self._right_hand_side(
                t, self._region, y, self._c, num_a_groups),
            [times[0], times[-1]], init_cond, method=method, t_eval=times)
        return sol

    def simulate(
            self, parameters, times, matrices_contact, time_changes_contact,
            regions, initial_r, matrices_region, time_changes_region,
            method):
        """
        """
        self._region = parameters[0]
        self._y_init = parameters[1:7]
        self._c = parameters[7:10]
        self._delta_t = parameters[10]
        self.infectivity_timeline = em.MultiTimesInfectivity(
            matrices_contact,
            time_changes_contact,
            regions,
            matrices_region,
            time_changes_region,
            initial_r,
            self._c[2],
            self._y_init[0])

        self._num_ages = matrices_contact[0]._num_a_groups

        if method == 'my-solver':
            sol = self._my_solver(times, self._num_ages)
        else:
            sol = self._RK45_solver(times, self._num_ages, method)

        output = sol['y']

        # Age-based total infected is infectious 'i' plus recovered 'r'
        total_infected = output[
            (3*self._num_ages):(4*self._num_ages), :] + output[
            (4*self._num_ages):(5*self._num_ages), :] + output[
                (5*self._num_ages):, :]

        # Number of incidences is the increase in total_infected
        # between the time points (add a 0 at the front to
        # make the length consistent with the solution
        n_incidence = np.zeros((self._num_ages, len(times)))
        n_incidence[:, 1:] = total_infected[:, 1:] - total_infected[:, :-1]

        # Append n_incidence to output
        # Output is a matrix with rows being S, E1, I1, R1 and Incidence
        output = np.concatenate((output, n_incidence), axis=0)

        # Get the selected outputs
        self._output_indices = np.arange(self._n_outputs * self._num_ages)
        output = output[self._output_indices, :]

        return output.transpose()


regions = ['London', 'Cornwall']
age_groups = ['0-10', '10-25']

# Initial state of the system
contact_data_matrix_0 = np.array([[1, 0], [0, 3]])
contact_data_matrix_1 = np.array([[10, 5.2], [0, 3]])

region_data_matrix_0_0 = np.array([[0.5, 0], [0, 6]])
region_data_matrix_0_1 = np.array([[1, 10], [1, 0]])
region_data_matrix_1_0 = np.array([[0.5, 1.2], [0.29, 6]])
region_data_matrix_1_1 = np.array([[0.85, 1], [0.9, 6]])

susceptibles = [[5, 6], [7, 8]]
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

parameters = [
    1, susceptibles, [[0, 0], [0, 0]], [[0, 0], [0, 0]],
    [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], 1, 4, 4,
    0.5]

times = [1, 1.5, 2, 2.5, 3]

print(PheSEIRModel().simulate(
    parameters, times, matrices_contact, time_changes_contact,
    regions, initial_r, matrices_region, time_changes_region,
    method='my-solver'))
