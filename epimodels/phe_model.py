#
# PheSEIRModel Class
#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""
This script contains code for modelling the extended SEIR model created by
Public Health England and Univerity of Cambridge and which is the official
model used by the UK government for policy making.

It uses an extended version of an SEIR model and contact and region specific
matrices.

"""

from itertools import chain

import numpy as np
from scipy.integrate import solve_ivp

import epimodels as em


class PheSEIRModel(object):
    r"""PheSEIRModel Class:
    Base class for constructing the ODE model: deterministic SEIR used by the
    Public Health England to model the Covid-19 epidemic in UK based on region.

    The population is structured according to their age-group (:math:`i`) and
    region (:math:`r`) and every individual will belong to one of the
    compartments of the SEIR model.

    The general SEIR Model has four compartments: susceptible individuals
    (:math:`S`), exposed but not yet infectious (:math:`E`), infectious
    (:math:`I`) and recovered (:math:`R`).

    In the PHE model framework, the exposed and infectious compartments:
    .. math::
       :nowrap:

        \begin{eqnarray}
            \frac{dS(r, t, i)}{dt} = -\lambda_{r, t, i} S(r, t, i) \\
            \frac{dE_1(r, t, i)}{dt} = \lambda_{r, t, i} S(
                r, t, i) - \kappa E_1(r, t, i) \\
            \frac{dE_2(r, t, i)}{dt} = \kappa E_1(r, t, i) - \kappa E_2(
                r, t, i) \\
            \frac{dI_1(r, t, i)}{dt} = \kappa E_2(r, t, i) - \gamma I_1(
                r, t, i) \\
            \frac{dI_2(r, t, i)}{dt} = \gamma I_1(r, t, i) - \gamma I_2(
                r, t, i) \\
            \frac{dR(r, t, i)}{dt} = \gamma I_2(r, t, i)
        \end{eqnarray}

    where :math:`S(0) = S_0, E(0) = E_0, I(O) = I_0, R(0) = R_0`
    are also parameters of the model (evaluation at 0 refers to the
    compartments' structure at intial time.

    The parameter :math:`\lambda_{r, t, i}` is the time, age and region-varying
    rate with which susceptible individuals become infected, which in the
    context of the PHE model depends on contact and region-specific relative
    susceptibility matrices. The other two parameters, :math:`\kappa` and
    :math:`\gamma` are virsus specific and so do not depend with region, age or
    time

    .. math::
        \kappa = \frac{2}{d_L}

    .. math::
        \gamma = \frac{2}{d_I}

    where :math:`d_L` refers to mean latent period until disease onset and
    :math:`d_I` to mean period of infection.

    """

    def __init__(self):
        super(PheSEIRModel, self).__init__()

        # Assign default values
        self._output_names = ['S', 'E1', 'E2', 'I1', 'I2', 'R', 'Incidence']
        self._parameter_names = [
            'S0', 'E10', 'E20', 'I10', 'I20', 'R0', 'beta', 'kappa', 'gamma']

        # The default number of outputs is 7,
        # i.e. S, E1, E2, I1, I2, R and Incidence
        self._n_outputs = len(self._output_names)
        # The default number of outputs is 7,
        # i.e. 6 initial conditions and 3 parameters
        self._n_parameters = len(self._parameter_names)

        self._output_indices = np.arange(self._n_outputs)

    def n_outputs(self):
        """
        Returns the number of outputs.

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
        r"""
        Constructs the RHS of the equations of the system of ODEs for given a
        region and time point. The :math:`\lambda` parameter that accompanies
        the susceptible numbers is dependent on the current number of
        infectives and is computed using the updated multi-step infectivity
        matrix of the system according to the following formula

        .. math::
            \lambda_{r, t, i} = 1 - \prod_{j=1}^{n_A}[
                (1-b_{r,ij}^{t})^{I1(r,t,j)+I2(r,t,j)}]

        Parameters
        ----------
        t
            (float) Time point at which we compute the evaluation.
        r
            (int) The index of the region to which the current instance of the
            ODEs system refers.
        y
            (array) Array of all the compartments of the ODE system, segregated
            by age-group. It assumes y = [S, E1, E2, I1, I2, R] where each
            letter actually refers to all compartment of that type. (e.g. S
            refers to the compartments of all ages of susceptibles).
        c
            (list) List values used to compute the parameters of the ODEs
            system. It assumes c = [beta, kappa, gamma], where :math:`beta`
            encaplsulates temporal fluctuations in transmition for all ages.
        num_a_groups
            (int) Number of age groups in which the population is split. It
            refers to the number of compartments of each type.

        """
        # Read in the number of age-groups
        a = num_a_groups

        # Split compartments into their types
        s, e1, e2, i1, i2, _ = (
            y[:a], y[a:(2*a)], y[(2*a):(3*a)],
            y[(3*a):(4*a)], y[(4*a):(5*a)], y[(5*a):])

        # Read parameters of the system
        beta, dL, dI = c
        kappa = 2/dL
        gamma = 2/dI

        # And identify the appropriate MultiTimesInfectivity matrix for the
        # ODE system
        pos = np.where(self._times <= t)
        ind = pos[-1][-1]
        b = self.infectivity_timeline.compute_prob_infectivity_matrix(
            r, t, s, beta[self._region-1][ind])

        # Compute the current time, age and region-varying
        # rate with which susceptible individuals become infected
        lam = np.empty_like(s)
        for i, l in enumerate(lam):
            prod = 0
            for j, _ in enumerate(lam):
                prod *= (1-b[i, j])**(i1[j]+i2[j])
            lam[i] = 1-prod

        # Write actual RHS
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
        Computes the values in each compartment of the PHE ODEs system using
        a 'homemade' solver in the context of the discretised time step version
        of the model, as suggested in the paper in which it is referenced.

        Parameters
        ----------
        times
            (list) List of time points at which we wish to evaluate the ODEs
            system.
        num_a_groups
            (int) Number of age groups in which the population is split. It
            refers to the number of compartments of each type.

        """
        # Split compartments into their types
        s, e1, e2, i1, i2, _ = np.asarray(self._y_init)[:, self._region-1]

        # Read parameters of the system
        beta, dL, dI = self._c
        delta_t = self._delta_t
        kappa = delta_t * 2/dL
        gamma = delta_t * 2/dI

        eval_times = np.around(np.arange(
            times[0], times[-1]+delta_t, delta_t, dtype=np.float64), 5)
        eval_indices = np.where(
            np.array([(t in times) for t in eval_times]))[0].tolist()

        ind_in_times = []
        j = 0
        for i, t in enumerate(eval_times):
            if i >= eval_indices[j+1]:
                j += 1
            ind_in_times.append(j)

        solution = np.ones((len(eval_times), num_a_groups*6))

        for ind, t in enumerate(eval_times):
            # Add present vlaues of the compartments to the solutions
            solution[ind, :] = tuple(chain(s, e1, e2, i1, i2, _))

            # And identify the appropriate MultiTimesInfectivity matrix for the
            # ODE system
            b = self.infectivity_timeline.compute_prob_infectivity_matrix(
                self._region, t, s, beta[self._region-1][ind_in_times[ind]])

            # Compute the current time, age and region-varying
            # rate with which susceptible individuals become infected
            lam = np.empty_like(s)
            for i, l in enumerate(lam):
                prod = 0
                for j, _ in enumerate(lam):
                    prod *= (1-b[i, j])**(i1[j]+i2[j])
                lam[i] = 1-prod

            # Write down ODE system and compute new values for all compartments
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

        solution = solution[tuple(eval_indices), :]

        return({'y': np.transpose(solution)})

    def _scipy_solver(self, times, num_a_groups, method):
        """
        Computes the values in each compartment of the PHE ODEs system using
        the 'off-the-shelf' solver of the IVP from :module:`scipy`.

        Parameters
        ----------
        times
            (list) List of time points at which we wish to evaluate the ODEs
            system.
        num_a_groups
            (int) Number of age groups in which the population is split. It
            refers to the number of compartments of each type.
        method
            (string) The type of solver implemented by the
            :meth:`scipy.solve_ivp`.

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
        r"""
        Computes the number of individuals in each compartment at the given
        time points and specified region.

        Parameters
        ----------
        parameters
            (list) List of quantities that characterise the PHE SEIR model in
            this order: index of region for which we wish to simulate,
            initial conditions matrices classifed by age (column name) and
            region (row name) for each type of compartment (s, e1, e2, i1, i2,
            r), temporal and regional fluctuation matrix :math:`\beta`,
            mean latent period :math:`d_L`, mean infection period :math:`d_I`
            and time step for the 'homemade' solver.
        times
            (list) List of time points at which we wish to evaluate the ODEs
            system.
        matrices_contact
            (list of ContactMatrix) Time-dependent contact matrices used for
            the modelling.
        time_changes_contact
            (list) Times at which the next contact matrix recorded starts to be
            used. In increasing order.
        regions
            (list) List of region names for the region-specific relative
            susceptibility matrices.
        matrices_region
            (list of lists of RegionMatrix)) Time-dependent and region-specific
            relative susceptibility matrices used for the modelling.
        time_changes_region
            (list) Times at which the next instances of region-specific
            relative susceptibility matrices recorded start to be used. In
            increasing order.
        initial_r
            (list) List of initial values of the reproduction number by region.
        method
            (string) The type of solver implemented by the simulator.

        """
        # Check correct format of output
        if not isinstance(times, list):
            raise TypeError('Time points of evaluation must be given in a list \
                format.')
        for _ in times:
            if not isinstance(_, (int, float)):
                raise TypeError('Time points of evaluation must be integer or \
                    float.')
            if _ <= 0:
                raise ValueError('Time points of evaluation must be > 0.')
        if not isinstance(parameters, list):
            raise TypeError('Parametrs must be given in a list format.')
        if len(parameters) != 11:
            raise ValueError('List of parameters needs to be of length 11.')
        if not isinstance(parameters[0], int):
            raise TypeError('Index of region to evaluate must be integer.')
        if parameters[0] <= 0:
            raise ValueError('Index of region to evaluate must be >= 1.')
        if parameters[0] > len(regions):
            raise ValueError('Index of region to evaluate is out of bounds.')
        for _ in range(1, 7):
            if np.asarray(parameters[_]).ndim != 2:
                raise ValueError(
                    'Storage format for the numbers in each type of compartment\
                        must be 2-dimensional.')
            if np.asarray(parameters[_]).shape[0] != len(regions):
                raise ValueError(
                    'Number of age-split compartments of each type does not match \
                        that of the regions.')
            if np.asarray(parameters[_]).shape[1] != len(
                    matrices_contact[0].ages):
                raise ValueError(
                    'Number of age compartments of each type for given region does not match \
                        that of age groups.')
        if np.asarray(parameters[7]).ndim != 2:
            raise ValueError(
                'Storage format for the temporal and regional fluctuations in transmition\
                    must be 2-dimensional.')
        if np.asarray(parameters[7]).shape[0] != len(regions):
            raise ValueError(
                'Number of temporal and regional fluctuations in transmition does not match \
                    that of the regions.')
        if np.asarray(parameters[7]).shape[1] != len(times):
            raise ValueError(
                'Number of temporal and regional fluctuations in transmition does not match \
                    that of time points.')
        if not isinstance(parameters[8], (float, int)):
            raise TypeError('Mean latent period must be float or integer.')
        if parameters[8] <= 0:
            raise ValueError('Mean latent period must be > 0.')
        if not isinstance(parameters[9], (float, int)):
            raise TypeError('Mean infection period must be float or integer.')
        if parameters[9] <= 0:
            raise ValueError('Mean infection period must be > 0.')
        if not isinstance(parameters[10], (float, int)):
            raise TypeError(
                'Time step for ODE solver must be float or integer.')
        if parameters[10] <= 0:
            raise ValueError('Time step for ODE solver must be > 0.')

        # Split parameters into the features of the model
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
        self._times = np.asarray(times)

        # Select method of simulation
        if method == 'my-solver':
            sol = self._my_solver(times, self._num_ages)
        else:
            sol = self._scipy_solver(times, self._num_ages, method)

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
        self._output_indices = np.arange(self._n_outputs)

        output_indices = []
        for i in self._output_indices:
            output_indices.extend(
                np.arange(i*self._num_ages, (i+1)*self._num_ages)
            )

        output = output[output_indices, :]

        return output.transpose()
