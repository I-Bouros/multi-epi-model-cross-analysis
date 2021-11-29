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
import pints
from scipy.stats import nbinom, binom
from scipy.integrate import solve_ivp

import epimodels as em


class PheSEIRModel(pints.ForwardModel):
    r"""PheSEIRModel Class:
    Base class for constructing the ODE model: deterministic SEIR used by the
    Public Health England to model the Covid-19 epidemic in UK based on region.

    The population is structured according to their age-group (:math:`i`) and
    region (:math:`r`) and every individual will belong to one of the
    compartments of the SEIR model.

    The general SEIR Model has four compartments - susceptible individuals
    (:math:`S`), exposed but not yet infectious (:math:`E`), infectious
    (:math:`I`) and recovered (:math:`R`).

    In the PHE model framework, the exposed and infectious compartments are
    split in two:

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

    where :math:`S(0) = S_0, E(0) = E_0, I(O) = I_0, R(0) = R_0` are also
    parameters of the model (evaluation at 0 refers to the compartments'
    structure at intial time.

    The parameter :math:`\lambda_{r, t, i}` is the time, age and region-varying
    rate with which susceptible individuals become infected, which in the
    context of the PHE model depends on contact and region-specific relative
    susceptibility matrices. The other two parameters, :math:`\kappa` and
    :math:`\gamma` are virsus specific and so do not depend with region, age or
    time:

    .. math::
        :nowrap:

        \begin{eqnarray}
            \kappa = \frac{2}{d_L} \\
            \gamma = \frac{2}{d_I}
        \end{eqnarray}

    where :math:`d_L` refers to mean latent period until disease onset and
    :math:`d_I` to mean period of infection.

    Extends :class:`pints.ForwardModel`.

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

    def set_regions(self, regions):
        """
        Sets region names.

        """
        self.regions = regions

    def region_names(self):
        """
        Returns the regions names.

        """
        return self.regions

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

    def _compute_lambda(self, s, i1, i2, b):
        """
        Computes the current time, age and region-varying rate with which
        susceptible individuals become infected.

        Parameters
        ----------
        s
            vector of susceptibles by age group.
        i1
            vector of 1st infective by age group.
        i2
            vector of 2nd infective by age group.
        b
            probability matrix of infectivity.

        """
        lam = np.empty_like(s)
        for i, l in enumerate(lam):
            prod = 1
            for j, _ in enumerate(lam):
                prod *= (1-b[i, j])**(i1[j]+i2[j])
            lam[i] = 1-prod

        return lam

    def _compute_evaluation_moments(self, times):
        """
        Returns the points at which we keep the evaluations of the ODE system.

        """
        eval_times = np.around(
            np.arange(
                times[0], times[-1]+self._delta_t, self._delta_t,
                dtype=np.float64),
            5)

        eval_indices = np.where(
            np.array([(t in times) for t in eval_times]))[0].tolist()

        ind_in_times = []
        j = 0
        for i, t in enumerate(eval_times):
            if i >= eval_indices[j+1]:
                j += 1
            ind_in_times.append(j)

        return eval_times, ind_in_times

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
            encaplsulates temporal fluctuations in transmission for all ages.
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
        lam = self._compute_lambda(s, i1, i2, b)

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
        s, e1, e2, i1, i2, r = np.asarray(self._y_init)[:, self._region-1]

        # Read parameters of the system
        beta, dL, dI = self._c
        kappa = self._delta_t * 2/dL
        gamma = self._delta_t * 2/dI

        eval_times, ind_in_times = \
            self._compute_evaluation_moments(times)

        solution = np.ones((len(times), num_a_groups*6))

        for ind, t in enumerate(eval_times):
            # Add present vlaues of the compartments to the solutions
            if t in times:
                solution[ind_in_times[ind]] = tuple(
                    chain(s, e1, e2, i1, i2, r))

            # And identify the appropriate MultiTimesInfectivity matrix for the
            # ODE system
            b = self.infectivity_timeline.compute_prob_infectivity_matrix(
                self._region, t, s, beta[self._region-1][ind_in_times[ind]])

            # Compute the current time, age and region-varying
            # rate with which susceptible individuals become infected
            lam = self._compute_lambda(s, i1, i2, b)

            # Write down ODE system and compute new values for all compartments
            s_ = np.multiply(
                np.asarray(s), (np.ones_like(lam) - self._delta_t * lam))
            e1_ = (1 - kappa) * np.asarray(e1) + np.multiply(
                np.asarray(s), self._delta_t * lam)
            e2_ = (1 - kappa) * np.asarray(e2) + kappa * np.asarray(e1)
            i1_ = (1 - gamma) * np.asarray(i1) + kappa * np.asarray(e2)
            i2_ = (1 - gamma) * np.asarray(i2) + gamma * np.asarray(i1)
            r_ = gamma * np.asarray(i2) + r

            s, e1, e2, i1, i2, r = (
                s_.tolist(), e1_.tolist(), e2_.tolist(),
                i1_.tolist(), i2_.tolist(), r_.tolist())

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

    def _simulate(
            self, parameters, times, initial_r, method):
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
        regions
            (list) List of region names for the region-specific relative
            susceptibility matrices.
        initial_r
            (list) List of initial values of the reproduction number by region.
        method
            (string) The type of solver implemented by the simulator.

        """
        # Split parameters into the features of the model
        self._region = parameters[0]
        self._y_init = parameters[1:7]
        self._c = parameters[7:10]
        self._delta_t = parameters[10]
        self.infectivity_timeline = em.MultiTimesInfectivity(
            self.matrices_contact,
            self.time_changes_contact,
            self.regions,
            self.matrices_region,
            self.time_changes_region,
            initial_r,
            self._c[2],
            self._y_init[0])

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

    def read_contact_data(self, matrices_contact, time_changes_contact):
        """
        Reads in tthe timelines of contact data used for the modelling.

        Parameters
        ----------
        matrices_contact
            (list of ContactMatrix) Time-dependent contact matrices used for
            the modelling.
        time_changes_contact
            (list) Times at which the next contact matrix recorded starts to be
            used. In increasing order.

        """
        self.matrices_contact = matrices_contact
        self.time_changes_contact = time_changes_contact

    def read_regional_data(self, matrices_region, time_changes_region):
        """
        Reads in tthe timelines of regional data used for the modelling.

        Parameters
        ----------
        matrices_region
            (list of lists of RegionMatrix)) Time-dependent and region-specific
            relative susceptibility matrices used for the modelling.
        time_changes_region
            (list) Times at which the next instances of region-specific
            relative susceptibility matrices recorded start to be used. In
            increasing order.

        """
        self.matrices_region = matrices_region
        self.time_changes_region = time_changes_region

    def simulate(self, parameters, times):
        r"""
        PINTS-configured wrapper for the simulation method of the PHE model.

        Extends the :meth:`_simulation`. Always apply methods
        :meth:`set_regions`, :meth:`read_contact_data` and
        :meth:`read_regional_data` before running the
        :meth:`PheSEIRModel.simulate`.

        Parameters
        ----------
        parameters
            Long vector format of the quantities that characterise the PHE
            SEIR model in this order:
            (1) initial values of the reproduction number
            by region,
            (2) index of region for which we wish to simulate,
            (3) initial conditions matrices classifed by age (column name) and
            region (row name) for each type of compartment (s, e1, e2, i1, i2,
            r),
            (4) temporal and regional fluctuation matrix :math:`\beta`,
            (5) mean latent period :math:`d_L`,
            (6) mean infection period :math:`d_I`,
            (7) time step for the 'homemade' solver and
            (8) (string) the type of solver implemented by the simulator.
            Splited into the formats necessary for the :meth:`_simulate`
            method.
        times
            (list) List of time points at which we wish to evaluate the ODEs
            system.

        """
        # Number of regions and age groups
        self._num_ages = self.matrices_contact[0]._num_a_groups

        n_ages = self._num_ages
        n_reg = len(self.regions)

        start_index = n_reg * (1 + (len(self._output_names)-1) * n_ages) + 1
        finish_index = start_index + n_reg * len(times)

        self._check_input_simulate(
            parameters, times, finish_index + 4, n_reg)

        # Read initial reproduction numbers
        initial_r = parameters[:n_reg]

        # Separate list of parameters into the structures needed for the
        # simulation
        my_parameters = []

        # Add index of region
        my_parameters.append(parameters[n_reg])

        # Add initial conditions for the s, e1, e2, i1, i2, r compartments
        for c in range(len(self._output_names)-1):
            initial_cond_comp = []
            for r in range(n_reg):
                ind = n_reg + r * n_ages + n_reg * c * n_ages + 1
                initial_cond_comp.append(
                    parameters[ind:(ind + n_ages)])
            my_parameters.append(initial_cond_comp)

        # Add beta parameters
        beta_param = np.array(
            parameters[start_index:finish_index]).reshape(n_reg, -1)

        my_parameters.append(beta_param.tolist())

        # Add mean latent period, mean infection period and delta_t
        my_parameters.extend(parameters[finish_index:(finish_index + 3)])

        # Add method
        method = parameters[finish_index + 3]

        return self._simulate(my_parameters,
                              times,
                              initial_r,
                              method)

    def _check_input_simulate(
            self, parameters, times, L, n_reg):
        """
        Check correct format of input of simulate method.

        """
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
            raise TypeError('Parameters must be given in a list format.')
        if len(parameters) != L:
            raise ValueError('List of parameters has wrong length.')
        if not isinstance(parameters[n_reg], int):
            raise TypeError('Index of region to evaluate must be integer.')
        if parameters[n_reg] <= 0:
            raise ValueError('Index of region to evaluate must be >= 1.')
        if parameters[n_reg] > n_reg:
            raise ValueError('Index of region to evaluate is out of bounds.')
        if not isinstance(parameters[-4], (float, int)):
            raise TypeError('Mean latent period must be float or integer.')
        if parameters[-4] <= 0:
            raise ValueError('Mean latent period must be > 0.')
        if not isinstance(parameters[-3], (float, int)):
            raise TypeError('Mean infection period must be float or integer.')
        if parameters[-3] <= 0:
            raise ValueError('Mean infection period must be > 0.')
        if not isinstance(parameters[-2], (float, int)):
            raise TypeError(
                'Time step for ODE solver must be float or integer.')
        if parameters[-2] <= 0:
            raise ValueError('Time step for ODE solver must be > 0.')
        if not isinstance(parameters[-1], str):
            raise TypeError('Simulation method must be a string.')
        if parameters[-1] not in (
                'my-solver', 'RK45', 'RK23', 'Radau',
                'BDF', 'LSODA', 'DOP853'):
            raise ValueError('Simulation method not available.')

    def _check_output_format(self, output):
        """
        Checks correct format of the output matrix.

        """
        if np.asarray(output).ndim != 2:
            raise ValueError(
                'Model output storage format must be 2-dimensional.')
        if np.asarray(output).shape[0] != self._times.shape[0]:
            raise ValueError(
                    'Wrong number of rows for the model output.')
        if np.asarray(output).shape[1] != 7 * self._num_ages:
            raise ValueError(
                    'Wrong number of columns for the model output.')
        for r in np.asarray(output):
            for _ in r:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'Model output elements must be integer or float.')

    def new_infections(self, output):
        """
        Computes number of new infections at each time step in specified
        region, given the simulated timeline of susceptible number of
        individuals, for all age groups in the model.

        It uses an output of the simulation method for the PheSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        output
            (numpy.array) Output of the simulation method for the PheSEIRModel.

        Notes
        -----
        Always run :meth:`PheSEIRModel.simulate` before running this one.

        """
        # Check correct format of parameters
        self._check_output_format(output)

        beta, dL, dI = self._c
        d_infec = np.empty((self._times.shape[0], self._num_ages))

        for ind, t in enumerate(self._times.tolist()):
            # Read from output
            s = output[ind, :][:self._num_ages]
            i1 = output[ind, :][(3*self._num_ages):(4*self._num_ages)]
            i2 = output[ind, :][(4*self._num_ages):(5*self._num_ages)]

            b = self.infectivity_timeline.compute_prob_infectivity_matrix(
                self._region, t, s, beta[self._region-1][ind])

            # Compute the current time, age and region-varying
            # rate with which susceptible individuals become infected
            lam = self._compute_lambda(s, i1, i2, b)

            # fraction of new infectives in delta_t time step
            d_infec[ind, :] = np.multiply(np.asarray(s), lam*self._delta_t)

        return d_infec

    def loglik_deaths(
            self, obs_death, output, fatality_ratio, time_to_death, niu, k):
        r"""
        Computes the log-likelihood for the number of deaths at time step
        :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of deaths is assumed to be distributed according to
        a negative binomial distribution with mean

        .. math::
            \mu_{r,t_k,i} = p_i \sum_{l=0}^{k} f_{k-l} \delta_{r,t_l,i}^{infec}

        and variance :math:`\mu_{r,t_k,i} (\nu + 1)`, where :math:`p_i` is the
        age-specific fatality ratio for age group :math:`i`, :math:`f_{k-l}`
        is the probability of demise :math:`k-l` days after infection and
        :math:`\delta_{r,t_l,i}^{infec}` is the number of new infections
        in specified region, for age group :math:`i` on day :math:`t_l`.

        It uses an output of the simulation method for the PheSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        obs_death
            List of number of observed deaths by age group at time point k.
        output
            (numpy.array) Output of the simulation method for the PheSEIRModel.
        fatality_ratio
            List of age-specific fatality ratios.
        time_to_death
            List of probabilities of death of individual k days after
            infection.
        niu
            Dispersion factor for the negative binomial distribution
        k
            Index of day for which we intend to sample the number of deaths for
            by age group.

        Returns
        -------
        Array of log-likelihoods for the obsereved number of deaths for
        each age group in specified region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`PheSEIRModel.simulate` before running this one.

        """
        self._check_death_format(
            output, fatality_ratio, time_to_death, niu, k)

        # Check correct format for observed number of deaths
        if np.asarray(obs_death).ndim != 1:
            raise ValueError('Observed number of deaths by age category storage \
                format is 1-dimensional.')
        if np.asarray(obs_death).shape[0] != self._num_ages:
            raise ValueError('Wrong number of age groups for observed number \
                of deaths.')
        for _ in obs_death:
            if not isinstance(_, (int, np.integer)):
                raise TypeError('Observed number of deaths must be integer.')
            if _ < 0:
                raise ValueError('Observed number of deaths must be => 0.')

        # Compute mean of negative-binomial
        d_infec = self.new_infections(output)
        log_lik_death = np.empty(self._num_ages)
        for _ in range(self._num_ages):
            mean = self.mean_deaths(
                fatality_ratio, time_to_death, k, _, d_infec)
            log_lik_death[_] = nbinom.logpmf(
                k=obs_death[_],
                n=mean*niu,
                p=niu/(1+niu))

        return log_lik_death

    def _check_death_format(
            self, output, fatality_ratio, time_to_death, niu, k):
        """
        Checks correct format of the inputs of number of death calculation.

        """
        self._check_output_format(output)
        if not isinstance(niu, (int, float)):
            raise TypeError('Dispersion factor must be integer or float.')
        if niu <= 0:
            raise ValueError('Dispersion factor must be > 0.')
        if not isinstance(k, int):
            raise TypeError('Index of time of computation of the log-likelihood \
                must be integer.')
        if k < 0:
            raise ValueError('Index of time of computation of the log-likelihood \
                must be >= 0.')
        if k >= self._times.shape[0]:
            raise ValueError('Index of time of computation of the log-likelihood \
                must be within those considered in the output.')
        if np.asarray(fatality_ratio).ndim != 1:
            raise ValueError('Fatality ratios by age category storage \
                format is 1-dimensional.')
        if np.asarray(fatality_ratio).shape[0] != self._num_ages:
            raise ValueError('Wrong number of age groups for fatality ratios.')
        for _ in fatality_ratio:
            if not isinstance(_, (int, float)):
                raise TypeError('Fatality ratio must be integer or \
                    float.')
            if(_ < 0) or (_ > 1):
                raise ValueError('Fatality ratio must be => 0 and <=1.')
        if np.asarray(time_to_death).ndim != 1:
            raise ValueError('Probabilities of death of individual k days after \
                infection storage format is 1-dimensional.')
        if np.asarray(time_to_death).shape[0] != len(self._times):
            raise ValueError('Wrong number of probabilities of death of individual\
                k days after infection.')
        for _ in time_to_death:
            if not isinstance(_, (int, float)):
                raise TypeError('Probabilities of death of individual k days after \
                    infection must be integer or float.')
            if (_ < 0) or (_ > 1):
                raise ValueError('Probabilities of death of individual k days after \
                    infection must be => 0 and <=1.')

    def mean_deaths(self, fatality_ratio, time_to_death, k, a, d_infec):
        """
        Computes the mean of the negative binomial distribution used to
        calculate number of deaths for specified age group.

        """
        mean = fatality_ratio[a] * np.sum(np.multiply(
            d_infec[:(k+1), a], np.asarray(time_to_death[:(k+1)][::-1])))

        return mean

    def samples_deaths(
            self, output, fatality_ratio, time_to_death, niu, k):
        r"""
        Computes samples for the number of deaths at time step
        :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of deaths is assumed to be distributed according to
        a negative binomial distribution with mean

        .. math::
            \mu_{r,t_k,i} = p_i \sum_{l=0}^{k} f_{k-l} \delta_{r,t_l,i}^{infec}

        and variance :math:`\mu_{r,t_k,i} (\nu + 1)`, where :math:`p_i` is the
        age-specific fatality ratio for age group :math:`i`, :math:`f_{k-l}`
        is the probability of demise :math:`k-l` days after infection and
        :math:`\delta_{r,t_l,i}^{infec}` is the number of new infections
        in specified region, for age group :math:`i` on day :math:`t_l`.

        It uses an output of the simulation method for the PheSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        output
            (numpy.array) Output of the simulation method for the PheSEIRModel.
        fatality_ratio
            List of age-specific fatality ratios.
        time_to_death
            List of probabilities of death of individual k days after
            infection.
        niu
            Dispesion factor for the negative binomial distribution
        k
            Index of day for which we intend to sample the number of deaths for
            by age group.

        Returns
        -------
        Array of log-likelihoods for the obsereved number of deaths for
        each age group in specified region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`PheSEIRModel.simulate` before running this one.

        """
        self._check_death_format(
            output, fatality_ratio, time_to_death, niu, k)

        # Compute mean of negative-binomial
        d_infec = self.new_infections(output)
        sample_death = np.empty(self._num_ages)
        for _ in range(self._num_ages):
            mean = self.mean_deaths(
                fatality_ratio, time_to_death, k, _, d_infec)
            sample_death[_] = nbinom.rvs(
                n=mean*niu,
                p=niu/(1+niu))

        return sample_death

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

        It uses an output of the simulation method for the PheSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        obs_pos
            List of number of observed positive test results by age group at
            time point k.
        output
            (numpy.array) Output of the simulation method for the PheSEIRModel.
        tests
            List of conducted tests in specified region and at time point k
            classifed by age groups.
        sens
            Sensitivity of the test (or ratio of true positives).
        spec
            Specificity of the test (or ratio of true negatives).
        k
            Index of day for which we intend to sample the number of deaths for
            by age group.

        Returns
        -------
        Array of log-likelihoods for the obsereved number of positive test
        results for each age group in specified region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`PheSEIRModel.simulate` before running this one.

        """
        self._check_positives_format(output, tests, sens, spec, k)

        # Check correct format for observed number of positive results
        if np.asarray(obs_pos).ndim != 1:
            raise ValueError('Observed number of postive tests results by age category \
                storage format is 1-dimensional.')
        if np.asarray(obs_pos).shape[0] != self._num_ages:
            raise ValueError('Wrong number of age groups for observed number \
                of postive tests results.')
        for _ in obs_pos:
            if not isinstance(_, (int, np.integer)):
                raise TypeError('Observed number of postive tests results must be \
                    integer.')
            if _ < 0:
                raise ValueError('Observed number of postive tests results must \
                    be => 0.')

        # Check correct format for number of tests based on the observed number
        # of positive results
        for i, _ in enumerate(tests):
            if _ < obs_pos[i]:
                raise ValueError('Not enough performed tests for the number \
                    of observed positives.')

        a = self._num_ages
        # Compute parameters of binomial
        suscep = output[k, :a]
        pop = output[k, :a]
        for i in range(6):
            pop += output[k, (i*a):((i+1)*a)]

        log_lik_pos = np.empty(self._num_ages)
        for _ in range(self._num_ages):
            prob = self.mean_positives(sens, spec, _, suscep, pop)
            log_lik_pos[_] = binom.logpmf(
                k=obs_pos[_],
                n=tests[_],
                p=prob)

        return log_lik_pos

    def _check_positives_format(self, output, tests, sens, spec, k):
        """
        Checks correct format of the inputs of number of positie test results
        calculation.

        """
        self._check_output_format(output)
        if np.asarray(tests).ndim != 1:
            raise ValueError('Number of tests conducted by age category storage \
                format is 1-dimensional.')
        if np.asarray(tests).shape[0] != self._num_ages:
            raise ValueError('Wrong number of age groups for observed number \
                of tests conducted.')
        for _ in tests:
            if not isinstance(_, (int, np.integer)):
                raise TypeError('Number of tests conducted must be integer.')
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
        if not isinstance(k, int):
            raise TypeError('Index of time of computation of the log-likelihood \
                must be integer.')
        if k < 0:
            raise ValueError('Index of time of computation of the log-likelihood \
                must be >= 0.')
        if k >= self._times.shape[0]:
            raise ValueError('Index of time of computation of the log-likelihood \
                must be within those considered in the output.')

    def mean_positives(self, sens, spec, a, suscep, pop):
        """
        Computes the mean of the binomial distribution used to
        calculate number of positive test results for specified age group.

        """
        prob = sens * (1-suscep[a]/pop[a]) + (
                1-spec) * suscep[a]/pop[a]
        return prob

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

        It uses an output of the simulation method for the PheSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        output
            (numpy.array) Output of the simulation method for the PheSEIRModel.
        tests
            List of conducted tests in specified region and at time point k
            classifed by age groups.
        sens
            Sensitivity of the test (or ratio of true positives).
        spec
            Specificity of the test (or ratio of true negatives).
        k
            Index of day for which we intend to sample the number of deaths for
            by age group.

        Returns
        -------
        Array of log-likelihoods for the obsereved number of positive test
        results for each age group in specified region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`PheSEIRModel.simulate` before running this one.

        """
        self._check_positives_format(output, tests, sens, spec, k)

        a = self._num_ages
        # Compute parameters of binomial
        suscep = output[k, :a]
        pop = output[k, :a]
        for i in range(6):
            pop += output[k, (i*a):((i+1)*a)]

        sample_pos = np.empty(self._num_ages)
        for _ in range(self._num_ages):
            prob = self.mean_positives(sens, spec, _, suscep, pop)
            sample_pos[_] = binom.rvs(
                n=tests[_],
                p=prob)

        return sample_pos
