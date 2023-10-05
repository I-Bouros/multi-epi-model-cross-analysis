#
# SEIRDModel Class
#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""
This script contains code for modelling the simple SEIRD model.

"""

import numpy as np
import pints
from scipy.stats import nbinom, binom
from scipy.integrate import solve_ivp


class SEIRDModel(pints.ForwardModel):
    r"""SEIRDModel Class:
    Base class for constructing the SEIRD model.

    The population is structured according to their age-group (:math:`i`) and
    region (:math:`r`) and every individual will belong to one of the
    compartments of the SEIRD model.

    The SEIRD Model has five compartments: susceptible individuals (:math:`S`),
    exposed but not yet infectious (:math:`E`), infectious (:math:`I`),
    recovered (:math:`R`) and dead (:math:`D`):

    .. math::
        \frac{dS_i(t)}{dt} = -\beta S(t)I(t),
    .. math::
        \frac{dE_i(t)}{dt} = \beta S(t)I(t) - \kappa E(t),
    .. math::
        \frac{dI_i(t)}{dt} = \kappa E(t) - \gamma I(t),
    .. math::
        \frac{dR_i(t)}{dt} = \gamma (1-P_d) I(t),
    .. math::
        \frac{dD_i(t)}{dt} = \gamma P_d I(t),

    where :math:`i` is the age group of the individual, math:`\beta,
    \kappa, \gamma` represent the transmission rates for all ages,
    :math:`P_d` is the propotion of infectious people that go on to die.
    :math:`S(0) = S_0, E(0) = E_0, I(O) = I_0, R(0) = R_0`
    are also parameters of the model.

    Extends :class:`pints.ForwardModel`.

    """
    def __init__(self):
        super(SEIRDModel, self).__init__()

        # Assign default values
        self._output_names = ['S', 'E', 'I', 'R', 'D', 'Incidence']
        self._parameter_names = [
            'S0', 'E0', 'I0', 'R0', 'D0', 'beta', 'kappa', 'gamma', 'Pd']

        # The default number of outputs is 6,
        # i.e. S, E, I, R, D and Incidence
        self._n_outputs = len(self._output_names)
        # The default number of parameters is 9,
        # i.e. 5 initial conditions and 4 parameters
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

    def region_names(self):
        """
        Returns the regions names.

        Returns
        -------
        list
            List of the regions names.

        """
        return self.regions

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

    def _right_hand_side(self, t, r, y, c):
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
            Array of all the compartments of the ODE system. It assumes
            y = [S, E, I, R, D] where each letter actually refers to all
            compartment of that type. (e.g. S refers to the compartments
            of susceptibles).
        c : list
            List of values used to compute the parameters of the ODEs
            system. It assumes c = [beta, kappa, gamma, Pd], where :math:`beta,
            kappa, gamma` represent the transmission rates and :math:`Pd` is
            the propotion of infectious people that go on to die.

        Returns
        -------
        numpy.array
            Age-structured matrix representation of the RHS of the ODEs system.

        """

        # Split compartments into their types
        s, e, i, _, d = y

        # Read parameters of the system
        beta, kappa, gamma, Pd = c

        # Write actual RHS
        dydt = np.array([
            -beta * s * i / self._N,
            beta * s * i / self._N - kappa * e,
            kappa * e - gamma * i,
            gamma * (1 - Pd) * i,
            gamma * Pd * i,
        ])

        return dydt

    def _scipy_solver(self, times, method):
        """
        Computes the values in each compartment of the SEIRD ODEs system using
        the 'off-the-shelf' solver of the IVP from :module:`scipy`.

        Parameters
        ----------
        times : list
            List of time points at which we wish to evaluate the ODEs system.
        method : str
            The type of solver implemented by the :meth:`scipy.solve_ivp`.

        Returns
        -------
        dict
            Solution of the ODE system at the time points provided.

        """
        # Initial conditions
        init_cond = np.asarray(self._y_init)[:, self._region-1].tolist()

        # Solve the system of ODEs
        sol = solve_ivp(
            lambda t, y: self._right_hand_side(
                t, self._region, y, self._c),
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
            List of quantities that characterise the SEIRD model in
            this order: index of region for which we wish to simulate,
            initial conditions matrices classifed by age (column name) and
            region (row name) for each type of compartment (s, e, i, r, d),
            r), the rates of progression to different infection stages (beta,
            kappa, gamma) and the propotion of infectious people that go on to
            die (Pd).
        times : list
            List of time points at which we wish to evaluate the ODEs system.
        method : str
            The type of solver implemented by the simulator.

        Returns
        -------
        numpy.array
            Age-structured output matrix of the simulation for the specified
            region.

        """
        # Split parameters into the features of the model
        self._region = parameters[0]
        self._y_init = parameters[1:6]
        self._N = np.sum(np.asarray(self._y_init))
        self._c = parameters[6:]

        self._times = np.asarray(times)

        # Select method of simulation
        sol = self._scipy_solver(times, method)

        output = sol['y']

        # Age-based total infected is infectious 'i' plus recovered 'r'
        total_infected = output[2, :] + output[3, :]

        # Number of incidences is the increase in total_infected
        # between the time points (add a 0 at the front to
        # make the length consistent with the solution
        n_incidence = np.zeros(len(times))
        n_incidence[1:] = total_infected[1:] - total_infected[:-1]

        # Append n_incidence to output
        # Output is a matrix with rows being S, E1, I1, R1 and Incidence
        output = np.vstack((output, n_incidence))

        # Get the selected outputs
        self._output_indices = np.arange(self._n_outputs)

        output = output[self._output_indices, :]

        return output.transpose()

    def simulate(self, parameters):
        """
        Simulates the SEIRD model using a :class:`SEIRDParametersController`
        for the model parameters.

        Extends the :meth:`_split_simulate`. Always apply methods
        :meth:`set_regions`, :meth:`read_contact_data`
        and :meth:`read_regional_data` before running the
        :meth:`SEIRDModel.simulate`.

        Parameters
        ----------
        parameters : SEIRDParametersController
            Controller class for the parameters used by the forward simulation
            of the model.

        Returns
        -------
        numpy.array
            Output matrix of the simulation for the specified
            region.

        """
        return self._simulate(
            parameters(), parameters.simulation_parameters.times)

    def _simulate(self, parameters, times):
        r"""
        PINTS-configured wrapper for the simulation method of the SEIRD model.

        Extends the :meth:`_split_simulate`. Always apply methods
        :meth:`set_regions`, :meth:`read_contact_data`
        and :meth:`read_regional_data` before running the
        :meth:`SEIRDModel.simulate`.

        Parameters
        ----------
        parameters : list
            Long vector format of the quantities that characterise the SEIRD
            model in this order:
            (1) index of region for which we wish to simulate,
            (2) initial conditions matrices classifed by region (row name) for
            each type of compartment (s, e, i, r, d),
            (3) the rates of progression to different infection stages (beta,
            kappa, gamma) and
            (4) the propotion of infectious people that go on to die (Pd).
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
        n_reg = len(self.regions)

        start_index = n_reg * (len(self._output_names)-1) + 1

        # Separate list of parameters into the structures needed for the
        # simulation
        my_parameters = []

        # Add index of region
        my_parameters.append(parameters[0])

        # Add initial conditions for the s, e, i, r, d compartments
        for c in range(len(self._output_names)-1):
            initial_cond_comp = []
            for r in range(n_reg):
                ind = r + n_reg * c + 1
                initial_cond_comp.append(
                    parameters[ind])
            my_parameters.append(initial_cond_comp)

        # Add other parameters
        my_parameters.extend(parameters[start_index:(start_index + 4)])

        # Add method
        method = parameters[start_index + 4]

        return self._split_simulate(my_parameters,
                                    times,
                                    method)

    def _check_output_format(self, output):
        """
        Checks correct format of the output matrix.

        Parameters
        ----------
        output : numpy.array
            Output matrix of the simulation method
            for the SEIRDModel.

        """
        if np.asarray(output).ndim != 2:
            raise ValueError(
                'Model output storage format must be 2-dimensional.')
        if np.asarray(output).shape[0] != self._times.shape[0]:
            raise ValueError(
                    'Wrong number of rows for the model output.')
        if np.asarray(output).shape[1] != 6:
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
        individuals.

        It uses an output of the simulation method for the SEIRDModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        output : numpy.array
            Output of the simulation method for the
            SEIRDModel.

        Returns
        -------
        nunmpy.array
            Matrix of the number of new infections from the
            simulation method for the SEIRDModel.

        Notes
        -----
        Always run :meth:`SEIRDModel.simulate` before running this one.

        """
        # Check correct format of parameters
        self._check_output_format(output)

        kappa = self._c[1]
        d_infec = np.empty(self._times.shape[0])

        for ind, t in enumerate(self._times.tolist()):
            # Read from output
            e = output[ind][1]

            # fraction of new infectives in delta_t time step
            d_infec[ind] = kappa * e

            if np.any(d_infec[ind] < 0):  # pragma: no cover
                d_infec[ind] = np.zeros_like(d_infec[ind])

        return d_infec

    def _check_new_deaths_format(self, new_deaths):
        """
        Checks correct format of the new deaths matrix.

        Parameters
        ----------
        new_deaths : numpy.array
            Matrix of the number of new deaths from the
            simulation method for the SEIRDModel.

        """
        if np.asarray(new_deaths).ndim != 1:
            raise ValueError(
                'Model new infections storage format must be 1-dimensional.')
        if np.asarray(new_deaths).shape[0] != self._times.shape[0]:
            raise ValueError(
                    'Wrong number of rows for the model new infections.')
        for _ in np.asarray(new_deaths):
            if not isinstance(_, (np.integer, np.floating)):
                raise TypeError(
                    'Model new deaths elements must be integer or \
                        float.')

    def new_deaths(self, output):
        """
        Computes number of new deaths at each time step in specified
        region, given the simulated timeline of susceptible number of
        individuals.

        It uses an output of the simulation method for the SEIRDModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        output : numpy.array
            Output of the simulation method for the
            SEIRDModel.

        Returns
        -------
        nunmpy.array
            Matrix of the number of new deaths from the
            simulation method for the SEIRDModel.

        Notes
        -----
        Always run :meth:`SEIRDModel.simulate` before running this one.

        """
        # Check correct format of parameters
        self._check_output_format(output)

        # Check correct format of parameters
        # Age-based total dead is dead 'd'
        n_daily_deaths = np.zeros(self._times.shape[0])
        total_dead = output[:, 4]
        n_daily_deaths[1:] = total_dead[1:] - total_dead[:-1]

        for ind, t in enumerate(self._times.tolist()):  # pragma: no cover
            if np.any(n_daily_deaths[ind] < 0):
                n_daily_deaths[ind] = np.zeros_like(n_daily_deaths[ind])

        return n_daily_deaths

    def loglik_deaths(self, obs_death, new_deaths, niu, k):
        r"""
        Computes the log-likelihood for the number of deaths at time step
        :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals.

        The number of deaths is assumed to be distributed according to
        a negative binomial distribution with mean :math:`\mu_{r,t_k,i}`
        and variance :math:`\mu_{r,t_k,i} (\nu + 1)`, where
        :math:`\mu_{r,t_k,i}` is the number of new deaths in specified region,
        for age group :math:`i` on day :math:`t_k`.

        It uses new_infections output of the simulation method for the
        SEIRDModel, taking all the rest of the parameters necessary for
        the computation from the way its simulation has been fitted.

        Parameters
        ----------
        obs_death : int or float
            Number of observed deaths at time point k.
        new_deaths : numpy.array
            Matrix of the number of new deaths from the
            simulation method for the SEIRDModel.
        niu : float
            Dispersion factor for the negative binomial distribution.
        k : int
            Index of day for which we intend to sample the number of deaths for
            by age group.

        Returns
        -------
        numpy.array
            Matrix of log-likelihoods for the observed number
            of deaths in specified region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`SEIRDModel.new_infections` and
        :meth:`SEIRDModel.check_death_format` before running this one.

        """
        self._check_time_step_format(k)

        # Check correct format for observed number of deaths
        if not isinstance(obs_death, (int, np.integer)):
            raise TypeError('Observed number of deaths must be integer.')
        if obs_death < 0:
            raise ValueError('Observed number of deaths must be => 0.')

        if not hasattr(self, 'actual_deaths'):
            self.actual_deaths = [0] * 150
        self.actual_deaths[k] = self.mean_deaths(k, new_deaths)

        # Compute mean of negative-binomial
        if k != 0:
            if np.sum(self.mean_deaths(k, new_deaths)) != 0:
                return nbinom.logpmf(
                    k=obs_death,
                    n=(1/niu) * self.mean_deaths(k, new_deaths),
                    p=1/(1+niu))
            else:
                return np.zeros(1)
        else:
            return np.zeros(1)

    def check_death_format(self, new_deaths, niu):
        """
        Checks correct format of the inputs of number of death calculation.

        Parameters
        ----------
        new_deaths : numpy.array
            Matrix of the number of new deaths from the
            simulation method for the SEIRDModel.
        niu : float
            Dispersion factor for the negative binomial distribution.

        """
        self._check_new_deaths_format(new_deaths)
        if not isinstance(niu, (int, float)):
            raise TypeError('Dispersion factor must be integer or float.')
        if niu <= 0:
            raise ValueError('Dispersion factor must be > 0.')

    def mean_deaths(self, k, new_deaths):
        """
        Computes the mean of the negative binomial distribution used to
        calculate number of deaths.

        Parameters
        ----------
        k : int
            Index of day for which we intend to sample the number of deaths for
            by age group.
        new_deaths : numpy.array
            Matrix of the number of new deaths from the
            simulation method for the SEIRDModel.

        Returns
        -------
        numpy.array
            Matrix of the expected number of deaths to be
            observed in specified region at time :math:`t_k`.

        """
        return new_deaths[k] + 1e-20

    def samples_deaths(self, new_deaths, niu, k):
        r"""
        Computes samples for the number of deaths at time step
        :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals.

        The number of deaths is assumed to be distributed according to
        a negative binomial distribution with mean :math:`\mu_{r,t_k,i}`
        and variance :math:`\mu_{r,t_k,i} (\nu + 1)`, where
        :math:`\mu_{r,t_k,i}` is the number of new deaths in specified region,
        for age group :math:`i` on day :math:`t_k`.

        It uses an output of the simulation method for the SEIRDModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        new_deaths : numpy.array
            Matrix of the number of new deaths from the
            simulation method for the SEIRDModel.
        niu : float
            Dispersion factor for the negative binomial distribution.
        k : int
            Index of day for which we intend to sample the number of deaths for
            by age group.

        Returns
        -------
        numpy.array
            Matrix of sampled number of deaths in specified region at time
            :math:`t_k`.

        Notes
        -----
        Always run :meth:`SEIRDModel.new_infections` and
        :meth:`SEIRDModel.check_death_format` before running this one.

        """
        self._check_time_step_format(k)

        # Compute mean of negative-binomial
        if k != 0:
            if np.sum(self.mean_deaths(k, new_deaths)) != 0:
                return nbinom.rvs(
                    n=(1/niu) * self.mean_deaths(k, new_deaths),
                    p=1/(1+niu))
            else:
                return np.zeros(1)
        else:
            return np.zeros_like(self.mean_deaths(k, new_deaths))

    def loglik_positive_tests(self, obs_pos, output, tests, sens, spec, k):
        r"""
        Computes the log-likelihood for the number of positive tests at time
        step :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals.

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

        It uses an output of the simulation method for the SEIRDModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        obs_pos : int or float
            Number of observed positive test results by age group at
            time point k.
        output : numpy.array
            Output matrix of the simulation method for the SEIRDModel.
        tests : int or float
            Conducted tests in specified region and at time point k.
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
            Matrix of log-likelihoods for the obsereved number
            of positive test results for each age group in specified region at
            time :math:`t_k`.

        Notes
        -----
        Always run :meth:`SEIRDModel.simulate` and
        :meth:`SEIRDModel.check_positives_format` before running this one.

        """
        self._check_time_step_format(k)

        # Check correct format for observed number of positive results
        if not isinstance(obs_pos, (int, np.integer)):
            raise TypeError('Observed number of postive tests results must\
                be integer.')
        if obs_pos < 0:
            raise ValueError('Observed number of postive tests results \
                must be => 0.')

        # Check correct format for number of tests based on the observed number
        # of positive results
        if tests < obs_pos:
            raise ValueError('Not enough performed tests for the number \
                of observed positives.')

        # Compute parameters of binomial
        suscep = output[k, 0]
        pop = np.sum(output[k, :6])

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
            Output matrix of the simulation method
            for the SEIRDModel.
        tests : list
            List of conducted tests in specified region and at time point k.
        sens : float or int
            Sensitivity of the test (or ratio of true positives).
        spec : float or int
            Specificity of the test (or ratio of true negatives).

        """
        self._check_output_format(output)
        if np.asarray(tests).ndim != 1:
            raise ValueError('Number of tests conducted storage format is \
                1-dimensional.')
        for _ in tests:
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
            Matrix of the current number of susceptibles
            in the population.
        pop : numpy.array
            Matrix of the current number of individuals
            in the population.

        Returns
        -------
        numpy.array
            Matrix of the expected number of positive test
            results to be observed in specified region at time :math:`t_k`.

        """
        return sens * (1 - suscep / pop) + (1 - spec) * suscep / pop

    def samples_positive_tests(self, output, tests, sens, spec, k):
        r"""
        Computes the samples for the number of positive tests at time
        step :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals.

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

        It uses an output of the simulation method for the SEIRDModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        output : numpy.array
            Output matrix of the simulation method
            for the SEIRDModel.
        tests : list
            List of conducted tests in specified region and at time point k.
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
            Matrix of sampled number of positive test results
            in specified region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`SEIRDModel.simulate` and
        :meth:`SEIRDModel.check_positives_format` before running this one.

        """
        self._check_time_step_format(k)

        # Compute parameters of binomial
        suscep = output[k, 0]
        pop = np.sum(output[k, :6])

        return binom.rvs(
            n=tests,
            p=self.mean_positives(sens, spec, suscep, pop))
