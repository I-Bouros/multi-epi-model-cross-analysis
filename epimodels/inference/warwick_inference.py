#
# WarwickLogLik Class
#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""
This script contains code for parameter inference of the extended SEIR model
created by Public Health England and Univerity of Cambridge. This is one of the
official models used by the UK government for policy making.

It uses an extended version of an SEIR model with contact and region-specific
matrices.

"""

import numpy as np
import pints
from iteration_utilities import deepflatten

import epimodels as em


class WarwickLogLik(pints.LogPDF):
    """WarwickLogLik Class:
    Controller class to construct the log-likelihood needed for optimisation or
    inference in a PINTS framework.

    Parameters
    ----------
    model : WarwickSEIRModel
        The model for which we solve the optimisation or inference problem.
    extended_susceptibles

    extended_infectives

    extended_house_cont_mat : ContactMatrix
        Initial contact matrix with more age groups used for the modelling,
        underlying household interactions.
    extended_school_cont_mat : ContactMatrix
        Initial contact matrix with more age groups used for the modelling,
        underlying school interactions.
    extended_work_cont_mat : ContactMatrix
            Initial contact matrix with more age groups used for the modelling,
        underlying workplace interactions.
    extended_other_cont_mat : ContactMatrix
        Initial contact matrix with more age groups used for the modelling,
        underlying other non-household interactions.
    susceptibles_data : list
        List of regional age-structured lists of the initial number of
        susceptibles.
    infectives_data : list
        List of regional age-structured lists of the initial number of
        infectives in the presymptomatic infectives compartments.
    times : list
        List of time points at which we have data for the log-likelihood
        computation.
    deaths_data : numpy.array
        List of region-specific age-structured number of deaths as a matrix.
        Each column represents an age group.
    deaths_times : numpy.array
        Matrix of timepoints for which deaths data is available.
    tests_data : list of numpy.array
        List of region-specific age-structured number of tests conducted as a
        matrix. Each column represents an age group.
    positives_data : list of numpy.array
        List of region-specific age-structured number of positive test results
        as a matrix. Each column represents an age group.
    serology_times : numpy.array
        Matrix of timepoints for which serology data is available.
    sens : float or int
        Sensitivity of the test (or ratio of true positives).
    spec : float or int
        Specificity of the test (or ratio of true negatives).
    wd : float or int
        Proportion of the contribution of the deaths data to the
        log-likelihood.
    wp : float or int
        Proportion of the contribution of the serology data to the
        log-likelihood.

    """
    def __init__(self, model, extended_susceptibles, extended_infectives,
                 extended_house_cont_mat, extended_school_cont_mat,
                 extended_work_cont_mat, extended_other_cont_mat,
                 susceptibles_data, infectives_data, times,
                 deaths, deaths_times, tests_data, positives_data,
                 serology_times, sens, spec, wd=1, wp=1):
        # Set the prerequisites for the inference wrapper
        # Model and ICs data
        self._model = model
        self._times = times

        self._susceptibles = susceptibles_data
        self._infectives = infectives_data

        # Death data
        self._deaths = deaths
        self._deaths_times = deaths_times

        # Serology data
        self._total_tests = tests_data
        self._positive_tests = positives_data
        self._serology_times = serology_times
        self._sens = sens
        self._spec = spec

        # Contribution parameters
        self._wd = wd
        self._wp = wp

        # Set fixed parameters of the model
        self.set_fixed_parameters()

        # Extended population structure and contact matrices
        self._extended_susceptibles = np.array(extended_susceptibles)
        self._extended_infectives = np.array(extended_infectives)
        self._pop = self._extended_susceptibles + self._extended_susceptibles

        # Identify the appropriate contact matrix for the ODE system
        house_cont_mat = extended_house_cont_mat
        school_cont_mat = extended_school_cont_mat
        work_cont_mat = extended_work_cont_mat
        other_cont_mat = extended_other_cont_mat

        # Read the social distancing parameters of the system
        theta, phi, q_H, q_S, q_W, q_O = self._model.social_distancing_param

        house_cont_mat = (1 - phi - phi * q_H) * house_cont_mat
        self._nonhouse_cont_mat = (1 - phi - phi * q_S) * school_cont_mat + \
            ((1 - phi - phi * q_W) *
             (1 - theta + theta * (1 - phi - phi * q_O))) * work_cont_mat + \
            ((1 - phi - phi * q_O)**2) * other_cont_mat

    def n_parameters(self):
        """
        Returns number of parameters for log-likelihood object.

        Returns
        -------
        int
            Number of parameters for log-likelihood object.

        """
        return 3

    def _update_age_groups(self, parameter_vector):
        """
        """
        new_vector = np.empty(8)

        ind_old = [
            np.array([0]),
            np.array([0]),
            np.array(range(1, 3)),
            np.array(range(3, 5)),
            np.array(range(5, 9)),
            np.array(range(9, 13)),
            np.array(range(13, 15)),
            np.array(range(15, 21))]

        for _ in range(8):
            new_vector[_] = np.average(
                parameter_vector[ind_old[_][:, None]],
                weights=self._pop[ind_old[_][:, None]])

        return new_vector

    def _compute_updated_Q(self, Q, alpha, tau):
        """
        Updates step-wise the current guess of the auxiliary parameter value Q
        used in computing the model parameters for the guesses of alpha and
        tau.

        Parameters
        ----------
        Q : int or float
            The current guess of the auxiliary parameter value Q.
        alpha : int or float
            The current guess of the auxiliary scenario weight parameter alpha.
        tau : int or float
            The current guess of the reduction in the transmission rate of
            infection for asymptomatic individuals.

        Returns
        -------
        tuple of lists
            Tuple of the updated guess of the auxiliary parameter value Q
            and force of infection vector.
        """
        symp_cases = self._extended_infectives
        # Compute symptom probability vector
        d = np.power(Q, 1-alpha)

        # Compute asymptomatic cases U from known symptomatic cases D
        asymp_cases = np.matmul(np.divide(1-d, d), symp_cases)

        # New unnormalised value for Q
        transmission = np.dot(
            self._nonhouse_cont_mat, symp_cases + tau * asymp_cases)
        nQ = np.divide(symp_cases, transmission)

        # Normalise new value of Q
        nQ = nQ / np.max(nQ)

        # Return updated guess of Q based on prior value and transmission
        # vector
        return 0.9 * Q + 0.1 * nQ, transmission

    def _log_likelihood(self, var_parameters):
        """
        Computes the log-likelihood of the non-fixed parameters
        using death and serology data.

        Parameters
        ----------
        var_parameters : list
            List of varying parameters of the model for which
            the log-likelihood is computed for.

        Returns
        -------
        float
            Value of the log-likelihood for the given choice of
            free parameters.

        """
        # Update parameters
        # alpha
        alpha = var_parameters[-3]
        # tau
        self._parameters[-6] = var_parameters[-2]
        # epsilon
        self._parameters[-5] = var_parameters[-1]

        # Compute Q
        Q_guess = [
            0.0185, 0.0019, 0.0029, 0.0041, 0.0200, 0.0355, 0.0383,
            0.0319, 0.0368, 0.0507, 0.0947, 0.1497, 0.1939, 0.4396,
            0.5789, 0.4939, 0.7038, 0.9309, 0.9818, 0.8767, 1.0000]

        for _ in range(100):
            Q_guess, transmission = self._compute_updated_Q(
                Q_guess, alpha, self._parameters[-6])

        # Compute d
        d = 0.9 * np.power(Q_guess, 1-alpha)

        # Compute sigma
        sigma = (1/0.9) * np.power(Q_guess, alpha)

        # Compute gamma
        self._parameters[-4] = transmission[-1]

        # Recompute d and sigma with correct number of age groups
        self._parameters[-3] = self._update_age_groups(d)
        self._parameters[-7] = self._update_age_groups(sigma)

        total_log_lik = 0

        # Compute log-likelihood
        try:
            for r, _ in enumerate(self._model.regions):
                self._parameters[0] = r+1

                model_output = self._model._simulate(
                    parameters=list(deepflatten(self._parameters, ignore=str)),
                    times=self._times
                    )
                model_new_deaths = self._model.new_deaths(model_output)

                # Check the input of log-likelihoods fixed data
                self._model.check_death_format(model_new_deaths, self._niu)

                self._model.check_positives_format(
                    model_output,
                    self._total_tests[r],
                    self._sens,
                    self._spec)

                # Log-likelihood contribution from death data
                for t, time in enumerate(self._deaths_times):
                    total_log_lik += self._wd * self._model.loglik_deaths(
                        obs_death=self._deaths[r][t, :],
                        new_deaths=model_new_deaths,
                        niu=self._niu,
                        k=time-1
                    )

                # Log-likelihood contribution from serology data
                for t, time in enumerate(self._serology_times):
                    total_log_lik += self._wp * \
                        self._model.loglik_positive_tests(
                            obs_pos=self._positive_tests[r][t, :],
                            output=model_output,
                            tests=self._total_tests[r][t, :],
                            sens=self._sens,
                            spec=self._spec,
                            k=time-1
                        )

            return np.sum(total_log_lik)

        except ValueError:  # pragma: no cover
            return -np.inf

    def set_fixed_parameters(self):
        """
        Sets the non-changing parameters of the model in the class structure
        to save time in the evaluation of the log-likelihood.

        """
        # Use prior mean for the over-dispersion parameter
        self._niu = 10**(-5)

        # Initial Conditions
        susceptibles = self._susceptibles

        exposed_f = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed_sd = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed_su = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed_q = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        detected_f = self._infectives

        detected_qf = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        detected_sd = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        detected_su = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        detected_qs = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        undetected_f = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        undetected_s = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        undetected_q = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        recovered = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        # Regional household quarantine proportions
        h = [0.8, 0.8]

        # Disease-specific parameters
        tau = 0.4
        d = 0.4 * np.ones(self._model._num_ages)

        # Transmission parameters
        epsilon = 0.5
        gamma = 1
        sigma = 0.5 * np.ones(self._model._num_ages)

        self._parameters = [
            0, susceptibles, exposed_f, exposed_sd, exposed_su,
            exposed_q, detected_f, detected_qf, detected_sd, detected_su,
            detected_qs, undetected_f, undetected_s, undetected_q, recovered,
            sigma, tau, epsilon, gamma, d, h, 'RK45'
        ]

    def __call__(self, x):
        """
        Evaluates the log-likelihood in a PINTS framework.

        Parameters
        ----------
        x : list
            List of free parameters used for computing the log-likelihood.

        Returns
        -------
        float
            Value of the log-likelihood at the given point in the free
            parameter space.

        """
        return self._log_likelihood(x)

#
# WarwickLogPrior Class
#


class WarwickLogPrior(pints.LogPrior):
    """WarwickLogPrior Class:
    Controller class to construct the log-prior needed for optimisation or
    inference in a PINTS framework.

    Parameters
    ----------
    model : WarwickSEIRModel
        The model for which we solve the optimisation or inference problem.
    times : list
        List of time points at which we have data for the log-likelihood
        computation.

    """
    def __init__(self, model, times):
        super(WarwickLogPrior, self).__init__()
        # Set the prerequisites for the inference wrapper
        # Model
        self._model = model
        self._times = times

    def n_parameters(self):
        """
        Returns number of parameters for log-prior object.

        Returns
        -------
        int
            Number of parameters for log-prior object.

        """
        return 3

    def __call__(self, x):
        """
        Evaluates the log-prior in a PINTS framework.

        Parameters
        ----------
        x : list
            List of free parameters used for computing the log-prior.

        Returns
        -------
        float
            Value of the log-prior at the given point in the free
            parameter space.

        """
        # Prior contribution of alpha
        log_prior = pints.UniformLogPrior([0], [1])(x[0])

        # Prior contribution of tau
        log_prior += pints.UniformLogPrior([0], [10])(x[1])

        # Prior contribution of epsilon
        log_prior += pints.UniformLogPrior([0], [10])(x[2])

        return log_prior

#
# WarwickSEIRInfer Class
#


class WarwickSEIRInfer(object):
    """WarwickSEIRInfer Class:
    Controller class for the optimisation or inference of parameters of the
    Warwick model in a PINTS framework.

    Parameters
    ----------
    model : WarwickSEIRModel
        The model for which we solve the optimisation or inference problem.

    """
    def __init__(self, model):
        super(WarwickSEIRInfer, self).__init__()

        # Assign model for inference or optimisation
        if not isinstance(model, em.WarwickSEIRModel):
            raise TypeError('Wrong model type for parameters inference.')

        self._model = model

    def read_model_data(
            self, susceptibles_data, infectives_data):
        """
        Sets the initial data used for the model's parameters optimisation or
        inference.

        Parameters
        ----------
        susceptibles_data : list
            List of regional age-structured lists of the initial number of
            susceptibles.
        infectives_data : list
            List of regional age-structured lists of the initial number of
            infectives in the presymptomatic infectives compartment.

        """
        self._susceptibles_data = susceptibles_data
        self._infectives_data = infectives_data

    def read_extended_population_structure(
            self, extended_susceptibles, extended_infectives):
        """
        Sets the initial data with more age groups used for the model's
        parameters optimisation or inference.

        Parameters
        ----------
        extended_susceptibles : list
            List of regional age-structured lists of the initial number of
            susceptibles with more age groups.
        extended_infectives : list
            List of regional age-structured lists of the initial number of
            infectives with more age groups.

        """
        self._extended_susceptibles = extended_susceptibles
        self._extended_infectives = extended_infectives

    def read_extended_contact_matrices(
            self, extended_house_cont_mat, extended_school_cont_mat,
            extended_work_cont_mat, extended_other_cont_mat):
        """
        Sets the initial contact matrix with more age groups used
        for the model's parameters optimisation or inference.

        Parameters
        ----------
        extended_house_cont_mat : ContactMatrix
            Initial contact matrix with more age groups used for the modelling,
            underlying household interactions.
        extended_school_cont_mat : ContactMatrix
            Initial contact matrix with more age groups used for the modelling,
            underlying school interactions.
        extended_work_cont_mat : ContactMatrix
            Initial contact matrix with more age groups used for the modelling,
            underlying workplace interactions.
        extended_other_cont_mat : ContactMatrix
            Initial contact matrix with more age groups used for the modelling,
            underlying other non-household interactions.

        """
        self._extended_house_cont_mat = extended_house_cont_mat
        self._extended_school_cont_mat = extended_school_cont_mat
        self._extended_work_cont_mat = extended_work_cont_mat
        self._extended_other_cont_mat = extended_other_cont_mat

    def read_serology_data(
            self, tests_data, positives_data, serology_times, sens, spec):
        """
        Sets the serology data used for the model's parameters inference.

        Parameters
        ----------
        tests_data: list of numpy.array
            List of region-specific age-structured number of tests conducted
            as a matrix. Each column represents an age group.
        positives_data : list of numpy.array
            List of region-specific age-structured number of positive test
            results as a matrix. Each column represents an age group.
        serology_times : numpy.array
            Matrix of timepoints for which serology data is available.
        sens : float or int
            Sensitivity of the test (or ratio of true positives).
        spec : float or int
            Specificity of the test (or ratio of true negatives).

        """
        self._total_tests = tests_data
        self._positive_tests = positives_data
        self._serology_times = serology_times
        self._sens = sens
        self._spec = spec

    def read_deaths_data(
            self, deaths_data, deaths_times):
        """
        Sets the serology data used for the model's parameters inference.

        Parameters
        ----------
        deaths_data : numpy.array
            List of region-specific age-structured number of deaths as a
            matrix. Each column represents an age group.
        deaths_times : numpy.array
            Matrix of timepoints for which deaths data is available.

        """
        self._deaths = deaths_data
        self._deaths_times = deaths_times

    def return_loglikelihood(self, times, x, wd=1, wp=1):
        """
        Return the log-likelihood used for the optimisation or inference.

        Parameters
        ----------
        times : list
            List of time points at which we have data for the log-likelihood
            computation.
        x : list
            List of free parameters used for computing the log-likelihood.
        wd : float or int
            Proportion of the contribution of the deaths data to the
            log-likelihood.
        wp : float or int
            Proportion of the contribution of the serology data to the
            log-likelihood.

        Returns
        -------
        float
            Value of the log-likelihood at the given point in the free
            parameter space.

        """
        loglikelihood = WarwickLogLik(
            self._model, self._extended_susceptibles,
            self._extended_infectives, self._extended_house_cont_mat,
            self._extended_school_cont_mat, self._extended_work_cont_mat,
            self._extended_other_cont_mat,
            self._susceptibles_data, self._infectives_data, times,
            self._deaths, self._deaths_times,
            self._total_tests, self._positive_tests, self._serology_times,
            self._sens, self._spec, wd, wp)
        return loglikelihood(x)

    def _create_posterior(self, times, wd, wp):
        """
        Runs the initial conditions optimisation routine for the Warwick model.

        Parameters
        ----------
        times : list
            List of time points at which we have data for the log-likelihood
            computation.
        wd : float or int
            Proportion of the contribution of the deaths data to the
            log-likelihood.
        wp : float or int
            Proportion of the contribution of the serology data to the
            log-likelihood.

        """
        # Create a likelihood
        loglikelihood = WarwickLogLik(
            self._model, self._extended_susceptibles,
            self._extended_infectives, self._extended_house_cont_mat,
            self._extended_school_cont_mat, self._extended_work_cont_mat,
            self._extended_other_cont_mat,
            self._susceptibles_data, self._infectives_data, times,
            self._deaths, self._deaths_times,
            self._total_tests, self._positive_tests, self._serology_times,
            self._sens, self._spec, wd, wp)

        # Create a prior
        log_prior = WarwickLogPrior(self._model, times)

        self.ll = loglikelihood

        # Create a posterior log-likelihood (log(likelihood * prior))
        self._log_posterior = pints.LogPosterior(loglikelihood, log_prior)

    def inference_problem_setup(self, times, num_iter, wd=1, wp=1):
        """
        Runs the parameter inference routine for the Warwick model.

        Parameters
        ----------
        times : list
            List of time points at which we have data for the log-likelihood
            computation.
        num_iter : integer
            Number of iterations the MCMC sampler algorithm is run for.
        wd : float or int
            Proportion of the contribution of the deaths data to the
            log-likelihood.
        wp : float or int
            Proportion of the contribution of the serology data to the
            log-likelihood.

        Returns
        -------
        numpy.array
            3D-matrix of the proposed parameters for each iteration for
            each of the chains of the MCMC sampler.

        """
        # Starting points using optimisation object
        x0 = [self.optimisation_problem_setup(times, wd, wp)[0].tolist()]*3

        # Create MCMC routine
        mcmc = pints.MCMCController(
            self._log_posterior, 3, x0)
        mcmc.set_max_iterations(num_iter)
        mcmc.set_log_to_screen(True)
        mcmc.set_parallel(True)

        print('Running...')
        chains = mcmc.run()
        print('Done!')

        param_names = ['alpha', 'tau', 'epsilon']

        # Check convergence and other properties of chains
        results = pints.MCMCSummary(
            chains=chains, time=mcmc.time(),
            parameter_names=param_names)
        print(results)

        return chains

    def optimisation_problem_setup(self, times, wd=1, wp=1):
        """
        Runs the initial conditions optimisation routine for the Warwick model.

        Parameters
        ----------
        times : list
            List of time points at which we have data for the log-likelihood
            computation.
        wd : float or int
            Proportion of the contribution of the deaths data to the
            log-likelihood.
        wp : float or int
            Proportion of the contribution of the serology data to the
            log-likelihood.

        Returns
        -------
        numpy.array
            Matrix of the optimised parameters at the end of the optimisation
            procedure.
        float
            Value of the log-posterior at the optimised point in the free
            parameter space.

        """
        self._create_posterior(times, wd, wp)

        # Starting points
        x0 = [1, 0, 0.5]

        # Create optimisation routine
        optimiser = pints.OptimisationController(
            self._log_posterior, x0, method=pints.CMAES)

        optimiser.set_max_unchanged_iterations(100, 1)

        found_ics, found_posterior_val = optimiser.run()
        print(found_ics, found_posterior_val)

        print("Optimisation phase is finished.")

        return found_ics, found_posterior_val
