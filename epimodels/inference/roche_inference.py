#
# RocheLogLik Class
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


class RocheLogLik(pints.LogPDF):
    """RocheLogLik Class:
    Controller class to construct the log-likelihood needed for optimisation or
    inference in a PINTS framework.

    Parameters
    ----------
    model : RocheSEIRModel
        The model for which we solve the optimisation or inference problem.
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
    time_to_death : list
        List of probabilities of death of individual d days after infection.
    deaths_times : numpy.array
        Matrix of timepoints for which deaths data is available.
    fatality_ratio : list
        List of age-specific fatality ratios.
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
    max_levels_npi : list of int
        List of maximum levels the non-pharmaceutical interventions can
        reach.
    targeted_npi : list of bool
        List of the targeted non-pharmaceutical interventions.
    general_npi : list of list of int
        List of the general values of the targeted non-pharmaceutical
        interventions. In chronological order.
    reg_levels_npi : list of list of int
        List of region-specific levels the non-pharmaceutical interventions
        changes. In chronological order.
    time_changes_npi : list
        List of times at which the next instances of region-specific
        non-pharmaceutical interventions start to be used. In
        increasing order.
    time_changes_flag : list
        List of times at which the next instances of region-specific
        non-pharmaceutical interventions start to be used. In
        increasing order.
    wd : float or int
        Proportion of the contribution of the deaths data to the
        log-likelihood.
    wp : float or int
        Proportion of the contribution of the serology data to the
        log-likelihood.

    """
    def __init__(self, model, susceptibles_data, infectives_data, times,
                 deaths, time_to_death, deaths_times, fatality_ratio,
                 tests_data, positives_data, serology_times, sens, spec,
                 max_levels_npi, targeted_npi, general_npi,
                 reg_levels_npi, time_changes_npi, time_changes_flag,
                 wd=1, wp=1):
        # Set the prerequisites for the inference wrapper
        # Model and ICs data
        self._model = model
        self._times = times

        self._susceptibles = susceptibles_data
        self._infectives = infectives_data

        # Death data
        self._deaths = deaths
        self._deaths_times = deaths_times
        self._time_to_death = time_to_death
        self._fatality_ratio = fatality_ratio

        # Serology data
        self._total_tests = tests_data
        self._positive_tests = positives_data
        self._serology_times = serology_times
        self._sens = sens
        self._spec = spec

        # Non-pharmaceutical interventions data
        self._max_levels_npi = max_levels_npi
        self._targeted_npi = targeted_npi
        self._general_npi = general_npi
        self._reg_levels_npi = reg_levels_npi
        self._time_changes_npi = time_changes_npi
        self._time_changes_flag = time_changes_flag

        # Compute the additional weight for a policy of general scope
        self._w = self._model._compute_add_pol_weight(
            max_levels_npi, targeted_npi)

        # Contribution parameters
        self._wd = wd
        self._wp = wp

        # Set fixed parameters of the model
        self.set_fixed_parameters()

    def n_parameters(self):
        """
        Returns number of parameters for log-likelihood object.

        Returns
        -------
        int
            Number of parameters for log-likelihood object.

        """
        return 4 + self._model._num_ages

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
        Prop = var_parameters[0]

        self._parameters[3] = \
            ((1 - Prop) * np.asarray(self._infectives)).tolist()
        self._parameters[6] = \
            (Prop * np.asarray(self._infectives)).tolist()

        # # Pa
        # self._parameters[-9] = var_parameters[
        #     (-4-2*self._model._num_ages):(-4-self._model._num_ages)]
        # # Pss
        # self._parameters[-8] = var_parameters[-4-self._model._num_ages]
        # Pd
        self._parameters[-7] = var_parameters[(-3-self._model._num_ages):(-3)]
        # beta_min
        self._parameters[-5] = var_parameters[-3]
        # beta_max
        self._parameters[-4] = var_parameters[-2]
        # bss
        self._parameters[-3] = var_parameters[-1]

        total_log_lik = 0

        # Compute log-likelihood
        try:
            for r, _ in enumerate(self._model.regions):
                self._parameters[0] = r+1

                model_output = self._model._simulate(
                    parameters=list(deepflatten(self._parameters, ignore=str)),
                    times=self._times
                    )
                model_new_infections = self._model.new_infections(model_output)

                # Check the input of log-likelihoods fixed data
                self._model.check_death_format(
                    model_new_infections,
                    self._fatality_ratio,
                    self._time_to_death,
                    self._niu)

                self._model.check_positives_format(
                    model_output,
                    self._total_tests[r],
                    self._sens,
                    self._spec)

                # Log-likelihood contribution from death data
                for t, time in enumerate(self._deaths_times):
                    total_log_lik += self._wd * self._model.loglik_deaths(
                        obs_death=self._deaths[r][t, :],
                        new_infections=model_new_infections,
                        fatality_ratio=self._fatality_ratio,
                        time_to_death=self._time_to_death,
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
        self._niu = 5

        # Initial Conditions
        susceptibles = self._susceptibles

        exposed = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        infectives_pre = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        infectives_pre_ss = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        infectives_asym = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        infectives_asym_ss = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        infectives_sym = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        infectives_sym_ss = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        infectives_q = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        recovered = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        recovered_asym = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        dead = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        # Average times in compartments
        k = 4.5
        kS = 1
        kQ = 1
        kR = 9 * np.ones(self._model._num_ages)
        kRI = 10 * np.ones(self._model._num_ages)

        # Proportion of asymptomatic, super-spreader and dead cases
        Pa = 0.716 * np.ones(self._model._num_ages)
        Pss = 0.106
        Pd = 0.05 * np.ones(self._model._num_ages)

        # Transmission parameters
        beta_min = 0.228
        beta_max = 0.928
        bss = 3.11
        gamma = 0.5
        s50 = 51.

        self._parameters = [
            0, susceptibles, exposed, infectives_pre,
            infectives_asym, infectives_sym, infectives_pre_ss,
            infectives_asym_ss, infectives_sym_ss, infectives_q,
            recovered, recovered_asym, dead,
            k, kS, kQ, kR, kRI, Pa, Pss, Pd,
            beta_min, beta_max, bss, gamma, s50, 'RK45'
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
# RocheLogPrior Class
#


class RocheLogPrior(pints.LogPrior):
    """RocheLogPrior Class:
    Controller class to construct the log-prior needed for optimisation or
    inference in a PINTS framework.

    Parameters
    ----------
    model : RocheSEIRModel
        The model for which we solve the optimisation or inference problem.
    times : list
        List of time points at which we have data for the log-likelihood
        computation.

    """
    def __init__(self, model, times):
        super(RocheLogPrior, self).__init__()
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
        return 4 + self._model._num_ages

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
        # Prior contribution of proportion of super-spreaders in intial
        # infectives
        log_prior = pints.UniformLogPrior([0], [1])(x[0])

        # Prior contribution of Pa, Pss, Pd
        for param in range(self._model._num_ages):
            log_prior += pints.UniformLogPrior([0], [1])(x[param + 1])

        # Prior contribution of beta_min
        log_prior += pints.UniformLogPrior([0], [5])(x[-3])

        # Prior contribution of beta_max
        log_prior += pints.UniformLogPrior([0], [5])(x[-2])

        # Prior contribution of bss
        log_prior += pints.UniformLogPrior([0], [10])(x[-1])

        return log_prior

#
# RocheSEIRInfer Class
#


class RocheSEIRInfer(object):
    """RocheSEIRInfer Class:
    Controller class for the optimisation or inference of parameters of the
    Roche model in a PINTS framework.

    Parameters
    ----------
    model : RocheSEIRModel
        The model for which we solve the optimisation or inference problem.

    """
    def __init__(self, model):
        super(RocheSEIRInfer, self).__init__()

        # Assign model for inference or optimisation
        if not isinstance(model, em.RocheSEIRModel):
            raise TypeError('Wrong model type for parameters inference.')

        self._model = model

    def read_model_data(
            self, susceptibles_data, infectives_data):
        """
        Sets the serology data used for the model's parameters optimisation or
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
            self, deaths_data, deaths_times, time_to_death, fatality_ratio):
        """
        Sets the serology data used for the model's parameters inference.

        Parameters
        ----------
        deaths_data : numpy.array
            List of region-specific age-structured number of deaths as a
            matrix. Each column represents an age group.
        deaths_times : numpy.array
            Matrix of timepoints for which deaths data is available.
        time_to_death : list
            List of probabilities of death of individual d days after
            infection.
        fatality_ratio : list
            List of age-specific fatality ratios.

        """
        self._deaths = deaths_data
        self._deaths_times = deaths_times
        self._time_to_death = time_to_death
        self._fatality_ratio = fatality_ratio

    def read_npis_data(self, max_levels_npi, targeted_npi, general_npi,
                       reg_levels_npi, time_changes_npi, time_changes_flag):
        """
        Sets the non-pharmaceutical interventions data used for the model's
        parameters inference.

        Parameters
        ----------
        max_levels_npi : list of int
            List of maximum levels the non-pharmaceutical interventions can
            reach.
        targeted_npi : list of bool
            List of the targeted non-pharmaceutical interventions.
        general_npi : list of list of int
            List of the general values of the targeted non-pharmaceutical
            interventions. In chronological order.
        reg_levels_npi : list of list of int
            List of region-specific levels the non-pharmaceutical interventions
            changes. In chronological order.
        time_changes_npi : list
            List of times at which the next instances of region-specific
            non-pharmaceutical interventions start to be used. In
            increasing order.
        time_changes_flag : list
            List of times at which the next instances of region-specific
            non-pharmaceutical interventions start to be used. In
            increasing order.

        """
        self._max_levels_npi = max_levels_npi
        self._targeted_npi = targeted_npi
        self._general_npi = general_npi
        self._reg_levels_npi = reg_levels_npi
        self._time_changes_npi = time_changes_npi
        self._time_changes_flag = time_changes_flag

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
        loglikelihood = RocheLogLik(
            self._model, self._susceptibles_data, self._infectives_data, times,
            self._deaths, self._time_to_death, self._deaths_times,
            self._fatality_ratio,
            self._total_tests, self._positive_tests, self._serology_times,
            self._sens, self._spec,
            self._max_levels_npi, self._targeted_npi, self._general_npi,
            self._reg_levels_npi, self._time_changes_npi,
            self._time_changes_flag, wd, wp)
        return loglikelihood(x)

    def _create_posterior(self, times, wd, wp):
        """
        Runs the initial conditions optimisation routine for the Roche model.

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
        loglikelihood = RocheLogLik(
            self._model, self._susceptibles_data, self._infectives_data, times,
            self._deaths, self._time_to_death, self._deaths_times,
            self._fatality_ratio,
            self._total_tests, self._positive_tests, self._serology_times,
            self._sens, self._spec,
            self._max_levels_npi, self._targeted_npi, self._general_npi,
            self._reg_levels_npi, self._time_changes_npi,
            self._time_changes_flag, wd, wp)

        # Create a prior
        log_prior = RocheLogPrior(self._model, times)

        # Create a posterior log-likelihood (log(likelihood * prior))
        self._log_posterior = pints.LogPosterior(loglikelihood, log_prior)

    def inference_problem_setup(self, times, num_iter, wd=1, wp=1):
        """
        Runs the parameter inference routine for the Roche model.

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

        param_names = ['Initial Pss']
        for age in self._model.age_groups:
            param_names.extend('Pa_{}'.format(age))
        param_names.extend('Pss')

        for age in self._model.age_groups:
            param_names.extend('Pd_{}'.format(age))

        param_names.extend(['beta_min', 'beta_max', 'bss'])

        # Check convergence and other properties of chains
        results = pints.MCMCSummary(
            chains=chains, time=mcmc.time(),
            parameter_names=param_names)
        print(results)

        return chains

    def optimisation_problem_setup(self, times, wd=1, wp=1):
        """
        Runs the initial conditions optimisation routine for the Roche model.

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
        x0 = [0.5]
        x0 += self._model._num_ages * [0.05]
        x0 += [0.135] + [1.08] + [3]

        # Create optimisation routine
        optimiser = pints.OptimisationController(
            self._log_posterior, x0, method=pints.CMAES)

        optimiser.set_max_unchanged_iterations(100, 1)

        found_ics, found_posterior_val = optimiser.run()
        print(found_ics, found_posterior_val)

        print("Optimisation phase is finished.")

        return found_ics, found_posterior_val
