#
# SEIRDLogLik Class
#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""
This script contains code for parameter inference of the simple SEIRD model.

"""

import numpy as np
import pints
from iteration_utilities import deepflatten

import epimodels as em


class SEIRDLogLik(pints.LogPDF):
    """SEIRDLogLik Class:
    Controller class to construct the log-likelihood needed for optimisation or
    inference in a PINTS framework.

    Parameters
    ----------
    model : SEIRDModel
        The model for which we solve the optimisation or inference problem.
    susceptibles_data : list
        List of regional age-structured lists of the initial number of
        susceptibles.
    infectives_data : list
        List of regional age-structured lists of the initial number of
        infectives.
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
    Pd : float or int
        Proportion of dead cases.
    wd : float or int
        Proportion of the contribution of the deaths data to the
        log-likelihood.
    wp : float or int
        Proportion of the contribution of the serology data to the
        log-likelihood.

    """
    def __init__(self, model, susceptibles_data, infectives_data, times,
                 deaths, deaths_times,
                 tests_data, positives_data, serology_times, sens, spec,
                 Pd, wd=1, wp=1):
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

        # Probability of dying
        self._Pd = Pd

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
        return 2

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
        # # Update parameters
        # beta
        self._parameters[-5] = var_parameters[0]
        # kappa
        self._parameters[-4] = var_parameters[1]
        # # gamma
        # self._parameters[-3] = var_parameters[1]

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
                        obs_death=self._deaths[r][t],
                        new_deaths=model_new_deaths,
                        niu=self._niu,
                        k=time-1
                    )

                # Log-likelihood contribution from serology data
                for t, time in enumerate(self._serology_times):
                    total_log_lik += self._wp * \
                        self._model.loglik_positive_tests(
                            obs_pos=self._positive_tests[r][t],
                            output=model_output,
                            tests=self._total_tests[r][t],
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

        exposed = np.zeros(len(self._model.regions)).tolist()

        infectives = self._infectives

        recovered = np.zeros(len(self._model.regions)).tolist()

        dead = np.zeros(len(self._model.regions)).tolist()

        # Transmission parameters
        beta = 5
        kappa = 0.25
        gamma = 0.25

        self._parameters = [
            0, susceptibles, exposed, infectives,
            recovered, dead, beta, kappa, gamma, self._Pd, 'RK45'
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
# SEIRDLogPrior Class
#


class SEIRDLogPrior(pints.LogPrior):
    """SEIRDLogPrior Class:
    Controller class to construct the log-prior needed for optimisation or
    inference in a PINTS framework.

    Parameters
    ----------
    model : SEIRDModel
        The model for which we solve the optimisation or inference problem.
    times : list
        List of time points at which we have data for the log-likelihood
        computation.

    """
    def __init__(self, model, times):
        super(SEIRDLogPrior, self).__init__()
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
        return 2

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
        # Prior contribution of beta
        log_prior = pints.UniformLogPrior([0], [20])(x[0])

        # Prior contribution of kappa
        log_prior += pints.UniformLogPrior([0], [20])(x[1])

        # # Prior contribution of gamma
        # log_prior += pints.UniformLogPrior([0], [20])(x[1])

        return log_prior

#
# SEIRDInfer Class
#


class SEIRDInfer(object):
    """SEIRDInfer Class:
    Controller class for the optimisation or inference of parameters of the
    SEIRD model in a PINTS framework.

    Parameters
    ----------
    model : SEIRDModel
        The model for which we solve the optimisation or inference problem.
    Pd : float or int
        Proportion of dead cases.

    """
    def __init__(self, model, Pd):
        super(SEIRDInfer, self).__init__()

        # Assign model for inference or optimisation
        if not isinstance(model, em.SEIRDModel):
            raise TypeError('Wrong model type for parameters inference.')

        self._model = model
        self._Pd = Pd

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
        loglikelihood = SEIRDLogLik(
            self._model, self._susceptibles_data, self._infectives_data, times,
            self._deaths, self._deaths_times,
            self._total_tests, self._positive_tests, self._serology_times,
            self._sens, self._spec, self._Pd, wd, wp)
        return loglikelihood(x)

    def _create_posterior(self, times, wd, wp):
        """
        Runs the initial conditions optimisation routine for the SEIRD model.

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
        loglikelihood = SEIRDLogLik(
            self._model, self._susceptibles_data, self._infectives_data, times,
            self._deaths, self._deaths_times,
            self._total_tests, self._positive_tests, self._serology_times,
            self._sens, self._spec, self._Pd, wd, wp)

        # Create a prior
        log_prior = SEIRDLogPrior(self._model, times)

        self.ll = loglikelihood

        # Create a posterior log-likelihood (log(likelihood * prior))
        self._log_posterior = pints.LogPosterior(loglikelihood, log_prior)

    def inference_problem_setup(self, times, num_iter, wd=1, wp=1):
        """
        Runs the parameter inference routine for the SEIRD model.

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
        x0 = [self.optimisation_problem_setup(times, wd, wp)[0].tolist()]*1

        # Create MCMC routine
        mcmc = pints.MCMCController(
            self._log_posterior, 1, x0)
        mcmc.set_max_iterations(num_iter)
        mcmc.set_log_to_screen(True)
        mcmc.set_parallel(True)

        print('Running...')
        chains = mcmc.run()
        print('Done!')

        param_names = ['beta', 'kappa']

        # Check convergence and other properties of chains
        results = pints.MCMCSummary(
            chains=chains, time=mcmc.time(),
            parameter_names=param_names)
        print(results)

        return chains

    def optimisation_problem_setup(self, times, wd=1, wp=1):
        """
        Runs the initial conditions optimisation routine for the SEIRD model.

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
        x0 = [4, 0.2]

        # Create optimisation routine
        optimiser = pints.OptimisationController(
            self._log_posterior, x0, method=pints.CMAES)

        optimiser.set_max_unchanged_iterations(100, 1)

        found_ics, found_posterior_val = optimiser.run()
        print(found_ics, found_posterior_val)

        print("Optimisation phase is finished.")

        return found_ics, found_posterior_val
