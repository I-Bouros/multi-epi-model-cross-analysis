#
# PHELogLik Class
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
from scipy.stats import norm, gamma

import epimodels as em


class PHELogLik(pints.LogPDF):
    """PHELogLik Class:
    Controller class to construct the log-likelihood needed for optimisation or
    inference in a PINTS framework.

    Parameters
    ----------
    model : PheSEIRModel
        The model for which we solve the optimisation or inference problem.
    susceptibles_data : list
        List of regional age-structured lists of the initial number of
        susceptibles.
    infectives_data : list
        List of regional age-structured lists of the initial number of
        infectives in the first infectives compartment.
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
        return 2+len(np.arange(44, len(self._times), 7)) * len(
            self._model.regions)

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
        self._parameters[0] = [var_parameters[0]] * len(self._model.regions)

        LEN = len(np.arange(44, len(self._times), 7))

        betas = np.array(self._parameters[8])
        for r in range(len(self._model.regions)):
            for d, day in enumerate(np.arange(44, len(self._times), 7)):
                betas[r, day:(day+7)] = var_parameters[r*LEN+d+1]

        self._parameters[8] = betas.tolist()

        total_log_lik = 0

        # Compute log-likelihood
        try:
            for r, _ in enumerate(self._model.regions):
                self._parameters[1] = r+1

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
        self._niu = 10**(-5)

        # Initial Conditions
        susceptibles = self._susceptibles

        exposed1 = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed2 = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        infectives1 = self._infectives

        infectives2 = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        recovered = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        # Beta multipliers
        betas = np.ones((len(self._model.regions), len(self._times))).tolist()

        # Other Parameters
        dI = 4
        dL = 4
        delta_t = 0.5

        self._parameters = [
            np.zeros(len(self._model.regions)).tolist(),
            0,
            susceptibles, exposed1, exposed2, infectives1, infectives2,
            recovered,
            betas,
            dL,
            dI,
            delta_t,
            'RK45'
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
        return self._log_likelihood(x[:-1])

#
# PHELogPrior Class
#


class PHELogPrior(pints.LogPrior):
    """PHELogPrior Class:
    Controller class to construct the log-prior needed for optimisation or
    inference in a PINTS framework.

    Parameters
    ----------
    model : PheSEIRModel
        The model for which we solve the optimisation or inference problem.
    times : list
        List of time points at which we have data for the log-likelihood
        computation.

    """
    def __init__(self, model, times):
        super(PHELogPrior, self).__init__()
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
        return 2+len(np.arange(44, len(self._times), 7)) * len(
            self._model.regions)

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
        # Prior contribution of initial R
        log_prior = pints.UniformLogPrior([0], [5])(x[0])

        # Variance for betas
        sigma_b = x[-1]

        log_prior += gamma.logpdf(sigma_b, 1, scale=1/100)

        # Prior contribution of betas
        LEN = len(np.arange(44, len(self._times), 7))
        for r in range(len(self._model.regions)):
            log_prior += norm.logpdf(
                    np.log(x[r*LEN+1]),
                    loc=0,
                    scale=sigma_b)
            for d in range(1, LEN):
                log_prior += norm.logpdf(
                    np.log(x[r*LEN+d+1]),
                    loc=np.log(x[r*LEN+d]),
                    scale=sigma_b)

        return log_prior

#
# PheSEIRInfer Class
#


class PheSEIRInfer(object):
    """PheSEIRInfer Class:
    Controller class for the optimisation or inference of parameters of the
    PHE model in a PINTS framework.

    Parameters
    ----------
    model : PheSEIRModel
        The model for which we solve the optimisation or inference problem.

    """
    def __init__(self, model):
        super(PheSEIRInfer, self).__init__()

        # Assign model for inference or optimisation
        if not isinstance(model, em.PheSEIRModel):
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
            infectives in the first infectives compartment.

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
        loglikelihood = PHELogLik(
            self._model, self._susceptibles_data, self._infectives_data, times,
            self._deaths, self._time_to_death, self._deaths_times,
            self._fatality_ratio,
            self._total_tests, self._positive_tests, self._serology_times,
            self._sens, self._spec, wd, wp)
        return loglikelihood(x)

    def _create_posterior(self, times, wd, wp):
        """
        Runs the initial conditions optimisation routine for the PHE model.

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
        loglikelihood = PHELogLik(
            self._model, self._susceptibles_data, self._infectives_data, times,
            self._deaths, self._time_to_death, self._deaths_times,
            self._fatality_ratio,
            self._total_tests, self._positive_tests, self._serology_times,
            self._sens, self._spec, wd, wp)

        # Create a prior
        log_prior = PHELogPrior(self._model, times)

        # Create a posterior log-likelihood (log(likelihood * prior))
        self._log_posterior = pints.LogPosterior(loglikelihood, log_prior)

    def inference_problem_setup(self, times, num_iter, wd=1, wp=1):
        """
        Runs the parameter inference routine for the PHE model.

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

        param_names = ['initial_r']
        for region in self._model.regions:
            param_names.extend(['beta_W{}_{}'.format(
                i+1, region) for i in range(
                    len(np.arange(44, len(times), 7)))])
        param_names.extend(['sigma_b'])

        # Check convergence and other properties of chains
        results = pints.MCMCSummary(
            chains=chains, time=mcmc.time(),
            parameter_names=param_names)
        print(results)

        return chains

    def optimisation_problem_setup(self, times, wd=1, wp=1):
        """
        Runs the initial conditions optimisation routine for the PHE model.

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
        x0 = [3]
        x0.extend([1]*(len(np.arange(44, len(times), 7)) * len(
            self._model.regions)))
        x0.extend([0.1])

        # Create optimisation routine
        optimiser = pints.OptimisationController(
            self._log_posterior, x0, method=pints.CMAES)

        optimiser.set_max_unchanged_iterations(100, 1)

        found_ics, found_posterior_val = optimiser.run()
        print(found_ics, found_posterior_val)

        print("Optimisation phase is finished.")

        return found_ics, found_posterior_val
