#
# PheSEIRInfer Class
#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""
This script contains code for parameter inference of the extended SEIR model
created by Public Health England and Univerity of Cambridge and which is the
official model used by the UK government for policy making.

It uses an extended version of an SEIR model and contact and region specific
matrices.

"""

import numpy as np
import pints
from iteration_utilities import deepflatten
from scipy.stats import norm

import epimodels as em


class PHELogLik(pints.LogPDF):
    """
    Constructs the log-likelihood needed for optimisation in a PINTS framework.

    Parameters
    ----------
    model
        (PheSEIRModel) The model for which we solve the optimisation
        problem.
    times
        (list) List of time points at which we have data for the
        log-likelihood computation.
    deaths_data
        (Numpy array) List of regional numpy arrays of the daily number
        of deaths, split by age category. Each column represents an age
        group.
    time_to_death
        (list) List of probabilities of death of individual d days after
        infection.
    deaths_times
        (Numpy array) List of timepoints for which deaths data is
        available.
    fatality_ratio
        List of age-specific fatality ratios.
    tests_data
        (Numpy array) List of regional numpy arrays of the daily number
        of tests conducted, split by age category. Each column represents
        an age group.
    positives_data
        (Numpy array) List of regional numpy arrays of the daily number
        of positive test results, split by age category. Each column
        represents an age group.
    serology_times
        (Numpy array) List of timepoints for which serology data is
        available.
    sens
        Sensitivity of the test (or ratio of true positives).
    spec
        Specificity of the test (or ratio of true negatives).
    wd
        Proportion of contribution of the deaths_data to the log-likelihood.
    wp
        Proportion of contribution of the poritives_data to the log-likelihood.

    """
    def __init__(self, model, times,
                 deaths, time_to_death, deaths_times, fatality_ratio,
                 tests_data, positives_data, serology_times, sens, spec,
                 wd=1, wp=1):
        # Set the prerequsites for the inference wrapper
        # Model
        self._model = model
        self._times = times

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
        """
        return 1+len(np.arange(44, len(self._times), 7)) * len(
            self._model.regions)

    def _log_likelihood(self, var_parameters):
        """
        Computes the log-likelihood of the non-fixed parameters
        using death and serology data.

        Parameters
        ----------
        var_parameters
            (list) List of varying parameters of the model for which
            the log-likelihood is computed for.

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

        try:
            for r, _ in enumerate(self._model.regions):
                self._parameters[1] = r+1

                model_output = self._model.simulate(
                    parameters=list(deepflatten(self._parameters, ignore=str)),
                    times=self._times
                    )
                model_new_infections = self._model.new_infections(model_output)

                # Check input of log-likelihoods fixed data
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

        except ValueError:
            return -np.inf

    def set_fixed_parameters(self):
        """
        Sets the non-changing parameters of the model in the class structure
        to save time in the evaluation of the log-likelihood.

        """
        # Use prior mean for the over-dispersion parameter
        self._niu = 5

        # Initial Conditions
        susceptibles = [
            #[68124, 299908, 773741, 668994, 1554740, 1632059, 660187, 578319],  # noqa
            [117840, 488164, 1140597, 1033029, 3050671, 2050173, 586472, 495043],  # noqa
            #[116401, 508081, 1321675, 1319046, 2689334, 2765974, 1106091, 943363],  # noqa
            #[85845, 374034, 978659, 1005275, 2036049, 2128261, 857595, 707190],  # noqa
            #[81258, 348379, 894662, 871907, 1864807, 1905072, 750263, 624848],  # noqa
            #[95825, 424854, 1141632, 1044242, 2257437, 2424929, 946459, 844757],  # noqa
            #[53565, 237359, 641486, 635602, 1304264, 1499291, 668999, 584130]  # noqa
            ]

        exposed1 = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed2 = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        infectives1 = [
            # [0, 0, 0, 0, 0, 1, 0, 0],  # noqa
            [10, 10, 50, 100, 150, 50, 50, 50],  # noqa
            # [0, 0, 0, 0, 0, 0, 1, 0],  # noqa
            # [0, 0, 0, 0, 1, 0, 0, 0],  # noqa
            # [0, 0, 0, 0, 0, 1, 0, 0],  # noqa
            # [0, 0, 0, 1, 0, 0, 0, 0],  # noqa
            # [0, 0, 0, 0, 0, 1, 0, 0]  # noqa
            ]

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

        # [var_r] * reg
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

        """
        return self._log_likelihood(x)


class PHELogPrior(pints.LogPrior):
    """
    Constructs the log-prior needed for optimisation in a PINTS framework.

    Parameters
    ----------
    model
        (PheSEIRModel) The model for which we solve the optimisation
        problem.
    times
        (list) List of time points at which we have data for the
        log-likelihood computation.

    """
    def __init__(self, model, times):
        super(PHELogPrior, self).__init__()
        # Set the prerequsites for the inference wrapper
        # Model
        self._model = model
        self._times = times

    def n_parameters(self):
        """
        Returns number of parameters for log-prior object.
        """
        return 1+len(np.arange(44, len(self._times), 7)) * len(
            self._model.regions)

    def __call__(self, x):
        """
        Evaluates the log-prior in a PINTS framework.

        """
        # Prior contribution for initial R
        log_prior = pints.UniformLogPrior([0], [5])(x[0])

        # Variance for betas
        sigma_b = 1/100

        # Prior contriubution for betas
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


class PheSEIRInfer(object):
    """PheSEIRInfer Class:
    Controller class for the inference of parameters of the PHE model.

    """
    def __init__(self, model):
        super(PheSEIRInfer, self).__init__()

        # Assign model for inference
        if not isinstance(model, em.PheSEIRModel):
            raise TypeError('Wrong model type for parameters inference.')

        self._model = model

    def read_serology_data(
            self, tests_data, positives_data, serology_times, sens, spec):
        """
        Sets the serology data used for the model's parameters inference.

        Paramaters
        ----------
        tests_data
            (Numpy array) List of regional numpy arrays of the daily number
            of tests conducted, split by age category. Each column represents
            an age group.
        positives_data
            (Numpy array) List of regional numpy arrays of the daily number
            of positive test results, split by age category. Each column
            represents an age group.
        serology_times
            (Numpy array) List of timepoints for which serology data is
            available.
        sens
            Sensitivity of the test (or ratio of true positives).
        spec
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

        Paramaters
        ----------
        deaths_data
            (Numpy array) List of regional numpy arrays of the daily number
            of deaths, split by age category. Each column represents an age
            group.
        deaths_times
            (Numpy array) List of timepoints for which deaths data is
            available.
        time_to_death
            (list) List of probabilities of death of individual d days after
            infection.
        fatality_ratio
            List of age-specific fatality ratios.

        """
        self._deaths = deaths_data
        self._deaths_times = deaths_times
        self._time_to_death = time_to_death
        self._fatality_ratio = fatality_ratio

    def return_loglikelihood(self, times, x, wd=1, wp=1):
        """
        Return the log-likelihood used for the inference.

        """
        loglikelihood = PHELogLik(
            self._model, times,
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
        times
            (list) List of time points at which we have data for the
            log-likelihood computation.
        wd
            Proportion of contribution of the deaths_data to the
            log-likelihood.
        wp
            Proportion of contribution of the poritives_data to the
            log-likelihood.

        """
        # Create a likelihood
        loglikelihood = PHELogLik(
            self._model, times,
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
        times
            (list) List of time points at which we have data for the
            log-likelihood computation.
        num_iter
            Number of iterations the MCMC sampler algorithm is run for.
        wd
            Proportion of contribution of the deaths_data to the
            log-likelihood.
        wp
            Proportion of contribution of the poritives_data to the
            log-likelihood.

        """
        # Starting points using optimisation object
        x0 = [self.optimisation_problem_setup(times, wd, wp).tolist()]*3

        # Create MCMC routine
        mcmc = pints.MCMCController(
            self._log_posterior, 3, x0)
        mcmc.set_max_iterations(num_iter)
        mcmc.set_log_to_screen(True)

        print('Running...')
        chains = mcmc.run()
        print('Done!')

        param_names = ['initial_r']
        param_names.extend(['beta_W{}'.format(
            i+1) for i in range(len(np.arange(44, len(times), 7)))])

        # Check convergence and other properties of chains
        results = pints.MCMCSummary(
            chains=chains, time=mcmc.time(),
            parameter_names=param_names)
        print(results)

        return chains

    def optimisation_problem_setup(self, times,  wd=1, wp=1):
        """
        Runs the initial conditions optimisation routine for the PHE model.

        Parameters
        ----------
        times
            (list) List of time points at which we have data for the
            log-likelihood computation.
        wd
            Proportion of contribution of the deaths_data to the
            log-likelihood.
        wp
            Proportion of contribution of the poritives_data to the
            log-likelihood.

        """
        self._create_posterior(times, wd, wp)

        # Starting points
        x0 = [3]
        x0.extend([1]*len(np.arange(44, len(times), 7)))

        # Create Optimisation routine
        optimiser = pints.OptimisationController(
            self._log_posterior, x0, method=pints.CMAES)

        optimiser.set_max_unchanged_iterations(100, 1)

        found_ics, found_posterior_val = optimiser.run()
        print(found_ics, found_posterior_val)

        print("Optimisation phase is finished.")

        return found_ics, found_posterior_val
