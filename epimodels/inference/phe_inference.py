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

import epimodels as em


class InferLogLikelihood(pints.LogPDF):
    """
    Parameters
    ----------
    model
        (PheSEIRModel) The model for which we solve the inference
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
    tests_data
        (Numpy array) List of regional numpy arrays of the daily number
        of tests conducted, split by age category. Each column represents
        an age group.
    positives_data
        (Numpy array) List of regional numpy arrays of the daily number
        of positive test results, split by age category. Each column
        represents an age group.
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
                 deaths, time_to_death, deaths_times,
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
        return 1+self._model._num_ages

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
        fatality_ratio = (np.asarray(var_parameters[1:]) * 10**(-4)).tolist()

        total_log_lik = 0

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
                fatality_ratio,
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
                    fatality_ratio=fatality_ratio,
                    time_to_death=self._time_to_death,
                    niu=self._niu,
                    k=time-1
                )

            # Log-likelihood contribution from serology data
            for t, time in enumerate(self._serology_times):
                total_log_lik += self._wp * self._model.loglik_positive_tests(
                    obs_pos=self._positive_tests[r][t, :],
                    output=model_output,
                    tests=self._total_tests[r][t, :],
                    sens=self._sens,
                    spec=self._spec,
                    k=time-1
                )

        return np.sum(total_log_lik)

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
            [81258, 348379, 894662, 871907, 1864807, 1905072, 750263, 624848],  # noqa
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
            [0, 0, 0, 0, 2, 0, 0, 0],  # noqa
            # [0, 0, 0, 0, 0, 0, 1, 0],  # noqa
            # [0, 0, 0, 0, 1, 0, 0, 0],  # noqa
            [0, 0, 0, 0, 0, 1, 0, 0],  # noqa
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
        Evaluates the log-lokelihood in a PINTS framework.

        """
        return self._log_likelihood(x)


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

    def read_deaths_data(self, deaths_data, deaths_times, time_to_death):
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

        """
        self._deaths = deaths_data
        self._deaths_times = deaths_times
        self._time_to_death = time_to_death

    def return_loglikelihood(self, times, x, wd=1, wp=1):
        """
        Return the log-likelihood used for the inference.

        """
        loglikelihood = InferLogLikelihood(
            self._model, times,
            self._deaths, self._time_to_death,
            self._total_tests, self._positive_tests, self._sens, self._spec,
            wd, wp)
        return loglikelihood(x)

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
        loglikelihood = InferLogLikelihood(
            self._model, times,
            self._deaths, self._time_to_death, self._deaths_times,
            self._total_tests, self._positive_tests, self._serology_times,
            self._sens, self._spec, wd, wp)

        upper_bd = [5]
        upper_bd.extend([10**4] * self._model._num_ages)

        uniform_log_prior = pints.UniformLogPrior(
            [0] * (self._model._num_ages+1),
            upper_bd)

        # Create a posterior log-likelihood (log(likelihood * prior))
        log_posterior = pints.LogPosterior(loglikelihood, uniform_log_prior)

        # Starting points
        x0 = [
            [3, 0.16, 0.16, 0.43, 1.9, 8.975, 81.5, 310, 605],
            [3, 0.16, 0.16, 0.43, 1.9, 8.975, 81.5, 310, 605],
            [3, 0.16, 0.16, 0.43, 1.9, 8.975, 81.5, 310, 605],
        ]

        # print(loglikelihood(x0[0]))

        # Create MCMC routine
        mcmc = pints.MCMCController(
            log_posterior, 3, x0)
        mcmc.set_max_iterations(num_iter)
        mcmc.set_log_to_screen(True)

        print('Running...')
        chains = mcmc.run()
        print('Done!')

        param_names = ['initial_r']
        param_names.extend(['fatal_ratio_{}'.format(
            i+1) for i in range(self._model._num_ages)])

        # Check convergence and other properties of chains
        results = pints.MCMCSummary(
            chains=chains, time=mcmc.time(),
            parameter_names=param_names)
        print(results)

        return chains
