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
    fatality_ratio
        (list) List of age-specific fatality ratios.
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

    """
    def __init__(self, model, times,
                 deaths, fatality_ratio, time_to_death,
                 tests_data, positives_data, sens, spec):
        # Set the prerequsites for the inference wrapper
        self._model = model
        self._times = times
        self._deaths = deaths
        self._fatality_ratio = fatality_ratio
        self._time_to_death = time_to_death
        self._total_tests = tests_data
        self._positive_tests = positives_data
        self._sens = sens
        self._spec = spec

        # Set fixed parameters of the model
        self.set_fixed_parameters()

    def n_parameters(self):
        return 1

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
        self._parameters[0] = [var_parameters] * len(self._model.regions)

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
                self._fatality_ratio,
                self._time_to_death,
                self._niu)

            self._model.check_positives_format(
                model_output,
                self._total_tests[r],
                self._sens,
                self._spec)

            for t, _ in enumerate(self._times):
                total_log_lik += self._model.loglik_deaths(
                    obs_death=self._deaths[r][t, :],
                    new_infections=model_new_infections,
                    fatality_ratio=self._fatality_ratio,
                    time_to_death=self._time_to_death,
                    niu=self._niu,
                    k=t
                ) + self._model.loglik_positive_tests(
                    obs_pos=self._positive_tests[r][t, :],
                    output=model_output,
                    tests=self._total_tests[r][t, :],
                    sens=self._sens,
                    spec=self._spec,
                    k=t
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
            [68124, 299908, 773741, 668994, 1554740, 1632059, 660187, 578319],  # noqa
            [117840, 488164, 1140597, 1033029, 3050671, 2050173, 586472, 495043],  # noqa
            [116401, 508081, 1321675, 1319046, 2689334, 2765974, 1106091, 943363],  # noqa
            [85845, 374034, 978659, 1005275, 2036049, 2128261, 857595, 707190],  # noqa
            [81258, 348379, 894662, 871907, 1864807, 1905072, 750263, 624848],  # noqa
            [95825, 424854, 1141632, 1044242, 2257437, 2424929, 946459, 844757],  # noqa
            [53565, 237359, 641486, 635602, 1304264, 1499291, 668999, 584130]]  # noqa

        exposed1 = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed2 = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        infectives1 = [
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]]

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
        return self._log_likelihood(x[0])


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

    def read_serology_data(self, tests_data, positives_data, sens, spec):
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
        sens
            Sensitivity of the test (or ratio of true positives).
        spec
            Specificity of the test (or ratio of true negatives).

        """
        self._total_tests = tests_data
        self._positive_tests = positives_data
        self._sens = sens
        self._spec = spec

    def read_deaths_data(self, deaths_data, fatality_ratio, time_to_death):
        """
        Sets the serology data used for the model's parameters inference.

        Paramaters
        ----------
        deaths_data
            (Numpy array) List of regional numpy arrays of the daily number
            of deaths, split by age category. Each column represents an age
            group.
        fatality_ratio
            (list) List of age-specific fatality ratios.
        time_to_death
            (list) List of probabilities of death of individual d days after
            infection.

        """
        self._deaths = deaths_data
        self._fatality_ratio = fatality_ratio
        self._time_to_death = time_to_death

    def inference_problem_setup(self, times):
        """
        Runs the parameter inference routine for the PHE model.

        Parameters
        ----------
        times
            (list) List of time points at which we have data for the
            log-likelihood computation.
        """
        loglikelihood = InferLogLikelihood(
            self._model, times,
            self._deaths, self._fatality_ratio, self._time_to_death,
            self._total_tests, self._positive_tests, self._sens, self._spec)

        # Starting points
        x0 = [
            [3],
            [3],
            [3],
        ]

        # print(loglikelihood(x0[0]))

        # Create MCMC routine
        mcmc = pints.MCMCController(
            loglikelihood, 3, x0, method=pints.HaarioACMC)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)

        print('Running...')
        chains = mcmc.run()
        print('Done!')

        # Check convergence and other properties of chains
        results = pints.MCMCSummary(
            chains=chains, time=mcmc.time(),
            parameter_names=[
                'initial_r'])
        print(results)
