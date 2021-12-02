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

    def _log_likelihood(self, times, var_parameters):
        """
        Computes the log-likelihood of the non-fixed parameters
        using death and serology data.

        Parameters
        ----------
        times
            (list) List of time points at which we have data for the
            log-likelihood computation.
        var_parameters
            List of values for the model paramaters to infer.

        """
        # Use prior mean for the over-dispersion parameter
        niu = 5

        # Set fixed parameters
        # Initial Conditions
        susceptibles = [
            [68124, 299908, 773741, 668994, 1554740, 1632059, 660187, 578319],
            [117840, 488164, 1140597, 1033029, 3050671, 2050173, 586472, 495043],  # noqa
            [116401, 508081, 1321675, 1319046, 2689334, 2765974, 1106091, 943363],  # noqa
            [85845, 374034, 978659, 1005275, 2036049, 2128261, 857595, 707190],
            [81258, 348379, 894662, 871907, 1864807, 1905072, 750263, 624848],
            [95825, 424854, 1141632, 1044242, 2257437, 2424929, 946459, 844757],  # noqa
            [53565, 237359, 641486, 635602, 1304264, 1499291, 668999, 584130]]

        exposed1 = np.zeros((
            len(self._model.regions),
            len(self._model._num_ages))).tolist()

        exposed2 = np.zeros((
            len(self._model.regions),
            len(self._model._num_ages))).tolist()

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
            len(self._model._num_ages))).tolist()

        recovered = np.zeros((
            len(self._model.regions),
            len(self._model._num_ages))).tolist()

        # Beta multipliers
        betas = np.ones((len(self._model.regions), len(times))).tolist()

        # Other parameters
        dI = 4
        dL = 4
        delta_t = 0.5

        # [var_r] * reg
        parameters = [
            var_parameters,
            0,
            susceptibles, exposed1, exposed2, infectives1, infectives2,
            recovered,
            betas,
            dL,
            dI,
            delta_t,
            'RK45'
        ]

        total_log_lik = 0

        for r, _ in enumerate(self._model.regions):
            parameters[1] = r+1

            model_output = self._model.simulate(
                parameters=list(deepflatten(parameters, ignore=str)),
                times=times
            )

            for t in times:
                total_log_lik += self._model.loglik_deaths(
                    obs_death=self._deaths[r][t, :],
                    output=model_output,
                    fatality_ratio=self._fatality_ratio,
                    time_to_death=self._time_to_death,
                    niu=niu,
                    k=t
                ) + self._model.loglik_positive_tests(
                    obs_pos=self._positive_tests[r][t, :],
                    output=model_output,
                    tests=self._total_tests[r][t, :],
                    sens=self._sens,
                    spec=self._spec,
                    k=t
                )

        return total_log_lik

    def inference_problem_setup(self, times, var_parameters):
        """
        Runs the parameter inference routine for the PHE model.
        """
        loglikelihood = self._log_likelihood(times, var_parameters)

        # Starting points
        x0 = [
            [0.001, 0.20, 52, 3, 3],
            [0.05, 0.34, 34, 3, 3],
            [0.02, 0.18, 20, 3, 3],
        ]

        # Create MCMC routine
        mcmc = pints.MCMCController(loglikelihood, 3, x0)
        mcmc.set_max_iterations(3000)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()

        # Check convergence and other properties of chains
        results = pints.MCMCSummary(
            chains=chains, time=mcmc.time(),
            parameter_names=[
                'gamma', 'v', 'S_0', 'sigma infected',
                'sigma recovery'])
        print(results)
