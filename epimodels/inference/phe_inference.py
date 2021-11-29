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

import pints

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

    def read_serology_data(self, tests_data, positives_data):
        """
        """
        self._total_tests = tests_data
        self._positive_tests = positives_data

    def read_deaths_data(self, deaths_data):
        """
        """
        self._deaths = deaths_data

    def _log_likelihood(self):
        """
        """
        pass

    def _inference_problem_setup(self, times, observed_output):
        """
        """
        self._infer_problem = pints.MultiOutputProblem(
            self._model, times, observed_output)

        self._score = pints.SumOfSquaresError(self._infer_problem)

        loglikelihood = self._log_likelihood()

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
