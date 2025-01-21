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
created by the University of Warwick. This is one of the official models used
by the UK government for policy making.

It uses an extended version of an SEIR model with contact and region-specific
matrices and can be used to model the effects of within-household dynamics on
the epidemic trajectory in different countries. It also differentiates between
asymptomatic and symptomatic infections.

"""

from iteration_utilities import deepflatten

import os
import pandas as pd
import numpy as np
import pints

import epimodels as em


class WarwickLogLik(pints.LogPDF):
    """WarwickLogLik Class:
    Controller class to construct the log-likelihood needed for optimisation or
    inference in a PINTS framework.

    Parameters
    ----------
    model : WarwickSEIRModel
        The model for which we solve the optimisation or inference problem.
    extended_susceptibles : list
        List country-level initial number of
        susceptibles organised in an extended age classification.
    extended_infectives_prop : list
        List country-level initial proportions of
        infective individuals in each age group when the population is
        organised in an extended age classification.
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
    pDtoH : list
        Age-dependent fractions of the number of symptomatic cases that
        end up hospitalised.
    dDtoH : list
        Distribution of the delay between onset of symptoms and
        hospitalisation. Must be normalised.
    pHtoDeath : list
        Age-dependent fractions of the number of hospitalised cases that
        die.
    dHtoDeath : list
        Distribution of the delay between onset of hospitalisation and
        death. Must be normalised.
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
    def __init__(self, model, extended_susceptibles, extended_infectives_prop,
                 extended_house_cont_mat, extended_school_cont_mat,
                 extended_work_cont_mat, extended_other_cont_mat,
                 pDtoH, dDtoH, pHtoDeath, dHtoDeath,
                 susceptibles_data, infectives_data, times,
                 deaths, deaths_times, tests_data, positives_data,
                 serology_times, sens, spec, wd=1, wp=1):
        # Set the prerequisites for the inference wrapper
        # Model and ICs data
        self._model = model
        self._times = times

        self._susceptibles = susceptibles_data
        self._infectives = infectives_data

        # Probablities and delay distributions to hospitalisation and death
        self._pDtoH = pDtoH
        self._dDtoH = dDtoH
        self._pHtoDeath = pHtoDeath
        self._dHtoDeath = dHtoDeath

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

        # Extended population structure and contact matrices
        self._extended_susceptibles = np.array(extended_susceptibles)
        self._extended_infectives_prop = np.array(extended_infectives_prop)
        self._pop = self._extended_susceptibles

        self._N = np.sum(self._pop)

        # Identify the appropriate contact matrix for the ODE system
        house_cont_mat = extended_house_cont_mat
        school_cont_mat = extended_school_cont_mat
        work_cont_mat = extended_work_cont_mat
        other_cont_mat = extended_other_cont_mat

        self._house_cont_mat = house_cont_mat
        self._other_cont_mat = other_cont_mat
        self._nonhouse_cont_mat = \
            school_cont_mat + work_cont_mat + other_cont_mat
        self._total_cont_mat = \
            self._house_cont_mat + self._nonhouse_cont_mat

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
        # return 8
        # return 7
        return 2

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

    def _stack_age_groups(self, parameter_vector, r):
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

        if np.asarray(self._susceptibles).ndim != 1:
            new_vector[0] = \
                parameter_vector[0] * self._susceptibles[r][0] / (
                    self._susceptibles[r][0] + self._susceptibles[r][1])

            new_vector[1] = \
                parameter_vector[0] * self._susceptibles[r][1] / (
                    self._susceptibles[r][0] + self._susceptibles[r][1])

        else:
            new_vector[0] = \
                parameter_vector[0] * self._susceptibles[0] / (
                    self._susceptibles[0] + self._susceptibles[1])

            new_vector[1] = \
                parameter_vector[0] * self._susceptibles[1] / (
                    self._susceptibles[0] + self._susceptibles[1])

        for _ in range(2, 8):
            new_vector[_] = np.sum(parameter_vector[ind_old[_][:, None]])

        return new_vector

    def _compute_updated_param(self, alpha, tau):
        """
        Computes updated parameters values based on current values of
        alpha and tau.

        Parameters
        ----------
        alpha : int or float
            The current guess of the auxiliary scenario weight parameter alpha.
        tau : int or float
            The current guess of the reduction in the transmission rate of
            infection for asymptomatic individuals.

        Returns
        -------
        tuple of lists
            Tuple of the updated values of the d, sigma and gamma parameters
            of the model.

        """
        # Compute Q
        Q_guess = np.array([
            0.0185, 0.0019, 0.0029, 0.0041, 0.0200, 0.0355, 0.0383,
            0.0319, 0.0368, 0.0507, 0.0947, 0.1497, 0.1939, 0.4396,
            0.5789, 0.4939, 0.7038, 0.9309, 0.9818, 0.8767, 1.0000])

        for _ in range(100):
            Q_guess = self._compute_updated_Q(
                Q_guess, alpha, tau)

        # Compute d
        d = 0.9 * np.power(Q_guess, 1-alpha)

        # Compute sigma
        sigma = (1/0.9) * np.power(Q_guess, alpha)

        return d, sigma

    def _compute_updated_Q(self, Q, alpha, tau):
        """
        Updates stepwise the current guess of the auxiliary parameter value Q
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
        # Compute symptom probability vector
        d = np.power(Q, 1-alpha)

        # Compute asymptomatic cases U from known symptomatic cases D
        symp_cases = self._extended_infectives_prop
        asymp_cases = np.multiply(np.divide(1, d), symp_cases) - symp_cases

        # New unnormalised value for Q
        transmission = np.dot(
            symp_cases + tau * asymp_cases,
            self._total_cont_mat)

        nQ = np.divide(symp_cases, transmission)

        # Normalise new value of Q
        nQ = nQ / np.max(nQ)

        # Return updated guess of Q based on prior value and transmission
        # vector
        return 0.9 * Q + 0.1 * nQ

    def _compute_r0_age_structure(self, house_cont_mat, nonhouse_cont_mat,
                                  d, sigma, tau, gamma):
        """
        """
        # Compute eigenvalues and vectors of the
        M_from_to_HAT = self._compute_next_gen_matrix(
            house_cont_mat, nonhouse_cont_mat,
            d, sigma, tau)

        eigvals, eigvecs = np.linalg.eig(M_from_to_HAT)

        reprod_number_0, i = np.max(
            np.absolute(eigvals)), np.argmax(abs(eigvals))

        reprod_number_0 = reprod_number_0 / gamma
        Age_structure = abs(eigvecs[:, i])

        return Age_structure, reprod_number_0

    def _compute_next_gen_matrix(self, house_cont_mat, other_cont_mat,
                                 d, sigma, tau):
        """
        """
        M_from_to = house_cont_mat + other_cont_mat

        M_from_to_HAT = np.zeros_like(M_from_to)
        k = np.shape(M_from_to_HAT)[0]

        tau = tau * np.ones(k)

        for f in range(k):
            for t in range(k):
                M_from_to_HAT[f, t] = \
                    M_from_to[f, t] * d[t] * sigma[t] * (
                        1 + tau[f] * (1 - d[f]) / d[f])

        return M_from_to_HAT

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
        # number_E_states = 3

        # # H
        # self._parameters[-2] = \
        #     [var_parameters[0]] * len(self._model.regions)
        # # d
        # self._parameters[-3] = var_parameters[1] * np.ones(
        #     self._model._num_ages)
        # # sigma
        # self._parameters[-7] = var_parameters[2] * np.ones(
        #     self._model._num_ages)
        # tau
        self._parameters[-6] = var_parameters[-2]
        # gamma
        self._parameters[-4] = var_parameters[-1]
        # # epsilon
        # self._parameters[-5] = var_parameters[-1]

        # Update parameters
        # # alpha
        # alpha = var_parameters[0]

        # # E0
        # E0_multiplier = var_parameters[2]
        # # phi
        # self._model.social_distancing_param[1] = var_parameters[4]

        # d, sigma = self._compute_updated_param(alpha, var_parameters[1])

        # Age_structure, reprod_number_0 = self._compute_r0_age_structure(
        #     self._house_cont_mat * 0,
        #     self._parameters[-4] * (
        #         self._house_cont_mat + self._nonhouse_cont_mat),
        #     d, sigma, var_parameters[1], self._parameters[-4])

        # exposed_0 = Age_structure / np.sum(Age_structure)
        # detected_0 = Age_structure / np.sum(Age_structure)
        # undetected_0 = Age_structure / np.sum(Age_structure)

        # # Assign updated initial conditions
        # # Exposed_1_f
        # self._parameters[2] = E0_multiplier * np.asarray(
        #     [self._stack_age_groups(exposed_0 / number_E_states, r)
        #      for r in range(len(self._model.regions))])

        # # Exposed_2_f
        # self._parameters[6] = E0_multiplier * np.asarray(
        #     [self._stack_age_groups(exposed_0 / number_E_states, r)
        #      for r in range(len(self._model.regions))])

        # # Exposed_3_f
        # self._parameters[10] = E0_multiplier * np.asarray(
        #     [self._stack_age_groups(exposed_0 / number_E_states, r)
        #      for r in range(len(self._model.regions))])

        # # Detected_f
        # self._parameters[14] = E0_multiplier * np.asarray(
        #     [self._stack_age_groups(detected_0, r)
        #      for r in range(len(self._model.regions))])

        # # Undetected_f
        # self._parameters[19] = E0_multiplier * np.asarray(
        #     [self._stack_age_groups(undetected_0, r)
        #      for r in range(len(self._model.regions))])

        # # Recompute d and sigma with correct number of age groups
        # d = self._update_age_groups(d)
        # sigma = self._update_age_groups(sigma)

        # Update rest of parameters

        # Hs and Ds
        # Hs = var_parameters[6]
        Hs = 1
        # Ds = var_parameters[7]
        Ds = 1

        total_log_lik = 0

        # Compute log-likelihood
        try:
            for r, _ in enumerate(self._model.regions):
                self._parameters[0] = r+1

                model_output = self._model._simulate(
                    parameters=list(deepflatten(self._parameters, ignore=str)),
                    times=self._times
                    )

                model_new_infec = self._model.new_infections(model_output)
                model_new_hosp = self._model.new_hospitalisations(
                    model_new_infec,
                    (Hs * np.array(self._pDtoH)).tolist(),
                    self._dDtoH)
                model_new_deaths = self._model.new_deaths(
                    model_new_hosp,
                    (Ds * np.array(self._pHtoDeath)).tolist(),
                    self._dHtoDeath)

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

        exposed_1_f = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed_1_sd = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed_1_su = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed_1_q = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed_2_f = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed_2_sd = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed_2_su = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed_2_q = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed_3_f = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed_3_sd = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed_3_su = np.zeros((
            len(self._model.regions),
            self._model._num_ages)).tolist()

        exposed_3_q = np.zeros((
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
        h = [0.9] * len(self._model.regions)

        # Disease-specific parameters
        tau = 0.4
        d = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__),
                '../data/risks_death/Risks_United Kingdom.csv'),
            dtype=np.float64)['symptom_risk'].tolist()

        d = 1.43 * self._update_age_groups(np.array(d))

        # Transmission parameters
        epsilon = 0.2
        gamma = 0.083
        sigma = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__),
                '../data/risks_death/Risks_United Kingdom.csv'),
            dtype=np.float64)['susceptibility'].tolist()

        sigma = 1.09 * self._update_age_groups(np.array(sigma))

        self._parameters = [
            0, susceptibles, exposed_1_f, exposed_1_sd, exposed_1_su,
            exposed_1_q, exposed_2_f, exposed_2_sd, exposed_2_su,
            exposed_2_q, exposed_3_f, exposed_3_sd, exposed_3_su,
            exposed_3_q, detected_f, detected_qf, detected_sd, detected_su,
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
        # return 8
        # return 7
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
        # # Prior contribution of H
        # log_prior = pints.UniformLogPrior([0.5], [0.9])(x[0])

        # Prior contribution of tau
        log_prior = pints.UniformLogPrior([0], [0.5])(x[0])

        # Prior contribution of gamma
        log_prior += pints.UniformLogPrior([0.05], [0.5])(x[1])

        # # Prior contribution of epsilon
        # log_prior += pints.UniformLogPrior([0.1], [0.3])(x[3])

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
            self, extended_susceptibles, extended_infectives_prop):
        """
        Sets the initial data with more age groups used for the model's
        parameters optimisation or inference.

        Parameters
        ----------
        extended_susceptibles : list
            List country-level initial number of
            susceptibles organised in an extended age classification.
        extended_infectives_prop : list
            List country-level initial proportions of
            infective individuals in each age group when the population is
            organised in an extended age classification.

        """
        self._extended_susceptibles = extended_susceptibles
        self._extended_infectives_prop = extended_infectives_prop

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

    def read_delay_data(self, pDtoH, dDtoH, pHtoDeath, dHtoDeath):
        """
        Sets the hospitalisation and death delays data used for the model's
        parameters inference.

        Parameters
        ----------
        pDtoH : list
            Age-dependent fractions of the number of symptomatic cases that
            end up hospitalised.
        dDtoH : list
            Distribution of the delay between onset of symptoms and
            hospitalisation. Must be normalised.
        pHtoDeath : list
            Age-dependent fractions of the number of hospitalised cases that
            die.
        dHtoDeath : list
            Distribution of the delay between onset of hospitalisation and
            death. Must be normalised.
        """
        self._pDtoH = pDtoH
        self._dDtoH = dDtoH
        self._pHtoDeath = pHtoDeath
        self._dHtoDeath = dHtoDeath

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
            self._extended_infectives_prop, self._extended_house_cont_mat,
            self._extended_school_cont_mat, self._extended_work_cont_mat,
            self._extended_other_cont_mat,
            self._pDtoH, self._dDtoH, self._pHtoDeath, self._dHtoDeath,
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
            self._extended_infectives_prop, self._extended_house_cont_mat,
            self._extended_school_cont_mat, self._extended_work_cont_mat,
            self._extended_other_cont_mat,
            self._pDtoH, self._dDtoH, self._pHtoDeath, self._dHtoDeath,
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

        # param_names = [
        #     'alpha', 'tau', 'epsilon', 'E0', 'phi', 'sigmaR',
        #     'Hs', 'Ds']

        # param_names = [
        #     'alpha', 'tau', 'epsilon', 'E0', 'sigmaR',
        #     'Hs', 'Ds']

        param_names = [
            'tau', 'gamma']

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
        # x0 = [0.9, 0, 0.2, 15, 0.5, 1, 1, 1]
        # x0 = [0.9, 0, 0.2, 15, 1, 1, 1]
        x0 = [0.5, 0.2]

        # Create optimisation routine
        optimiser = pints.OptimisationController(
            self._log_posterior, x0, method=pints.CMAES)

        optimiser.set_max_unchanged_iterations(100, 1)

        found_ics, found_posterior_val = optimiser.run()
        print(found_ics, found_posterior_val)

        print("Optimisation phase is finished.")

        return found_ics, found_posterior_val
