#
# RocheSEIRModel Class
#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""
This script contains code for modelling the extended SEIRD model created by
F. Hoffmann-La Roche Ltd and can be used to model the effects of
non-pharmaceutical interventions (NPIs) on the epidemic dynamics.

It uses an extended version of an SEIRD model which differentiates between
symptomatic and asymptomatic, as well as super-spreaders infectives.

"""

from itertools import chain

import numpy as np
import pints
from scipy.stats import nbinom, binom
from scipy.integrate import solve_ivp

import epimodels as em


class RocheSEIRModel(pints.ForwardModel):
    r"""RocheSEIRModel Class:
    Base class for constructing the ODE model: deterministic SEIRD used by the
    F. Hoffmann-La Roche Ltd to model the Covid-19 epidemic and the effects of
    non-pharmaceutical interventions (NPIs) on the epidemic dynamic in
    different countries.

    The population is structured such that every individual will belong to one
    of the compartments of the extended SEIRD model.

    The general SEIRD Model has five compartments - susceptible individuals
    (:math:`S`), exposed but not yet infectious (:math:`E`), infectious
    (:math:`I`), recovered (:math:`R`) and dead (:math:`D`).

    In the Roche model framework, the infectious compartment is split into
    6 distinct ones: depending on whether they are super-spreader or not, and
    whether are in the presymptomatic phase, which can than evolve into either
    symptomatic or asymptomatic infectious. We also consider a population
    divided in age groups, as we expect people of different ages to interact
    diferently between themselves and to be affected differently by the virus,
    i.e. have different death and recovery rates and propostions of
    asymptomatic, dead an recovered individuals. The model structure now
    becomes, for each region:

    .. math::
       :nowrap:

        \begin{eqnarray}
            \frac{dS_i}{dt} &=& \sum_{j} C_{ij}(- \frac{\beta_a}{N} S_i
                {I_a}_j - \frac{\beta_{aa}}{N} S_i {I_{aa}}_j -
                \frac{\beta_s}{N} S_i {I_s}_j - \frac{\beta_{as}}{N} S_i
                {I_{as}}_j - \frac{\beta_{aas}}{N} S_i {I_{aas}}_j -
                \frac{\beta_{ss}}{N} S_i {I_{ss}}_j) \\
            \frac{dE_i}{dt} &=& -\gamma_E E_i + \sum_{j} C_{ij}(
                \frac{\beta_a}{N} S_i {I_a}_j + \frac{\beta_{aa}}{N} S_i
                {I_{aa}}_j + \frac{\beta_s}{N} S_i {I_s}_j +
                \frac{\beta_{as}}{N} S_i {I_{as}}_j + \frac{\beta_{aas}}{N}
                S_i {I_{aas}}_j + \frac{\beta_{ss}}{N} S_i {I_{ss}}_j) \\
            \frac{d{I_a}_i}{dt} &=& (1 - P_{ss}) \gamma_E E_i -
                \gamma_s {I_a}_i \\
            \frac{d{I_{aa}}_i}{dt} &=& {P_a}_i \gamma_s {I_a}_i -
                {\gamma_{ra}}_i {I_{aa}}_i \\
            \frac{d{I_s}_i}{dt} &=& (1 - {P_a}_i) \gamma_s {I_a}_i -
                \gamma_q {I_s}_i \\
            \frac{d{I_{as}}_i}{dt} &=& P_{ss} \gamma_E E_i -
                \gamma_s {I_{as}}_i \\
            \frac{d{I_{aas}}_i}{dt} &=& {P_a}_i \gamma_s {I_{as}}_i -
                {\gamma_{ra}}_i {I_{aas}}_i \\
            \frac{d{I_{ss}}_i}{dt} &=& (1 - {P_a}_i) \gamma_s {I_{as}}_i -
                \gamma_q {I_{ss}}_i \\
            \frac{d{I_q}_i}{dt} &=& \gamma_q {I_{ss}}_i + \gamma_q {I_s}_i -
                {\gamma_r}_i {I_q}_i\\
            \frac{dR_i}{dt} &=& (1 - {P_d}_i) {\gamma_r}_i {I_q}_i \\
            \frac{d{R_a}_i}{dt} &=& {\gamma_{ra}}_i {I_{aas}}_i +
                {\gamma_{ra}}_i {I_{aa}}_i \\
            \frac{dD_i}{dt} &=& {P_d}_i {\gamma_r}_i {I_q}_i
        \end{eqnarray}

    where :math:`i` is the age group of the individual, :math:`C_{ij}` is
    the :math:`(i,j)` th element of the regional contact matrix, and
    represents the expected number of new infections in age group :math:`i`
    caused by an infectious in age group :math:`j`. :math:`N` is the total
    population size.

    The transmission parameters are the rates with which different types of
    infectious individual infects susceptible ones.

    The transmission rates for the different types of infectious vectors are:

        * :math:`\beta_a`: presymptomatic infectious;
        * :math:`\beta_{aa}`: asymptomatic infectious;
        * :math:`\beta_s`: symptomatic infectious;
        * :math:`\beta_{as}`: presymptomatic super-spreader infectious;
        * :math:`\beta_{aas}`: asymptomatic super-spreader infectious;
        * :math:`\beta_{ss}`: symptomatic super-spreader infectious.

    The transmission rates depend on each other according to the following
    formulae:

    .. math::
        :nowrap:

        \begin{eqnarray}
            \beta_a &=& \beta_{aa} = \frac{\beta_s}{2} \\
            \beta_{as} &=& \beta_{aas} = \frac{\beta_{ss}}{2} \\
            \beta_{s} &=& \beta_{max} - (\beta_{max} - \beta_{min})\frac{
                SI^\gamma}{SI^\gamma + SI_50^\gamma} \\
            \beta_{as} &=& (1 + b_{ss})\beta_a \\
            \beta_{aas} &=& (1 + b_{ss})\beta_{aa} \\
            \beta_{ss} &=& (1 + b_{ss})\beta_s \\
        \end{eqnarray}

    where :math:`b_{ss}` represents the relative increase in transmission of a
    super-spreader case and :math:`\gamma` is the sharpness of the
    intervention wave used for function continuity purposes. Larger values of
    this parameter cause the curve to more closely approach the step function.

    The :math:`P_a`, :math:`P_{ss}` and :math:`P_d` parameters represent the
    propotions of people that go on to become asymptomatic, super-spreaders
    or dead, respectively. Because we expect older people to be more likely to
    die and younger people to be more likely to be asymptomatic, we consider
    :math:`P_a` and :math:`P_d` to be age dependent.

    The rates of progessions through the different
    stages of the illness are:

        * :math:`\gamma_E`: exposed to presymptomatic infectious status;
        * :math:`\gamma_s`: presymptomatic to (a)symptomatic infectious status;
        * :math:`\gamma_q`: symptomatic to quarantined infectious status;
        * :math:`\gamma_r`: quarantined infectious to recovered (or dead)
          status;
        * :math:`\gamma_{ra}`: asymptomatic to recovered (or dead) status.

    Because we expect older and younger people to recover diferently from the
    virus we consider :math:`\gamma_r` and :math:`\gamma_{ra}` to be age
    dependent. These rates are computed according to the following formulae:

    .. math::
        :nowrap:

        \begin{eqnarray}
            \gamma_E &=& \frac{1}{k} \\
            \gamma_s &=& \frac{1}{k_s} \\
            \gamma_q &=& \frac{1}{k_q} \\
            {\gamma_r}_i &=& \frac{1}{{k_r}_i} \\
            {\gamma_{ra}}_i &=& \frac{1}{{k_{ri}}_i} \\
        \end{eqnarray}

    where :math:`k` refers to mean incubation period until disease onset (i.e.
    from exposed to presymptomatic infection), :math:`k_s` the average time to
    developing symptoms since disease onset, :math:`k_q` the average time until
    the case is quarantined once the symptoms appear, :math:`k_r` the average
    time until recovery since the start of the quaranrining period and
    :math:`k_{ri}` the average time to recovery since the end of the
    presymptomatic stage for an asymptomatic case.

    :math:`S(0) = S_0, E(0) = E_0, I(0) = I_0, R(0) = R_0, D(0) = D_0` are also
    parameters of the model (evaluation at 0 refers to the compartments'
    structure at intial time.

    Extends :class:`pints.ForwardModel`.

    """
    def __init__(self):
        super(RocheSEIRModel, self).__init__()

        # Assign default values
        self._output_names = [
            'S', 'E', 'Ia', 'Iaa', 'Is', 'Ias', 'Iaas', 'Iss', 'Iq', 'R', 'Ra',
            'D', 'Incidence']
        self._parameter_names = [
            'S0', 'E0', 'Ia0', 'Iaa0', 'Is0', 'Ias0', 'Iaas0', 'Iss0', 'Iq0',
            'R0', 'Ra0', 'D0', 'k', 'kS', 'kQ', 'kR', 'kRI', 'Pa', 'Pss', 'Pd',
            'beta_min', 'beta_max', 'bss', 'gamma', 's50']

        # The default number of outputs is 13,
        # i.e. S, E, Ia, Iaa, Is, Ias, Iaas, Iss, Iq, R, Ra, D and Incidence
        self._n_outputs = len(self._output_names)
        # The default number of parameters is 25,
        # i.e. 12 initial conditions and 13 parameters
        self._n_parameters = len(self._parameter_names)

        self._output_indices = np.arange(self._n_outputs)

    def n_outputs(self):
        """
        Returns the number of outputs.

        Returns
        -------
        int
            Number of outputs.

        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the number of parameters.

        Returns
        -------
        int
            Number of parameters.

        """
        return self._n_parameters

    def output_names(self):
        """
        Returns the (selected) output names.

        Returns
        -------
        list
            List of the (selected) output names.

        """
        names = [self._output_names[x] for x in self._output_indices]
        return names

    def parameter_names(self):
        """
        Returns the parameter names.

        Returns
        -------
        list
            List of the parameter names.

        """
        return self._parameter_names

    def set_regions(self, regions):
        """
        Sets region names.

        Parameters
        ----------
        regions : list
            List of region names considered by the model.

        """
        self.regions = regions

    def set_age_groups(self, age_groups):
        """
        Sets age group names and counts their number.

        Parameters
        ----------
        age_groups : list
            List of age group names considered by the model.

        """
        self.age_groups = age_groups
        self._num_ages = len(self.age_groups)

    def region_names(self):
        """
        Returns the regions names.

        Returns
        -------
        list
            List of the regions names.

        """
        return self.regions

    def age_groups_names(self):
        """
        Returns the age group names.

        Returns
        -------
        list
            List of the age group names.

        """
        return self.age_groups

    def set_outputs(self, outputs):
        """
        Checks existence of outputs and selects only those remaining.

        Parameters
        ----------
        outputs : list
            List of output names that are selected.

        """
        for output in outputs:
            if output not in self._output_names:
                raise ValueError(
                    'The output names specified must be in correct forms')

        output_indices = []
        for output_id, output in enumerate(self._output_names):
            if output in outputs:
                output_indices.append(output_id)

        # Remember outputs
        self._output_indices = output_indices
        self._n_outputs = len(outputs)

    def _compute_betas(self, beta_min, beta_max, bss, gamma, SI, S50=0.353):
        """
        Computes the current time, age and region-varying rates with which
        susceptible individuals become infected, depending on the type of
        infective vector.

        Parameters
        ----------
        beta_min : int of float
            Minimum transmission rate of the virus when all non-pharmaceutical
            interventions are turned-on to the maximum level.
        beta_max : int of float
            Maximum transmission rate of the virus when all non-pharmaceutical
            interventions are turned-off.
        bss : int or float
            Addistional increase in transmission due to the infective vector
            being a super-spreader.
        gamma : int or float
            Sharpness of the intervention wave used for function
            continuity purposes. Larger values of this parameter cause
            the curve to more closely approach the step function.
        SI : int or float
            Stringency index representing the effect of all the
            non-pharmaceutical interventions put in place at the time point.
        S50 : int or float
            Stringency index needed to reach 50% of the maximum effect on the
            infection rate.

        """
        bS = beta_max - (beta_max - beta_min) * (SI ** gamma) / \
            (SI ** gamma + S50 ** gamma)

        bA = bS / 2
        bAA = bA
        bSS = (1 + bss) * bS
        bAS = bSS / 2
        bAAS = bAS

        return bA, bS, bAA, bAS, bSS, bAAS

    def _compute_SI(self, r, t):
        """
        Computes the stringency index depending on the present state of
        non-pharmaceutical intervention levels in a given region and at a
        specified timepoint.

        Parameters
        ----------
        t : float
            Time point at which we compute the evaluation.
        r : int
            The index of the region to which the current instance of the ODEs
            system refers.

        Returns
        -------
        int or float
            Stringency index in the given region and at the specified
            timepoint.

        """
        # Identify the current time and region NPIs levels
        pos = np.where(np.asarray(self.time_changes_npi) <= t)
        current_npis = self.reg_levels_npi[r-1][pos[-1][-1]]

        # Identify the current time and region NPIs flags
        pos = np.where(np.asarray(self.time_changes_flag) <= t)
        current_flags = self.general_npi[pos[-1][-1]]

        # Compute the sub-indices of each of the different NPIs
        sub_indeces = [100 * (current_npis[j] * (1 - self._w *
                       self.targeted_npi[j]) / self.max_levels_npi[j] +
                       self._w * self.targeted_npi[j] * current_flags[j])
                       for j in range(len(current_npis))]

        return self.formula_SI(sub_indeces)

    def formula_SI(self, sub_indeces):
        r"""
        Formula for computing the stringency index using the sub-indeces
        computed using the levels prescribed for the non-pharmaceutical
        interventions.

        In the case of the Roche model the stringency index is computed
        according to the following formula

        .. math::
            SI = \frac{1}{7}(I_1 + I_2 + max(I_3, I_4) + I_5 + max(I_6,
                I_7) + I_8 + I_9)

        where :math:`I_j` represents sub-index computed for the :math:`j` th
        intervention. For the Roche model, the interventions are defined as
        in the table bellow:

        .. csv-table::
            :header: Intervention, Max Level :math:`N_j`, Targeted, "General
             Value"

            "School closing", "3 (0, 1, 2, 3)", "Yes", "1"
            "Workplace closing", "3 (0, 1, 2, 3)", "Yes", "1"
            "Cancel public events", "2 (0, 1, 2)", "Yes", "1"
            "Restrictions on gatherings", "4 (0, 1, 2, 3, 4)", "Yes", "1"
            "Close public transport", "2 (0, 1, 2)", "Yes", "1"
            "Stay at home requirements", "3 (0, 1, 2, 3)", "Yes", "1"
            "Restrictions on internal movement", "2 (0, 1, 2)", "Yes", "1"
            "International travel controls", "4 (0, 1, 2, 3, 4)", "No", "0"
            "Public information campaigns", "2 (0, 1, 2)", "Yes", "1"

        Parameters
        ----------
        sub_indeces : list
            List of sub-indeces values of strength of each intervention.

        """
        # Formula for the SI using the fact that max(a, b) = a + b - min(a, b)
        formula = (np.sum(sub_indeces) - np.min(sub_indeces[2:4]) - np.min(
                sub_indeces[5:7])) / 7

        return formula

    def _right_hand_side(self, t, r, y, c, num_a_groups):
        r"""
        Constructs the RHS of the equations of the system of ODEs for given a
        region and time point.

        Parameters
        ----------
        t : float
            Time point at which we compute the evaluation.
        r : int
            The index of the region to which the current instance of the ODEs
            system refers.
        y : numpy.array
            Array of all the compartments of the ODE system, segregated
            by age-group. It assumes y = [S, E, Ia, Iaa, Is, Ias, Iaas, Iss,
            Iq, R, Ra, D] where each letter actually refers to all compartment
            of that type. (e.g. S refers to the compartments of all ages of
            susceptibles).
        c : list
            List of values used to compute the parameters of the ODEs
            system. It assumes c = [k, kS, kQ, kR, kRI, Pa, Pss, Pd, beta_min,
            beta_max, bss, gamma, s50], where :math:`k, kS, kQ, kR, kRI`
            represent the average time spent in the different stages of the
            illness, :math:`Pa, Pss, Pd` are the propotion of people that go
            on to be asymptomatic, super-spreaders or dead,
            :math:`beta_min, beta_max` encaplsulates the minimum and maximum
            possible transmission rate of the virus, :math:`bss` is the
            relative increase in transmission of a superspreader case,
            :math:`gamma` represents the sharpness of the intervention wave
            and s50 is the stringency index needed to reach 50% of the maximum
            effect on the infection rate.
        num_a_groups : int
            Number of age groups in which the population is split. It
            refers to the number of compartments of each type.

        Returns
        -------
        numpy.array
            Age-structured matrix representation of the RHS of the ODEs system.

        """
        # Read in the number of age-groups
        a = num_a_groups

        # Split compartments into their types
        s, e, iA, iAA, iS, iAS, iAAS, iSS, iQ, _, rA, d = (  # noqa
            y[:a], y[a:(2*a)], y[(2*a):(3*a)],
            y[(3*a):(4*a)], y[(4*a):(5*a)], y[(5*a):(6*a)],
            y[(6*a):(7*a)], y[(7*a):(8*a)], y[(8*a):(9*a)],
            y[(9*a):(10*a)], y[(10*a):(11*a)], y[(11*a):])

        # Read parameters of the system
        k, kS, kQ, kR, kRI, Pa, Pss, Pd = c[:8]
        beta_min, beta_max, bss, gamma, s50 = c[8:]

        s_index = self._compute_SI(r, t)

        # Compute transmission rates of the system
        bA, bS, bAA, bAS, bSS, bAAS = \
            self._compute_betas(beta_min, beta_max, bss, gamma, s_index, s50)
        gE, gS, gQ, gR, gRA = \
            1/k, 1/kS, 1/kQ, [1/x for x in kR], [1/x for x in kRI]

        # Identify the appropriate contact matrix for the ODE system
        cont_mat = self.contacts_timeline.identify_current_contacts(r, t)

        # Write actual RHS
        lam = bA * np.asarray(iA) + bAA * np.asarray(iAA) + bS * \
            np.asarray(iS) + bAS * np.asarray(iAS) + bAAS * np.asarray(iAAS) \
            + bSS * np.asarray(iSS)
        lam_times_s = np.multiply(s, (1 / self._N) * np.dot(cont_mat, lam))

        dydt = np.concatenate((
            -lam_times_s, lam_times_s - gE * np.asarray(e),
            (1 - Pss) * gE * np.asarray(e) - gS * np.asarray(iA),
            gS * np.multiply(Pa, iA) - np.multiply(gRA, iAA),
            gS * np.multiply((1 - np.asarray(Pa)), iA) - gQ * np.asarray(iS),
            Pss * gE * np.asarray(e) - gS * np.asarray(iAS),
            gS * np.multiply(Pa, iAS) - np.multiply(gRA, iAAS),
            gS * np.multiply((1 - np.asarray(Pa)), iAS) - gQ * np.asarray(iSS),
            gQ * (np.asarray(iS) + np.asarray(iSS)) - np.multiply(gR, iQ),
            np.multiply((1 - np.asarray(Pd)), np.multiply(gR, iQ)),
            np.multiply(gRA, np.asarray(iAA) + np.asarray(iAAS)),
            np.multiply(Pd, np.multiply(gR, iQ))
            ))

        return dydt

    def _scipy_solver(self, times, num_a_groups, method):
        """
        Computes the values in each compartment of the Roche ODEs system using
        the 'off-the-shelf' solver of the IVP from :module:`scipy`.

        Parameters
        ----------
        times : list
            List of time points at which we wish to evaluate the ODEs system.
        num_a_groups : int
            Number of age groups in which the population is split. It
            refers to the number of compartments of each type.
        method : str
            The type of solver implemented by the :meth:`scipy.solve_ivp`.

        Returns
        -------
        dict
            Solution of the ODE system at the time points provided.

        """
        # Initial conditions
        si, ei, iAi, iAAi, iSi, iASi, iAASi, iSSi, iQi, _i, rAi, di \
            = np.asarray(self._y_init)[:, self._region-1]
        init_cond = list(
            chain(
                si.tolist(), ei.tolist(), iAi.tolist(),
                iAAi.tolist(), iSi.tolist(), iASi.tolist(),
                iAASi.tolist(), iSSi.tolist(), iQi.tolist(),
                _i.tolist(), rAi.tolist(), di.tolist()))

        # Solve the system of ODEs
        sol = solve_ivp(
            lambda t, y: self._right_hand_side(
                t, self._region, y, self._c, num_a_groups),
            [times[0], times[-1]], init_cond, method=method, t_eval=times)
        return sol

    def _split_simulate(
            self, parameters, times, method):
        r"""
        Computes the number of individuals in each compartment at the given
        time points and specified region.

        Parameters
        ----------
        parameters : list
            List of quantities that characterise the Roche SEIR model in
            this order: index of region for which we wish to simulate,
            initial conditions matrices classifed by age (column name) and
            region (row name) for each type of compartment (s, e, iA, iAA, iS,
            iAS, iAAS, iSS, iQ, r, rA, d), the average times spent in the
            different stages of the illness (k, kS, kQ, kR, kRI - kR and kRI
            are age-dependent, while k, kS and kQ are not), the propotions of
            people that go on to be asymptomatic, super-spreaders or dead (Pa,
            Pss, Pd - Pa and Pd are age-dependent, while Pss is not), the
            minimum (beta_min) and maximum (beta_max) possible transmission
            rate of the virus and the relative increase in transmission of a
            super-spreader case (bss), the sharpness of the intervention wave
            used for function continuity purposes (gamma) and the stringency
            index needed to reach 50% of the maximum effect on the infection
            rate (s50).
        times : list
            List of time points at which we wish to evaluate the ODEs system.
        method : str
            The type of solver implemented by the :meth:`scipy.solve_ivp`.

        Returns
        -------
        numpy.array
            Age-structured output matrix of the simulation for the specified
            region.

        """
        # Split parameters into the features of the model
        self._region = parameters[0]
        self._y_init = parameters[1:13]
        self._N = np.sum(np.asarray(self._y_init))
        self._c = parameters[13:26]
        self.contacts_timeline = em.MultiTimesContacts(
            self.matrices_contact,
            self.time_changes_contact,
            self.regions,
            self.matrices_region,
            self.time_changes_region)

        self._times = np.asarray(times)

        # Simulation using the scipy solver
        sol = self._scipy_solver(times, self._num_ages, method)

        output = sol['y']

        # Age-based total infected is infectious 'i' plus recovered 'r'
        total_infected = output[
            (2*self._num_ages):(3*self._num_ages), :] + output[
            (3*self._num_ages):(4*self._num_ages), :] + output[
            (4*self._num_ages):(5*self._num_ages), :] + output[
            (5*self._num_ages):(6*self._num_ages), :] + output[
            (6*self._num_ages):(7*self._num_ages), :] + output[
            (7*self._num_ages):(8*self._num_ages), :] + output[
            (8*self._num_ages):(9*self._num_ages), :] + output[
            (9*self._num_ages):(10*self._num_ages), :] + output[
            (10*self._num_ages):(11*self._num_ages), :]

        # Number of incidences is the increase in total_infected
        # between the time points (add a 0 at the front to
        # make the length consistent with the solution
        n_incidence = np.zeros((self._num_ages, len(times)))
        n_incidence[:, 1:] = total_infected[:, 1:] - total_infected[:, :-1]

        # Append n_incidence to output
        # Output is a matrix with rows being S, E1, I1, R1 and Incidence
        output = np.concatenate((output, n_incidence), axis=0)

        # Get the selected outputs
        self._output_indices = np.arange(self._n_outputs)

        output_indices = []
        for i in self._output_indices:
            output_indices.extend(
                np.arange(i*self._num_ages, (i+1)*self._num_ages)
            )

        output = output[output_indices, :]

        return output.transpose()

    def read_contact_data(self, matrices_contact, time_changes_contact):
        """
        Reads in the timelines of contact data used for the modelling.

        Parameters
        ----------
        matrices_contact : list of ContactMatrix
            List of time-dependent contact matrices used for the modelling.
        time_changes_contact : list
            List of times at which the next contact matrix recorded starts to
            be used. In increasing order.

        """
        self.matrices_contact = matrices_contact
        self.time_changes_contact = time_changes_contact

    def read_regional_data(self, matrices_region, time_changes_region):
        """
        Reads in the timelines of regional data used for the modelling.

        Parameters
        ----------
        matrices_region : lists of RegionMatrix
            List of ime-dependent and region-specific relative susceptibility
            matrices used for the modelling.
        time_changes_region : list
            List of times at which the next instances of region-specific
            relative susceptibility matrices recorded start to be used. In
            increasing order.

        """
        self.matrices_region = matrices_region
        self.time_changes_region = time_changes_region

    def read_npis_data(self, max_levels_npi, targeted_npi, general_npi,
                       reg_levels_npi, time_changes_npi, time_changes_flag):
        """
        Reads in the timelines of non-pharmaceutical interventions used for
        the modelling. These are expressed as levels of severity for each
        different type of NPI, e.g. for a "school closer" measure implemented
        we can assign it a value for 0, 1, 2 or 3 with 3 for the case with most
        restrictions in place.

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
        # Check the data for the NPIs is in the correct format
        self._check_npis_data(max_levels_npi, targeted_npi, general_npi,
                              reg_levels_npi, time_changes_npi,
                              time_changes_flag)

        self.max_levels_npi = max_levels_npi
        self.targeted_npi = targeted_npi
        self.general_npi = general_npi
        self.reg_levels_npi = reg_levels_npi
        self.time_changes_npi = time_changes_npi
        self.time_changes_flag = time_changes_flag

        # Compute the additional weight for a policy of general scope
        self._w = self._compute_add_pol_weight(max_levels_npi, targeted_npi)

    def _check_npis_data(self, max_levels_npi, targeted_npi, general_npi,
                         reg_levels_npi, time_changes_npi, time_changes_flag):
        """
        Check correct format of input of non-pharmaceutical interventions data.

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
        # Times of changes NPI flags:
        if not isinstance(time_changes_flag, list):
            raise TypeError('Time points of changes in non-pharmaceutical \
                    interventions flags must be given in a list format.')
        for _ in time_changes_flag:
            if not isinstance(_, (int, float)):
                raise TypeError('Time points of changes in non-pharmaceutical \
                    interventions flags must be integer or float.')
            if _ < 0:
                raise ValueError('Time points of changes in non-pharmaceutical\
                    interventions flags must be => 0.')

        # Times of changes NPIs
        if not isinstance(time_changes_npi, list):
            raise TypeError('Time points of changes in non-pharmaceutical \
                    interventions must be given in a list format.')
        for _ in time_changes_npi:
            if not isinstance(_, (int, float)):
                raise TypeError('Time points of changes in non-pharmaceutical \
                    interventions must be integer or float.')
            if _ < 0:
                raise ValueError('Time points of changes in\
                non-pharmaceutical interventions must be => 0.')

        # Maximum Levels NPIs
        if not isinstance(max_levels_npi, list):
            raise TypeError('Maximum levels the non-pharmaceutical \
                    interventions must be given in a list format.')
        for _ in max_levels_npi:
            if not isinstance(_, int):
                raise TypeError('Maximum levels the non-pharmaceutical \
                    interventions must be integer.')
            if _ <= 0:
                raise ValueError('Maximum levels the non-pharmaceutical \
                    interventions must be > 0.')

        # Targeted NPIs
        if not isinstance(targeted_npi, list):
            raise TypeError('The targeted non-pharmaceutical \
                    interventions must be given in a list format.')
        if len(targeted_npi) != len(max_levels_npi):
            raise ValueError('Wrong number of targeted interventions.')
        for _ in targeted_npi:
            if not isinstance(_, bool):
                raise TypeError('The targeted non-pharmaceutical \
                    interventions must be boolean.')

        # General value of targeted NPIs
        if not isinstance(general_npi, list):
            raise TypeError('The general value of non-pharmaceutical \
                    interventions must be given in a list format.')
        if len(general_npi) != len(time_changes_flag):
            raise ValueError('Wrong number of general value of \
                non-pharmaceutical interventions changes.')
        for flags_npi in general_npi:
            if not isinstance(flags_npi, list):
                raise TypeError('Each change in levels the non-pharmaceutical \
                    interventions must be given in a list format.')
            if len(flags_npi) != len(max_levels_npi):
                raise ValueError('Wrong number of the general value of \
                    interventions.')
            for ind, _ in enumerate(flags_npi):
                if not isinstance(_, bool):
                    raise TypeError('The general value of the \
                        non-pharmaceutical interventions must be boolean.')

        # Regional time-dependent NPIs
        if not isinstance(reg_levels_npi, list):
            raise TypeError('Regional changes in levels the non-pharmaceutical\
                    interventions must be given in a list format.')
        if len(reg_levels_npi) != len(self.regions):
            raise ValueError('Wrong number of regions for the regional changes\
                in levels the non-pharmaceutical interventions.')
        for levels_npi in reg_levels_npi:
            if not isinstance(levels_npi, list):
                raise TypeError('Each change in levels the non-pharmaceutical \
                    interventions must be given in a list format.')
            if len(levels_npi) != len(time_changes_npi):
                raise ValueError('Wrong number of time changes for the\
                     regional changes in levels the non-pharmaceutical\
                         interventions.')
            for inst_npis in levels_npi:
                if len(inst_npis) != len(max_levels_npi):
                    raise ValueError('Wrong number of interventions for the \
                        regional changes in levels the non-pharmaceutical \
                            interventions.')
                for ind, _ in enumerate(inst_npis):
                    if not isinstance(_, int):
                        raise TypeError('Levels the non-pharmaceutical \
                            interventions must be integer.')
                    if _ < 0:
                        raise ValueError('Levels the non-pharmaceutical \
                            interventions must be => 0.')
                    if _ > max_levels_npi[ind]:
                        raise ValueError('Levels the non-pharmaceutical \
                            interventions cannot exceed maximum threshold.')

    def _compute_add_pol_weight(self, max_levels_npi, targeted_npi):
        r"""
        Computes the additional weight for a policy of general scope is
        defined in relation to the number of ordinal points of all the
        indicators that have the targeted/general flags, that is

        .. math::
            w = frac{1}{\sum_{j=1}^{n} \delta_j} \sum_{j=1}^{n} \frac{1}{
                N_j+1} \delta_j

        where :math:`N_j` and :math:`\delta_j` represents the maximum severity
        level and indicator function of the targeted status of the:math:`j`th
        intevention and :math:`n` is the total number of interventions
        considered.

        Parameters
        ----------
        max_levels_npi : list of int
            List of maximum levels the non-pharmaceutical interventions can
            reach.
        targeted_npi : list of bool
            List of the targeted non-pharmaceutical interventions.

        Returns
        -------
        float
            The additional weight for a targeted policy.

        """
        inverse_vals = [1 / (1 + lev) for lev in max_levels_npi]
        inverse_sumand = np.multiply(targeted_npi, inverse_vals)

        return np.sum(inverse_sumand) / np.sum(targeted_npi)

    def simulate(self, parameters):
        """
        Simulates the Roche model using a :class:`RocheParametersController`
        for the model parameters.

        Extends the :meth:`_split_simulate`. Always apply methods
        :meth:`set_regions`, :meth:`set_age_groups`, :meth:`read_contact_data`
        and :meth:`read_regional_data` before running the
        :meth:`RocheSEIRModel.simulate`.

        Parameters
        ----------
        parameters : RocheParametersController
            Controller class for the parameters used by the forward simulation
            of the model.

        Returns
        -------
        numpy.array
            Age-structured output matrix of the simulation for the specified
            region.

        """
        return self._simulate(
            parameters(), parameters.simulation_parameters.times)

    def _simulate(self, parameters, times):
        r"""
        PINTS-configured wrapper for the simulation method of the Roche model.

        Extends the :meth:`_split_simulate`. Always apply methods
        :meth:`set_regions`, :meth:`set_age_groups`, :meth:`read_contact_data`
        and :meth:`read_regional_data` before running the
        :meth:`RocheSEIRModel.simulate`.

        Parameters
        ----------
        parameters : list
            Long vector format of the quantities that characterise the Roche
            SEIR model in this order:
            (1) index of region for which we wish to simulate,
            (2) initial conditions matrices classifed by age (column name) and
            region (row name) for each type of compartment (s, e, iA, iAA, iS,
            iAS, iAAS, iSS, iQ, r, rA, d),
            (3) the average times spent in the different stages of the illness
            (k, kS, kQ, kR, kRI) - kR and kRI are age-dependent, while k, kS
            and kQ are not,
            (4) the propotions of people that go on to be asymptomatic, super-
            spreaders or dead (Pa, Pss, Pd) - Pa and Pd are age-dependent,
            while Pss is not,
            (5) the minimum (beta_min) and maximum (beta_max) possible
            transmission rate of the virus,
            (6) the relative increase in transmission of a super-spreader case
            (bss),
            (7) the sharpness of the intervention wave used for function
            continuity purposes (gamma),
            (8) the stringency index needed to reach 50% of the maximum effect
            on the infection rate (s50) and
            (9) the type of solver implemented by the :meth:`scipy.solve_ivp`.
            Splited into the formats necessary for the :meth:`_simulate`
            method.
        times : list
            List of time points at which we wish to evaluate the ODEs
            system.

        Returns
        -------
        numpy.array
            Age-structured output matrix of the simulation for the specified
            region.

        """
        # Number of regions and age groups
        self._num_ages = self.matrices_contact[0]._num_a_groups

        n_ages = self._num_ages
        n_reg = len(self.regions)

        start_index = n_reg * ((len(self._output_names)-1) * n_ages) + 1

        # Separate list of parameters into the structures needed for the
        # simulation
        my_parameters = []

        # Add index of region
        my_parameters.append(parameters[0])

        # Add initial conditions for the s, e, iA, iAA, iS, iAS, iAAS, iSS,
        # iQ, r, rA, d compartments
        for c in range(len(self._output_names)-1):
            initial_cond_comp = []
            for r in range(n_reg):
                ind = r * n_ages + n_reg * c * n_ages + 1
                initial_cond_comp.append(
                    parameters[ind:(ind + n_ages)])
            my_parameters.append(initial_cond_comp)

        # Add other parameters
        my_parameters.extend(parameters[start_index:(start_index + 3)])
        my_parameters.append(parameters[
            (start_index + 3):(start_index + 3 + n_ages)])
        my_parameters.append(parameters[
            (start_index + 3 + n_ages):(start_index + 3 + 2 * n_ages)])
        my_parameters.append(parameters[
            (start_index + 3 + 2 * n_ages):(start_index + 3 + 3 * n_ages)])
        my_parameters.append(parameters[start_index + 3 + 3 * n_ages])
        my_parameters.append(parameters[
            (start_index + 4 + 3 * n_ages):(start_index + 4 + 4 * n_ages)])
        my_parameters.extend(parameters[
            (start_index + 4 + 4 * n_ages):(start_index + 9 + 4 * n_ages)])

        # Add method
        method = parameters[start_index + 9 + 4 * n_ages]

        return self._split_simulate(my_parameters,
                                    times,
                                    method)

    def _check_output_format(self, output):
        """
        Checks correct format of the output matrix.

        Parameters
        ----------
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the RocheSEIRModel.

        """
        if np.asarray(output).ndim != 2:
            raise ValueError(
                'Model output storage format must be 2-dimensional.')
        if np.asarray(output).shape[0] != self._times.shape[0]:
            raise ValueError(
                    'Wrong number of rows for the model output.')
        if np.asarray(output).shape[1] != 13 * self._num_ages:
            raise ValueError(
                    'Wrong number of columns for the model output.')
        for r in np.asarray(output):
            for _ in r:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'Model output elements must be integer or float.')

    def new_infections(self, output):
        """
        Computes number of new infections at each time step in specified
        region, given the simulated timeline of susceptible number of
        individuals, for all age groups in the model.

        It uses an output of the simulation method for the RocheSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        output : numpy.array
            Age-structured output of the simulation method for the
            RocheSEIRModel.

        Returns
        -------
        nunmpy.array
            Age-structured matrix of the number of new infections from the
            simulation method for the RocheSEIRModel.

        Notes
        -----
        Always run :meth:`RocheSEIRModel.simulate` before running this one.

        """
        # Check correct format of parameters
        self._check_output_format(output)

        beta_min, beta_max, bss, gamma, s50 = self._c[8:]
        d_infec = np.empty((self._times.shape[0], self._num_ages))

        for ind, t in enumerate(self._times.tolist()):
            # Read from output
            s = output[ind, :][:self._num_ages]
            iA = output[ind, :][(2*self._num_ages):(3*self._num_ages)]
            iAA = output[ind, :][(3*self._num_ages):(4*self._num_ages)]
            iS = output[ind, :][(4*self._num_ages):(5*self._num_ages)]
            iAS = output[ind, :][(5*self._num_ages):(6*self._num_ages)]
            iAAS = output[ind, :][(6*self._num_ages):(7*self._num_ages)]
            iSS = output[ind, :][(7*self._num_ages):(8*self._num_ages)]

            # Compute the current time, age and region-varying
            # rate with which susceptible individuals become infected
            si = self._compute_SI(self._region, t)
            bA, bS, bAA, bAS, bSS, bAAS = \
                self._compute_betas(beta_min, beta_max, bss, gamma, si, s50)

            # Identify the appropriate contact matrix for the ODE system
            cont_mat = self.contacts_timeline.identify_current_contacts(
                self._region, t)

            # Write actual RHS
            lam = bA * np.asarray(iA) + bAA * np.asarray(iAA) + bS * \
                np.asarray(iS) + bAS * np.asarray(iAS) + bAAS * \
                np.asarray(iAAS) + bSS * np.asarray(iSS)

            # fraction of new infectives in delta_t time step
            d_infec[ind, :] = np.multiply(
                np.asarray(s), (1 / self._N) * np.dot(cont_mat, lam))

            if np.any(d_infec[ind, :] < 0):  # pragma: no cover
                d_infec[ind, :] = np.zeros_like(d_infec[ind, :])

        return d_infec

    def _check_new_deaths_format(self, new_deaths):
        """
        Checks correct format of the new deaths matrix.

        Parameters
        ----------
        new_deaths : numpy.array
            Age-structured matrix of the number of new deaths from the
            simulation method for the RocheSEIRModel.

        """
        if np.asarray(new_deaths).ndim != 2:
            raise ValueError(
                'Model new infections storage format must be 2-dimensional.')
        if np.asarray(new_deaths).shape[0] != self._times.shape[0]:
            raise ValueError(
                    'Wrong number of rows for the model new infections.')
        if np.asarray(new_deaths).shape[1] != self._num_ages:
            raise ValueError(
                    'Wrong number of columns for the model new infections.')
        for r in np.asarray(new_deaths):
            for _ in r:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'Model new infections elements must be integer or \
                            float.')

    def new_deaths(self, output):
        """
        Computes number of new deaths at each time step in specified
        region, given the simulated timeline of susceptible number of
        individuals, for all age groups in the model.

        It uses an output of the simulation method for the RocheSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        output : numpy.array
            Age-structured output of the simulation method for the
            RocheSEIRModel.

        Returns
        -------
        nunmpy.array
            Age-structured matrix of the number of new deaths from the
            simulation method for the RocheSEIRModel.

        Notes
        -----
        Always run :meth:`RocheSEIRModel.simulate` before running this one.

        """
        # Check correct format of parameters
        self._check_output_format(output)

        # Check correct format of parameters
        # Age-based total dead is dead 'd'
        n_daily_deaths = np.zeros((self._times.shape[0], self._num_ages))
        total_dead = output[:, (11*self._num_ages):(12*self._num_ages)]
        n_daily_deaths[1:, :] = total_dead[1:, :] - total_dead[:-1, :]

        for ind, t in enumerate(self._times.tolist()):  # pragma: no cover
            if np.any(n_daily_deaths[ind, :] < 0):
                n_daily_deaths[ind, :] = np.zeros_like(n_daily_deaths[ind, :])

        return n_daily_deaths

    def loglik_deaths(self, obs_death, new_deaths, niu, k):
        r"""
        Computes the log-likelihood for the number of deaths at time step
        :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of deaths is assumed to be distributed according to
        a negative binomial distribution with mean :math:`\mu_{r,t_k,i}`
        and variance :math:`\mu_{r,t_k,i} (\nu + 1)`, where
        :math:`\mu_{r,t_k,i}` is the number of new deaths in specified region,
        for age group :math:`i` on day :math:`t_k`.

        It uses new_infections output of the simulation method for the
        RocheSEIRModel, taking all the rest of the parameters necessary for
        the computation from the way its simulation has been fitted.

        Parameters
        ----------
        obs_death : list
            List of number of observed deaths by age group at time point k.
        new_deaths : numpy.array
            Age-structured matrix of the number of new deaths from the
            simulation method for the RocheSEIRModel.
        niu : float
            Dispersion factor for the negative binomial distribution.
        k : int
            Index of day for which we intend to sample the number of deaths for
            by age group.

        Returns
        -------
        numpy.array
            Age-structured matrix of log-likelihoods for the observed number
            of deaths in specified region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`RocheSEIRModel.new_infections` and
        :meth:`RocheSEIRModel.check_death_format` before running this one.

        """
        self._check_time_step_format(k)

        # Check correct format for observed number of deaths
        if np.asarray(obs_death).ndim != 1:
            raise ValueError('Observed number of deaths by age category \
                storage format is 1-dimensional.')
        if np.asarray(obs_death).shape[0] != self._num_ages:
            raise ValueError('Wrong number of age groups for observed number \
                of deaths.')
        for _ in obs_death:
            if not isinstance(_, (int, np.integer)):
                raise TypeError('Observed number of deaths must be integer.')
            if _ < 0:
                raise ValueError('Observed number of deaths must be => 0.')

        if not hasattr(self, 'actual_deaths'):
            self.actual_deaths = [0] * 150
        self.actual_deaths[k] = sum(self.mean_deaths(k, new_deaths))

        # Compute mean of negative-binomial
        if k != 0:
            if np.sum(self.mean_deaths(k, new_deaths)) != 0:
                return nbinom.logpmf(
                    k=obs_death,
                    n=(1/niu) * self.mean_deaths(k, new_deaths),
                    p=1/(1+niu))
            else:
                return np.zeros(self._num_ages)
        else:
            return np.zeros(self._num_ages)

    def check_death_format(self, new_deaths, niu):
        """
        Checks correct format of the inputs of number of death calculation.

        Parameters
        ----------
        new_deaths : numpy.array
            Age-structured matrix of the number of new deaths from the
            simulation method for the RocheSEIRModel.
        niu : float
            Dispersion factor for the negative binomial distribution.

        """
        self._check_new_deaths_format(new_deaths)
        if not isinstance(niu, (int, float)):
            raise TypeError('Dispersion factor must be integer or float.')
        if niu <= 0:
            raise ValueError('Dispersion factor must be > 0.')

    def mean_deaths(self, k, new_deaths):
        """
        Computes the mean of the negative binomial distribution used to
        calculate number of deaths for specified age group.

        Parameters
        ----------
        k : int
            Index of day for which we intend to sample the number of deaths for
            by age group.
        new_deaths : numpy.array
            Age-structured matrix of the number of new deaths from the
            simulation method for the RocheSEIRModel.

        Returns
        -------
        numpy.array
            Age-structured matrix of the expected number of deaths to be
            observed in specified region at time :math:`t_k`.

        """
        return new_deaths[k, :]

    def samples_deaths(self, new_deaths, niu, k):
        r"""
        Computes samples for the number of deaths at time step
        :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of deaths is assumed to be distributed according to
        a negative binomial distribution with mean :math:`\mu_{r,t_k,i}`
        and variance :math:`\mu_{r,t_k,i} (\nu + 1)`, where
        :math:`\mu_{r,t_k,i}` is the number of new deaths in specified region,
        for age group :math:`i` on day :math:`t_k`.

        It uses an output of the simulation method for the RocheSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        new_deaths : numpy.array
            Age-structured matrix of the number of new deaths from the
            simulation method for the RocheSEIRModel.
        niu : float
            Dispersion factor for the negative binomial distribution.
        k : int
            Index of day for which we intend to sample the number of deaths for
            by age group.

        Returns
        -------
        numpy.array
            Age-structured matrix of sampled number of deaths in specified
            region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`RocheSEIRModel.new_infections` and
        :meth:`RocheSEIRModel.check_death_format` before running this one.

        """
        self._check_time_step_format(k)

        # Compute mean of negative-binomial
        if k != 0:
            if np.sum(self.mean_deaths(k, new_deaths)) != 0:
                return nbinom.rvs(
                    n=(1/niu) * self.mean_deaths(k, new_deaths),
                    p=1/(1+niu))
            else:
                return np.zeros(self._num_ages)
        else:
            return np.zeros_like(self.mean_deaths(k, new_deaths))

    def loglik_positive_tests(self, obs_pos, output, tests, sens, spec, k):
        r"""
        Computes the log-likelihood for the number of positive tests at time
        step :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of positive tests is assumed to be distributed according to
        a binomial distribution with parameters :math:`n = n_{r,t_k,i}` and

        .. math::
            p = k_{sens} (1-\frac{S_{r,t_k,i}}{N_{r,i}}) + (
                1-k_{spec}) \frac{S_{r,t_k,i}}{N_{r,i}}

        where :math:`n_{r,t_k,i}` is the number of tests conducted for
        people in age group :math:`i` in specified region :math:`r` at time
        atep :math:`t_k`, :math:`k_{sens}` and :math:`k_{spec}` are the
        sensitivity and specificity respectively of a test, while
        is the probability of demise :math:`k-l` days after infection and
        :math:`\delta_{r,t_l,i}^{infec}` is the number of new infections
        in specified region, for age group :math:`i` on day :math:`t_l`.

        It uses an output of the simulation method for the RocheSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        obs_pos : list
            List of number of observed positive test results by age group at
            time point k.
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the RocheSEIRModel.
        tests : list
            List of conducted tests in specified region and at time point k
            classifed by age groups.
        sens : float or int
            Sensitivity of the test (or ratio of true positives).
        spec : float or int
            Specificity of the test (or ratio of true negatives).
        k : int
            Index of day for which we intend to sample the number of positive
            test results by age group.

        Returns
        -------
        numpy.array
            Age-structured matrix of log-likelihoods for the obsereved number
            of positive test results for each age group in specified region at
            time :math:`t_k`.

        Notes
        -----
        Always run :meth:`RocheSEIRModel.simulate` and
        :meth:`RocheSEIRModel.check_positives_format` before running this one.

        """
        self._check_time_step_format(k)

        # Check correct format for observed number of positive results
        if np.asarray(obs_pos).ndim != 1:
            raise ValueError('Observed number of postive tests results by age \
                category storage format is 1-dimensional.')
        if np.asarray(obs_pos).shape[0] != self._num_ages:
            raise ValueError('Wrong number of age groups for observed number \
                of postive tests results.')
        for _ in obs_pos:
            if not isinstance(_, (int, np.integer)):
                raise TypeError('Observed number of postive tests results must\
                    be integer.')
            if _ < 0:
                raise ValueError('Observed number of postive tests results \
                    must be => 0.')

        # Check correct format for number of tests based on the observed number
        # of positive results
        for i, _ in enumerate(tests):
            if _ < obs_pos[i]:
                raise ValueError('Not enough performed tests for the number \
                    of observed positives.')

        a = self._num_ages
        # Compute parameters of binomial
        suscep = output[k, :a]
        pop = 0
        for i in range(6):
            pop += output[k, (i*a):((i+1)*a)]

        return binom.logpmf(
            k=obs_pos,
            n=tests,
            p=self.mean_positives(sens, spec, suscep, pop))

    def _check_time_step_format(self, k):
        if not isinstance(k, int):
            raise TypeError('Index of time of computation of the \
                log-likelihood must be integer.')
        if k < 0:
            raise ValueError('Index of time of computation of the \
                log-likelihood must be >= 0.')
        if k >= self._times.shape[0]:
            raise ValueError('Index of time of computation of the \
                log-likelihood must be within those considered in the output.')

    def check_positives_format(self, output, tests, sens, spec):
        """
        Checks correct format of the inputs of number of positive test results
        calculation.

        Parameters
        ----------
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the RocheSEIRModel.
        tests : list
            List of conducted tests in specified region and at time point k
            classifed by age groups.
        sens : float or int
            Sensitivity of the test (or ratio of true positives).
        spec : float or int
            Specificity of the test (or ratio of true negatives).

        """
        self._check_output_format(output)
        if np.asarray(tests).ndim != 2:
            raise ValueError('Number of tests conducted by age category \
                storage format is 2-dimensional.')
        if np.asarray(tests).shape[1] != self._num_ages:
            raise ValueError('Wrong number of age groups for observed number \
                of tests conducted.')
        for i in tests:
            for _ in i:
                if not isinstance(_, (int, np.integer)):
                    raise TypeError('Number of tests conducted must be \
                        integer.')
                if _ < 0:
                    raise ValueError('Number of tests conducted ratio must \
                        be => 0.')
        if not isinstance(sens, (int, float)):
            raise TypeError('Sensitivity must be integer or float.')
        if (sens < 0) or (sens > 1):
            raise ValueError('Sensitivity must be >= 0 and <=1.')
        if not isinstance(spec, (int, float)):
            raise TypeError('Specificity must be integer or float.')
        if (spec < 0) or (spec > 1):
            raise ValueError('Specificity must be >= 0 and >=1.')

    def mean_positives(self, sens, spec, suscep, pop):
        """
        Computes the mean of the binomial distribution used to
        calculate number of positive test results for specified age group.

        Parameters
        ----------
        sens : float or int
            Sensitivity of the test (or ratio of true positives).
        spec : float or int
            Specificity of the test (or ratio of true negatives).
        suscep : numpy.array
            Age-structured matrix of the current number of susceptibles
            in the population.
        pop : numpy.array
            Age-structured matrix of the current number of individuals
            in the population.

        Returns
        -------
        numpy.array
            Age-structured matrix of the expected number of positive test
            results to be observed in specified region at time :math:`t_k`.

        """
        return sens * (1-np.divide(suscep, pop)) + (1-spec) * np.divide(
            suscep, pop)

    def samples_positive_tests(self, output, tests, sens, spec, k):
        r"""
        Computes the samples for the number of positive tests at time
        step :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of positive tests is assumed to be distributed according to
        a binomial distribution with parameters :math:`n = n_{r,t_k,i}` and

        .. math::
            p = k_{sens} (1-\frac{S_{r,t_k,i}}{N_{r,i}}) + (
                1-k_{spec}) \frac{S_{r,t_k,i}}{N_{r,i}}

        where :math:`n_{r,t_k,i}` is the number of tests conducted for
        people in age group :math:`i` in specified region :math:`r` at time
        atep :math:`t_k`, :math:`k_{sens}` and :math:`k_{spec}` are the
        sensitivity and specificity respectively of a test, while
        is the probability of demise :math:`k-l` days after infection and
        :math:`\delta_{r,t_l,i}^{infec}` is the number of new infections
        in specified region, for age group :math:`i` on day :math:`t_l`.

        It uses an output of the simulation method for the RocheSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the RocheSEIRModel.
        tests : list
            List of conducted tests in specified region and at time point k
            classifed by age groups.
        sens : float or int
            Sensitivity of the test (or ratio of true positives).
        spec : float or int
            Specificity of the test (or ratio of true negatives).
        k : int
            Index of day for which we intend to sample the number of positive
            test results by age group.

        Returns
        -------
        numpy.array
            Age-structured matrix of sampled number of positive test results
            in specified region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`RocheSEIRModel.simulate` and
        :meth:`RocheSEIRModel.check_positives_format` before running this one.

        """
        self._check_time_step_format(k)

        a = self._num_ages
        # Compute parameters of binomial
        suscep = output[k, :a]
        pop = 0
        for i in range(6):
            pop += output[k, (i*a):((i+1)*a)]

        return binom.rvs(
            n=tests,
            p=self.mean_positives(sens, spec, suscep, pop))
