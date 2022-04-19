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
    symptomatic or asymptomatic infectious. The model structure now becomes:

    .. math::
       :nowrap:

        \begin{eqnarray}
            \frac{dS}{dt} = - \frac{\beta_a}{N} S I_a - \frac{\beta_{aa}}{N} S
                I_{aa} - \frac{\beta_s}{N} S I_s - \frac{\beta_{as}}{N} S
                I_{as} - \frac{\beta_{aas}}{N} S I_{aas} - \frac{\beta_{ss}}{N}
                S I_{ss} \\
            \frac{dE}{dt} = -\gamma_E E + \frac{\beta_a}{N} S I_a +
                \frac{\beta_{aa}}{N} S I_{aa} + \frac{\beta_s}{N} S I_s +
                \frac{\beta_{as}}{N} S I_{as} + \frac{\beta_{aas}}{N} S I_{aas}
                + \frac{\beta_{ss}}{N} S I_{ss} \\
            \frac{dI_a}{dt} = (1-P_{ss}) \gamma_E E - \gamma_s I_a \\
            \frac{dI_{aa}}{dt} = P_a \gamma_s I_a - \gamma_{ra} I_{aa} \\
            \frac{dI_{s}}{dt} = (1-P_a) \gamma_s I_a - \gamma_q I_s \\
            \frac{dI_{as}}{dt} = P_{ss} \gamma_E E - \gamma_s I_{as} \\
            \frac{dI_{aas}}{dt} = P_a \gamma_s I_{as} - \gamma_{ra} I_{aas} \\
            \frac{dI_{ss}}{dt} = (1-P_a) \gamma_s I_{as} - \gamma_q I_{ss} \\
            \frac{dI_q}{dt} = \gamma_q I_{ss} + \gamma_q I_s - \gamma_r I_q \\
            \frac{dR}{dt} = (1-P_d) \gamma_r I_q \\
            \frac{dR_a}{dt} = \gamma_{ra} I_{aas} + \gamma_{ra} I_{aa} \\
            \frac{dD}{dt} = P_d \gamma_r I_q
        \end{eqnarray}

    where :math:`S(0) = S_0, E(0) = E_0, I(O) = I_0, R(0) = R_0` are also
    parameters of the model (evaluation at 0 refers to the compartments'
    structure at intial time.

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
            \beta_a = \beta_{aa} = \frac{\beta_s}{2} \\
            \beta_{as} = \beta_{aas} = \frac{\beta_{ss}}{2} \\
            \beta_{s} = \beta_{max} - (\beta_{max} - \beta_{min})\frac{
                SI^\gamma}{SI^\gamma + SI_50^\gamma} \\
            \beta_{as} = (1 + b_{ss})\beta_a \\
            \beta_{aas} = (1 + b_{ss})\beta_{aa} \\
            \beta_{ss} = (1 + b_{ss})\beta_s \\
        \end{eqnarray}

    where :math:`b_{ss}` represents ... and :math:`\gamma` is the ... .

    The :math:`P_a`, :math:`P_{ss}` and :math:`P_d` parameters represent the
    propotions of people that go on to become asymptomatic, super-spreaders
    or dead, respectively.

    The rates of progessions through the different
    stages of the illness are:

        * :math:`\gamma_E`: exposed to presymptomatic infectious status;
        * :math:`\gamma_s`: presymptomatic to (a)symptomatic infectious status;
        * :math:`\gamma_q`: symptomatic to quarantined infectious status;
        * :math:`\gamma_r`: quarantined infectious to recovered (or dead)
          status;
        * :math:`\gamma_{ra}`: asymptomatic to recovered (or dead) status.

    These rates are computed according to the following formulae:

    .. math::
        :nowrap:

        \begin{eqnarray}
            \gamma_E = \frac{1}{k} \\
            \gamma_s = \frac{1}{k_s} \\
            \gamma_q = \frac{1}{k_q} \\
            \gamma_r = \frac{1}{k_r} \\
            \gamma_{ra} = \frac{1}{k_{ri}} \\
        \end{eqnarray}

    where :math:`k` refers to mean incubation period until disease onset (i.e.
    from exposed to presymptomatic infection), :math:`k_s` the average time to
    developing symptoms since disease onset, :math:`k_q` the average time until
    the case is quarantined once the symptoms appear, :math:`k_r` the average
    time until recovery since the start of the quaranrining period and
    :math:`k_{ri}` the average time to recovery since the end of the
    presymptomatic stage for an asymptomatic case.

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
            'R0', 'Ra0', 'beta', 'kappa', 'gamma']

        # The default number of outputs is 13,
        # i.e. S, E, Ia, Iaa, Is, Ias, Iaas, Iss, Iq, R, Ra, D and Incidence
        self._n_outputs = len(self._output_names)
        # The default number of parameters is 12,
        # i.e. 11 initial conditions and 3 parameters
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

    def _right_hand_side(self, t, r, y, c, num_a_groups):
        r"""
        Constructs the RHS of the equations of the system of ODEs for given a
        region and time point. The :math:`\lambda` parameter that accompanies
        the susceptible numbers is dependent on the current number of
        infectives and is computed using the updated multi-step infectivity
        matrix of the system according to the following formula

        .. math::
            \lambda_{r, t, i} = 1 - \prod_{j=1}^{n_A}[
                (1-b_{r,ij}^{t})^{I1(r,t,j)+I2(r,t,j)}]

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
            beta_max, bss, gamma], where :math:`k, kS, kQ, kR, kRI` represent
            the average time spent in the different stages of the illness,
            :math:`Pa, Pss, Pd` are the propotion of people that go on to be
            asymptomatic, super-spreaders or dead, :math:`beta_min, beta_max`
            encaplsulates the minimum and maximum possible transmission rate
            of the virus, :math:`bss` is the relative increase in transmission
            of a superspreader case and :math:`gamma` represents the sharpness
            of the intervention wave.
        num_a_groups : int
            Number of age groups in which the population is split. It
            refers to the number of compartments of each type.

        Returns
        -------
        numpy.array
            Age-strictured matrix representation of the RHS of the ODEs system.

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
        beta_min, beta_max, bss, gamma = c[8:]

        s_index, s50 = 4

        # Compute transmission rates of the system
        bA, bS, bAA, bAS, bSS, bAAS = \
            self._compute_betas(beta_min, beta_max, bss, gamma, s_index, s50)
        gE, gS, gQ, gR, gRA = 1/k, 1/kS, 1/kQ, 1/kR, 1/kRI

        # Identify the appropriate contact matrix for the ODE system
        cont_mat = self.contacts_timeline.identify_current_contacts(r, t)

        # Write actual RHS
        lam = bA * iA + bAA * iAA + bS * iS + bAS * iAS + bAAS * iAAS \
            + bSS * iSS
        lam_times_s = np.multiply(s, (1 / self._N) * np.dot(cont_mat, lam))

        dydt = np.concatenate((
            -lam_times_s, lam_times_s - gE * np.asarray(e),
            (1 - Pss) * gE * np.asarray(e) - gS * np.asarray(iA),
            Pa * gS * np.asarray(iA) - gRA * np.asarray(iAA),
            (1 - Pa) * gS * np.asarray(iA) - gQ * np.asarray(iS),
            Pss * gE * np.asarray(e) - gS * np.asarray(iAS),
            Pa * gS * np.asarray(iAS) - gRA * np.asarray(iAAS),
            (1 - Pa) * gS * np.asarray(iAS) - gQ * np.asarray(iSS),
            gQ * (np.asarray(iS) + np.asarray(iSS)) - gR * np.asarray(iQ),
            (1 - Pd) * gR * np.asarray(iQ),
            gRA * (np.asarray(iAA) + np.asarray(iAAS)),
            Pd * gR * np.asarray(iQ)
            ))

        return dydt

    def _scipy_solver(self, times, num_a_groups, method):
        """
        Computes the values in each compartment of the PHE ODEs system using
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

    def _simulate(
            self, parameters, times, gamma, method):
        r"""
        Computes the number of individuals in each compartment at the given
        time points and specified region.

        Parameters
        ----------
        parameters : list
            List of quantities that characterise the PHE SEIR model in
            this order: index of region for which we wish to simulate,
            initial conditions matrices classifed by age (column name) and
            region (row name) for each type of compartment (s, e, iA, iAA, iS,
            iAS, iAAS, iSS, iQ, r, rA, d), the average times spent in the
            different stages of the illness (k, kS, kQ, kR, kRI), the
            propotions of people that go on to be asymptomatic, super-spreaders
            or dead (Pa, Pss, Pd), the minimum (beta_min) and maximum
            (beta_max) possible transmission rate of the virus and the relative
            increase in transmission of a super-spreader case (bss).
        times : list
            List of time points at which we wish to evaluate the ODEs system.
        gamma
            Sharpness of the intervention wave used for function
            continuity purposes. Larger values of this parameter cause
            the curve to more closely approach the step function.
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
        self._c = parameters[13:25]
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
            (3*self._num_ages):(4*self._num_ages), :] + output[
            (4*self._num_ages):(5*self._num_ages), :] + output[
                (5*self._num_ages):, :]

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
        Reads in tthe timelines of contact data used for the modelling.

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
        Reads in tthe timelines of regional data used for the modelling.

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

    def simulate(self, parameters, times):
        r"""
        PINTS-configured wrapper for the simulation method of the PHE model.

        Extends the :meth:`_simulation`. Always apply methods
        :meth:`set_regions`, :meth:`read_contact_data` and
        :meth:`read_regional_data` before running the
        :meth:`PheSEIRModel.simulate`.

        Parameters
        ----------
        parameters : list
            Long vector format of the quantities that characterise the PHE
            SEIR model in this order:
            (1) index of region for which we wish to simulate,
            (2) initial conditions matrices classifed by age (column name) and
            region (row name) for each type of compartment (s, e, iA, iAA, iS,
            iAS, iAAS, iSS, iQ, r, rA, d),
            (3) the average times spent in the different stages of the illness
            (k, kS, kQ, kR, kRI),
            (4) the propotions of people that go on to be asymptomatic, super-
            spreaders or dead (Pa, Pss, Pd),
            (5) the minimum (beta_min) and maximum (beta_max) possible
            transmission rate of the virus,
            (6) the relative increase in transmission of a super-spreader case
            (bss) and
            (7) the type of solver implemented by the :meth:`scipy.solve_ivp`.
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

        self._check_input_simulate(
            parameters, times, start_index + 13, n_reg)

        # Separate list of parameters into the structures needed for the
        # simulation
        my_parameters = []

        # Add index of region
        my_parameters.append(parameters[0])

        # Add initial conditions for the s, e1, e2, i1, i2, r compartments
        for c in range(len(self._output_names)-1):
            initial_cond_comp = []
            for r in range(n_reg):
                ind = r * n_ages + n_reg * c * n_ages + 1
                initial_cond_comp.append(
                    parameters[ind:(ind + n_ages)])
            my_parameters.append(initial_cond_comp)

        # Add other parameters
        my_parameters.extend(parameters[start_index:(start_index + 11)])

        # Add gamma
        gamma = parameters[start_index + 11]

        # Add method
        method = parameters[start_index + 12]

        return self._simulate(my_parameters,
                              times,
                              gamma,
                              method)

    def _check_input_simulate(
            self, parameters, times, L, n_reg):
        """
        Check correct format of input of simulate method.

        Parameters
        ----------
        parameters : list
            Long vector format of the quantities that characterise the PHE
            SEIR model in this order:
            (1) index of region for which we wish to simulate,
            (2) initial conditions matrices classifed by age (column name) and
            region (row name) for each type of compartment (s, e, iA, iAA, iS,
            iAS, iAAS, iSS, iQ, r, rA, d),
            (3) the average times spent in the different stages of the illness
            (k, kS, kQ, kR, kRI),
            (4) the propotions of people that go on to be asymptomatic, super-
            spreaders or dead (Pa, Pss, Pd),
            (5) the minimum (beta_min) and maximum (beta_max) possible
            transmission rate of the virus,
            (6) the relative increase in transmission of a super-spreader case
            (bss) and
            (7) the type of solver implemented by the :meth:`scipy.solve_ivp`.
            Splited into the formats necessary for the :meth:`_simulate`
            method.
        times : list
            List of time points at which we wish to evaluate the ODEs
            system.
        L : int
            Number of parameters considered in the model.
        n_reg : int
            Number of regions considered in the model.

        """
        if not isinstance(times, list):
            raise TypeError('Time points of evaluation must be given in a list \
                format.')
        for _ in times:
            if not isinstance(_, (int, float)):
                raise TypeError('Time points of evaluation must be integer or \
                    float.')
            if _ <= 0:
                raise ValueError('Time points of evaluation must be > 0.')

        if not isinstance(parameters, list):
            raise TypeError('Parameters must be given in a list format.')
        if len(parameters) != L:
            raise ValueError('List of parameters has wrong length.')
        if not isinstance(parameters[0], int):
            raise TypeError('Index of region to evaluate must be integer.')
        if parameters[0] <= 0:
            raise ValueError('Index of region to evaluate must be >= 1.')
        if parameters[0] > n_reg:
            raise ValueError('Index of region to evaluate is out of bounds.')
        for param in parameters[-13:(-8)]:
            if not isinstance(param, (float, int)):
                raise TypeError('The average times spent in the different stages \
                    of the illness must be float or integer.')
            if param <= 0:
                raise ValueError('The average times spent in the different stages \
                    of the illness must be > 0.')
        for param in parameters[-8:(-5)]:
            if not isinstance(param, (float, int)):
                raise TypeError('The propotions of people that go on to be \
                    asymptomatic, super-spreaders or dead must be float or \
                        integer.')
            if param <= 0:
                raise ValueError('The propotions of people that go on to be \
                    asymptomatic, super-spreaders or dead must be > 0.')
        for param in parameters[-5:(-3)]:
            if not isinstance(param, (float, int)):
                raise TypeError('The minimum and maximum possible transmission \
                    rate must be float or integer.')
            if param <= 0:
                raise ValueError('The minimum and maximum possible transmission \
                    rate must be > 0.')
        if not isinstance(parameters[-3], (float, int)):
            raise TypeError('The relative increase in transmission of a \
                super-spreader must be float or integer.')
        if parameters[-3] <= 0:
            raise ValueError('The relative increase in transmission of a \
                super-spreader must be > 0.')
        if not isinstance(parameters[-2], (float, int)):
            raise TypeError(
                'The sharpness of the intervention wave must be float or \
                    integer.')
        if parameters[-2] <= 0:
            raise ValueError('The sharpness of the intervention wave must be \
                > 0.')
        if not isinstance(parameters[-1], str):
            raise TypeError('Simulation method must be a string.')
        if parameters[-1] not in (
                'RK45', 'RK23', 'Radau', 'BDF', 'LSODA', 'DOP853'):
            raise ValueError('Simulation method not available.')

    def _check_output_format(self, output):
        """
        Checks correct format of the output matrix.

        Parameters
        ----------
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the PheSEIRModel.

        """
        if np.asarray(output).ndim != 2:
            raise ValueError(
                'Model output storage format must be 2-dimensional.')
        if np.asarray(output).shape[0] != self._times.shape[0]:
            raise ValueError(
                    'Wrong number of rows for the model output.')
        if np.asarray(output).shape[1] != 7 * self._num_ages:
            raise ValueError(
                    'Wrong number of columns for the model output.')
        for r in np.asarray(output):
            for _ in r:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'Model output elements must be integer or float.')

    def _check_new_infections_format(self, new_infections):
        """
        Checks correct format of the new infections matrix.

        Parameters
        ----------
        new_infections : numpy.array
            Age-structured matrix of the number of new infections from the
            simulation method for the PheSEIRModel.

        """
        if np.asarray(new_infections).ndim != 2:
            raise ValueError(
                'Model new infections storage format must be 2-dimensional.')
        if np.asarray(new_infections).shape[0] != self._times.shape[0]:
            raise ValueError(
                    'Wrong number of rows for the model new infections.')
        if np.asarray(new_infections).shape[1] != self._num_ages:
            raise ValueError(
                    'Wrong number of columns for the model new infections.')
        for r in np.asarray(new_infections):
            for _ in r:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'Model new infections elements must be integer or \
                            float.')

    def new_infections(self, output):
        """
        Computes number of new infections at each time step in specified
        region, given the simulated timeline of susceptible number of
        individuals, for all age groups in the model.

        It uses an output of the simulation method for the PheSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        output : numpy.array
            Age-structured output of the simulation method for the
            PheSEIRModel.

        Returns
        -------
        nunmpy.array
            Age-structured matrix of the number of new infections from the
            simulation method for the PheSEIRModel.

        Notes
        -----
        Always run :meth:`PheSEIRModel.simulate` before running this one.

        """
        # Check correct format of parameters
        self._check_output_format(output)

        beta, dL, dI = self._c
        d_infec = np.empty((self._times.shape[0], self._num_ages))

        for ind, t in enumerate(self._times.tolist()):
            # Read from output
            s = output[ind, :][:self._num_ages]
            i1 = output[ind, :][(3*self._num_ages):(4*self._num_ages)]
            i2 = output[ind, :][(4*self._num_ages):(5*self._num_ages)]

            b = self.infectivity_timeline.compute_prob_infectivity_matrix(
                self._region, t, s, beta[self._region-1][ind])

            # Compute the current time, age and region-varying
            # rate with which susceptible individuals become infected
            lam = self._compute_lambda(s, i1, i2, b)

            # fraction of new infectives in delta_t time step
            d_infec[ind, :] = np.multiply(np.asarray(s), lam*self._delta_t)

            if np.any(d_infec[ind, :] < 0):  # pragma: no cover
                d_infec[ind, :] = np.zeros_like(d_infec[ind, :])

        return d_infec

    def loglik_deaths(
            self, obs_death, new_infections, fatality_ratio, time_to_death,
            niu, k):
        r"""
        Computes the log-likelihood for the number of deaths at time step
        :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of deaths is assumed to be distributed according to
        a negative binomial distribution with mean

        .. math::
            \mu_{r,t_k,i} = p_i \sum_{l=0}^{k} f_{k-l} \delta_{r,t_l,i}^{infec}

        and variance :math:`\mu_{r,t_k,i} (\nu + 1)`, where :math:`p_i` is the
        age-specific fatality ratio for age group :math:`i`, :math:`f_{k-l}`
        is the probability of demise :math:`k-l` days after infection and
        :math:`\delta_{r,t_l,i}^{infec}` is the number of new infections
        in specified region, for age group :math:`i` on day :math:`t_l`.

        It uses new_infections output of the simulation method for the
        PheSEIRModel, taking all the rest of the parameters necessary for
        the computation from the way its simulation has been fitted.

        Parameters
        ----------
        obs_death : list
            List of number of observed deaths by age group at time point k.
        new_infections : numpy.array
            Age-structured matrix of the number of new infections from the
            simulation method for the PheSEIRModel.
        fatality_ratio : list
            List of age-specific fatality ratios.
        time_to_death : list
            List of probabilities of death of individual k days after
            infection.
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
        Always run :meth:`PheSEIRModel.new_infections` and
        :meth:`PheSEIRModel.check_death_format` before running this one.

        """
        self._check_time_step_format(k)

        # Check correct format for observed number of deaths
        if np.asarray(obs_death).ndim != 1:
            raise ValueError('Observed number of deaths by age category storage \
                format is 1-dimensional.')
        if np.asarray(obs_death).shape[0] != self._num_ages:
            raise ValueError('Wrong number of age groups for observed number \
                of deaths.')
        for _ in obs_death:
            if not isinstance(_, (int, np.integer)):
                raise TypeError('Observed number of deaths must be integer.')
            if _ < 0:
                raise ValueError('Observed number of deaths must be => 0.')

        # Compute mean of negative-binomial
        return nbinom.logpmf(
            k=obs_death,
            n=niu * self.mean_deaths(
                fatality_ratio, time_to_death, k, new_infections),
            p=niu/(1+niu))

    def check_death_format(
            self, new_infections, fatality_ratio, time_to_death, niu):
        """
        Checks correct format of the inputs of number of death calculation.

        Parameters
        ----------
        new_infections : numpy.array
            Age-structured matrix of the number of new infections from the
            simulation method for the PheSEIRModel.
        fatality_ratio : list
            List of age-specific fatality ratios.
        time_to_death : list
            List of probabilities of death of individual k days after
            infection.
        niu : float
            Dispersion factor for the negative binomial distribution.

        """
        self._check_new_infections_format(new_infections)
        if not isinstance(niu, (int, float)):
            raise TypeError('Dispersion factor must be integer or float.')
        if niu <= 0:
            raise ValueError('Dispersion factor must be > 0.')
        if np.asarray(fatality_ratio).ndim != 1:
            raise ValueError('Fatality ratios by age category storage \
                format is 1-dimensional.')
        if np.asarray(fatality_ratio).shape[0] != self._num_ages:
            raise ValueError('Wrong number of age groups for fatality ratios.')
        for _ in fatality_ratio:
            if not isinstance(_, (int, float)):
                raise TypeError('Fatality ratio must be integer or \
                    float.')
            if(_ < 0) or (_ > 1):
                raise ValueError('Fatality ratio must be => 0 and <=1.')
        if np.asarray(time_to_death).ndim != 1:
            raise ValueError('Probabilities of death of individual k days after \
                infection storage format is 1-dimensional.')
        if np.asarray(time_to_death).shape[0] != len(self._times):
            raise ValueError('Wrong number of probabilities of death of individual\
                k days after infection.')
        for _ in time_to_death:
            if not isinstance(_, (int, float)):
                raise TypeError('Probabilities of death of individual k days after \
                    infection must be integer or float.')
            if (_ < 0) or (_ > 1):
                raise ValueError('Probabilities of death of individual k days after \
                    infection must be => 0 and <=1.')

    def mean_deaths(self, fatality_ratio, time_to_death, k, d_infec):
        """
        Computes the mean of the negative binomial distribution used to
        calculate number of deaths for specified age group.

        Parameters
        ----------
        fatality_ratio : list
            List of age-specific fatality ratios.
        time_to_death : list
            List of probabilities of death of individual k days after
            infection.
        k : int
            Index of day for which we intend to sample the number of deaths for
            by age group.
        d_infec : numpy.array
            Age-structured matrix of the number of new infections from the
            simulation method for the PheSEIRModel.

        Returns
        -------
        numpy.array
            Age-structured matrix of the expected number of deaths to be
            observed in specified region at time :math:`t_k`.

        """
        if k >= 30:
            return np.array(fatality_ratio) * np.sum(np.matmul(
                np.diag(time_to_death[:31][::-1]),
                d_infec[(k-30):(k+1), :]), axis=0)
        else:
            return np.array(fatality_ratio) * np.sum(np.matmul(
                np.diag(time_to_death[:(k+1)][::-1]), d_infec[:(k+1), :]),
                axis=0)

    def samples_deaths(
            self, new_infections, fatality_ratio, time_to_death, niu, k):
        r"""
        Computes samples for the number of deaths at time step
        :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of deaths is assumed to be distributed according to
        a negative binomial distribution with mean

        .. math::
            \mu_{r,t_k,i} = p_i \sum_{l=0}^{k} f_{k-l} \delta_{r,t_l,i}^{infec}

        and variance :math:`\mu_{r,t_k,i} (\nu + 1)`, where :math:`p_i` is the
        age-specific fatality ratio for age group :math:`i`, :math:`f_{k-l}`
        is the probability of demise :math:`k-l` days after infection and
        :math:`\delta_{r,t_l,i}^{infec}` is the number of new infections
        in specified region, for age group :math:`i` on day :math:`t_l`.

        It uses an output of the simulation method for the PheSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        new_infections : numpy.array
            Age-structured matrix of the number of new infections from the
            simulation method for the PheSEIRModel.
        fatality_ratio : list
            List of age-specific fatality ratios.
        time_to_death : list
            List of probabilities of death of individual k days after
            infection.
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
        Always run :meth:`PheSEIRModel.new_infections` and
        :meth:`PheSEIRModel.check_death_format` before running this one.

        """
        self._check_time_step_format(k)

        # Compute mean of negative-binomial
        return nbinom.rvs(
            n=niu * self.mean_deaths(
                fatality_ratio, time_to_death, k, new_infections),
            p=niu/(1+niu))

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

        It uses an output of the simulation method for the PheSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        obs_pos : list
            List of number of observed positive test results by age group at
            time point k.
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the PheSEIRModel.
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
        Always run :meth:`PheSEIRModel.simulate` and
        :meth:`PheSEIRModel.check_positives_format` before running this one.

        """
        self._check_time_step_format(k)

        # Check correct format for observed number of positive results
        if np.asarray(obs_pos).ndim != 1:
            raise ValueError('Observed number of postive tests results by age category \
                storage format is 1-dimensional.')
        if np.asarray(obs_pos).shape[0] != self._num_ages:
            raise ValueError('Wrong number of age groups for observed number \
                of postive tests results.')
        for _ in obs_pos:
            if not isinstance(_, (int, np.integer)):
                raise TypeError('Observed number of postive tests results must be \
                    integer.')
            if _ < 0:
                raise ValueError('Observed number of postive tests results must \
                    be => 0.')

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
            raise TypeError('Index of time of computation of the log-likelihood \
                must be integer.')
        if k < 0:
            raise ValueError('Index of time of computation of the log-likelihood \
                must be >= 0.')
        if k >= self._times.shape[0]:
            raise ValueError('Index of time of computation of the log-likelihood \
                must be within those considered in the output.')

    def check_positives_format(self, output, tests, sens, spec):
        """
        Checks correct format of the inputs of number of positive test results
        calculation.

        Parameters
        ----------
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the PheSEIRModel.
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
            raise ValueError('Number of tests conducted by age category storage \
                format is 2-dimensional.')
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

        It uses an output of the simulation method for the PheSEIRModel,
        taking all the rest of the parameters necessary for the computation
        from the way its simulation has been fitted.

        Parameters
        ----------
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the PheSEIRModel.
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
        Always run :meth:`PheSEIRModel.simulate` and
        :meth:`PheSEIRModel.check_positives_format` before running this one.

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
