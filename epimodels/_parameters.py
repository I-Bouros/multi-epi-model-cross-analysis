#
# Phe Parameter Classes
#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""
This script contains code for the Parameter classes for all the epidemiological
model include in the `epimodels` Python module.

The Parameter classes store the model parameters as class features and with the
object then fed into the model when :meth:`simulate` is called on the model
class.

"""

import numpy as np
from iteration_utilities import deepflatten

import epimodels as em

#
# PheICs Class
#


class PheICs(object):
    """PheICs:
    Base class for the ICs of the PHE model: a deterministic SEIR used
    by the Public Health England to model the Covid-19 epidemic in UK based on
    region.

    Parameters
    ----------
    susceptibles_IC : list of lists
        Initial number of susceptibles classifed by age (column name) and
        region (row name).
    exposed1_IC : list of lists
        Initial number of exposed of the first type classifed by age
        (column name) and region (row name).
    exposed2_IC : list of lists
        Initial number of exposed of the second type classifed by age
        (column name) and region (row name).
    infectives1_IC :list of lists
        Initial number of infectives of the first type classifed by age
        (column name) and region (row name).
    infectives2_IC : list of lists
        Initial number of infectives of the second type classifed by age
        (column name) and region (row name).
    recovered_IC : list of lists
        Initial number of recovered classifed by age (column name) and
        region (row name).

    """
    def __init__(self, model, susceptibles_IC, exposed1_IC, exposed2_IC,
                 infectives1_IC, infectives2_IC, recovered_IC):
        super(PheICs, self).__init__()

        # Set model
        if not isinstance(model, em.PheSEIRModel):
            raise TypeError('The model must be a PHE SEIR Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(
            susceptibles_IC, exposed1_IC, exposed2_IC, infectives1_IC,
            infectives2_IC, recovered_IC)

        # Set ICs parameters
        self.susceptibles = susceptibles_IC
        self.exposed1 = exposed1_IC
        self.exposed2 = exposed2_IC
        self.infectives1 = infectives1_IC
        self.infectives2 = infectives2_IC
        self.recovered = recovered_IC

    def _check_parameters_input(self, susceptibles_IC, exposed1_IC,
                                exposed2_IC, infectives1_IC, infectives2_IC,
                                recovered_IC):
        """
        Check correct format of ICs input.

        Parameters
        ----------
        susceptibles_IC : list of lists
            Initial number of susceptibles classifed by age (column name) and
            region (row name).
        exposed1_IC : list of lists
            Initial number of exposed of the first type classifed by age
            (column name) and region (row name).
        exposed2_IC : list of lists
            Initial number of exposed of the second type classifed by age
            (column name) and region (row name).
        infectives1_IC :list of lists
            Initial number of infectives of the first type classifed by age
            (column name) and region (row name).
        infectives2_IC : list of lists
            Initial number of infectives of the second type classifed by age
            (column name) and region (row name).
        recovered_IC : list of lists
            Initial number of recovered classifed by age (column name) and
            region (row name).

        """
        if np.asarray(susceptibles_IC).ndim != 2:
            raise ValueError('The inital numbers of susceptibles storage format \
                must be 2-dimensional.')
        if np.asarray(susceptibles_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        susceptibles.')
        if np.asarray(susceptibles_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        susceptibles.')
        for ic in np.asarray(susceptibles_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of susceptibles must be integer or \
                            float.')

        if np.asarray(exposed1_IC).ndim != 2:
            raise ValueError('The inital numbers of exposed of the first type storage format \
                must be 2-dimensional.')
        if np.asarray(exposed1_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed of the first type.')
        if np.asarray(exposed1_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed of the first type.')
        for ic in np.asarray(exposed1_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of exposed of the first type must be integer or \
                            float.')

        if np.asarray(exposed2_IC).ndim != 2:
            raise ValueError('The inital numbers of exposed of the second type storage format \
                must be 2-dimensional.')
        if np.asarray(exposed2_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed of the second type.')
        if np.asarray(exposed2_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed of the second type.')
        for ic in np.asarray(exposed2_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of exposed of the second type must be integer or \
                            float.')

        if np.asarray(infectives1_IC).ndim != 2:
            raise ValueError('The inital numbers of infectives of the first type storage format \
                must be 2-dimensional.')
        if np.asarray(infectives1_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        infectives of the first type.')
        if np.asarray(infectives1_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        infectives of the first type.')
        for ic in np.asarray(infectives1_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of infectives of the first type must be integer or \
                            float.')

        if np.asarray(infectives2_IC).ndim != 2:
            raise ValueError('The inital numbers of infectives of the second type storage format \
                must be 2-dimensional.')
        if np.asarray(infectives2_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        infectives of the second type.')
        if np.asarray(infectives2_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        infectives of the second type.')
        for ic in np.asarray(infectives2_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of infectives of the second type must be integer or \
                            float.')

        if np.asarray(recovered_IC).ndim != 2:
            raise ValueError('The inital numbers of recovered storage format \
                must be 2-dimensional.')
        if np.asarray(recovered_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        recovered.')
        if np.asarray(recovered_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        recovered.')
        for ic in np.asarray(recovered_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of recovered must be integer or \
                            float.')

    def __call__(self):
        """
        Returns the initial conditions of the :class:`PheSEIRModel` the
        class relates to.

        Returns
        -------
        List of lists
            List of the initial conditions of the :class:`PheSEIRModel`
            the class relates to.

        """
        return [self.susceptibles, self.exposed1, self.exposed2,
                self.infectives1, self.infectives2, self.recovered]

#
# PheRegParameters Class
#


class PheRegParameters(object):
    """PheRegParameters:
    Base class for the regional and time dependent parameters of the PHE model:
    a deterministic SEIR used by the Public Health England to model the
    Covid-19 epidemic in UK based on region.

    Parameters
    ----------
    inital_r : list
        Initial values of the reproduction number by region.
    region_index : int
        Index of region for which we wish to simulate.
    betas : list of lists
        Temporal and regional fluctuation matrix.
    times : list
        List of time points at which we wish to evaluate the ODEs
        system.

    """
    def __init__(self, model, initial_r, region_index, betas, times):
        super(PheRegParameters, self).__init__()

        # Set model
        if not isinstance(model, em.PheSEIRModel):
            raise TypeError('The model must be a PHE SEIR Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(initial_r, region_index, betas, times)

        # Set regional and time dependent parameters
        self.initial_r = initial_r
        self.region_index = region_index
        self.betas = betas
        self.times = times

    def _check_parameters_input(self, initial_r, region_index, betas, times):
        """
        Check correct format of the regional and time dependent parameters
        input.

        Parameters
        ----------
        inital_r : list
            Initial values of the reproduction number by region.
        region_index : int
            Index of region for which we wish to simulate.
        betas : list of lists
            Temporal and regional fluctuation matrix.
        times : list
            List of time points at which we wish to evaluate the ODEs
            system.

        """
        # Check times format
        if not isinstance(times, list):
            raise TypeError('Time points of evaluation must be given in a list \
                format.')
        for _ in times:
            if not isinstance(_, (int, float)):
                raise TypeError('Time points of evaluation must be integer or \
                    float.')
            if _ <= 0:
                raise ValueError('Time points of evaluation must be > 0.')

        if np.asarray(initial_r).ndim != 1:
            raise ValueError('The inital reproduction numbers storage format \
                must be 1-dimensional.')
        if np.asarray(initial_r).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital reproduction \
                    numbers.')
        for _ in np.asarray(initial_r):
            if not isinstance(_, (np.integer, np.floating)):
                raise TypeError(
                    'The inital reproduction numbers must be integer or \
                        float.')

        if not isinstance(region_index, int):
            raise TypeError('Index of region to evaluate must be integer.')
        if region_index <= 0:
            raise ValueError('Index of region to evaluate must be >= 1.')
        if region_index > len(self.model.regions):
            raise ValueError('Index of region to evaluate is out of bounds.')

        if np.asarray(betas).ndim != 2:
            raise ValueError('The temporal and regional fluctuations storage format \
                must be 2-dimensional.')
        if np.asarray(betas).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the temporal and regional \
                        fluctuation.')
        if np.asarray(betas).shape[1] != len(times):
            raise ValueError(
                    'Wrong number of columns for the temporal and regional \
                        fluctuation.')

    def __call__(self):
        """
        Returns the regional and time dependent parameters of the
        :class:`PheSEIRModel` the class relates to.

        Returns
        -------
        List of lists
            List of the regional and time dependent parameters of the
            :class:`PheSEIRModel` the class relates to.

        """
        return [self.initial_r, self.region_index, self.betas]

#
# PheDiseaseParameters Class
#


class PheDiseaseParameters(object):
    """PheDiseaseParameters:
    Base class for the disease-specific parameters of the PHE model:
    a deterministic SEIR used by the Public Health England to model the
    Covid-19 epidemic in UK based on region.

    Parameters
    ----------
    dL : int or float
        Mean latent period.
    dI : int or float
        Mean infection period.

    """
    def __init__(self, model, dL, dI):
        super(PheDiseaseParameters, self).__init__()

        # Set model
        if not isinstance(model, em.PheSEIRModel):
            raise TypeError('The model must be a PHE SEIR Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(dL, dI)

        # Set disease-specific parameters
        self.dL = dL
        self.dI = dI

    def _check_parameters_input(self, dL, dI):
        """
        Check correct format of the disease-specific parameters input.

        Parameters
        ----------
        dL : int or float
            Mean latent period.
        dI : int or float
            Mean infection period.

        """
        if not isinstance(dL, (float, int)):
            raise TypeError('Mean latent period must be float or integer.')
        if dL <= 0:
            raise ValueError('Mean latent period must be > 0.')
        if not isinstance(dI, (float, int)):
            raise TypeError('Mean infection period must be float or integer.')
        if dI <= 0:
            raise ValueError('Mean infection period must be > 0.')

    def __call__(self):
        """
        Returns the disease-specific parameters of the :class:`PheSEIRModel`
        the class relates to.

        Returns
        -------
        List of lists
            List of the disease-specific parameters of the
            :class:`PheSEIRModel` the class relates to.

        """
        return [self.dL, self.dI]

#
# PheSimParameters Class
#


class PheSimParameters(object):
    """PheSimParameters:
    Base class for the simulation method's parameters of the PHE model:
    a deterministic SEIR used by the Public Health England to model the
    Covid-19 epidemic in UK based on region.

    Parameters
    ----------
    delta_t : float
        Time step for the 'homemade' solver.
    method: str
        The type of solver implemented by the simulator.

    """
    def __init__(self, model, delta_t, method):
        super(PheSimParameters, self).__init__()

        # Set model
        if not isinstance(model, em.PheSEIRModel):
            raise TypeError('The model must be a PHE SEIR Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(delta_t, method)

        # Set other simulation parameters
        self.delta_t = delta_t
        self.method = method

    def _check_parameters_input(self, delta_t, method):
        """
        Check correct format of the simulation method's parameters input.

        Parameters
        ----------
        delta_t : float
            Time step for the 'homemade' solver.
        method: str
            The type of solver implemented by the simulator.

        """
        if not isinstance(delta_t, (float, int)):
            raise TypeError(
                'Time step for ODE solver must be float or integer.')
        if delta_t <= 0:
            raise ValueError('Time step for ODE solver must be > 0.')
        if not isinstance(method, str):
            raise TypeError('Simulation method must be a string.')
        if method not in (
                'my-solver', 'RK45', 'RK23', 'Radau',
                'BDF', 'LSODA', 'DOP853'):
            raise ValueError('Simulation method not available.')

    def __call__(self):
        """
        Returns the simulation method's parameters of the :class:`PheSEIRModel`
        the class relates to.

        Returns
        -------
        List of lists
            List of the simulation method's parameters of the
            :class:`PheSEIRModel` the class relates to.

        """
        return [self.delta_t, self.method]

#
# PheParametersController Class
#


class PheParametersController(object):
    """PheParametersController Class:
    Base class for the paramaters of the PHE model: a deterministic SEIR used
    by the Public Health England to model the Covid-19 epidemic in UK based on
    region.

    In order to simulate using the PHE model, the following parameters are
    required, which are stored as part of this class.

    Parameters
    ----------
    model : PheSEIRModel
        The model whose parameters are stored.
    regional_parameters : PheRegParameters
        Class of the regional and time dependent parameters used in the
        simulation of the model.
    ICs : PheICs
        Class of the Ics used in the simulation of the model.
    disease_parameters : PheDiseaseParameters
        Class of the disease-specific parameters used in the simulation of
        the model.
    simulation_parameters : PheSimParameters
        Class of the simulation method's parameters used in the simulation of
        the model.

    """
    def __init__(
            self, model, regional_parameters, ICs,
            disease_parameters, simulation_parameters):
        # Instantiate class
        super(PheParametersController, self).__init__()

        # Set model
        if not isinstance(model, em.PheSEIRModel):
            raise TypeError('The model must be a PHE SEIR Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(
            regional_parameters, ICs, disease_parameters,
            simulation_parameters)

        # Set regional and time dependent parameters
        self.regional_parameters = regional_parameters

        # Set ICs parameters
        self.ICs = ICs

        # Set disease-specific parameters
        self.disease_parameters = disease_parameters

        # Set other simulation parameters
        self.simulation_parameters = simulation_parameters

    def _check_parameters_input(
            self, regional_parameters, ICs, disease_parameters,
            simulation_parameters):
        """
        Check correct format of input of simulate method.

        Parameters
        ----------
        regional_parameters : PheRegParameters
            Class of the regional and time dependent parameters used in the
            simulation of the model.
        ICs : PheICs
            Class of the Ics used in the simulation of the model.
        disease_parameters : PheDiseaseParameters
            Class of the disease-specific parameters used in the simulation of
            the model.
        simulation_parameters : PheSimParameters
            Class of the simulation method's parameters used in the simulation
            of the model.

        """
        if not isinstance(regional_parameters, PheRegParameters):
            raise TypeError('The model`s regional and time dependent parameters must \
                be a of a PHE SEIR Model.')
        if regional_parameters.model != self.model:
            raise ValueError('The regional and time dependent parameters do \
                not correspond to the right model.')

        if not isinstance(ICs, PheICs):
            raise TypeError('The model`s ICs parameters must be a of a PHE SEIR \
                Model.')
        if ICs.model != self.model:
            raise ValueError('ICs do not correspond to the right model.')

        if not isinstance(disease_parameters, PheDiseaseParameters):
            raise TypeError('The model`s disease-specific parameters must be a \
                of a PHE SEIR Model.')
        if disease_parameters.model != self.model:
            raise ValueError('The disease-specific parameters do not correspond \
                to the right model.')

        if not isinstance(simulation_parameters, PheSimParameters):
            raise TypeError('The model`s simulation method`s parameters must be a \
                of a PHE SEIR Model.')
        if simulation_parameters.model != self.model:
            raise ValueError('The simulation method`s parameters do not correspond \
                to the right model.')

    def __call__(self):
        """
        Returns the list of all the parameters used for the simulation of the
        PHE model in their order, which will be then separated within the
        :class:`PheSEIRModel` class.

        Returns
        -------
        list
            List of all the parameters used for the simulation of the
            PHE model in their order.

        """
        parameters = []

        # Add the regional and time dependent parameters
        parameters.extend(self.regional_parameters()[:2])

        # Add ICs
        parameters.extend(self.ICs())

        # Add betas
        parameters.extend(self.regional_parameters()[2])

        # Add disease-specific parameters
        parameters.extend(self.disease_parameters())

        # Add other simulation parameters
        parameters.extend(self.simulation_parameters())

        return list(deepflatten(parameters, ignore=str))

#
# Roche Parameter Classes
#

#
# RocheICs Class
#


class RocheICs(object):
    """RocheICs:
    Base class for the ICs of the Roche model: deterministic SEIRD used by the
    F. Hoffmann-La Roche Ltd to model the Covid-19 epidemic and the effects of
    non-pharmaceutical interventions (NPIs) on the epidemic dynamic in
    different countries.

    Parameters
    ----------
    susceptibles_IC : list of lists
        Initial number of susceptibles classifed by age (column name) and
        region (row name).
    exposed_IC : list of lists
        Initial number of exposed classifed by age (column name) and region
        (row name).
    infectives_pre_IC :list of lists
        Initial number of presymptomatic infectives classifed by age
        (column name) and region (row name).
    infectives_asym_IC :list of lists
        Initial number of asymptomatic infectives classifed by age
        (column name) and region (row name).
    infectives_sym_IC :list of lists
        Initial number of symptomatic infectives classifed by age
        (column name) and region (row name).
    infectives_pre_ss_IC : list of lists
        Initial number of presymptomatic superspreader infectives classifed by
        age (column name) and region (row name).
    infectives_asym_ss_IC : list of lists
        Initial number of asymptomatic superspreader infectives classifed by
        age (column name) and region (row name).
    infectives_sym_ss_IC : list of lists
        Initial number of symptomatic superspreader infectives classifed by
        age (column name) and region (row name).
    infectives_q_IC : list of lists
        Initial number of symptomatic infectives quarantined classifed by
        age (column name) and region (row name).
    recovered_IC : list of lists
        Initial number of symptomatic recovered classifed by age (column name)
        and region (row name).
    recovered_asym_IC : list of lists
        Initial number of asymptomatic recovered classifed by age (column name)
        and region (row name).
    dead_IC: list of lists
        Initial number of dead classifed by age (column name) and region
        (row name).

    """
    def __init__(self, model, susceptibles_IC, exposed_IC, infectives_pre_IC,
                 infectives_asym_IC, infectives_sym_IC, infectives_pre_ss_IC,
                 infectives_asym_ss_IC, infectives_sym_ss_IC, infectives_q_IC,
                 recovered_IC, recovered_asym_IC, dead_IC):
        super(RocheICs, self).__init__()

        # Set model
        if not isinstance(model, em.RocheSEIRModel):
            raise TypeError('The model must be a Roche SEIRD Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(
            susceptibles_IC, exposed_IC, infectives_pre_IC, infectives_asym_IC,
            infectives_sym_IC, infectives_pre_ss_IC, infectives_asym_ss_IC,
            infectives_sym_ss_IC, infectives_q_IC, recovered_IC,
            recovered_asym_IC, dead_IC)

        # Set ICs parameters
        self.susceptibles = susceptibles_IC
        self.exposed = exposed_IC
        self.infectives_pre = infectives_pre_IC
        self.infectives_asym = infectives_asym_IC
        self.infectives_sym = infectives_sym_IC
        self.infectives_pre_ss = infectives_pre_ss_IC
        self.infectives_asym_ss = infectives_asym_ss_IC
        self.infectives_sym_ss = infectives_sym_ss_IC
        self.infectives_q = infectives_q_IC
        self.recovered = recovered_IC
        self.recovered_asym = recovered_asym_IC
        self.dead = dead_IC

    def _check_parameters_input(self, susceptibles_IC, exposed_IC,
                                infectives_pre_IC, infectives_asym_IC,
                                infectives_sym_IC, infectives_pre_ss_IC,
                                infectives_asym_ss_IC, infectives_sym_ss_IC,
                                infectives_q_IC, recovered_IC,
                                recovered_asym_IC, dead_IC):
        """
        Check correct format of ICs input.

        Parameters
        ----------
        susceptibles_IC : list of lists
            Initial number of susceptibles classifed by age (column name) and
            region (row name).
        exposed_IC : list of lists
            Initial number of exposed classifed by age (column name) and region
            (row name).
        infectives_pre_IC :list of lists
            Initial number of presymptomatic infectives classifed by age
            (column name) and region (row name).
        infectives_asym_IC :list of lists
            Initial number of asymptomatic infectives classifed by age
            (column name) and region (row name).
        infectives_sym_IC :list of lists
            Initial number of symptomatic infectives classifed by age
            (column name) and region (row name).
        infectives_pre_ss_IC : list of lists
            Initial number of presymptomatic superspreader infectives
            classifed by age (column name) and region (row name).
        infectives_asym_ss_IC : list of lists
            Initial number of asymptomatic superspreader infectives classifed
            by age (column name) and region (row name).
        infectives_sym_ss_IC : list of lists
            Initial number of symptomatic superspreader infectives classifed by
            age (column name) and region (row name).
        infectives_q_IC : list of lists
            Initial number of symptomatic infectives quarantined classifed by
            age (column name) and region (row name).
        recovered_IC : list of lists
            Initial number of symptomatic recovered classifed by age
            (column name) and region (row name).
        recovered_asym_IC : list of lists
            Initial number of asymptomatic recovered classifed by age
            (column name) and region (row name).
        dead_IC: list of lists
            Initial number of dead classifed by age (column name) and region
            (row name).

        """
        if np.asarray(susceptibles_IC).ndim != 2:
            raise ValueError('The inital numbers of susceptibles storage format \
                must be 2-dimensional.')
        if np.asarray(susceptibles_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        susceptibles.')
        if np.asarray(susceptibles_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        susceptibles.')
        for ic in np.asarray(susceptibles_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of susceptibles must be integer or \
                            float.')

        if np.asarray(exposed_IC).ndim != 2:
            raise ValueError('The inital numbers of exposed storage format \
                must be 2-dimensional.')
        if np.asarray(exposed_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed.')
        if np.asarray(exposed_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed.')
        for ic in np.asarray(exposed_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of exposed must be integer or \
                            float.')

        if np.asarray(infectives_pre_IC).ndim != 2:
            raise ValueError('The inital numbers of presymptomatic infectives storage format \
                must be 2-dimensional.')
        if np.asarray(infectives_pre_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        presymptomatic infectives.')
        if np.asarray(infectives_pre_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        presymptomatic infectives.')
        for ic in np.asarray(infectives_pre_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of presymptomatic infectives must be integer or \
                            float.')

        if np.asarray(infectives_asym_IC).ndim != 2:
            raise ValueError('The inital numbers of asymptomatic infectives storage format \
                must be 2-dimensional.')
        if np.asarray(infectives_asym_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        asymptomatic infectives.')
        if np.asarray(infectives_asym_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        asymptomatic infectives.')
        for ic in np.asarray(infectives_asym_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of asymptomatic infectives must be integer or \
                            float.')

        if np.asarray(infectives_sym_IC).ndim != 2:
            raise ValueError('The inital numbers of symptomatic infectives storage format \
                must be 2-dimensional.')
        if np.asarray(infectives_sym_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        symptomatic infectives.')
        if np.asarray(infectives_sym_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        symptomatic infectives.')
        for ic in np.asarray(infectives_sym_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of symptomatic infectives must be integer or \
                            float.')

        if np.asarray(infectives_pre_ss_IC).ndim != 2:
            raise ValueError('The inital numbers of presymptomatic super-spreader \
                infectives storage format must be 2-dimensional.')
        if np.asarray(infectives_pre_ss_IC).shape[0] != len(
                self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        presymptomatic super-spreader infectives.')
        if np.asarray(infectives_pre_ss_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        presymptomatic super-spreader infectives.')
        for ic in np.asarray(infectives_pre_ss_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of presymptomatic super-spreader \
                            infectives must be integer or float.')

        if np.asarray(infectives_asym_ss_IC).ndim != 2:
            raise ValueError('The inital numbers of asymptomatic super-spreader \
                infectives storage format must be 2-dimensional.')
        if np.asarray(infectives_asym_ss_IC).shape[0] != len(
                self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        asymptomatic super-spreader infectives.')
        if np.asarray(infectives_asym_ss_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        asymptomatic super-spreader infectives.')
        for ic in np.asarray(infectives_asym_ss_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of asymptomatic super-spreader \
                            infectives must be integer or float.')

        if np.asarray(infectives_sym_ss_IC).ndim != 2:
            raise ValueError('The inital numbers of symptomatic super-spreader \
                infectives storage format must be 2-dimensional.')
        if np.asarray(infectives_sym_ss_IC).shape[0] != len(
                self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        symptomatic super-spreader infectives.')
        if np.asarray(infectives_sym_ss_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        symptomatic super-spreader infectives.')
        for ic in np.asarray(infectives_sym_ss_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of symptomatic super-spreader \
                            infectives must be integer or float.')

        if np.asarray(infectives_q_IC).ndim != 2:
            raise ValueError('The inital numbers of quarantined \
                infectives storage format must be 2-dimensional.')
        if np.asarray(infectives_q_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        quarantined infectives.')
        if np.asarray(infectives_q_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        quarantined infectives.')
        for ic in np.asarray(infectives_q_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of quarantined \
                            infectives must be integer or float.')

        if np.asarray(recovered_IC).ndim != 2:
            raise ValueError('The inital numbers of symptomatic recovered storage format \
                must be 2-dimensional.')
        if np.asarray(recovered_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        symptomatic recovered.')
        if np.asarray(recovered_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        symptomatic recovered.')
        for ic in np.asarray(recovered_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of symptomatic recovered must be integer or \
                            float.')

        if np.asarray(recovered_asym_IC).ndim != 2:
            raise ValueError('The inital numbers of asymptomatic recovered storage format \
                must be 2-dimensional.')
        if np.asarray(recovered_asym_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        asymptomatic recovered.')
        if np.asarray(recovered_asym_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        asymptomatic recovered.')
        for ic in np.asarray(recovered_asym_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of asymptomatic recovered must be integer or \
                            float.')

        if np.asarray(dead_IC).ndim != 2:
            raise ValueError('The inital numbers of dead storage format \
                must be 2-dimensional.')
        if np.asarray(dead_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        dead.')
        if np.asarray(dead_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        dead.')
        for ic in np.asarray(dead_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of dead must be integer or \
                            float.')

    def __call__(self):
        """
        Returns the initial conditions of the :class:`RocheSEIRModel` the
        class relates to.

        Returns
        -------
        List of lists
            List of the initial conditions of the :class:`RocheSEIRModel`
            the class relates to.

        """
        return [self.susceptibles, self.exposed, self.infectives_pre,
                self.infectives_asym, self.infectives_sym,
                self.infectives_pre_ss, self.infectives_asym_ss,
                self.infectives_sym_ss, self.infectives_q,
                self.recovered, self.recovered_asym, self.dead]

#
# RocheCompartmentTimes Class
#


class RocheCompartmentTimes(object):
    """RocheCompartmentTimes:
    Base class for the average-time-in-compartment parameters of the Roche
    model: deterministic SEIRD used by the F. Hoffmann-La Roche Ltd to model
    the Covid-19 epidemic and the effects of non-pharmaceutical interventions
    (NPIs) on the epidemic dynamic in different countries.

    Parameters
    ----------
    k : float or int
        The average time it takes for an individual to become infectious once
        exposed. Non age-dependent.
    kS : float or int
        The average time it takes for an individual to develop symptoms
        (or remain asymptomatic) once they becomes infectious. Non
        age-dependent.
    kQ : float or int
        The average time it takes for an individual to enter quarantine once
        they develop symptoms. Non age-dependent.
    kR : float or int or list
        The average time it takes for a quarantined individual to recover or
        die. Age-dependent.
    kRI : float or int or list
        The average time it takes for an asymptomatic individual to recover.
        Age-dependent.

    """
    def __init__(self, model, k, kS, kQ, kR, kRI):
        super(RocheCompartmentTimes, self).__init__()

        # Set model
        if not isinstance(model, em.RocheSEIRModel):
            raise TypeError('The model must be a Roche SEIRD Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(k, kS, kQ, kR, kRI)

        # Set average-time-in-compartment parameters
        self.k = k
        self.kS = kS
        self.kQ = kQ

        if isinstance(kR, (float, int)):
            self.kR = kR * np.ones(self.model._num_ages)
        else:
            self.kR = kR

        if isinstance(kRI, (float, int)):
            self.kRI = kRI * np.ones(self.model._num_ages)
        else:
            self.kRI = kRI

    def _check_parameters_input(self, k, kS, kQ, kR, kRI):
        """
        Check correct format of the average-time-in-compartment parameters
        input.

        Parameters
        ----------
        k : float or int
            The average time it takes for an individual to become infectious
            once exposed. Non age-dependent.
        kS : float or int
            The average time it takes for an individual to develop symptoms
            (or remain asymptomatic) once they becomes infectious. Non
            age-dependent.
        kQ : float or int
            The average time it takes for an individual to enter quarantine
            once they develop symptoms. Non age-dependent.
        kR : float or int
            The average time it takes for a quarantined individual to recover
            or die. Age-dependent.
        kRI : float or int
            The average time it takes for an asymptomatic individual to
            recover. Age-dependent.

        """
        if not isinstance(k, (float, int)):
            raise TypeError('The average time it takes for an individual to become \
                infectious once exposed must be float or integer.')
        if k <= 0:
            raise ValueError('The average time it takes for an individual to become \
                infectious once exposed must be > 0.')

        if not isinstance(kS, (float, int)):
            raise TypeError('The average time it takes for an individual to develop symptoms \
                (or remain asymptomatic) once they becomes infectious must be \
                    float or integer.')
        if kS <= 0:
            raise ValueError('The average time it takes for an individual to develop symptoms \
                (or remain asymptomatic) once they becomes infectious must be \
                    > 0.')

        if not isinstance(kQ, (float, int)):
            raise TypeError('The average time it takes for an individual to enter quarantine \
                once they develop symptoms must be float or integer.')
        if kQ <= 0:
            raise ValueError('The average time it takes for an individual to enter quarantine \
                once they develop symptoms must be > 0.')

        if isinstance(kR, (float, int)):
            kR = [kR]
        if np.asarray(kR).ndim != 1:
            raise ValueError('The average time it takes for a quarantined individual to \
                recover or die storage format must be 1-dimensional.')
        if (np.asarray(kR).shape[0] != self.model._num_ages) and (
                np.asarray(kR).shape[0] != 1):
            raise ValueError(
                    'Wrong number of age groups for the average time it takes for a quarantined \
                    individual to recover or die.')
        for _ in kR:
            if not isinstance(_, (float, int)):
                raise TypeError('The average time it takes for a quarantined individual to \
                    recover or die must be float or integer.')
            if _ <= 0:
                raise ValueError('The average time it takes for a quarantined individual to \
                    recover or die must be > 0.')

        if isinstance(kRI, (float, int)):
            kRI = [kRI]
        if np.asarray(kRI).ndim != 1:
            raise ValueError('The average times it takes for an asymptomatic individual to \
                recover storage format must be 1-dimensional.')
        if (np.asarray(kRI).shape[0] != self.model._num_ages) and (
                np.asarray(kRI).shape[0] != 1):
            raise ValueError(
                    'Wrong number of age groups for the average times it takes for \
                    an asymptomatic individual to recover.')
        for _ in kRI:
            if not isinstance(_, (float, int)):
                raise TypeError('The average times it takes for an asymptomatic individual to \
                    recover must be float or integer.')
            if _ <= 0:
                raise ValueError('The average times it takes for an asymptomatic individual to \
                    recover must be > 0.')

    def __call__(self):
        """
        Returns the average-time-in-compartment parameters of the
        :class:`RocheSEIRModel` the class relates to.

        Returns
        -------
        List of lists
            List of the average-time-in-compartment parameters of the
            :class:`RocheSEIRModel` the class relates to.

        """
        return [self.k, self.kS, self.kQ, self.kR, self.kRI]

#
# RocheProportions Class
#


class RocheProportions(object):
    """RocheProportions:
    Base class for the proportions of asymptomatic, super-spreader and dead
    cases parameters of the Roche model: deterministic SEIRD used by the F.
    Hoffmann-La Roche Ltd to model the Covid-19 epidemic and the effects of
    non-pharmaceutical interventions (NPIs) on the epidemic dynamic in
    different countries.

    Parameters
    ----------
    Pa : int or float or list
        Proportion of asymptomatic cases.
    Pss : int or float
        Proportion of super-spreader cases.
    Pd : int or float or list
        Proportion of dead cases.

    """
    def __init__(self, model, Pa, Pss, Pd):
        super(RocheProportions, self).__init__()

        # Set model
        if not isinstance(model, em.RocheSEIRModel):
            raise TypeError('The model must be a Roche SEIRD Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(Pa, Pss, Pd)

        # Set proportions of asymptomatic, super-spreader and dead cases
        if isinstance(Pa, (float, int)):
            self.Pa = Pa * np.ones(self.model._num_ages)
        else:
            self.Pa = Pa

        self.Pss = Pss

        if isinstance(Pd, (float, int)):
            self.Pd = Pd * np.ones(self.model._num_ages)
        else:
            self.Pd = Pd

    def _check_parameters_input(self, Pa, Pss, Pd):
        """
        Check correct format of the proportions of asymptomatic,
        super-spreader and dead cases parameters input.

        Parameters
        ----------
        Pa : int or float or list
            Proportion of asymptomatic cases.
        Pss : int or float
            Proportion of super-spreader cases.
        Pd : int or float or list
            Proportion of dead cases.

        """
        if isinstance(Pa, (float, int)):
            Pa = [Pa]
        if np.asarray(Pa).ndim != 1:
            raise ValueError('The propotions of people that go on to be \
                    asymptomatic storage format must be 1-dimensional.')
        if (np.asarray(Pa).shape[0] != self.model._num_ages) and (
                np.asarray(Pa).shape[0] != 1):
            raise ValueError(
                    'Wrong number of age groups for the propotions of people \
                        that go on to be asymptomatic.')
        for _ in Pa:
            if not isinstance(_, (float, int)):
                raise TypeError('The propotions of people that go on to be \
                    asymptomatic must be float or integer.')
            if _ < 0:
                raise ValueError('The propotions of people that go on to be \
                    asymptomatic must be >= 0.')
            if _ > 1:
                raise ValueError('The propotions of people that go on to be \
                    asymptomatic must be <= 1.')

        if not isinstance(Pss, (float, int)):
            raise TypeError('The propotions of people that go on to be \
                super-spreaders must be float or integer.')
        if Pss < 0:
            raise ValueError('The propotions of people that go on to be \
                super-spreaders must be >= 0.')
        if Pss > 1:
            raise ValueError('The propotions of people that go on to be \
                super-spreaders must be <= 1.')

        if isinstance(Pd, (float, int)):
            Pd = [Pd]
        if np.asarray(Pd).ndim != 1:
            raise ValueError('The propotions of people that go on to be \
                    dead storage format must be 1-dimensional.')
        if (np.asarray(Pd).shape[0] != self.model._num_ages) and (
                np.asarray(Pd).shape[0] != 1):
            raise ValueError(
                    'Wrong number of age groups for the propotions of people \
                        that go on to be dead.')
        for _ in Pd:
            if not isinstance(_, (float, int)):
                raise TypeError('The propotions of people that go on to be \
                    dead must be float or integer.')
            if _ < 0:
                raise ValueError('The propotions of people that go on to be \
                    dead must be >= 0.')
            if _ > 1:
                raise ValueError('The propotions of people that go on to be \
                    dead must be <= 1.')

    def __call__(self):
        """
        Returns the proportions of asymptomatic, super-spreader and dead
        cases  parameters of the :class:`RocheSEIRModel` the class relates to.

        Returns
        -------
        List of lists
            List of the proportions of asymptomatic, super-spreader and dead
            cases parameters of the :class:`RocheSEIRModel` the class relates
            to.

        """
        return [self.Pa, self.Pss, self.Pd]

#
# RocheTransmission Class
#


class RocheTransmission(object):
    """RocheTransmission:
    Base class for the transmission-specific parameters of the Roche
    model: deterministic SEIRD used by the F. Hoffmann-La Roche Ltd to model
    the Covid-19 epidemic and the effects of non-pharmaceutical interventions
    (NPIs) on the epidemic dynamic in different countries.

    Parameters
    ----------
    beta_min : int or float
        The minimum possible transmission rate of the virus.
    beta_max : int or float
        The maximum possible transmission rate of the virus.
    bss : int or float
        The relative increase in transmission of a super-spreader case.
    gamma :  : int or float
        The sharpness of the intervention wave used for function continuity
        purposes.
    s50 : int or float
        The stringency index needed to reach 50% of the maximum effect on the
        infection rate.

    """
    def __init__(self, model, beta_min, beta_max, bss, gamma, s50):
        super(RocheTransmission, self).__init__()

        # Set model
        if not isinstance(model, em.RocheSEIRModel):
            raise TypeError('The model must be a Roche SEIRD Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(beta_min, beta_max, bss, gamma, s50)

        # Set transmission-specific parameters
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.bss = bss
        self.gamma = gamma
        self.s50 = s50

    def _check_parameters_input(self, beta_min, beta_max, bss, gamma, s50):
        """
        Check correct format of the transmission-specific parameters input.

        Parameters
        ----------
        beta_min : int or float
            The minimum possible transmission rate of the virus.
        beta_max : int or float
            The maximum possible transmission rate of the virus.
        bss : int or float
            The relative increase in transmission of a super-spreader case.
        gamma :  : int or float
            The sharpness of the intervention wave used for function continuity
            purposes.
        s50 : int or float
            The stringency index needed to reach 50% of the maximum effect on
            the infection rate.

        """
        if not isinstance(beta_min, (float, int)):
            raise TypeError('The minimum possible transmission rate must be \
                float or integer.')
        if beta_min <= 0:
            raise ValueError('The minimum and maximum possible transmission \
                rate must be > 0.')

        if not isinstance(beta_max, (float, int)):
            raise TypeError('The maximum possible transmission rate must be \
                float or integer.')
        if beta_max <= 0:
            raise ValueError('The minimum and maximum possible transmission \
                rate must be > 0.')

        if not isinstance(bss, (float, int)):
            raise TypeError('The relative increase in transmission of a \
                super-spreader must be float or integer.')
        if bss <= 0:
            raise ValueError('The relative increase in transmission of a \
                super-spreader must be > 0.')

        if not isinstance(gamma, (float, int)):
            raise TypeError(
                'The sharpness of the intervention wave must be float or \
                    integer.')
        if gamma < 0:
            raise ValueError('The sharpness of the intervention wave must be \
                => 0.')

        if not isinstance(s50, (float, int)):
            raise TypeError(
                'The half-effect stringency index must be float or integer.')
        if s50 <= 0:
            raise ValueError('The half-effect stringency index must be > 0.')
        if s50 > 100:
            raise ValueError('The half-effect stringency index must be <=100.')

    def __call__(self):
        """
        Returns the transmission-specific parameters of the
        :class:`RocheSEIRModel` the class relates to.

        Returns
        -------
        List of lists
            List of the transmission-specific parameters of the
            :class:`RocheSEIRModel` the class relates to.

        """
        return [self.beta_min, self.beta_max, self.bss, self.gamma, self.s50]

#
# RocheSimParameters Class
#


class RocheSimParameters(object):
    """RocheSimParameters:
    Base class for the simulation method's parameters of the Roche
    model: deterministic SEIRD used by the F. Hoffmann-La Roche Ltd to model
    the Covid-19 epidemic and the effects of non-pharmaceutical interventions
    (NPIs) on the epidemic dynamic in different countries.

    Parameters
    ----------
    region_index : int
        Index of region for which we wish to simulate.
    method: str
        The type of solver implemented by the simulator.
    times : list
        List of time points at which we wish to evaluate the ODEs
        system.

    """
    def __init__(self, model, region_index, method, times):
        super(RocheSimParameters, self).__init__()

        # Set model
        if not isinstance(model, em.RocheSEIRModel):
            raise TypeError('The model must be a Roche SEIRD Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(region_index, method, times)

        # Set other simulation parameters
        self.region_index = region_index
        self.method = method
        self.times = times

    def _check_parameters_input(self, region_index, method, times):
        """
        Check correct format of the simulation method's parameters input.

        Parameters
        ----------
        region_index : int
            Index of region for which we wish to simulate.
        method: str
            The type of solver implemented by the simulator.
        times : list
            List of time points at which we wish to evaluate the ODEs
            system.

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

        if not isinstance(region_index, int):
            raise TypeError('Index of region to evaluate must be integer.')
        if region_index <= 0:
            raise ValueError('Index of region to evaluate must be >= 1.')
        if region_index > len(self.model.regions):
            raise ValueError('Index of region to evaluate is out of bounds.')

        if not isinstance(method, str):
            raise TypeError('Simulation method must be a string.')
        if method not in (
                'RK45', 'RK23', 'Radau', 'BDF', 'LSODA', 'DOP853'):
            raise ValueError('Simulation method not available.')

    def __call__(self):
        """
        Returns the simulation method's parameters of the
        :class:`RocheSEIRModel` the class relates to.

        Returns
        -------
        List of lists
            List of the simulation method's parameters of the
            :class:`RocheSEIRModel` the class relates to.

        """
        return [self.region_index, self.method]

#
# RocheParametersController Class
#


class RocheParametersController(object):
    """RocheParametersController Class:
    Base class for the paramaters of the Roche model: deterministic SEIRD used
    by the F. Hoffmann-La Roche Ltd to model the Covid-19 epidemic and the
    effects of non-pharmaceutical interventions (NPIs) on the epidemic dynamic
    in different countries.

    In order to simulate using the Roche model, the following parameters are
    required, which are stored as part of this class.

    Parameters
    ----------
    model : RocheSEIRModel
        The model whose parameters are stored.
    ICs : RocheICs
        Class of the Ics used in the simulation of the model.
    compartment_times : RocheCompartmentTimes
        Class of the average-time-in-compartment parameters used in the
        simulation of the model.
    proportion_parameters : RocheProportions
            Class of the proportions of asymptomatic, super-spreader and dead
            cases parameters  used in the simulation of the model.
    proportion_parameters : RocheProportions
            Class of the proportions of asymptomatic, super-spreader and dead
            cases parameters used in the simulation of the model.
    simulation_parameters : RocheSimParameters
        Class of the simulation method's parameters used in the simulation of
        the model.

    """
    def __init__(
            self, model, ICs, compartment_times, proportion_parameters,
            transmission_parameters, simulation_parameters):
        # Instantiate class
        super(RocheParametersController, self).__init__()

        # Set model
        if not isinstance(model, em.RocheSEIRModel):
            raise TypeError('The model must be a Roche SEIRD Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(
            ICs, compartment_times, proportion_parameters,
            transmission_parameters, simulation_parameters)

        # Set ICs parameters
        self.ICs = ICs

        # Set average-time-in-compartment parameters
        self.compartment_times = compartment_times

        # Set proportions of asymptomatic, super-spreader and dead
        # cases parameters
        self.proportion_parameters = proportion_parameters

        # Set transmission-specific parameters
        self.transmission_parameters = transmission_parameters

        # Set other simulation parameters
        self.simulation_parameters = simulation_parameters

    def _check_parameters_input(
            self, ICs, compartment_times, proportion_parameters,
            transmission_parameters, simulation_parameters):
        """
        Check correct format of input of simulate method.

        Parameters
        ----------
        ICs : RocheICs
            Class of the Ics used in the simulation of the model.
        compartment_times : RocheCompartmentTimes
            Class of the average-time-in-compartment parameters used in the
            simulation of the model.
        proportion_parameters : RocheProportions
            Class of the proportions of asymptomatic, super-spreader and dead
            cases parameters used in the simulation of the model.
        transmission_parameters : RocheTransmission
            Class of the proportions of asymptomatic, super-spreader and dead
            cases parameters used in the simulation of the model.
        simulation_parameters : RocheSimParameters
            Class of the simulation method's parameters used in the simulation
            of the model.

        """
        if not isinstance(ICs, RocheICs):
            raise TypeError('The model`s ICs parameters must be a of a Roche SEIRD \
                Model.')
        if ICs.model != self.model:
            raise ValueError('ICs do not correspond to the right model.')

        if not isinstance(compartment_times, RocheCompartmentTimes):
            raise TypeError('The model`s average-time-in-compartment parameters must \
                be a of a Roche SEIRD Model.')
        if compartment_times.model != self.model:
            raise ValueError('The average-time-in-compartment parameters do \
                not correspond to the right model.')

        if not isinstance(proportion_parameters, RocheProportions):
            raise TypeError('The model`s proportions of asymptomatic, super-spreader and dead \
                cases parameters parameters must be a of a Roche SEIRD Model.')
        if proportion_parameters.model != self.model:
            raise ValueError('The proportions of asymptomatic, super-spreader and dead cases \
                parameters parameters do not correspond to the right model.')

        if not isinstance(transmission_parameters, RocheTransmission):
            raise TypeError('The model`s transmission-specific parameters must be a \
                of a Roche SEIRD Model.')
        if transmission_parameters.model != self.model:
            raise ValueError('The transmission-specific parameters do not correspond \
                to the right model.')

        if not isinstance(simulation_parameters, RocheSimParameters):
            raise TypeError('The model`s simulation method`s parameters must be a \
                of a Roche SEIR Model.')
        if simulation_parameters.model != self.model:
            raise ValueError('The simulation method`s parameters do not correspond \
                to the right model.')

    def __call__(self):
        """
        Returns the list of all the parameters used for the simulation of the
        Roche model in their order, which will be then separated within the
        :class:`RocheSEIRModel` class.

        Returns
        -------
        list
            List of all the parameters used for the simulation of the
            Roche model in their order.

        """
        parameters = []

        # Add the region index parameters
        parameters.append(self.simulation_parameters()[0])

        # Add ICs
        parameters.extend(self.ICs())

        # Add average-time-in-compartment parameters
        parameters.extend(self.compartment_times())

        # Add proportions of asymptomatic, super-spreader and dead
        # cases parameters
        parameters.extend(self.proportion_parameters())

        # Add transmission-specific parameters
        parameters.extend(self.transmission_parameters())

        # Add other simulation parameters
        parameters.append(self.simulation_parameters()[1])

        return list(deepflatten(parameters, ignore=str))
