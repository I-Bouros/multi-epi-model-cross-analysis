#
# PheICs Class
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
        super(PheICs, self).__init__()

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
        if np.asarray(initial_r).ndim != 1:
            raise ValueError('The inital reproduction numbers storage format \
                must be 1-dimensional.')
        if np.asarray(initial_r).shape[0] != self.model._num_ages:
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

        if np.asarray(betas).ndim != 2:
            raise ValueError('The temporal and regional fluctuations storage format \
                must be 2-dimensional.')
        if np.asarray(betas).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the temporal and regional \
                        fluctuation.')

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
        super(PheICs, self).__init__()

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
        Check correct format of the rdisease-specific parameters input.

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
        super(PheICs, self).__init__()

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
        Class of the disease-specific parameters used in the simulation of
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
            Class of the disease-specific parameters used in the simulation of
            the model.

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
            raise TypeError('The model`s disease-specific parameters must be a \
                of a PHE SEIR Model.')
        if simulation_parameters.model != self.model:
            raise ValueError('The disease-specific parameters do not correspond \
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
        parameters.extend(self.disease_parameters)

        # Add other simulation parameters
        parameters.extend(self.simulation_parameters)

        return list(deepflatten(parameters, ignore=str))
