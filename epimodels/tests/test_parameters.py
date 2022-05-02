#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#

import unittest

import numpy as np
import numpy.testing as npt

import epimodels as em
from epimodels.phe_model import PheSEIRModel


examplePHEmodel = PheSEIRModel()
examplePHEmodel2 = PheSEIRModel()

# Populate the model
regions = ['London', 'Cornwall']
age_groups = ['0-10', '10-25']

# Initial state of the system
contact_data_matrix_0 = np.array([[1, 0], [0, 3]])
contact_data_matrix_1 = np.array([[10, 5.2], [0, 3]])

region_data_matrix_0_0 = np.array([[0.5, 0], [0, 6]])
region_data_matrix_0_1 = np.array([[1, 10], [1, 0]])
region_data_matrix_1_0 = np.array([[0.5, 1.2], [0.29, 6]])
region_data_matrix_1_1 = np.array([[0.85, 1], [0.9, 6]])

contacts_0 = em.ContactMatrix(age_groups, contact_data_matrix_0)
contacts_1 = em.ContactMatrix(age_groups, contact_data_matrix_1)
regional_0_0 = em.RegionMatrix(
    regions[0], age_groups, region_data_matrix_0_0)
regional_0_1 = em.RegionMatrix(
    regions[1], age_groups, region_data_matrix_0_1)
regional_1_0 = em.RegionMatrix(
    regions[0], age_groups, region_data_matrix_1_0)
regional_1_1 = em.RegionMatrix(
    regions[1], age_groups, region_data_matrix_1_1)

# Matrices contact
matrices_contact = [contacts_0, contacts_1]
time_changes_contact = [1, 3]
matrices_region = [
    [regional_0_0, regional_0_1],
    [regional_1_0, regional_1_1]]
time_changes_region = [1, 2]

matrices_contact2 = [contacts_0, contacts_1]
time_changes_contact2 = [1, 5]
matrices_region2 = [
    [regional_0_0, regional_0_1],
    [regional_1_0, regional_1_1]]
time_changes_region2 = [1, 10]

examplePHEmodel.set_regions(regions)
examplePHEmodel.set_age_groups(age_groups)
examplePHEmodel.read_contact_data(matrices_contact, time_changes_contact)
examplePHEmodel.read_regional_data(matrices_region, time_changes_region)

examplePHEmodel2.set_regions(regions)
examplePHEmodel2.set_age_groups(age_groups)
examplePHEmodel2.read_contact_data(matrices_contact2, time_changes_contact2)
examplePHEmodel2.read_regional_data(matrices_region2, time_changes_region2)


#
# Test PheICs Class
#


class TestPheICs(unittest.TestCase):
    """
    Test the 'PheICs' class.
    """
    def test__init__(self):
        model = examplePHEmodel

        susceptibles = [[5, 6], [7, 8]]
        exposed1 = [[10, 2], [3, 0]]
        exposed2 = [[5, 9], [8, 8]]
        infectives1 = [[1, 1], [1, 0]]
        infectives2 = [[5, 5], [0, 0]]
        recovered = [[0, 0], [0, 0]]

        ICs = em.PheICs(
            model=model,
            susceptibles_IC=susceptibles,
            exposed1_IC=exposed1,
            exposed2_IC=exposed2,
            infectives1_IC=infectives1,
            infectives2_IC=infectives2,
            recovered_IC=recovered)

        self.assertEqual(ICs.model, model)

        npt.assert_array_equal(np.array(ICs.susceptibles),
                               np.array([[5, 6], [7, 8]]))

        npt.assert_array_equal(np.array(ICs.exposed1),
                               np.array([[10, 2], [3, 0]]))

        npt.assert_array_equal(np.array(ICs.exposed2),
                               np.array([[5, 9], [8, 8]]))

        npt.assert_array_equal(np.array(ICs.infectives1),
                               np.array([[1, 1], [1, 0]]))

        npt.assert_array_equal(np.array(ICs.infectives2),
                               np.array([[5, 5], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.recovered),
                               np.array([[0, 0], [0, 0]]))

        with self.assertRaises(TypeError):
            model1 = 0

            em.PheICs(
                model=model1,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                infectives1_IC=infectives1,
                infectives2_IC=infectives2,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            susceptibles1 = [5, 6]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles1,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                infectives1_IC=infectives1,
                infectives2_IC=infectives2,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            susceptibles1 = [[5, 6], [7, 8], [9, 10]]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles1,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                infectives1_IC=infectives1,
                infectives2_IC=infectives2,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            susceptibles1 = [[5, 6, 7], [8, 9, 10]]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles1,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                infectives1_IC=infectives1,
                infectives2_IC=infectives2,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed11 = [10, 2]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed11,
                exposed2_IC=exposed2,
                infectives1_IC=infectives1,
                infectives2_IC=infectives2,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed11 = [[10, 2], [3, 0], [0, 0]]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed11,
                exposed2_IC=exposed2,
                infectives1_IC=infectives1,
                infectives2_IC=infectives2,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed11 = [[10, 2, 3], [0, 0, 0]]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed11,
                exposed2_IC=exposed2,
                infectives1_IC=infectives1,
                infectives2_IC=infectives2,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed21 = [5, 9]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed21,
                infectives1_IC=infectives1,
                infectives2_IC=infectives2,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed21 = [[5, 9], [8, 8], [0, 0]]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed21,
                infectives1_IC=infectives1,
                infectives2_IC=infectives2,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed21 = [[5, 9, 8], [8, 0, 0]]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed21,
                infectives1_IC=infectives1,
                infectives2_IC=infectives2,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            infectives11 = [1, 1]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                infectives1_IC=infectives11,
                infectives2_IC=infectives2,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            infectives11 = [[1, 1], [1, 0], [0, 0]]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                infectives1_IC=infectives11,
                infectives2_IC=infectives2,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            infectives11 = [[1, 1, 1], [0, 0, 0]]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                infectives1_IC=infectives11,
                infectives2_IC=infectives2,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            infectives21 = [5, 5]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                infectives1_IC=infectives1,
                infectives2_IC=infectives21,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            infectives21 = [[5, 5], [0, 0], [0, 0]]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                infectives1_IC=infectives1,
                infectives2_IC=infectives21,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            infectives21 = [[5, 5, 0], [0, 0, 0]]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                infectives1_IC=infectives1,
                infectives2_IC=infectives21,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            recovered1 = [0, 0]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                infectives1_IC=infectives1,
                infectives2_IC=infectives2,
                recovered_IC=recovered1)

        with self.assertRaises(ValueError):
            recovered1 = [[0, 0], [0, 0], [0, 0]]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                infectives1_IC=infectives1,
                infectives2_IC=infectives2,
                recovered_IC=recovered1)

        with self.assertRaises(ValueError):
            recovered1 = [[0, 0, 0], [0, 0, 0]]

            em.PheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                infectives1_IC=infectives1,
                infectives2_IC=infectives2,
                recovered_IC=recovered1)

    def test__call__(self):
        model = examplePHEmodel

        susceptibles = [[5, 6], [7, 8]]
        exposed1 = [[10, 2], [3, 0]]
        exposed2 = [[5, 9], [8, 8]]
        infectives1 = [[1, 1], [1, 0]]
        infectives2 = [[5, 5], [0, 0]]
        recovered = [[0, 0], [0, 0]]

        ICs = em.PheICs(
            model=model,
            susceptibles_IC=susceptibles,
            exposed1_IC=exposed1,
            exposed2_IC=exposed2,
            infectives1_IC=infectives1,
            infectives2_IC=infectives2,
            recovered_IC=recovered)

        self.assertEqual(
            ICs(),
            [susceptibles, exposed1, exposed2, infectives1, infectives2,
             recovered]
        )

#
# Test PheRegParameters Class
#


class TestPheRegParameters(unittest.TestCase):
    """
    Test the 'PheRegParameters' class.
    """
    def test__init__(self):
        model = examplePHEmodel

        initial_r = [0.5, 1]
        region_index = 2
        betas = [[1]*2, [1]*2]
        times = [1, 2]

        RegParameters = em.PheRegParameters(
            model=model,
            initial_r=initial_r,
            region_index=region_index,
            betas=betas,
            times=times
        )

        self.assertEqual(RegParameters.model, model)

        npt.assert_array_equal(np.array(RegParameters.initial_r),
                               np.array([0.5, 1]))

        self.assertEqual(RegParameters.region_index, 2)

        npt.assert_array_equal(np.array(RegParameters.betas),
                               np.array([[1, 1], [1, 1]]))

        npt.assert_array_equal(np.array(RegParameters.times),
                               np.array([1, 2]))

        with self.assertRaises(TypeError):
            model1 = '0'

            em.PheRegParameters(
                model=model1,
                initial_r=initial_r,
                region_index=region_index,
                betas=betas,
                times=times
            )

        with self.assertRaises(TypeError):
            times1 = '0'

            em.PheRegParameters(
                model=model,
                initial_r=initial_r,
                region_index=region_index,
                betas=betas,
                times=times1
            )

        with self.assertRaises(TypeError):
            times1 = ['1', 2]

            em.PheRegParameters(
                model=model,
                initial_r=initial_r,
                region_index=region_index,
                betas=betas,
                times=times1
            )

        with self.assertRaises(ValueError):
            times1 = [0, 1]

            em.PheRegParameters(
                model=model,
                initial_r=initial_r,
                region_index=region_index,
                betas=betas,
                times=times1
            )

        with self.assertRaises(TypeError):
            region_index1 = 0.5

            em.PheRegParameters(
                model=model,
                initial_r=initial_r,
                region_index=region_index1,
                betas=betas,
                times=times
            )

        with self.assertRaises(ValueError):
            region_index1 = 0

            em.PheRegParameters(
                model=model,
                initial_r=initial_r,
                region_index=region_index1,
                betas=betas,
                times=times
            )

        with self.assertRaises(ValueError):
            region_index1 = 3

            em.PheRegParameters(
                model=model,
                initial_r=initial_r,
                region_index=region_index1,
                betas=betas,
                times=times
            )

        with self.assertRaises(ValueError):
            betas1 = [[[1]*2, [1]*2], [[1]*2, [1]*2]]

            em.PheRegParameters(
                model=model,
                initial_r=initial_r,
                region_index=region_index,
                betas=betas1,
                times=times
            )
        with self.assertRaises(ValueError):
            betas1 = [[1]*2, [1]*2, [1]*2]

            em.PheRegParameters(
                model=model,
                initial_r=initial_r,
                region_index=region_index,
                betas=betas1,
                times=times
            )
        with self.assertRaises(ValueError):
            betas1 = [[1]*4, [1]*4]

            em.PheRegParameters(
                model=model,
                initial_r=initial_r,
                region_index=region_index,
                betas=betas1,
                times=times
            )

    def test__call__(self):
        model = examplePHEmodel

        initial_r = [0.5, 1]
        region_index = 2
        betas = [[1]*2, [1]*2]
        times = [1, 2]

        RegParameters = em.PheRegParameters(
            model=model,
            initial_r=initial_r,
            region_index=region_index,
            betas=betas,
            times=times
        )

        self.assertEqual(RegParameters(), [initial_r, region_index, betas])

#
# Test PheDiseaseParameters Class
#


class TestPheDiseaseParameters(unittest.TestCase):
    """
    Test the 'PheDiseaseParameters' class.
    """
    def test__init__(self):
        model = examplePHEmodel

        dL = 4
        dI = 4.5

        DiseaseParameters = em.PheDiseaseParameters(
            model=model,
            dL=dL,
            dI=dI
        )

        self.assertEqual(DiseaseParameters.model, model)
        self.assertEqual(DiseaseParameters.dL, 4)
        self.assertEqual(DiseaseParameters.dI, 4.5)

        with self.assertRaises(TypeError):
            model1 = [4]

            em.PheDiseaseParameters(
                model=model1,
                dL=dL,
                dI=dI
            )

        with self.assertRaises(TypeError):
            dL1 = '4'

            em.PheDiseaseParameters(
                model=model,
                dL=dL1,
                dI=dI
            )

        with self.assertRaises(ValueError):
            dL1 = -1

            em.PheDiseaseParameters(
                model=model,
                dL=dL1,
                dI=dI
            )

        with self.assertRaises(TypeError):
            dI1 = '4'

            em.PheDiseaseParameters(
                model=model,
                dL=dL,
                dI=dI1
            )

        with self.assertRaises(ValueError):
            dI1 = 0

            em.PheDiseaseParameters(
                model=model,
                dL=dL,
                dI=dI1
            )

    def test__call__(self):
        model = examplePHEmodel

        dL = 4
        dI = 4.5

        DiseaseParameters = em.PheDiseaseParameters(
            model=model,
            dL=dL,
            dI=dI
        )

        self.assertEqual(DiseaseParameters(), [4, 4.5])

#
# Test PheSimParameters Class
#


class TestPheSimParameterss(unittest.TestCase):
    """
    Test the 'PheSimParameters' class.
    """
    def test__init__(self):
        model = examplePHEmodel

        delta_t = 0.5
        method = 'RK45'

        SimParameters = em.PheSimParameters(
            model=model,
            delta_t=delta_t,
            method=method
        )

        self.assertEqual(SimParameters.model, model)
        self.assertEqual(SimParameters.delta_t, 0.5)
        self.assertEqual(SimParameters.method, 'RK45')

        with self.assertRaises(TypeError):
            model1 = {'0.005': 0}

            em.PheSimParameters(
                model=model1,
                delta_t=delta_t,
                method=method
            )

        with self.assertRaises(TypeError):
            delta_t1 = '0.005'

            em.PheSimParameters(
                model=model,
                delta_t=delta_t1,
                method=method
            )

        with self.assertRaises(ValueError):
            delta_t1 = 0

            em.PheSimParameters(
                model=model,
                delta_t=delta_t1,
                method=method
            )

        with self.assertRaises(TypeError):
            method1 = 3

            em.PheSimParameters(
                model=model,
                delta_t=delta_t,
                method=method1
            )

        with self.assertRaises(ValueError):
            method1 = 'my-solver2'

            em.PheSimParameters(
                model=model,
                delta_t=delta_t,
                method=method1
            )

    def test__call__(self):
        model = examplePHEmodel

        delta_t = 0.5
        method = 'my-solver'

        SimParameters = em.PheSimParameters(
            model=model,
            delta_t=delta_t,
            method=method
        )

        self.assertEqual(SimParameters(), [0.5, 'my-solver'])

#
# Test PheParametersController Class
#


class TestPheParametersController(unittest.TestCase):
    """
    Test the 'PheParametersController' class.
    """
    def test__init__(self):
        model = examplePHEmodel

        # Set regional and time dependent parameters
        regional_parameters = em.PheRegParameters(
            model=model,
            initial_r=[0.5, 1],
            region_index=2,
            betas=[[1]*2, [1]*2],
            times=[1, 2]
        )

        # Set ICs parameters
        ICs = em.PheICs(
            model=model,
            susceptibles_IC=[[5, 6], [7, 8]],
            exposed1_IC=[[10, 2], [3, 0]],
            exposed2_IC=[[5, 9], [8, 8]],
            infectives1_IC=[[1, 1], [1, 0]],
            infectives2_IC=[[5, 5], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.PheDiseaseParameters(
            model=model,
            dL=4,
            dI=4
        )

        # Set other simulation parameters
        simulation_parameters = em.PheSimParameters(
            model=model,
            delta_t=0.5,
            method='Radau'
        )

        # Set all parameters in the controller
        parameters = em.PheParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            simulation_parameters=simulation_parameters
        )

        self.assertEqual(parameters.model, model)
        self.assertEqual(parameters.ICs, ICs)
        self.assertEqual(parameters.regional_parameters, regional_parameters)
        self.assertEqual(parameters.disease_parameters, disease_parameters)
        self.assertEqual(parameters.simulation_parameters,
                         simulation_parameters)

        with self.assertRaises(TypeError):
            model1 = 0.3

            em.PheParametersController(
                model=model1,
                regional_parameters=regional_parameters,
                ICs=ICs,
                disease_parameters=disease_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(TypeError):
            regional_parameters1 = 0

            em.PheParametersController(
                model=model,
                regional_parameters=regional_parameters1,
                ICs=ICs,
                disease_parameters=disease_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(ValueError):
            regional_parameters1 = em.PheRegParameters(
                model=examplePHEmodel2,
                initial_r=[0.5, 1],
                region_index=2,
                betas=[[1]*2, [1]*2],
                times=[1, 2]
            )

            em.PheParametersController(
                model=model,
                regional_parameters=regional_parameters1,
                ICs=ICs,
                disease_parameters=disease_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(TypeError):
            ICs1 = '0'

            em.PheParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs1,
                disease_parameters=disease_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(ValueError):
            ICs1 = em.PheICs(
                model=examplePHEmodel2,
                susceptibles_IC=[[5, 6], [7, 8]],
                exposed1_IC=[[10, 2], [3, 0]],
                exposed2_IC=[[5, 9], [8, 8]],
                infectives1_IC=[[1, 1], [1, 0]],
                infectives2_IC=[[5, 5], [0, 0]],
                recovered_IC=[[0, 0], [0, 0]]
            )

            em.PheParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs1,
                disease_parameters=disease_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(TypeError):
            disease_parameters1 = [0]

            em.PheParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs,
                disease_parameters=disease_parameters1,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(ValueError):
            disease_parameters1 = em.PheDiseaseParameters(
                model=examplePHEmodel2,
                dL=4,
                dI=4
            )

            em.PheParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs,
                disease_parameters=disease_parameters1,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(TypeError):
            simulation_parameters1 = {'0': 0}

            em.PheParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs,
                disease_parameters=disease_parameters,
                simulation_parameters=simulation_parameters1
            )

        with self.assertRaises(ValueError):
            simulation_parameters1 = em.PheSimParameters(
                model=examplePHEmodel2,
                delta_t=0.5,
                method='Radau'
            )

            em.PheParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs,
                disease_parameters=disease_parameters,
                simulation_parameters=simulation_parameters1
            )

    def test__call__(self):
        model = examplePHEmodel

        # Set regional and time dependent parameters
        regional_parameters = em.PheRegParameters(
            model=model,
            initial_r=[0.5, 1],
            region_index=2,
            betas=[[1]*2, [1]*2],
            times=[1, 2]
        )

        # Set ICs parameters
        ICs = em.PheICs(
            model=model,
            susceptibles_IC=[[5, 6], [7, 8]],
            exposed1_IC=[[10, 2], [3, 0]],
            exposed2_IC=[[5, 9], [8, 8]],
            infectives1_IC=[[1, 1], [1, 0]],
            infectives2_IC=[[5, 5], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.PheDiseaseParameters(
            model=model,
            dL=4,
            dI=4
        )

        # Set other simulation parameters
        simulation_parameters = em.PheSimParameters(
            model=model,
            delta_t=0.5,
            method='my-solver'
        )

        # Set all parameters in the controller
        parameters = em.PheParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            simulation_parameters=simulation_parameters
        )

        self.assertEqual(
            parameters(),
            [0.5, 1, 2, 5, 6, 7, 8, 10, 2, 3, 0, 5, 9, 8, 8,
             1, 1, 1, 0, 5, 5, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
             4, 4, 0.5, 'my-solver']
        )
