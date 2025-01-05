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

#
# Phe Model Examples
#


examplePHEmodel = em.PheSEIRModel()
examplePHEmodel2 = em.PheSEIRModel()

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

        with self.assertRaises(TypeError):
            susceptibles1 = [[5, 6], [7, '8']]

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

        with self.assertRaises(TypeError):
            exposed11 = [['10', 2], [3, 0]]

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

        with self.assertRaises(TypeError):
            exposed21 = [[5, 9], [8, '8']]

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

        with self.assertRaises(TypeError):
            infectives11 = [[1, 1], ['1', 0]]

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

        with self.assertRaises(TypeError):
            infectives21 = [[5, '5'], [0, 0]]

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

        with self.assertRaises(TypeError):
            recovered1 = [[0, 0], ['0', 0]]

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

        with self.assertRaises(ValueError):
            initial_r1 = [[0.5], [1]]

            em.PheRegParameters(
                model=model,
                initial_r=initial_r1,
                region_index=region_index,
                betas=betas,
                times=times
            )

        with self.assertRaises(ValueError):
            initial_r1 = [0.5, 1, 1]

            em.PheRegParameters(
                model=model,
                initial_r=initial_r1,
                region_index=region_index,
                betas=betas,
                times=times
            )

        with self.assertRaises(TypeError):
            initial_r1 = [0.5, '1']

            em.PheRegParameters(
                model=model,
                initial_r=initial_r1,
                region_index=region_index,
                betas=betas,
                times=times
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

#
# Roche Model Examples
#


exampleRochemodel = em.RocheSEIRModel()
exampleRochemodel2 = em.RocheSEIRModel()

# NPIs data
max_levels_npi = [3, 3, 2, 4, 2, 3, 2, 4, 2]
targeted_npi = [True, True, True, True, True, True, True, False, True]
general_npi = [
            [True, False, True, True, False, False, False, False, False],
            [True, False, True, True, True, True, False, False, False]]
time_changes_flag = [1, 12]
reg_levels_npi = [
    [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]],
    [[0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 2, 4, 2, 3, 2, 4, 2]]]
time_changes_npi = [1, 14]

exampleRochemodel.set_regions(regions)
exampleRochemodel.set_age_groups(age_groups)
exampleRochemodel.read_contact_data(matrices_contact, time_changes_contact)
exampleRochemodel.read_regional_data(matrices_region, time_changes_region)
exampleRochemodel.read_npis_data(
    max_levels_npi, targeted_npi, general_npi, reg_levels_npi,
    time_changes_npi, time_changes_flag)


exampleRochemodel2.set_regions(regions)
exampleRochemodel2.set_age_groups(age_groups)
exampleRochemodel2.read_contact_data(matrices_contact2, time_changes_contact2)
exampleRochemodel2.read_regional_data(matrices_region2, time_changes_region2)
exampleRochemodel2.read_npis_data(
    max_levels_npi, targeted_npi, general_npi, reg_levels_npi,
    time_changes_npi, time_changes_flag)

#
# Test RocheICs Class
#


class TestRocheICs(unittest.TestCase):
    """
    Test the 'RocheICs' class.
    """
    def test__init__(self):
        model = exampleRochemodel

        # ICs parameters
        susceptibles = [[1500, 600], [700, 400]]
        exposed = [[0, 0], [0, 0]]
        infectives_pre = [[40, 20], [50, 32]]
        infectives_asym = [[0, 10], [0, 2]]
        infectives_sym = [[10, 20], [20, 32]]
        infectives_pre_ss = [[2, 3], [10, 0]]
        infectives_asym_ss = [[1, 1], [1, 0]]
        infectives_sym_ss = [[4, 5], [1, 2]]
        infectives_q = [[0, 0], [0, 0]]
        recovered = [[0, 0], [0, 0]]
        recovered_asym = [[0, 0], [0, 0]]
        dead = [[0, 0], [0, 0]]

        ICs = em.RocheICs(
            model=model,
            susceptibles_IC=susceptibles,
            exposed_IC=exposed,
            infectives_pre_IC=infectives_pre,
            infectives_asym_IC=infectives_asym,
            infectives_sym_IC=infectives_sym,
            infectives_pre_ss_IC=infectives_pre_ss,
            infectives_asym_ss_IC=infectives_asym_ss,
            infectives_sym_ss_IC=infectives_sym_ss,
            infectives_q_IC=infectives_q,
            recovered_IC=recovered,
            recovered_asym_IC=recovered_asym,
            dead_IC=dead)

        self.assertEqual(ICs.model, model)

        npt.assert_array_equal(np.array(ICs.susceptibles),
                               np.array([[1500, 600], [700, 400]]))

        npt.assert_array_equal(np.array(ICs.exposed),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.infectives_pre),
                               np.array([[40, 20], [50, 32]]))

        npt.assert_array_equal(np.array(ICs.infectives_asym),
                               np.array([[0, 10], [0, 2]]))

        npt.assert_array_equal(np.array(ICs.infectives_sym),
                               np.array([[10, 20], [20, 32]]))

        npt.assert_array_equal(np.array(ICs.infectives_pre_ss),
                               np.array([[2, 3], [10, 0]]))

        npt.assert_array_equal(np.array(ICs.infectives_asym_ss),
                               np.array([[1, 1], [1, 0]]))

        npt.assert_array_equal(np.array(ICs.infectives_sym_ss),
                               np.array([[4, 5], [1, 2]]))

        npt.assert_array_equal(np.array(ICs.infectives_q),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.recovered),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.recovered_asym),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.dead),
                               np.array([[0, 0], [0, 0]]))

        with self.assertRaises(TypeError):
            model1 = 0

            em.RocheICs(
                model=model1,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            susceptibles1 = [1500, 600]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles1,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            susceptibles1 = [[1500, 600], [700, 400], [100, 200]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles1,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            susceptibles1 = [[1500, 600, 700], [400, 100, 200]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles1,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(TypeError):
            susceptibles1 = [[1500, '600'], [700, 400]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles1,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            exposed1 = [0, 0]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed1,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            exposed1 = [[0, 0], [0, 0], [0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed1,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            exposed1 = [[0, 0, 0], [0, 0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed1,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(TypeError):
            exposed1 = [[0, '0'], [0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed1,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_pre1 = [40, 20]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre1,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_pre1 = [[40, 20], [50, 32], [0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre1,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_pre1 = [[40, 20, 50], [32, 0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre1,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(TypeError):
            infectives_pre1 = [[40, '20'], [50, 32]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre1,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_asym1 = [0, 10]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym1,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_asym1 = [[0, 10], [0, 2], [0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym1,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_asym1 = [[0, 10, 0], [2, 0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym1,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(TypeError):
            infectives_asym1 = [[0, '10'], [0, 2]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym1,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_sym1 = [10, 20]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym1,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_sym1 = [[10, 20], [20, 32], [0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym1,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_sym1 = [[10, 20, 20], [32, 0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym1,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(TypeError):
            infectives_sym1 = [[10, 20], [20, '32']]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym1,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_pre_ss1 = [2, 3]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss1,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_pre_ss1 = [[2, 3], [10, 0], [0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss1,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_pre_ss1 = [[2, 3, 10], [0, 0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss1,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(TypeError):
            infectives_pre_ss1 = [['2', 3], [10, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss1,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_asym_ss1 = [1, 1]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss1,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_asym_ss1 = [[1, 1], [1, 0], [0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss1,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_asym_ss1 = [[1, 1, 1], [0, 0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss1,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(TypeError):
            infectives_asym_ss1 = [[1, 1], [1, '0']]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss1,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_sym_ss1 = [4, 5]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss1,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_sym_ss1 = [[4, 5], [1, 2], [0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss1,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_sym_ss1 = [[4, 5, 1], [2, 0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss1,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(TypeError):
            infectives_sym_ss1 = [['4', 5], [1, 2]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss1,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_q1 = [0, 0]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q1,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_q1 = [[0, 0], [0, 0], [0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q1,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            infectives_q1 = [[0, 0, 0], [0, 0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q1,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(TypeError):
            infectives_q1 = [['0', 0], [0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q1,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            recovered1 = [0, 0]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered1,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            recovered1 = [[0, 0], [0, 0], [0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered1,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            recovered1 = [[0, 0, 0], [0, 0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered1,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(TypeError):
            recovered1 = [[0, 0], ['0', 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered1,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            recovered_asym1 = [0, 0]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym1,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            recovered_asym1 = [[0, 0], [0, 0], [0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym1,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            recovered_asym1 = [[0, 0, 0], [0, 0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym1,
                dead_IC=dead)

        with self.assertRaises(TypeError):
            recovered_asym1 = [[0, 0], ['0', 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym1,
                dead_IC=dead)

        with self.assertRaises(ValueError):
            dead1 = [0, 0]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead1)

        with self.assertRaises(ValueError):
            dead1 = [[0, 0], [0, 0], [0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead1)

        with self.assertRaises(ValueError):
            dead1 = [[0, 0, 0], [0, 0, 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead1)

        with self.assertRaises(TypeError):
            dead1 = [[0, 0], ['0', 0]]

            em.RocheICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_IC=exposed,
                infectives_pre_IC=infectives_pre,
                infectives_asym_IC=infectives_asym,
                infectives_sym_IC=infectives_sym,
                infectives_pre_ss_IC=infectives_pre_ss,
                infectives_asym_ss_IC=infectives_asym_ss,
                infectives_sym_ss_IC=infectives_sym_ss,
                infectives_q_IC=infectives_q,
                recovered_IC=recovered,
                recovered_asym_IC=recovered_asym,
                dead_IC=dead1)

    def test__call__(self):
        model = exampleRochemodel

        susceptibles = [[1500, 600], [700, 400]]
        exposed = [[0, 0], [0, 0]]
        infectives_pre = [[40, 20], [50, 32]]
        infectives_asym = [[0, 10], [0, 2]]
        infectives_sym = [[10, 20], [20, 32]]
        infectives_pre_ss = [[2, 3], [10, 0]]
        infectives_asym_ss = [[1, 1], [1, 0]]
        infectives_sym_ss = [[4, 5], [1, 2]]
        infectives_q = [[0, 0], [0, 0]]
        recovered = [[0, 0], [0, 0]]
        recovered_asym = [[0, 0], [0, 0]]
        dead = [[0, 0], [0, 0]]

        ICs = em.RocheICs(
            model=model,
            susceptibles_IC=susceptibles,
            exposed_IC=exposed,
            infectives_pre_IC=infectives_pre,
            infectives_asym_IC=infectives_asym,
            infectives_sym_IC=infectives_sym,
            infectives_pre_ss_IC=infectives_pre_ss,
            infectives_asym_ss_IC=infectives_asym_ss,
            infectives_sym_ss_IC=infectives_sym_ss,
            infectives_q_IC=infectives_q,
            recovered_IC=recovered,
            recovered_asym_IC=recovered_asym,
            dead_IC=dead)

        self.assertEqual(
            ICs(),
            [susceptibles, exposed, infectives_pre, infectives_asym,
             infectives_sym, infectives_pre_ss, infectives_asym_ss,
             infectives_sym_ss, infectives_q, recovered, recovered_asym, dead]
        )

#
# Test RocheCompartmentTimes Class
#


class TestRocheCompartmentTimes(unittest.TestCase):
    """
    Test the 'RocheCompartmentTimes' class.
    """
    def test__init__(self):
        model = exampleRochemodel

        # Average times in compartments
        k = 3.43
        kS = 2.57
        kQ = 1
        kR = 9 * np.ones(len(model.age_groups))
        kRI = 10 * np.ones(len(model.age_groups))

        CompartmentTimes = em.RocheCompartmentTimes(
            model=model,
            k=k,
            kS=kS,
            kQ=kQ,
            kR=kR,
            kRI=kRI
        )

        self.assertEqual(CompartmentTimes.model, model)

        self.assertEqual(CompartmentTimes.k, 3.43)

        self.assertEqual(CompartmentTimes.kS, 2.57)

        self.assertEqual(CompartmentTimes.kQ, 1)

        npt.assert_array_equal(np.array(CompartmentTimes.kR),
                               np.array([9, 9]))

        npt.assert_array_equal(np.array(CompartmentTimes.kRI),
                               np.array([10, 10]))

        # Single value for all ages parsed
        kR = 9
        kRI = 10

        CompartmentTimes = em.RocheCompartmentTimes(
            model=model,
            k=k,
            kS=kS,
            kQ=kQ,
            kR=kR,
            kRI=kRI
        )

        npt.assert_array_equal(np.array(CompartmentTimes.kR),
                               np.array([9, 9]))

        npt.assert_array_equal(np.array(CompartmentTimes.kRI),
                               np.array([10, 10]))

        with self.assertRaises(TypeError):
            model1 = 0

            em.RocheCompartmentTimes(
                model=model1,
                k=k,
                kS=kS,
                kQ=kQ,
                kR=kR,
                kRI=kRI
            )

        with self.assertRaises(TypeError):
            em.RocheCompartmentTimes(
                model=model,
                k='0',
                kS=kS,
                kQ=kQ,
                kR=kR,
                kRI=kRI
            )

        with self.assertRaises(ValueError):
            em.RocheCompartmentTimes(
                model=model,
                k=-1,
                kS=kS,
                kQ=kQ,
                kR=kR,
                kRI=kRI
            )

        with self.assertRaises(TypeError):
            em.RocheCompartmentTimes(
                model=model,
                k=k,
                kS='4',
                kQ=kQ,
                kR=kR,
                kRI=kRI
            )

        with self.assertRaises(ValueError):
            em.RocheCompartmentTimes(
                model=model,
                k=k,
                kS=0,
                kQ=kQ,
                kR=kR,
                kRI=kRI
            )

        with self.assertRaises(TypeError):
            em.RocheCompartmentTimes(
                model=model,
                k=k,
                kS=kS,
                kQ=[4],
                kR=kR,
                kRI=kRI
            )

        with self.assertRaises(ValueError):
            em.RocheCompartmentTimes(
                model=model,
                k=k,
                kS=kS,
                kQ=-1,
                kR=kR,
                kRI=kRI
            )

        with self.assertRaises(ValueError):
            em.RocheCompartmentTimes(
                model=model,
                k=k,
                kS=kS,
                kQ=kQ,
                kR=[[9], [9]],
                kRI=kRI
            )

        with self.assertRaises(ValueError):
            em.RocheCompartmentTimes(
                model=model,
                k=k,
                kS=kS,
                kQ=kQ,
                kR=[9, 9, 9],
                kRI=kRI
            )

        with self.assertRaises(TypeError):
            em.RocheCompartmentTimes(
                model=model,
                k=k,
                kS=kS,
                kQ=kQ,
                kR=['10', 9],
                kRI=kRI
            )

        with self.assertRaises(ValueError):
            em.RocheCompartmentTimes(
                model=model,
                k=k,
                kS=kS,
                kQ=kQ,
                kR=[-1, 3],
                kRI=kRI
            )

        with self.assertRaises(ValueError):
            em.RocheCompartmentTimes(
                model=model,
                k=k,
                kS=kS,
                kQ=kQ,
                kR=kR,
                kRI=[[3], [3]]
            )

        with self.assertRaises(ValueError):
            em.RocheCompartmentTimes(
                model=model,
                k=k,
                kS=kS,
                kQ=kQ,
                kR=kR,
                kRI=[3, 3, 3]
            )

        with self.assertRaises(TypeError):
            em.RocheCompartmentTimes(
                model=model,
                k=k,
                kS=kS,
                kQ=kQ,
                kR=kR,
                kRI=['3', 10]
            )

        with self.assertRaises(ValueError):
            em.RocheCompartmentTimes(
                model=model,
                k=k,
                kS=kS,
                kQ=kQ,
                kR=kR,
                kRI=[3, -1]
            )

    def test__call__(self):
        model = exampleRochemodel

        # Average times in compartments
        k = 3.43
        kS = 2.57
        kQ = 1
        kR = 9 * np.ones(len(model.age_groups))
        kRI = 10 * np.ones(len(model.age_groups))

        CompartmentTimes = em.RocheCompartmentTimes(
            model=model,
            k=k,
            kS=kS,
            kQ=kQ,
            kR=kR,
            kRI=kRI
        )

        self.assertEqual(len(CompartmentTimes()), 5)

        self.assertEqual(
            CompartmentTimes()[:3],
            [3.43, 2.57, 1])

        npt.assert_array_equal(
            np.asarray(CompartmentTimes()[3]), np.array([9, 9]))

        npt.assert_array_equal(
            np.asarray(CompartmentTimes()[4]), np.array([10, 10]))

#
# Test RocheProportions Class
#


class TestRocheProportions(unittest.TestCase):
    """
    Test the 'RocheProportions' class.
    """
    def test__init__(self):
        model = exampleRochemodel

        # Proportion of asymptomatic, super-spreader and dead cases
        Pa = 0.658 * np.ones(len(age_groups))
        Pss = 0.0955
        Pd = 0.05 * np.ones(len(age_groups))

        ProportionParam = em.RocheProportions(
            model=model,
            Pa=Pa,
            Pss=Pss,
            Pd=Pd
        )

        self.assertEqual(ProportionParam.model, model)

        self.assertEqual(ProportionParam.Pss, 0.0955)

        npt.assert_array_equal(np.array(ProportionParam.Pa),
                               np.array([0.658, 0.658]))

        npt.assert_array_equal(np.array(ProportionParam.Pd),
                               np.array([0.05, 0.05]))

        # Single value for all ages parsed
        Pa = 0.658
        Pd = 0.05

        ProportionParam = em.RocheProportions(
            model=model,
            Pa=Pa,
            Pss=Pss,
            Pd=Pd
        )

        npt.assert_array_equal(np.array(ProportionParam.Pa),
                               np.array([0.658, 0.658]))

        npt.assert_array_equal(np.array(ProportionParam.Pd),
                               np.array([0.05, 0.05]))

        with self.assertRaises(TypeError):
            model1 = [0]

            em.RocheProportions(
                model=model1,
                Pa=Pa,
                Pss=Pss,
                Pd=Pd
            )

        with self.assertRaises(ValueError):
            em.RocheProportions(
                model=model,
                Pa=[[0.5], [0.5]],
                Pss=Pss,
                Pd=Pd
            )

        with self.assertRaises(ValueError):
            em.RocheProportions(
                model=model,
                Pa=[0.5, 0.5, 0.5],
                Pss=Pss,
                Pd=Pd
            )

        with self.assertRaises(TypeError):
            em.RocheProportions(
                model=model,
                Pa=['0', 0],
                Pss=Pss,
                Pd=Pd
            )

        with self.assertRaises(ValueError):
            em.RocheProportions(
                model=model,
                Pa=[0.2, -1],
                Pss=Pss,
                Pd=Pd
            )

        with self.assertRaises(ValueError):
            em.RocheProportions(
                model=model,
                Pa=[1.5, 0.5],
                Pss=Pss,
                Pd=Pd
            )

        with self.assertRaises(TypeError):
            em.RocheProportions(
                model=model,
                Pa=Pa,
                Pss='0',
                Pd=Pd
            )

        with self.assertRaises(ValueError):
            em.RocheProportions(
                model=model,
                Pa=Pa,
                Pss=-1,
                Pd=Pd
            )

        with self.assertRaises(ValueError):
            em.RocheProportions(
                model=model,
                Pa=Pa,
                Pss=1.5,
                Pd=Pd
            )

        with self.assertRaises(ValueError):
            em.RocheProportions(
                model=model,
                Pa=Pa,
                Pss=Pss,
                Pd=[[0.5], [0.5]]
            )

        with self.assertRaises(ValueError):
            em.RocheProportions(
                model=model,
                Pa=Pa,
                Pss=Pss,
                Pd=[0.5, 0.5, 0.5]
            )

        with self.assertRaises(TypeError):
            em.RocheProportions(
                model=model,
                Pa=Pa,
                Pss=Pss,
                Pd=['0', '0']
            )

        with self.assertRaises(ValueError):
            em.RocheProportions(
                model=model,
                Pa=Pa,
                Pss=Pss,
                Pd=[0.2, -1]
            )

        with self.assertRaises(ValueError):
            em.RocheProportions(
                model=model,
                Pa=Pa,
                Pss=Pss,
                Pd=[1.5, 0.5]
            )

    def test__call__(self):
        model = exampleRochemodel

        # Proportion of asymptomatic, super-spreader and dead cases
        Pa = 0.658 * np.ones(len(age_groups))
        Pss = 0.0955
        Pd = 0.05 * np.ones(len(age_groups))

        ProportionParam = em.RocheProportions(
            model=model,
            Pa=Pa,
            Pss=Pss,
            Pd=Pd
        )

        self.assertEqual(len(ProportionParam()), 3)

        npt.assert_array_equal(
            np.asarray(ProportionParam()[0]), np.array([0.658, 0.658]))

        self.assertEqual(ProportionParam()[1], 0.0955)

        npt.assert_array_equal(
            np.asarray(ProportionParam()[2]), np.array([0.05, 0.05]))

#
# Test RocheTransmission Class
#


class TestRocheTransmission(unittest.TestCase):
    """
    Test the 'RocheTransmission' class.
    """
    def test__init__(self):
        model = exampleRochemodel

        # Transmission parameters
        beta_min = 0.228
        beta_max = 0.927
        bss = 3.11
        gamma = 1
        s50 = 35.3

        TransmissionParam = em.RocheTransmission(
            model=model,
            beta_min=beta_min,
            beta_max=beta_max,
            bss=bss,
            gamma=gamma,
            s50=s50
        )

        self.assertEqual(TransmissionParam.model, model)
        self.assertEqual(TransmissionParam.beta_min, 0.228)
        self.assertEqual(TransmissionParam.beta_max, 0.927)
        self.assertEqual(TransmissionParam.bss, 3.11)
        self.assertEqual(TransmissionParam.gamma, 1)
        self.assertEqual(TransmissionParam.s50, 35.3)

        with self.assertRaises(TypeError):
            model1 = '0'

            em.RocheTransmission(
                model=model1,
                beta_min=beta_min,
                beta_max=beta_max,
                bss=bss,
                gamma=gamma,
                s50=s50
            )

        with self.assertRaises(TypeError):
            em.RocheTransmission(
                model=model,
                beta_min='0',
                beta_max=beta_max,
                bss=bss,
                gamma=gamma,
                s50=s50
            )

        with self.assertRaises(ValueError):
            em.RocheTransmission(
                model=model,
                beta_min=-1,
                beta_max=beta_max,
                bss=bss,
                gamma=gamma,
                s50=s50
            )

        with self.assertRaises(TypeError):
            em.RocheTransmission(
                model=model,
                beta_min=beta_min,
                beta_max='1',
                bss=bss,
                gamma=gamma,
                s50=s50
            )

        with self.assertRaises(ValueError):
            em.RocheTransmission(
                model=model,
                beta_min=beta_min,
                beta_max=-1,
                bss=bss,
                gamma=gamma,
                s50=s50
            )

        with self.assertRaises(TypeError):
            em.RocheTransmission(
                model=model,
                beta_min=beta_min,
                beta_max=beta_max,
                bss='0',
                gamma=gamma,
                s50=s50
            )

        with self.assertRaises(ValueError):
            em.RocheTransmission(
                model=model,
                beta_min=beta_min,
                beta_max=beta_max,
                bss=-2,
                gamma=gamma,
                s50=s50
            )

        with self.assertRaises(TypeError):
            em.RocheTransmission(
                model=model,
                beta_min=beta_min,
                beta_max=beta_max,
                bss=bss,
                gamma='0',
                s50=s50
            )

        with self.assertRaises(ValueError):
            em.RocheTransmission(
                model=model,
                beta_min=beta_min,
                beta_max=beta_max,
                bss=bss,
                gamma=-1,
                s50=s50
            )

        with self.assertRaises(TypeError):
            em.RocheTransmission(
                model=model,
                beta_min=beta_min,
                beta_max=beta_max,
                bss=bss,
                gamma=gamma,
                s50='10'
            )

        with self.assertRaises(ValueError):
            em.RocheTransmission(
                model=model,
                beta_min=beta_min,
                beta_max=beta_max,
                bss=bss,
                gamma=gamma,
                s50=0
            )

        with self.assertRaises(ValueError):
            em.RocheTransmission(
                model=model,
                beta_min=beta_min,
                beta_max=beta_max,
                bss=bss,
                gamma=gamma,
                s50=150
            )

    def test__call__(self):
        model = exampleRochemodel

        # Transmission parameters
        beta_min = 0.228
        beta_max = 0.927
        bss = 3.11
        gamma = 1
        s50 = 35.3

        TransmissionParam = em.RocheTransmission(
            model=model,
            beta_min=beta_min,
            beta_max=beta_max,
            bss=bss,
            gamma=gamma,
            s50=s50
        )

        self.assertEqual(TransmissionParam(), [0.228, 0.927, 3.11, 1, 35.3])

#
# Test RocheSimParameters Class
#


class TestRocheSimParameters(unittest.TestCase):
    """
    Test the 'RocheSimParameters' class.
    """
    def test__init__(self):
        model = exampleRochemodel

        # Set other simulation parameters
        region_index = 2
        method = 'RK45'
        times = [1, 2]

        SimulationParam = em.RocheSimParameters(
            model=model,
            region_index=region_index,
            method=method,
            times=times
        )

        self.assertEqual(SimulationParam.model, model)
        self.assertEqual(SimulationParam.region_index, 2)
        self.assertEqual(SimulationParam.method, 'RK45')
        self.assertEqual(SimulationParam.times, [1, 2])

        with self.assertRaises(TypeError):
            model1 = {'0': 0}

            em.RocheSimParameters(
                model=model1,
                region_index=region_index,
                method=method,
                times=times
            )

        with self.assertRaises(TypeError):
            em.RocheSimParameters(
                model=model,
                region_index=0.5,
                method=method,
                times=times
            )

        with self.assertRaises(ValueError):
            em.RocheSimParameters(
                model=model,
                region_index=0,
                method=method,
                times=times
            )

        with self.assertRaises(ValueError):
            em.RocheSimParameters(
                model=model,
                region_index=3,
                method=method,
                times=times
            )

        with self.assertRaises(TypeError):
            em.RocheSimParameters(
                model=model,
                region_index=region_index,
                method=3,
                times=times
            )

        with self.assertRaises(ValueError):
            em.RocheSimParameters(
                model=model,
                region_index=region_index,
                method='my-solver',
                times=times
            )

        with self.assertRaises(TypeError):
            em.RocheSimParameters(
                model=model,
                region_index=region_index,
                method=method,
                times='0'
            )

        with self.assertRaises(TypeError):
            em.RocheSimParameters(
                model=model,
                region_index=region_index,
                method=method,
                times=[1, '2']
            )

        with self.assertRaises(ValueError):
            em.RocheSimParameters(
                model=model,
                region_index=region_index,
                method=method,
                times=[0, 1]
            )

    def test__call__(self):
        model = exampleRochemodel

        # Set other simulation parameters
        region_index = 2
        method = 'RK45'
        times = [1, 2]

        SimulationParam = em.RocheSimParameters(
            model=model,
            region_index=region_index,
            method=method,
            times=times
        )

        self.assertEqual(SimulationParam(), [2, 'RK45'])

#
# Test RocheParametersController Class
#


class TestRocheParametersController(unittest.TestCase):
    """
    Test the 'RocheParametersController' class.
    """
    def test__init__(self):
        model = exampleRochemodel

        # Set ICs parameters
        ICs = em.RocheICs(
            model=model,
            susceptibles_IC=[[1500, 600], [700, 400]],
            exposed_IC=[[0, 0], [0, 0]],
            infectives_pre_IC=[[40, 20], [50, 32]],
            infectives_asym_IC=[[0, 10], [0, 2]],
            infectives_sym_IC=[[10, 20], [20, 32]],
            infectives_pre_ss_IC=[[2, 3], [10, 0]],
            infectives_asym_ss_IC=[[1, 1], [1, 0]],
            infectives_sym_ss_IC=[[4, 5], [1, 2]],
            infectives_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]],
            recovered_asym_IC=[[0, 0], [0, 0]],
            dead_IC=[[0, 0], [0, 0]]
        )

        # Set average times in compartments
        compartment_times = em.RocheCompartmentTimes(
            model=model,
            k=3.43,
            kS=2.57,
            kQ=1,
            kR=9 * np.ones(len(model.age_groups)),
            kRI=10
        )

        # Set proportion of asymptomatic, super-spreader and dead cases
        proportion_parameters = em.RocheProportions(
            model=model,
            Pa=0.658 * np.ones(len(age_groups)),
            Pss=0.0955,
            Pd=0.05
        )

        # Set transmission parameters
        transmission_parameters = em.RocheTransmission(
            model=model,
            beta_min=0.228,
            beta_max=0.927,
            bss=3.11,
            gamma=1,
            s50=35.3
        )

        # Set other simulation parameters
        simulation_parameters = em.RocheSimParameters(
            model=model,
            region_index=2,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.RocheParametersController(
            model=model,
            ICs=ICs,
            compartment_times=compartment_times,
            proportion_parameters=proportion_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        self.assertEqual(parameters.model, model)
        self.assertEqual(parameters.ICs, ICs)

        self.assertEqual(parameters.compartment_times, compartment_times)
        self.assertEqual(parameters.proportion_parameters,
                         proportion_parameters)

        self.assertEqual(parameters.transmission_parameters,
                         transmission_parameters)
        self.assertEqual(parameters.simulation_parameters,
                         simulation_parameters)

        with self.assertRaises(TypeError):
            model1 = 0.3

            em.RocheParametersController(
                model=model1,
                ICs=ICs,
                compartment_times=compartment_times,
                proportion_parameters=proportion_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(TypeError):
            ICs1 = '0'

            em.RocheParametersController(
                model=model,
                ICs=ICs1,
                compartment_times=compartment_times,
                proportion_parameters=proportion_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(ValueError):
            ICs1 = em.RocheICs(
                model=exampleRochemodel2,
                susceptibles_IC=[[1500, 600], [700, 400]],
                exposed_IC=[[0, 0], [0, 0]],
                infectives_pre_IC=[[40, 20], [50, 32]],
                infectives_asym_IC=[[0, 10], [0, 2]],
                infectives_sym_IC=[[10, 20], [20, 32]],
                infectives_pre_ss_IC=[[2, 3], [10, 0]],
                infectives_asym_ss_IC=[[1, 1], [1, 0]],
                infectives_sym_ss_IC=[[4, 5], [1, 2]],
                infectives_q_IC=[[0, 0], [0, 0]],
                recovered_IC=[[0, 0], [0, 0]],
                recovered_asym_IC=[[0, 0], [0, 0]],
                dead_IC=[[0, 0], [0, 0]]
            )

            em.RocheParametersController(
                model=model,
                ICs=ICs1,
                compartment_times=compartment_times,
                proportion_parameters=proportion_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(TypeError):
            compartment_times1 = 0

            em.RocheParametersController(
                model=model,
                ICs=ICs,
                compartment_times=compartment_times1,
                proportion_parameters=proportion_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(ValueError):
            compartment_times1 = em.RocheCompartmentTimes(
                model=exampleRochemodel2,
                k=3.43,
                kS=2.57,
                kQ=1,
                kR=9 * np.ones(len(model.age_groups)),
                kRI=10
            )

            em.RocheParametersController(
                model=model,
                ICs=ICs,
                compartment_times=compartment_times1,
                proportion_parameters=proportion_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(TypeError):
            proportion_parameters1 = '0'

            em.RocheParametersController(
                model=model,
                ICs=ICs,
                compartment_times=compartment_times,
                proportion_parameters=proportion_parameters1,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(ValueError):
            proportion_parameters1 = em.RocheProportions(
                model=exampleRochemodel2,
                Pa=0.658 * np.ones(len(age_groups)),
                Pss=0.0955,
                Pd=0.05
            )

            em.RocheParametersController(
                model=model,
                ICs=ICs,
                compartment_times=compartment_times,
                proportion_parameters=proportion_parameters1,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(TypeError):
            transmission_parameters1 = [0]

            em.RocheParametersController(
                model=model,
                ICs=ICs,
                compartment_times=compartment_times,
                proportion_parameters=proportion_parameters,
                transmission_parameters=transmission_parameters1,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(ValueError):
            transmission_parameters1 = em.RocheTransmission(
                model=exampleRochemodel2,
                beta_min=0.228,
                beta_max=0.927,
                bss=3.11,
                gamma=1,
                s50=35.3
            )

            em.RocheParametersController(
                model=model,
                ICs=ICs,
                compartment_times=compartment_times,
                proportion_parameters=proportion_parameters,
                transmission_parameters=transmission_parameters1,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(TypeError):
            simulation_parameters1 = {'0': 0}

            em.RocheParametersController(
                model=model,
                ICs=ICs,
                compartment_times=compartment_times,
                proportion_parameters=proportion_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters1
            )

        with self.assertRaises(ValueError):
            simulation_parameters1 = em.RocheSimParameters(
                model=exampleRochemodel2,
                region_index=2,
                method='Radau',
                times=[1, 2]
            )

            em.RocheParametersController(
                model=model,
                ICs=ICs,
                compartment_times=compartment_times,
                proportion_parameters=proportion_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters1
            )

    def test__call__(self):
        model = exampleRochemodel

        # Set ICs parameters
        ICs = em.RocheICs(
            model=model,
            susceptibles_IC=[[1500, 600], [700, 400]],
            exposed_IC=[[0, 0], [0, 0]],
            infectives_pre_IC=[[40, 20], [50, 32]],
            infectives_asym_IC=[[0, 10], [0, 2]],
            infectives_sym_IC=[[10, 20], [20, 32]],
            infectives_pre_ss_IC=[[2, 3], [10, 0]],
            infectives_asym_ss_IC=[[1, 1], [1, 0]],
            infectives_sym_ss_IC=[[4, 5], [1, 2]],
            infectives_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]],
            recovered_asym_IC=[[0, 0], [0, 0]],
            dead_IC=[[0, 0], [0, 0]]
        )

        # Set average times in compartments
        compartment_times = em.RocheCompartmentTimes(
            model=model,
            k=3.43,
            kS=2.57,
            kQ=1,
            kR=9 * np.ones(len(model.age_groups)),
            kRI=10
        )

        # Set proportion of asymptomatic, super-spreader and dead cases
        proportion_parameters = em.RocheProportions(
            model=model,
            Pa=0.658 * np.ones(len(age_groups)),
            Pss=0.0955,
            Pd=0.05
        )

        # Set transmission parameters
        transmission_parameters = em.RocheTransmission(
            model=model,
            beta_min=0.228,
            beta_max=0.927,
            bss=3.11,
            gamma=1,
            s50=35.3
        )

        # Set other simulation parameters
        simulation_parameters = em.RocheSimParameters(
            model=model,
            region_index=2,
            method='RK45',
            times=[1, 2]
        )

        # Set all parameters in the controller
        parameters = em.RocheParametersController(
            model=model,
            ICs=ICs,
            compartment_times=compartment_times,
            proportion_parameters=proportion_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        self.assertEqual(
            parameters(),
            [2, 1500, 600, 700, 400, 0, 0, 0, 0, 40, 20, 50, 32, 0,
             10, 0, 2, 10, 20, 20, 32, 2, 3, 10, 0, 1, 1, 1, 0, 4, 5,
             1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.43,
             2.57, 1, 9, 9, 10, 10, 0.658, 0.658, 0.0955, 0.05, 0.05, 0.228,
             0.927, 3.11, 1, 35.3, 'RK45']
        )

#
# Warwick Model Examples
#


exampleWarwickmodel = em.WarwickSEIRModel()
exampleWarwickmodel2 = em.WarwickSEIRModel()

# Initial state of the system
house_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
house_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

school_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
school_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

work_contact_data_matrix_0 = 0.3 * np.array([[10, 5.2], [0, 3]])
work_contact_data_matrix_1 = 0.3 * np.array([[1, 0], [0, 3]])

other_contact_data_matrix_0 = 0.2 * np.array([[10, 5.2], [0, 3]])
other_contact_data_matrix_1 = 0.2 * np.array([[1, 0], [0, 3]])

house_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
house_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
house_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
house_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2], [0.29, 4.6]])

school_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
school_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
school_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
school_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

work_region_data_matrix_0_0 = 0.3 * np.array([[1, 10], [1, 6]])
work_region_data_matrix_0_1 = 0.3 * np.array([[0.5, 3], [0.3, 3]])
work_region_data_matrix_1_0 = 0.3 * np.array([[0.85, 1], [0.9, 6]])
work_region_data_matrix_1_1 = 0.3 * np.array([[0.5, 0.2], [0.29, 4.6]])

other_region_data_matrix_0_0 = 0.2 * np.array([[1, 10], [1, 6]])
other_region_data_matrix_0_1 = 0.2 * np.array([[0.5, 3], [0.3, 3]])
other_region_data_matrix_1_0 = 0.2 * np.array([[0.85, 1], [0.9, 6]])
other_region_data_matrix_1_1 = 0.2 * np.array([[0.5, 0.2], [0.29, 4.6]])


house_contacts_0 = em.ContactMatrix(age_groups, house_contact_data_matrix_0)
house_contacts_1 = em.ContactMatrix(age_groups, house_contact_data_matrix_1)
house_regional_0_0 = em.RegionMatrix(
    regions[0], age_groups, house_region_data_matrix_0_0)
house_regional_0_1 = em.RegionMatrix(
    regions[1], age_groups, house_region_data_matrix_0_1)
house_regional_1_0 = em.RegionMatrix(
    regions[0], age_groups, house_region_data_matrix_1_0)
house_regional_1_1 = em.RegionMatrix(
    regions[1], age_groups, house_region_data_matrix_1_1)

school_contacts_0 = em.ContactMatrix(
    age_groups, school_contact_data_matrix_0)
school_contacts_1 = em.ContactMatrix(
    age_groups, school_contact_data_matrix_1)
school_regional_0_0 = em.RegionMatrix(
    regions[0], age_groups, school_region_data_matrix_0_0)
school_regional_0_1 = em.RegionMatrix(
    regions[1], age_groups, school_region_data_matrix_0_1)
school_regional_1_0 = em.RegionMatrix(
    regions[0], age_groups, school_region_data_matrix_1_0)
school_regional_1_1 = em.RegionMatrix(
    regions[1], age_groups, school_region_data_matrix_1_1)

work_contacts_0 = em.ContactMatrix(
    age_groups, work_contact_data_matrix_0)
work_contacts_1 = em.ContactMatrix(
    age_groups, work_contact_data_matrix_1)
work_regional_0_0 = em.RegionMatrix(
    regions[0], age_groups, work_region_data_matrix_0_0)
work_regional_0_1 = em.RegionMatrix(
    regions[1], age_groups, work_region_data_matrix_0_1)
work_regional_1_0 = em.RegionMatrix(
    regions[0], age_groups, work_region_data_matrix_1_0)
work_regional_1_1 = em.RegionMatrix(
    regions[1], age_groups, work_region_data_matrix_1_1)

other_contacts_0 = em.ContactMatrix(
    age_groups, other_contact_data_matrix_0)
other_contacts_1 = em.ContactMatrix(
    age_groups, other_contact_data_matrix_1)
other_regional_0_0 = em.RegionMatrix(
    regions[0], age_groups, other_region_data_matrix_0_0)
other_regional_0_1 = em.RegionMatrix(
    regions[1], age_groups, other_region_data_matrix_0_1)
other_regional_1_0 = em.RegionMatrix(
    regions[0], age_groups, other_region_data_matrix_1_0)
other_regional_1_1 = em.RegionMatrix(
    regions[1], age_groups, other_region_data_matrix_1_1)


# Matrices contact
house_matrices_contact = [house_contacts_0, house_contacts_1]
school_matrices_contact = [school_contacts_0, school_contacts_1]
work_matrices_contact = [work_contacts_0, work_contacts_1]
other_matrices_contact = [other_contacts_0, other_contacts_1]
time_changes_contact = [1, 2]

house_matrices_region = [
    [house_regional_0_0, house_regional_0_1],
    [house_regional_1_0, house_regional_1_1]]
school_matrices_region = [
    [school_regional_0_0, school_regional_0_1],
    [school_regional_1_0, school_regional_1_1]]
work_matrices_region = [
    [work_regional_0_0, work_regional_0_1],
    [work_regional_1_0, work_regional_1_1]]
other_matrices_region = [
    [other_regional_0_0, other_regional_0_1],
    [other_regional_1_0, other_regional_1_1]]
time_changes_region = [1, 3]

house_matrices_contact2 = [house_contacts_0, house_contacts_1]
school_matrices_contact2 = [school_contacts_0, school_contacts_1]
work_matrices_contact2 = [work_contacts_0, work_contacts_1]
other_matrices_contact2 = [other_contacts_0, other_contacts_1]
time_changes_contact2 = [1, 5]

house_matrices_region2 = [
    [house_regional_0_0, house_regional_0_1],
    [house_regional_1_0, house_regional_1_1]]
house_matrices_region2 = [
    [house_regional_0_0, house_regional_0_1],
    [house_regional_1_0, house_regional_1_1]]
school_matrices_region2 = [
    [school_regional_0_0, school_regional_0_1],
    [school_regional_1_0, school_regional_1_1]]
work_matrices_region2 = [
    [work_regional_0_0, work_regional_0_1],
    [work_regional_1_0, work_regional_1_1]]
other_matrices_region2 = [
    [other_regional_0_0, other_regional_0_1],
    [other_regional_1_0, other_regional_1_1]]
time_changes_region = [1, 3]
time_changes_region2 = [1, 10]

exampleWarwickmodel.set_regions(regions)
exampleWarwickmodel.set_age_groups(age_groups)
exampleWarwickmodel.read_contact_data(
    house_matrices_contact, school_matrices_contact, work_matrices_contact,
    other_matrices_contact, time_changes_contact)
exampleWarwickmodel.read_regional_data(
    house_matrices_region, school_matrices_region, work_matrices_region,
    other_matrices_region, time_changes_region)

exampleWarwickmodel2.set_regions(regions)
exampleWarwickmodel2.set_age_groups(age_groups)
exampleWarwickmodel2.read_contact_data(
    house_matrices_contact2, school_matrices_contact2, work_matrices_contact2,
    other_matrices_contact2, time_changes_contact2)
exampleWarwickmodel2.read_regional_data(
    house_matrices_region2, school_matrices_region2, work_matrices_region2,
    other_matrices_region2, time_changes_region2)

#
# Test WarwickICs Class
#


class TestWarwickICs(unittest.TestCase):
    """
    Test the 'WarwickICs' class.
    """
    def test__init__(self):
        model = exampleWarwickmodel

        # ICs parameters
        susceptibles = [[1500, 600], [700, 400]]
        exposed_1_f = [[0, 0], [0, 0]]
        exposed_1_sd = [[0, 0], [0, 0]]
        exposed_1_su = [[0, 0], [0, 0]]
        exposed_1_q = [[0, 0], [0, 0]]
        exposed_2_f = [[0, 0], [0, 0]]
        exposed_2_sd = [[0, 0], [0, 0]]
        exposed_2_su = [[0, 0], [0, 0]]
        exposed_2_q = [[0, 0], [0, 0]]
        exposed_3_f = [[0, 0], [0, 0]]
        exposed_3_sd = [[0, 0], [0, 0]]
        exposed_3_su = [[0, 0], [0, 0]]
        exposed_3_q = [[0, 0], [0, 0]]
        detected_f = [[0, 10], [2, 4]]
        detected_qf = [[0, 0], [0, 0]]
        detected_sd = [[10, 20], [0, 5]]
        detected_su = [[20, 20], [15, 35]]
        detected_qs = [[0, 0], [0, 0]]
        undetected_f = [[7, 13], [14, 2]]
        undetected_s = [[0, 10], [0, 0]]
        undetected_q = [[0, 0], [0, 0]]
        recovered = [[0, 0], [0, 0]]

        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=susceptibles,
            exposed_1_f_IC=exposed_1_f,
            exposed_1_sd_IC=exposed_1_sd,
            exposed_1_su_IC=exposed_1_su,
            exposed_1_q_IC=exposed_1_q,
            exposed_2_f_IC=exposed_2_f,
            exposed_2_sd_IC=exposed_2_sd,
            exposed_2_su_IC=exposed_2_su,
            exposed_2_q_IC=exposed_2_q,
            exposed_3_f_IC=exposed_3_f,
            exposed_3_sd_IC=exposed_3_sd,
            exposed_3_su_IC=exposed_3_su,
            exposed_3_q_IC=exposed_3_q,
            detected_f_IC=detected_f,
            detected_qf_IC=detected_qf,
            detected_sd_IC=detected_sd,
            detected_su_IC=detected_su,
            detected_qs_IC=detected_qs,
            undetected_f_IC=undetected_f,
            undetected_s_IC=undetected_s,
            undetected_q_IC=undetected_q,
            recovered_IC=recovered
        )

        self.assertEqual(ICs.model, model)

        npt.assert_array_equal(np.array(ICs.susceptibles),
                               np.array([[1500, 600], [700, 400]]))

        npt.assert_array_equal(np.array(ICs.exposed_1_f),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.exposed_1_sd),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.exposed_1_su),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.exposed_1_q),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.exposed_2_f),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.exposed_2_sd),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.exposed_2_su),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.exposed_2_q),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.exposed_3_f),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.exposed_3_sd),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.exposed_3_su),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.exposed_3_q),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.detected_f),
                               np.array([[0, 10], [2, 4]]))

        npt.assert_array_equal(np.array(ICs.detected_qf),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.detected_sd),
                               np.array([[10, 20], [0, 5]]))

        npt.assert_array_equal(np.array(ICs.detected_su),
                               np.array([[20, 20], [15, 35]]))

        npt.assert_array_equal(np.array(ICs.detected_qs),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.undetected_f),
                               np.array([[7, 13], [14, 2]]))

        npt.assert_array_equal(np.array(ICs.undetected_s),
                               np.array([[0, 10], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.undetected_q),
                               np.array([[0, 0], [0, 0]]))

        npt.assert_array_equal(np.array(ICs.recovered),
                               np.array([[0, 0], [0, 0]]))

        with self.assertRaises(TypeError):
            model1 = 0

            em.WarwickICs(
                model=model1,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            susceptibles1 = [1500, 600]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles1,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            susceptibles1 = [[1500, 600], [700, 400], [100, 200]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles1,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            susceptibles1 = [[1500, 600, 700], [400, 100, 200]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles1,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            susceptibles1 = [[1500, '600'], [700, 400]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles1,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_1_f1 = [0, 0]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f1,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_1_f1 = [[0, 0], [0, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f1,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_1_f1 = [[0, 0, 0], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f1,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed_1_f1 = [[0, '0'], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f1,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_1_sd1 = [0, 0]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd1,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_1_sd1 = [[0, 0], [0, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd1,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_1_sd1 = [[0, 0, 0], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd1,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed_1_sd1 = [[0, '0'], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd1,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_1_su1 = [0, 0]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su1,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_1_su1 = [[0, 0], [0, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su1,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_1_su1 = [[0, 0, 0], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su1,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed_1_su1 = [[0, '0'], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su1,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_1_q1 = [0, 0]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q1,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_1_q1 = [[0, 0], [0, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q1,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_1_q1 = [[0, 0, 0], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q1,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed_1_q1 = [[0, '0'], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q1,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_2_f1 = [0, 0]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f1,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_2_f1 = [[0, 0], [0, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f1,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_2_f1 = [[0, 0, 0], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f1,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed_2_f1 = [[0, '0'], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f1,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_2_sd1 = [0, 0]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd1,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_2_sd1 = [[0, 0], [0, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd1,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_2_sd1 = [[0, 0, 0], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd1,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed_2_sd1 = [[0, '0'], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd1,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_2_su1 = [0, 0]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su1,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_2_su1 = [[0, 0], [0, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su1,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_2_su1 = [[0, 0, 0], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su1,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed_2_su1 = [[0, '0'], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su1,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_2_q1 = [0, 0]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q1,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_2_q1 = [[0, 0], [0, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q1,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_2_q1 = [[0, 0, 0], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q1,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed_2_q1 = [[0, '0'], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q1,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_3_f1 = [0, 0]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f1,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_3_f1 = [[0, 0], [0, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f1,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_3_f1 = [[0, 0, 0], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f1,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed_3_f1 = [[0, '0'], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f1,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_3_sd1 = [0, 0]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd1,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_3_sd1 = [[0, 0], [0, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd1,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_3_sd1 = [[0, 0, 0], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd1,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed_3_sd1 = [[0, '0'], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd1,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_3_su1 = [0, 0]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su1,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_3_su1 = [[0, 0], [0, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su1,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_3_su1 = [[0, 0, 0], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su1,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed_3_su1 = [[0, '0'], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su1,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_3_q1 = [0, 0]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q1,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_3_q1 = [[0, 0], [0, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q1,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed_3_q1 = [[0, 0, 0], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q1,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed_3_q1 = [[0, '0'], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q1,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            detected_f1 = [40, 20]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f1,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            detected_f1 = [[40, 20], [50, 32], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f1,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            detected_f1 = [[40, 20, 50], [32, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f1,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            detected_f1 = [[40, '20'], [50, 32]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f1,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            detected_qf1 = [0, 10]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf1,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            detected_qf1 = [[0, 10], [0, 2], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf1,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            detected_qf1 = [[0, 10, 0], [2, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf1,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            detected_qf1 = [[0, '10'], [0, 2]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf1,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            detected_sd1 = [10, 20]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd1,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            detected_sd1 = [[10, 20], [20, 32], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd1,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            detected_sd1 = [[10, 20, 20], [32, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd1,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            detected_sd1 = [[10, 20], [20, '32']]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd1,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            detected_su1 = [2, 3]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su1,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            detected_su1 = [[2, 3], [10, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su1,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            detected_su1 = [[2, 3, 10], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su1,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            detected_su1 = [['2', 3], [10, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su1,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            detected_qs1 = [1, 1]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs1,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            detected_qs1 = [[1, 1], [1, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs1,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            detected_qs1 = [[1, 1, 1], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs1,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            detected_qs1 = [[1, 1], [1, '0']]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs1,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            undetected_f1 = [4, 5]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f1,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            undetected_f1 = [[4, 5], [1, 2], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f1,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            undetected_f1 = [[4, 5, 1], [2, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f1,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            undetected_f1 = [['4', 5], [1, 2]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f1,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            undetected_s1 = [4, 5]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s1,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            undetected_s1 = [[4, 5], [1, 2], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s1,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            undetected_s1 = [[4, 5, 1], [2, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s1,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            undetected_s1 = [['4', 5], [1, 2]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s1,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            undetected_q1 = [0, 0]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q1,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            undetected_q1 = [[0, 0], [0, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q1,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            undetected_q1 = [[0, 0, 0], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q1,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            undetected_q1 = [['0', 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q1,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            recovered1 = [0, 0]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered1)

        with self.assertRaises(ValueError):
            recovered1 = [[0, 0], [0, 0], [0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered1)

        with self.assertRaises(ValueError):
            recovered1 = [[0, 0, 0], [0, 0, 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered1)

        with self.assertRaises(TypeError):
            recovered1 = [[0, 0], ['0', 0]]

            em.WarwickICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed_1_f_IC=exposed_1_f,
                exposed_1_sd_IC=exposed_1_sd,
                exposed_1_su_IC=exposed_1_su,
                exposed_1_q_IC=exposed_1_q,
                exposed_2_f_IC=exposed_2_f,
                exposed_2_sd_IC=exposed_2_sd,
                exposed_2_su_IC=exposed_2_su,
                exposed_2_q_IC=exposed_2_q,
                exposed_3_f_IC=exposed_3_f,
                exposed_3_sd_IC=exposed_3_sd,
                exposed_3_su_IC=exposed_3_su,
                exposed_3_q_IC=exposed_3_q,
                detected_f_IC=detected_f,
                detected_qf_IC=detected_qf,
                detected_sd_IC=detected_sd,
                detected_su_IC=detected_su,
                detected_qs_IC=detected_qs,
                undetected_f_IC=undetected_f,
                undetected_s_IC=undetected_s,
                undetected_q_IC=undetected_q,
                recovered_IC=recovered1)

    def test__call__(self):
        model = exampleWarwickmodel

        # ICs parameters
        susceptibles = [[1500, 600], [700, 400]]
        exposed_1_f = [[0, 0], [0, 0]]
        exposed_1_sd = [[0, 0], [0, 0]]
        exposed_1_su = [[0, 0], [0, 0]]
        exposed_1_q = [[0, 0], [0, 0]]
        exposed_2_f = [[0, 0], [0, 0]]
        exposed_2_sd = [[0, 0], [0, 0]]
        exposed_2_su = [[0, 0], [0, 0]]
        exposed_2_q = [[0, 0], [0, 0]]
        exposed_3_f = [[0, 0], [0, 0]]
        exposed_3_sd = [[0, 0], [0, 0]]
        exposed_3_su = [[0, 0], [0, 0]]
        exposed_3_q = [[0, 0], [0, 0]]
        detected_f = [[0, 10], [2, 4]]
        detected_qf = [[0, 0], [0, 0]]
        detected_sd = [[10, 20], [0, 5]]
        detected_su = [[20, 20], [15, 35]]
        detected_qs = [[0, 0], [0, 0]]
        undetected_f = [[7, 13], [14, 2]]
        undetected_s = [[0, 10], [0, 0]]
        undetected_q = [[0, 0], [0, 0]]
        recovered = [[0, 0], [0, 0]]

        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=susceptibles,
            exposed_1_f_IC=exposed_1_f,
            exposed_1_sd_IC=exposed_1_sd,
            exposed_1_su_IC=exposed_1_su,
            exposed_1_q_IC=exposed_1_q,
            exposed_2_f_IC=exposed_2_f,
            exposed_2_sd_IC=exposed_2_sd,
            exposed_2_su_IC=exposed_2_su,
            exposed_2_q_IC=exposed_2_q,
            exposed_3_f_IC=exposed_3_f,
            exposed_3_sd_IC=exposed_3_sd,
            exposed_3_su_IC=exposed_3_su,
            exposed_3_q_IC=exposed_3_q,
            detected_f_IC=detected_f,
            detected_qf_IC=detected_qf,
            detected_sd_IC=detected_sd,
            detected_su_IC=detected_su,
            detected_qs_IC=detected_qs,
            undetected_f_IC=undetected_f,
            undetected_s_IC=undetected_s,
            undetected_q_IC=undetected_q,
            recovered_IC=recovered
        )

        self.assertEqual(
            ICs(),
            [susceptibles, exposed_1_f, exposed_1_sd, exposed_1_su,
             exposed_1_q, exposed_2_f, exposed_2_sd, exposed_2_su,
             exposed_2_q, exposed_3_f, exposed_3_sd, exposed_3_su,
             exposed_3_q, detected_f, detected_qf, detected_sd, detected_su,
             detected_qs, undetected_f, undetected_s, undetected_q, recovered]
        )

#
# Test WarwickRegParameters Class
#


class TestWarwickRegParameters(unittest.TestCase):
    """
    Test the 'WarwickRegParameters' class.
    """
    def test__init__(self):
        model = exampleWarwickmodel

        region_index = 2
        H_all = [0.8, 0.7]

        RegParameters = em.WarwickRegParameters(
            model=model,
            region_index=region_index,
            H=H_all
        )

        self.assertEqual(RegParameters.model, model)

        self.assertEqual(RegParameters.region_index, 2)

        npt.assert_array_equal(np.array(RegParameters.H),
                               np.array([0.8, 0.7]))

        with self.assertRaises(TypeError):
            model1 = '0'

            em.WarwickRegParameters(
                model=model1,
                region_index=region_index,
                H=H_all
            )

        with self.assertRaises(TypeError):
            H_all1 = '0'

            em.WarwickRegParameters(
                model=model1,
                region_index=region_index,
                H=H_all1
            )

        with self.assertRaises(ValueError):
            H_all1 = [0.3, 0.4, 0.5]

            em.WarwickRegParameters(
                model=model,
                region_index=region_index,
                H=H_all1
            )

        with self.assertRaises(TypeError):
            H_all1 = ['0.1', 0.2]

            em.WarwickRegParameters(
                model=model,
                region_index=region_index,
                H=H_all1
            )

        with self.assertRaises(ValueError):
            H_all1 = [-0.3, 0.1]

            em.WarwickRegParameters(
                model=model,
                region_index=region_index,
                H=H_all1
            )

        with self.assertRaises(ValueError):
            H_all1 = [0.9, 1.1]

            em.WarwickRegParameters(
                model=model,
                region_index=region_index,
                H=H_all1
            )

        with self.assertRaises(TypeError):
            region_index1 = 0.5

            em.WarwickRegParameters(
                model=model,
                region_index=region_index1,
                H=H_all
            )

        with self.assertRaises(ValueError):
            region_index1 = 0

            em.WarwickRegParameters(
                model=model,
                region_index=region_index1,
                H=H_all
            )

        with self.assertRaises(ValueError):
            region_index1 = 3

            em.WarwickRegParameters(
                model=model,
                region_index=region_index1,
                H=H_all
            )

    def test__call__(self):
        model = exampleWarwickmodel

        region_index = 2
        H_all = [0.8, 0.7]

        RegParameters = em.WarwickRegParameters(
            model=model,
            region_index=region_index,
            H=H_all
        )

        self.assertEqual(RegParameters(), [region_index, H_all])

#
# Test WarwickDiseaseParameters Class
#


class TestWarwickDiseaseParameters(unittest.TestCase):
    """
    Test the 'WarwickDiseaseParameters' class.
    """
    def test__init__(self):
        model = exampleWarwickmodel

        tau = 0.4
        d = [0.05, 0.02]

        DiseaseParameters = em.WarwickDiseaseParameters(
            model=model,
            tau=tau,
            d=d
        )

        self.assertEqual(DiseaseParameters.model, model)
        self.assertEqual(DiseaseParameters.tau, 0.4)
        npt.assert_array_equal(DiseaseParameters.d, np.array([0.05, 0.02]))

        with self.assertRaises(TypeError):
            model1 = [4]

            em.WarwickDiseaseParameters(
                model=model1,
                tau=tau,
                d=d
            )

        with self.assertRaises(TypeError):
            tau1 = '4'

            em.WarwickDiseaseParameters(
                model=model,
                tau=tau1,
                d=d
            )

        with self.assertRaises(ValueError):
            tau1 = -1

            em.WarwickDiseaseParameters(
                model=model,
                tau=tau1,
                d=d
            )

        with self.assertRaises(ValueError):
            tau1 = 1.2

            em.WarwickDiseaseParameters(
                model=model,
                tau=tau1,
                d=d
            )

        with self.assertRaises(TypeError):
            d1 = ['0.4', 0.2]

            em.WarwickDiseaseParameters(
                model=model,
                tau=tau,
                d=d1
            )

        with self.assertRaises(ValueError):
            d1 = [[0.2], [0.4]]

            em.WarwickDiseaseParameters(
                model=model,
                tau=tau,
                d=d1
            )

        with self.assertRaises(ValueError):
            d1 = [0.2, 0.4, 0.7]

            em.WarwickDiseaseParameters(
                model=model,
                tau=tau,
                d=d1
            )

        with self.assertRaises(ValueError):
            d1 = -0.2

            em.WarwickDiseaseParameters(
                model=model,
                tau=tau,
                d=d1
            )

        with self.assertRaises(ValueError):
            d1 = [0.2, 1.3]

            em.WarwickDiseaseParameters(
                model=model,
                tau=tau,
                d=d1
            )

    def test__call__(self):
        model = exampleWarwickmodel

        tau = 0.4
        d = [0.05, 0.02]

        DiseaseParameters = em.WarwickDiseaseParameters(
            model=model,
            tau=tau,
            d=d
        )

        self.assertEqual(DiseaseParameters(), [0.4, [0.05, 0.02]])
#
# Test WarwickTransmission Class
#


class TestWarwickTransmission(unittest.TestCase):
    """
    Test the 'WarwickTransmission' class.
    """
    def test__init__(self):
        model = exampleWarwickmodel

        # Transmission parameters
        epsilon = 0.5
        gamma = 1
        sigma = 0.5

        TransmissionParam = em.WarwickTransmission(
            model=model,
            epsilon=epsilon,
            gamma=gamma,
            sigma=sigma
        )

        self.assertEqual(TransmissionParam.model, model)
        self.assertEqual(TransmissionParam.epsilon, 0.5)
        self.assertEqual(TransmissionParam.gamma, 1)
        npt.assert_array_equal(TransmissionParam.sigma, np.array([0.5, 0.5]))

        with self.assertRaises(TypeError):
            model1 = '0'

            em.WarwickTransmission(
                model=model1,
                epsilon=epsilon,
                gamma=gamma,
                sigma=sigma
            )

        with self.assertRaises(TypeError):
            em.WarwickTransmission(
                model=model,
                epsilon='0.5',
                gamma=gamma,
                sigma=sigma
            )

        with self.assertRaises(ValueError):
            em.WarwickTransmission(
                model=model,
                epsilon=-1,
                gamma=gamma,
                sigma=sigma
            )

        with self.assertRaises(TypeError):
            em.WarwickTransmission(
                model=model,
                epsilon=epsilon,
                gamma='0.2',
                sigma=sigma
            )

        with self.assertRaises(ValueError):
            em.WarwickTransmission(
                model=model,
                epsilon=epsilon,
                gamma=-1,
                sigma=sigma
            )

        with self.assertRaises(ValueError):
            em.WarwickTransmission(
                model=model,
                epsilon=epsilon,
                gamma=gamma,
                sigma={'0.9': 0}
            )

        with self.assertRaises(ValueError):
            em.WarwickTransmission(
                model=model,
                epsilon=epsilon,
                gamma=gamma,
                sigma=[[0.4], [2.3]]
            )

        with self.assertRaises(ValueError):
            em.WarwickTransmission(
                model=model,
                epsilon=epsilon,
                gamma=gamma,
                sigma=[0.5, 0.9, 2]
            )

        with self.assertRaises(TypeError):
            em.WarwickTransmission(
                model=model,
                epsilon=epsilon,
                gamma=gamma,
                sigma=[0.5, '2']
            )

        with self.assertRaises(ValueError):
            em.WarwickTransmission(
                model=model,
                epsilon=epsilon,
                gamma=gamma,
                sigma=[-3, 2]
            )

    def test__call__(self):
        model = exampleWarwickmodel

        # Transmission parameters
        epsilon = 0.5
        gamma = 1
        sigma = [0.5, 0.5]

        TransmissionParam = em.WarwickTransmission(
            model=model,
            epsilon=epsilon,
            gamma=gamma,
            sigma=sigma
        )

        self.assertEqual(TransmissionParam(), [0.5, 1, [0.5, 0.5]])

#
# Test WarwickSimParameters Class
#


class TestWarwickSimParameters(unittest.TestCase):
    """
    Test the 'WarwickSimParameters' class.
    """
    def test__init__(self):
        model = exampleWarwickmodel

        method = 'RK45'
        times = [1, 2]

        SimulationParam = em.WarwickSimParameters(
            model=model,
            method=method,
            times=times
        )

        self.assertEqual(SimulationParam.model, model)
        self.assertEqual(SimulationParam.method, 'RK45')
        self.assertEqual(SimulationParam.times, [1, 2])

        with self.assertRaises(TypeError):
            model1 = {'0': 0}

            em.WarwickSimParameters(
                model=model1,
                method=method,
                times=times
            )

        with self.assertRaises(TypeError):
            em.WarwickSimParameters(
                model=model,
                method=3,
                times=times
            )

        with self.assertRaises(ValueError):
            em.WarwickSimParameters(
                model=model,
                method='my-solver',
                times=times
            )

        with self.assertRaises(TypeError):
            em.WarwickSimParameters(
                model=model,
                method=method,
                times='0'
            )

        with self.assertRaises(TypeError):
            em.WarwickSimParameters(
                model=model,
                method=method,
                times=[1, '2']
            )

        with self.assertRaises(ValueError):
            em.WarwickSimParameters(
                model=model,
                method=method,
                times=[0, 1]
            )

    def test__call__(self):
        model = exampleWarwickmodel

        # Set other simulation parameters
        method = 'RK45'
        times = [1, 2]

        SimulationParam = em.WarwickSimParameters(
            model=model,
            method=method,
            times=times
        )

        self.assertEqual(SimulationParam(), 'RK45')

#
# Test WarwickSocDistParameters Class
#


class TestWarwickSocDistParameters(unittest.TestCase):
    """
    Test the 'WarwickSocDistParameters' class.
    """
    def test__init__(self):
        model = exampleWarwickmodel

        SocDistParam = em.WarwickSocDistParameters(
            model=model)

        self.assertEqual(SocDistParam.model, model)
        self.assertEqual(SocDistParam.theta, [0.3])
        self.assertEqual(SocDistParam.phi, [0])
        self.assertEqual(SocDistParam.q_H, [1.25])
        self.assertEqual(SocDistParam.q_S, [0.05])
        self.assertEqual(SocDistParam.q_W, [0.2])
        self.assertEqual(SocDistParam.q_O, [0.05])
        self.assertEqual(SocDistParam.times_npis, [1])

        theta = 0.3
        phi = 0.2
        q_H = 1.25
        q_S = 0.05
        q_W = 0.2
        q_O = 0.05
        times_npis = [1]

        with self.assertRaises(TypeError):
            model1 = {'0': 0}

            em.WarwickSocDistParameters(
                model=model1,
                theta=theta,
                phi=phi,
                q_H=q_H,
                q_S=q_S,
                q_W=q_W,
                q_O=q_O,
                times_npis=times_npis
            )

        with self.assertRaises(TypeError):
            em.WarwickSocDistParameters(
                model=model,
                theta='0.3',
                phi=phi,
                q_H=q_H,
                q_S=q_S,
                q_W=q_W,
                q_O=q_O,
                times_npis=times_npis
            )

        with self.assertRaises(ValueError):
            em.WarwickSocDistParameters(
                model=model,
                theta=-0.3,
                phi=phi,
                q_H=q_H,
                q_S=q_S,
                q_W=q_W,
                q_O=q_O,
                times_npis=times_npis
            )

        with self.assertRaises(ValueError):
            em.WarwickSocDistParameters(
                model=model,
                theta=1.1,
                phi=phi,
                q_H=q_H,
                q_S=q_S,
                q_W=q_W,
                q_O=q_O,
                times_npis=times_npis
            )

        with self.assertRaises(TypeError):
            em.WarwickSocDistParameters(
                model=model,
                theta=theta,
                phi='0',
                q_H=q_H,
                q_S=q_S,
                q_W=q_W,
                q_O=q_O,
                times_npis=times_npis
            )

        with self.assertRaises(ValueError):
            em.WarwickSocDistParameters(
                model=model,
                theta=theta,
                phi=-0.1,
                q_H=q_H,
                q_S=q_S,
                q_W=q_W,
                q_O=q_O,
                times_npis=times_npis
            )

        with self.assertRaises(ValueError):
            em.WarwickSocDistParameters(
                model=model,
                theta=theta,
                phi=2,
                q_H=q_H,
                q_S=q_S,
                q_W=q_W,
                q_O=q_O,
                times_npis=times_npis
            )

        with self.assertRaises(TypeError):
            em.WarwickSocDistParameters(
                model=model,
                theta=theta,
                phi=phi,
                q_H='0.05',
                q_S=q_S,
                q_W=q_W,
                q_O=q_O,
                times_npis=times_npis
            )

        with self.assertRaises(ValueError):
            em.WarwickSocDistParameters(
                model=model,
                theta=theta,
                phi=phi,
                q_H=0.05,
                q_S=q_S,
                q_W=q_W,
                q_O=q_O,
                times_npis=times_npis
            )

        with self.assertRaises(TypeError):
            em.WarwickSocDistParameters(
                model=model,
                theta=theta,
                phi=phi,
                q_H=q_H,
                q_S='0.2',
                q_W=q_W,
                q_O=q_O,
                times_npis=times_npis
            )

        with self.assertRaises(ValueError):
            em.WarwickSocDistParameters(
                model=model,
                theta=theta,
                phi=phi,
                q_H=q_H,
                q_S=-0.2,
                q_W=q_W,
                q_O=q_O,
                times_npis=times_npis
            )

        with self.assertRaises(ValueError):
            em.WarwickSocDistParameters(
                model=model,
                theta=theta,
                phi=phi,
                q_H=q_H,
                q_S=1.2,
                q_W=q_W,
                q_O=q_O,
                times_npis=times_npis
            )

        with self.assertRaises(TypeError):
            em.WarwickSocDistParameters(
                model=model,
                theta=theta,
                phi=phi,
                q_H=q_H,
                q_S=q_S,
                q_W='0.05',
                q_O=q_O,
                times_npis=times_npis
            )

        with self.assertRaises(ValueError):
            em.WarwickSocDistParameters(
                model=model,
                theta=theta,
                phi=phi,
                q_H=q_H,
                q_S=q_S,
                q_W=-0.5,
                q_O=q_O,
                times_npis=times_npis
            )

        with self.assertRaises(ValueError):
            em.WarwickSocDistParameters(
                model=model,
                theta=theta,
                phi=phi,
                q_H=q_H,
                q_S=q_S,
                q_W=1.4,
                q_O=q_O,
                times_npis=times_npis
            )

        with self.assertRaises(TypeError):
            em.WarwickSocDistParameters(
                model=model,
                theta=theta,
                phi=phi,
                q_H=q_H,
                q_S=q_S,
                q_W=q_W,
                q_O='0.9',
                times_npis=times_npis
            )

        with self.assertRaises(ValueError):
            em.WarwickSocDistParameters(
                model=model,
                theta=theta,
                phi=phi,
                q_H=q_H,
                q_S=q_S,
                q_W=q_W,
                q_O=-5,
                times_npis=times_npis
            )

        with self.assertRaises(ValueError):
            em.WarwickSocDistParameters(
                model=model,
                theta=theta,
                phi=phi,
                q_H=q_H,
                q_S=q_S,
                q_W=q_W,
                q_O=3,
                times_npis=times_npis
            )

    def test__call__(self):
        model = exampleWarwickmodel

        # Set other simulation parameters
        theta = 0.3
        phi = 0.2
        q_H = 1.25
        q_S = 0.05
        q_W = 0.2
        q_O = 0.05

        SocDistParam = em.WarwickSocDistParameters(
            model=model,
            theta=theta,
            phi=phi,
            q_H=q_H,
            q_S=q_S,
            q_W=q_W,
            q_O=q_O)

        self.assertEqual(SocDistParam(), [0.3, 0.2, 1.25, 0.05, 0.2, 0.05])

#
# Test WarwickParametersController Class
#


class TestWarwickParametersController(unittest.TestCase):
    """
    Test the 'WarwickParametersController' class.
    """
    def test__init__(self):
        model = exampleWarwickmodel

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=1,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[1500, 600], [700, 400]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 10], [2, 4]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[10, 20], [0, 5]],
            detected_su_IC=[[20, 20], [15, 35]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[7, 13], [14, 2]],
            undetected_s_IC=[[0, 10], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=np.arange(1, 20.5, 0.5).tolist()
        )

        # Set social distancing parameters to default

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters
        )

        self.assertEqual(parameters.model, model)
        self.assertEqual(parameters.ICs, ICs)
        self.assertEqual(parameters.regional_parameters, regional_parameters)
        self.assertEqual(parameters.disease_parameters, disease_parameters)
        self.assertEqual(parameters.transmission_parameters,
                         transmission_parameters)
        self.assertEqual(parameters.simulation_parameters,
                         simulation_parameters)
        self.assertEqual(parameters.soc_dist_parameters(),
                         [0.3, 0, 1.25, 0.05, 0.2, 0.05])

        # Set social distancing parameters
        soc_dist_parameters = em.WarwickSocDistParameters(
            model=model,
            theta=0.3,
            phi=0.2,
            q_H=1.25,
            q_S=0.05,
            q_W=0.2,
            q_O=0.05)

        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            soc_dist_parameters=soc_dist_parameters
        )

        self.assertEqual(parameters.soc_dist_parameters,
                         soc_dist_parameters)

        with self.assertRaises(TypeError):
            model1 = 0.3

            em.WarwickParametersController(
                model=model1,
                regional_parameters=regional_parameters,
                ICs=ICs,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(TypeError):
            regional_parameters1 = 0

            em.WarwickParametersController(
                model=model,
                regional_parameters=regional_parameters1,
                ICs=ICs,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(ValueError):
            regional_parameters1 = em.WarwickRegParameters(
                model=exampleWarwickmodel2,
                region_index=1,
                H=[0.8, 0.8]
            )

            em.WarwickParametersController(
                model=model,
                regional_parameters=regional_parameters1,
                ICs=ICs,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(TypeError):
            ICs1 = '0'

            em.WarwickParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs1,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(ValueError):
            ICs1 = em.WarwickICs(
                model=exampleWarwickmodel2,
                susceptibles_IC=[[1500, 600], [700, 400]],
                exposed_1_f_IC=[[0, 0], [0, 0]],
                exposed_1_sd_IC=[[0, 0], [0, 0]],
                exposed_1_su_IC=[[0, 0], [0, 0]],
                exposed_1_q_IC=[[0, 0], [0, 0]],
                exposed_2_f_IC=[[0, 0], [0, 0]],
                exposed_2_sd_IC=[[0, 0], [0, 0]],
                exposed_2_su_IC=[[0, 0], [0, 0]],
                exposed_2_q_IC=[[0, 0], [0, 0]],
                exposed_3_f_IC=[[0, 0], [0, 0]],
                exposed_3_sd_IC=[[0, 0], [0, 0]],
                exposed_3_su_IC=[[0, 0], [0, 0]],
                exposed_3_q_IC=[[0, 0], [0, 0]],
                detected_f_IC=[[0, 10], [2, 4]],
                detected_qf_IC=[[0, 0], [0, 0]],
                detected_sd_IC=[[10, 20], [0, 5]],
                detected_su_IC=[[20, 20], [15, 35]],
                detected_qs_IC=[[0, 0], [0, 0]],
                undetected_f_IC=[[7, 13], [14, 2]],
                undetected_s_IC=[[0, 10], [0, 0]],
                undetected_q_IC=[[0, 0], [0, 0]],
                recovered_IC=[[0, 0], [0, 0]]
            )

            em.WarwickParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs1,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(TypeError):
            disease_parameters1 = [0]

            em.WarwickParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs,
                disease_parameters=disease_parameters1,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(ValueError):
            disease_parameters1 = em.WarwickDiseaseParameters(
                model=exampleWarwickmodel2,
                tau=0.4,
                d=0.4 * np.ones(len(age_groups))
            )

            em.WarwickParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs,
                disease_parameters=disease_parameters1,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(TypeError):
            transmission_parameters1 = [0]

            em.WarwickParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters1,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(ValueError):
            transmission_parameters1 = em.WarwickTransmission(
                model=exampleWarwickmodel2,
                epsilon=0.5,
                gamma=1,
                sigma=0.5 * np.ones(len(age_groups))
            )

            em.WarwickParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters1,
                simulation_parameters=simulation_parameters
            )

        with self.assertRaises(TypeError):
            simulation_parameters1 = {'0': 0}

            em.WarwickParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters1
            )

        with self.assertRaises(ValueError):
            simulation_parameters1 = em.WarwickSimParameters(
                model=exampleWarwickmodel2,
                method='RK45',
                times=np.arange(1, 20.5, 0.5).tolist()
            )

            em.WarwickParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters1
            )

        with self.assertRaises(TypeError):
            soc_dist_parameters1 = {'0': 0}

            em.WarwickParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters,
                soc_dist_parameters=soc_dist_parameters1
            )

        with self.assertRaises(ValueError):
            soc_dist_parameters1 = em.WarwickSocDistParameters(
                model=exampleWarwickmodel2,
                theta=0.3,
                phi=0.2,
                q_H=1.25,
                q_S=0.05,
                q_W=0.2,
                q_O=0.05)

            em.WarwickParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters,
                soc_dist_parameters=soc_dist_parameters1
            )

    def test__call__(self):
        model = exampleWarwickmodel

        # Set regional and time dependent parameters
        regional_parameters = em.WarwickRegParameters(
            model=model,
            region_index=1,
            H=[0.8, 0.8]
        )

        # Set ICs parameters
        ICs = em.WarwickICs(
            model=model,
            susceptibles_IC=[[1500, 600], [700, 400]],
            exposed_1_f_IC=[[0, 0], [0, 0]],
            exposed_1_sd_IC=[[0, 0], [0, 0]],
            exposed_1_su_IC=[[0, 0], [0, 0]],
            exposed_1_q_IC=[[0, 0], [0, 0]],
            exposed_2_f_IC=[[0, 0], [0, 0]],
            exposed_2_sd_IC=[[0, 0], [0, 0]],
            exposed_2_su_IC=[[0, 0], [0, 0]],
            exposed_2_q_IC=[[0, 0], [0, 0]],
            exposed_3_f_IC=[[0, 0], [0, 0]],
            exposed_3_sd_IC=[[0, 0], [0, 0]],
            exposed_3_su_IC=[[0, 0], [0, 0]],
            exposed_3_q_IC=[[0, 0], [0, 0]],
            detected_f_IC=[[0, 10], [2, 4]],
            detected_qf_IC=[[0, 0], [0, 0]],
            detected_sd_IC=[[10, 20], [0, 5]],
            detected_su_IC=[[20, 20], [15, 35]],
            detected_qs_IC=[[0, 0], [0, 0]],
            undetected_f_IC=[[7, 13], [14, 2]],
            undetected_s_IC=[[0, 10], [0, 0]],
            undetected_q_IC=[[0, 0], [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = em.WarwickDiseaseParameters(
            model=model,
            tau=0.4,
            d=0.4 * np.ones(len(age_groups))
        )

        # Set transmission parameters
        transmission_parameters = em.WarwickTransmission(
            model=model,
            epsilon=0.5,
            gamma=1,
            sigma=0.5 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = em.WarwickSimParameters(
            model=model,
            method='RK45',
            times=np.arange(1, 20.5, 0.5).tolist()
        )

        # Set social distancing parameters
        soc_dist_parameters = em.WarwickSocDistParameters(
            model=model,
            theta=0.3,
            phi=0.2,
            q_H=1.25,
            q_S=0.05,
            q_W=0.2,
            q_O=0.05)

        # Set all parameters in the controller
        parameters = em.WarwickParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs=ICs,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            soc_dist_parameters=soc_dist_parameters
        )

        self.assertEqual(
            parameters(),
            [1, 1500, 600, 700, 400, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 10, 2, 4, 0, 0, 0, 0, 10, 20, 0, 5,
             20, 20, 15, 35, 0, 0, 0, 0, 7, 13, 14, 2,
             0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0.5, 0.5, 0.4, 0.5, 1, 0.4, 0.4, 0.8, 0.8, 'RK45']
        )
