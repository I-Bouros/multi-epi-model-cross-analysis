************************
Model Parameters Classes
************************

.. currentmodule:: epimodels

This page contains information about the different classes which contain parameters used for the forward simulation of the various
models currently included in the :currentmodule:`epimodels` library: *PHE Model*, *Roche Model*.

Overview:

- Initial Conditions Classes:
    - :class:`PheICs`
    - :class:`RocheICs`

- Regional and Time Dependent Parameters Classes:
    - :class:`PheRegParameters`

- Disease Specific Parameters Classes:
    - :class:`PheDiseaseParameters`
    - :class:`RocheCompartmentTimes`
    - :class:`RocheProportions`
    - :class:`RocheTransmission`

- Simulation Method Parameters Classes:
    - :class:`PheSimParameters`
    - :class:`RocheSimParameters`

- Parameters Controller Classes:
    - :class:`PheParametersController`
    - :class:`RocheParametersController`

Parameter Classes for the PHE Model
***********************************
    Below we list the methods for all the parameter classes associated with the forward simulation of the PHE model.

Initial Conditions Parameters
*****************************

.. autoclass:: PheICs
  :members:

Regional and Time Dependent Parameters
**************************************

.. autoclass:: PheRegParameters
  :members:

Disease Specific Parameters
***************************

.. autoclass:: PheDiseaseParameters
  :members:

Simulation Method Parameters
****************************

.. autoclass:: PheSimParameters
  :members:

Parameters Controller
*********************

.. autoclass:: PheParametersController
  :members:

Parameter Classes for the Roche Model
***********************************
    Below we list the methods for all the parameter classes associated with the forward simulation of the Roche model.

Initial Conditions Parameters
*****************************

.. autoclass:: RocheICs
  :members:

Average Times in Compartments Parameters
****************************************

.. autoclass:: RocheCompartmentTimes
  :members:

Proportions of Asymptomatic, Super-spreader and Dead Parameters
***************************************************************

.. autoclass:: RocheProportions
  :members:

Transmission Specific Parameters
********************************

.. autoclass:: RocheTransmission
  :members:

Simulation Method Parameters
****************************

.. autoclass:: RocheSimParameters
  :members:

Parameters Controller
*********************

.. autoclass:: RocheParametersController
  :members: