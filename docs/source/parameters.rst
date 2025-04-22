************************
Model Parameters Classes
************************

.. currentmodule:: epimodels

This page contains information about the different classes which contain parameters used for the forward simulation of the various
models currently included in the `epimodels` library: *PHE Model*, *Roche Model*, and *Warwick-Household Model*.

Overview:

- Initial Conditions Classes:
    - :class:`PheICs`
    - :class:`RocheICs`
    - :class:`WarwickICs`

- Regional and Time Dependent Parameters Classes:
    - :class:`PheRegParameters`
    - :class:`WarwickRegParameters`

- Disease Specific Parameters Classes:
    - :class:`PheDiseaseParameters`
    - :class:`RocheCompartmentTimes`
    - :class:`RocheProportions`
    - :class:`RocheTransmission`
    - :class:`WarwickDiseaseParameters`
    - :class:`WarwickTransmission`

- Simulation Method Parameters Classes:
    - :class:`PheSimParameters`
    - :class:`RocheSimParameters`
    - :class:`WarwickSimParameters`

- Social Distancing Parameters Classes:
    - :class:`WarwickSocDistParameters`

- Parameters Controller Classes:
    - :class:`PheParametersController`
    - :class:`RocheParametersController`
    - :class:`WarwickParametersController`

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

Parameter Classes for the Roche SEIRD Model
*******************************************
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

Parameter Classes for the Warwick-Household Model
*************************************************
    Below we list the methods for all the parameter classes associated with the forward simulation of the Warwick-Household model.

Initial Conditions Parameters
*****************************

.. autoclass:: WarwickICs
  :members:

Regional and Time Dependent Parameters
**************************************

.. autoclass:: WarwickRegParameters
  :members:

Disease Specific Parameters
***************************

.. autoclass:: WarwickDiseaseParameters
  :members:

Transmission Specific Parameters
********************************

.. autoclass:: WarwickTransmission
  :members:

Simulation Method Parameters
****************************

.. autoclass:: WarwickSimParameters
  :members:

Social Distancing Parameters
****************************

.. autoclass:: WarwickSocDistParameters
  :members:

Parameters Controller
*********************

.. autoclass:: WarwickParametersController
  :members:
