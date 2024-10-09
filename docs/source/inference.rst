***************************
List of Epidemiology Models
***************************

This section documents the classes used for the parameetr inference of epidemiological models curated in this collection.

.. currentmodule:: epimodels.inference

Overview:

- Inference & Optimisation Controller Classes:
    - :class:`PheSEIRInfer`
    - :class:`RocheSEIRInfer`
    - :class:`WarwickSEIRInfer`

- Log-likelihood Initial Conditions Classes:
    - :class:`PHELogLik`
    - :class:`RocheLogLik`
    - :class:`WarwickLogLik`

- Prior Classes:
    - :class:`PHELogPrior`
    - :class:`RocheLogPrior`
    - :class:`WarwickLogPrior`

Public Health England & Cambridge Model
***************************************

.. autoclass:: PheSEIRInfer
  :members:

.. autoclass:: PHELogLik
  :members:

.. autoclass:: PHELogPrior
  :members:

Roche SEIRD Model
*****************

.. autoclass:: RocheSEIRInfer
  :members:

.. autoclass:: RocheLogLik
  :members:

.. autoclass:: RocheLogPrior
  :members:

Warwick-Household Model
***********************

.. autoclass:: WarwickSEIRInfer
  :members:

.. autoclass:: WarwickLogLik
  :members:

.. autoclass:: WarwickLogPrior
  :members: