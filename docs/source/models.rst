***************************
List of Epidemiology Models
***************************

This section documents the epidemiological models curated in this collection.

The first class contains code for modelling the extended SEIR model created by
Public Health England and University of Cambridge. This is one of the
official models used by the UK government for policy making.

It uses an extended version of an SEIR model and contact and region specific
matrices.

The second class contains code for modelling the extended SEIRD model created by
F. Hoffmann-La Roche Ltd and can be used to model the effects of
non-pharmaceutical interventions (NPIs) on the epidemic dynamics.

It uses an extended version of an SEIRD model which differentiates between
symptomatic and asymptomatic, as well as super-spreaders infectives.

The third class contains code for modelling the extended SEIR model
created by the University of Warwick. This is one of the official models used
by the UK government for policy making.

It uses an extended version of an SEIR model with contact and region-specific
matrices and can be used to model the effects of within-household dynamics on
the epidemic trajectory in different countries. It also differentiates between
asymptomatic and symptomatic infections.

.. currentmodule:: epimodels

Overview:

- :class:`PheSEIRModel`
- :class:`RocheSEIRModel`
- :class:`WarwickSEIRModel`

Public Health England & Cambridge Model
***************************************

.. autoclass:: PheSEIRModel
  :members:

Roche SEIRD Model
*****************

.. autoclass:: RocheSEIRModel
  :members:

Warwick-Household Model
***********************

.. autoclass:: WarwickSEIRModel
  :members:
