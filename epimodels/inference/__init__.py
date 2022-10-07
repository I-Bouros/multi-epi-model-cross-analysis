#
# Inference submodule of the epimodels module.
# Provides access to classes used in the infernce of parameters for the
# different models considered(phe, roche, etc.).
#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""epimodels.inference is asubmodlue of the Epidemiology Modelling library
and it provides functionality for running parameter inference on all our
models using both optimisation and sampling methods, using the PINTS python
module.
"""

# Import inference classes
from .phe_inference import PheSEIRInfer, PHELogLik, PHELogPrior  # noqa
from .roche_inference import RocheSEIRInfer, RocheLogLik, RocheLogPrior  # noqa
