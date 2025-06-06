#
# Root of the epimodels module.
# Provides access to all shared functionality (phe, roche, etc.).
#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""epimodels is an Epidemiology Modelling library.
It contains functionality for creating regional, contact population matrices
as well as modelling of the number of cases of infections by compartment
during an outbreak of the SARS-Cov-2 virus.

The submodule epimodels.inference provides functionality for running parameter
inference on all our models using both optimisation and sampling methods, using
the PINTS python module.
"""

# Import version info
from .version_info import VERSION_INT, VERSION  # noqa

# Import inference submodule
from . import inference  # noqa

# Import models
from .phe_model import PheSEIRModel  # noqa
from .roche_model import RocheSEIRModel  # noqa
from .warwick_model import WarwickSEIRModel  # noqa

# Import auxiliary matrices
from ._setup_matrices import (  # noqa
    ContactMatrix,
    RegionMatrix,
    UniNextGenMatrix,
    MultiTimesContacts,
    UniInfectivityMatrix,
    MultiTimesInfectivity)

# Import model parameter controller classes
# For PHE
from._parameters import (  # noqa
    PheICs,
    PheRegParameters,
    PheDiseaseParameters,
    PheSimParameters,
    PheParametersController
)

# For Roche
from._parameters import (  # noqa
    RocheICs,
    RocheCompartmentTimes,
    RocheProportions,
    RocheTransmission,
    RocheSimParameters,
    RocheParametersController
)

# For Warwick
from._parameters import (  # noqa
    WarwickICs,
    WarwickRegParameters,
    WarwickDiseaseParameters,
    WarwickTransmission,
    WarwickSimParameters,
    WarwickSocDistParameters,
    WarwickParametersController
)
