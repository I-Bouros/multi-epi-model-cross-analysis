#
# Root of the epimodels module.
# Provides access to all shared functionality (phe, roche, etc.).
#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""epimodels is a Epidemiology Modelling library.
It contains functionality for creatinf region, contact population matrices
as well as modelling of the number of cases of infections by compartment
during an outbreak of the SARS-Cov-2 virus.
"""

# Import version info
from .version_info import VERSION_INT, VERSION  # noqa

# Import main classes
from ._contact_matrix import ContactMatrix  # noqa