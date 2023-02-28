""" Contains external typing information for local sisl usage

The intended audience for this submodule is that the represented
variables are used throughout sisl.
The added external typing modules should only expose specific
typing constructs from the sub-packages.
In some cases the sisl requirements are lower than the typing
interfaces provided by the external packages and hence they are
required to be handled via versioning or by other means.
This is easier to maintain in a single source file, rather
than for every line of sisl using that typing information.
"""
from . import numpy as npt
