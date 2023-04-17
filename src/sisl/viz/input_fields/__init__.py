# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""This submodule implements all input fields.

We have the basic input fields, that need a GUI implementation,
and the rest of input fields, which are just extensions of the
basic input fields.
"""
from .basic import *

from .aiida_node import AiidaNodeInput
from .programatic import ProgramaticInput, FunctionInput
from .sisl_obj import SislObjectInput, GeometryInput, BandStructureInput, PlotableInput, SileInput, DistributionInput
from .queries import QueriesInput
from .file import FilePathInput

from .atoms import AtomSelect, SpeciesSelect
from .axes import GeomAxisSelect
from .orbital import OrbitalsNameSelect, OrbitalQueries
from .spin import SpinSelect

from .energy import ErangeInput
