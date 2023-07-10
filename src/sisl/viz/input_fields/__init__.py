# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""This submodule implements all input fields.

We have the basic input fields, that need a GUI implementation,
and the rest of input fields, which are just extensions of the
basic input fields.
"""
from .aiida_node import AiidaNodeInput
from .atoms import AtomSelect, SpeciesSelect
from .axes import GeomAxisSelect
from .basic import *
from .energy import ErangeInput
from .file import FilePathInput
from .orbital import OrbitalQueries, OrbitalsNameSelect
from .programatic import FunctionInput, ProgramaticInput
from .queries import QueriesInput
from .sisl_obj import (
    BandStructureInput,
    DistributionInput,
    GeometryInput,
    PlotableInput,
    SileInput,
    SislObjectInput,
)
from .spin import SpinSelect
