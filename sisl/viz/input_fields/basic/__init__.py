# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
This submodule contains all the basic input fields. 

If a GUI wants to use sisl.viz, it should **build an implementation for 
the input fields here**. These are the building blocks that all input
fields use.

The rest of input fields are just extensions of the ones implemented here. 
Extensions only tweak details of the internal functionality (e.g. parsing)
but the graphical interface of the input needs no modification.
"""
from .text import TextInput
from .bool import BoolInput
from .number import FloatInput, IntegerInput
from .list import ListInput
from .array import Array1DInput, Array2DInput
from .dict import DictInput, CreatableDictInput

from .options import OptionsInput, CreatableOptionsInput
from .range import RangeInput, RangeSliderInput
from .color import ColorInput
