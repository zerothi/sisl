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
from .array import Array1DInput, Array2DInput
from .bool import BoolInput
from .color import ColorInput
from .dict import CreatableDictInput, DictInput
from .list import ListInput
from .number import FloatInput, IntegerInput
from .options import CreatableOptionsInput, OptionsInput
from .range import RangeInput, RangeSliderInput
from .text import TextInput
