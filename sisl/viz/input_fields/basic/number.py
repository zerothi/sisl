# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from dataclasses import dataclass
from typing import Optional, Union

from .._input_field import InputField, InputParams


@dataclass
class NumericInputParams(InputParams):
    """Parameters for the NumericInput class."""
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    step: Union[int,float] = 1

class NumericInput(InputField):
    """Simple input for a number.

    GUI indications
    ----------------
    If you have a `param` that uses a `NumericInput`, you will find a dictionary
    at `param.inputField["params"]` that has some specifications that your input
    field should fulfill. These are: {"min", "max", "step"}.

    E.g. if `param.inputField["params"]` is `{"min": 0, "max": 1, "step": 0.1}`,
    your input field needs to make sure that the value is always contained between
    0 and 1 and can be increased/decreased in steps of 0.1.
    """
    params: NumericInputParams

    _type = "number"


class IntegerInput(NumericInput):
    """Simple input for an integer.

    GUI indications
    ----------------
    No special implementation needed for this input field, if your `NumericInput`
    implementation supports "min", "max" and "step" correctly, you already
    have an `IntegerInput`. 
    """
    params = NumericInputParams(min=None, max=None, step=1)

class FloatInput(NumericInput):
    """Simple input for an integer.

    GUI indications
    ----------------
    No implementation needed for this input field, if your `NumericInput`
    implementation supports "min", "max" and "step" correctly, you already
    have a `FloatInput`. 
    """
    params = NumericInputParams(min=None, max=None, step=0.1)
