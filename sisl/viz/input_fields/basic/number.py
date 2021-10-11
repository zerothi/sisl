# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from sisl._help import isiterable

from ..._input_field import InputField


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

    _type = 'number'

    _default = {
        "default": 0,
        "params": {
            "min": 0,
        }
    }


class IntegerInput(NumericInput):
    """Simple input for an integer.

    GUI indications
    ----------------
    No implementation needed for this input field, if your `NumericInput`
    implementation supports "min", "max" and "step" correctly, you already
    have an `IntegerInput`. 
    """

    dtype = int

    _default = {
        **NumericInput._default,
        "params": {
            **NumericInput._default["params"],
            "step": 1
        }
    }

    def parse(self, val):
        if val is None:
            return val
        if isiterable(val):
            return np.array(val, dtype=int)
        return int(val)


class FloatInput(NumericInput):
    """Simple input for an integer.

    GUI indications
    ----------------
    No implementation needed for this input field, if your `NumericInput`
    implementation supports "min", "max" and "step" correctly, you already
    have a `FloatInput`. 
    """

    dtype = float

    _default = {
        **NumericInput._default,
        "params": {
            **NumericInput._default["params"],
            "step": 0.1
        }
    }

    def parse(self, val):
        if val is None:
            return val
        if isiterable(val):
            return np.array(val, dtype=float)
        return float(val)
