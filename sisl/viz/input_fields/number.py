# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from .._input_field import InputField


class NumericInput(InputField):

    _type = 'number'

    _default = {
        "width": "s50% m30% l30%",
        "default": 0,
        "params": {
            "min": 0,
        }
    }


class IntegerInput(NumericInput):

    dtype = int

    _default = {
        **NumericInput._default,
        "params": {
            **NumericInput._default["params"],
            "step": 1
        }
    }


class FloatInput(NumericInput):

    dtype = float

    _default = {
        **NumericInput._default,
        "params": {
            **NumericInput._default["params"],
            "step": 0.1
        }
    }
