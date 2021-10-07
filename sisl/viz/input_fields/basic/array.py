# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from ..._input_field import InputField


class ArrayNDInput(InputField):

    dtype = "array-like"

    _type = 'array'

    _default = {}

    def __init__(self, *args, **kwargs):

        # If the shape of the array was not provided, register it
        # This is important because otherwise when the setting is set to None
        # there is no way of knowing how to display the input field.
        # For variable shapes, a different input (ListInput) should be used
        try:
            kwargs["params"]["shape"]
        except:
            shape = np.array(kwargs["default"]).shape

            if kwargs.get("params", False):
                kwargs["params"]["shape"] = shape
            else:
                kwargs["params"] = {"shape": shape}

        super().__init__(*args, **kwargs)


class Array1DInput(ArrayNDInput):

    _type = 'vector'


class Array2DInput(ArrayNDInput):

    _type = "matrix"
