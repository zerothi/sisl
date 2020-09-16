import numpy as np

from .._input_field import InputField


class ArrayNDInput(InputField):

    dtype = "array-like"

    _type = 'array'

    _default = {
        "width": "s100% l50%"
    }

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
