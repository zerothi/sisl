import numpy as np

from .._input_field import InputField


class RangeSlider(InputField):

    dtype = "array-like of shape (2,)"

    _type = 'rangeslider'

    _default = {
        "width": "s100%",
        "params": {
            "min": -10,
            "max": 10,
            "step": 0.1,
        }
    }

    def update_marks(self, marks=None):
        """
        Updates the marks of the rangeslider.

        Parameters
        ----------
        marks: dict, optional
            a dict like {value: label, ...} for each mark that we want.

            If no marks are passed, the method will try to update the marks acoording to the current
            min and max values.
        """
        if marks is None:
            marks = {int(val): str(val) for val in np.arange(self.inputField["params"]["min"], self.inputField["params"]["max"], 1, dtype=int)}

        self.modify("inputField.params.marks", marks)
