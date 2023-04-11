# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from ..._input_field import InputField


class RangeInput(InputField):
    """Simple range input composed of two values, min and max.

    GUI indications
    ----------------
    This input field is the interface to an array of length 2 that specifies
    some range. E.g. a valid value could be `[0, 1]`, which would mean "from
    0 to 1". Some simple implementation can be just two numeric inputs.

    It should make sure that if `param.inputField["params"]` contains min and
    max, the values of the range never go beyond those limits. 
    """

    dtype = "array-like of shape (2,)"

    _type = 'range'

    _default = {
        "params": {
            'step': 0.1
        }
    }


class RangeSliderInput(InputField):
    """Slider that controls a range.

    GUI indications
    ----------------
    A slider that lets you select a range.

    It is used over `RangeInput` when the bounds of the range (min and max)
    are very well defined. The reason to prefer a slider is that visually is
    much better. However, if it's not possible to implement, this field can
    use the same interface as `RangeInput` without problem.

    It should make sure that if `param.inputField["params"]` contains min and
    max, the values of the range never go beyond those limits. Also,
    `param.inputField["params"]["marks"]` can contain a list of dictionaries 
    with the values and labels of the ticks that should appear in the slider. 
    I.e. `[{"value": 0, "label": "mark1"}]` indicates that there should be a 
    tick at 0 with label "mark1".
    """

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
        """Updates the marks of the rangeslider.

        Parameters
        ----------
        marks: dict, optional
            a dict like {value: label, ...} for each mark that we want.

            If no marks are passed, the method will try to update the marks acoording to the current
            min and max values.
        """
        if marks is None:
            marks = [{"value": int(val), "label": str(val)} for val in np.arange(
                self.inputField["params"]["min"], self.inputField["params"]["max"], 1, dtype=int)]

        self.modify("inputField.params.marks", marks)
