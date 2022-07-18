# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, TypedDict
import numpy as np

from .._input_field import InputField, InputParams

@dataclass
class RangeInputParams(InputParams):
    """These are the parameters that any implementation of RangeInput should use.
    
    Parameters
    ----------
    min: float
        Minimum value of the range.
    max: float
        Maximum value of the range.
    step: float
        Step size of the range.
    """
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = 0.1

class RangeInput(InputField):
    """Simple range input composed of two values, min and max.

    GUI indications
    ----------------
    This input field is the interface to an array of length 2 that specifies
    some range. E.g. a valid value could be `[0, 1]`, which would mean "from
    0 to 1". Some simple implementation can be just two numeric inputs.
    """
    params: RangeInputParams

class Mark(TypedDict):
    value: int | float
    label: str

@dataclass
class RangeSliderInputParams(InputParams):
    """These are the parameters that any implementation of RangeInput should use.
    
    Parameters
    ----------
    min: float
        Minimum value of the range.
    max: float
        Maximum value of the range.
    step: float
        Step size of the range.
    marks:
        List of dictionaries with the values and labels of the ticks that should 
        appear in the slider. I.e. `[{"value": 0, "label": "mark1"}]` indicates that 
        there should be a tick at 0 with label "mark1".
    """
    min: int | float = -10
    max: int | float = 10
    step: int | float = 0.1
    marks: Optional[Sequence[Mark]] = None

class RangeSliderInput(InputField):
    """Slider that controls a range.

    GUI indications
    ----------------
    A slider that lets you select a range.

    It is used over `RangeInput` when the bounds of the range (min and max)
    are very well defined. The reason to prefer a slider is that visually is
    much better. However, if it's not possible to implement, this field can
    use the same interface as `RangeInput` without problem.
    """
    params: RangeSliderInputParams

    @classmethod
    def from_typehint(cls, type_):
        marks = cls._default_marks(-10, 10)
        params = RangeSliderInputParams(min=-10, max=10, step=0.1, marks=marks)

        return cls(params=params)

    @staticmethod
    def _default_marks(min: int | float, max: int | float) -> list[Mark]:
        return [{"value": int(val), "label": str(val)} for val in np.arange(min, max, 1, dtype=int)]

    # def update_marks(self, marks=None):
    #     """Updates the marks of the rangeslider.

    #     Parameters
    #     ----------
    #     marks: dict, optional
    #         a dict like {value: label, ...} for each mark that we want.

    #         If no marks are passed, the method will try to update the marks acoording to the current
    #         min and max values.
    #     """
    #     if marks is None:
    #         marks = [{"value": int(val), "label": str(val)} for val in np.arange(
    #             self.inputField["params"]["min"], self.inputField["params"]["max"], 1, dtype=int)]

    #     self.modify("inputField.params.marks", marks)
