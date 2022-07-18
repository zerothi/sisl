# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from dataclasses import dataclass, field
import typing
from typing import Literal, Sequence, TypedDict, Any, overload
from collections.abc import Sequence as collections_Sequence

from sisl.viz.input_fields.basic.list import ListInput

from .._input_field import InputField, InputParams

class Option(TypedDict):
    label: str
    value: Any

@dataclass
class OptionsParams(InputParams):
    """These are the parameters that any implementation of OptionsInput should use.

    Parameters
    ----------
    placeholder: str
        Not meaningful in some implementations. The text shown if 
        there's no option chosen. This is optional to implement, it just 
        makes the input field more explicit.
    options: list of dicts like {"label": "value_label", "value": value}
        Each dictionary represents an available option. `"value"` contains
        the value that this option represents, while "label" may be a more
        human readable description of the value. The label is what should
        be shown to the user.
    multiple_choices: boolean
        Whether multiple options can be selected.
    clearable: boolean
        Whether the input field can have an empty value (all its options
        can be deselected).
    """
    placeholder: str = "Choose..."
    options: Sequence[Option] = field(default_factory=list)
    multiple_choices: bool = False
    clearable: bool = False

class OptionsInput(InputField):
    """Input to select between different options.

    GUI indications
    ---------------
    The interface of this input field is left to the choice of the GUI
    designer. Some possibilities are:
        - Checkboxes or radiobuttons.
        - A dropdown, which is better if there are many options.

    Whatever interface one chooses to implement, it should comply with 
    the properties described at OptionsParams.
    """
    params: OptionsParams

    @classmethod
    def from_typehint(cls, type_):
        # Here, we try to understand how the input field should behave based on the type
        # provided.
        origin_type = typing.get_origin(type_)
        if origin_type == Literal:
            # A literal containing the options has been provided
            options = typing.get_args(type_)
            multiple_choices = False
        elif origin_type in (Sequence, collections_Sequence, list):
            # A sequence type has been provided, which means that multiple
            # options can be selected. The selectable options should be
            # provided as the argument of Sequence.
            options = typing.get_args(typing.get_args(type_)[0])
            multiple_choices = True
        else:
            raise TypeError(f"OptionsInput can only be created from a generic Literal or Sequence[Literal], not {type_}")
        
        # Set the parameters, we don't provide the options to OptionsParams
        # to benefit from the set_options method, which will set them.
        params = OptionsParams(multiple_choices=multiple_choices)

        new_obj = cls(params=params)
        new_obj.set_options(options, infer_labels=True)

        return new_obj
    
    def parse(self, val: Any) -> list[Any] | Any:
        if self.params.multiple_choices:
            return ListInput().parse(val)
        else:
            return val

    @overload
    def get_options(self, with_labels: Literal[False]) -> list[Any]:
        ...
    
    @overload
    def get_options(self, with_labels: Literal[True]) -> list[Option]:
        ...

    def get_options(self, with_labels: bool = False) -> list[Any]:
        """Get the available options.
        
        Parameters
        ----------
        with_labels: bool, optional
            Whether to return the options with their labels.
        
        Returns
        ----------
        list
            If `with_labels` is True, each element is a dictionary with
            the keys "label" and "value". Otherwise, each element is
            just the value. 
        """
        return [opt if with_labels else opt["value"] for opt in self.params.options]

    def set_options(self, options: Sequence[Any], infer_labels: bool = True):
        """Set the options that can be selected.
        
        Parameters
        ----------
        options: list
            The options that can be selected.
        infer_labels: bool, optional
            If you are just providing the values of the options, set this to
            True so that the labels are generated from the values with a call
            to `str`.
        """
        if infer_labels:
            options = [{"label": str(option), "value": option} for option in options]

        self.params.options = options

        return self

class CreatableOptionsInput(OptionsInput):
    """Input to select between different options and potentially create new ones.

    GUI indications
    ---------------
    This field is very similar to `OptionsInput`. The implementation should follow
    the details described for `OptionsInput`. Additionally, it should **allow the
    creation of new options**.

    This input will be used when there's no specific set of options, but we want to
    cover some of the most common ones.
    """
