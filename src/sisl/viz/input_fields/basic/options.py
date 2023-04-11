# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ..._input_field import InputField


class OptionsInput(InputField):
    """Input to select between different options.

    GUI indications
    ---------------
    The interface of this input field is left to the choice of the GUI
    designer. Some possibilities are:
        - Checkboxes or radiobuttons.
        - A dropdown, better if there are many options.

    Whatever interface one chooses to implement, it should comply with 
    the following properties described at `param.inputField["params"]`:
    placeholder: str
        Not meaningful in some implementations. The text shown if 
        there's no option chosen. This is optional to implement, it just 
        makes the input field more explicit.
    options: list of dicts like {"label": "value_label", "value": value}
        Each dictionary represents an available option. `"value"` contains
        the value that this option represents, while "label" may be a more
        human readable description of the value. The label is what should
        be shown to the user.
    isMulti: boolean
        Whether multiple options can be selected.
    isClearable: boolean
        Whether the input field can have an empty value (all its options
        can be deselected).
    """

    _type = 'options'

    _default = {
        "params": {
            "placeholder": "Choose an option...",
            "options": [
            ],
            "isMulti": False,
            "isClearable": True,
            "isSearchable": True,
        }
    }

    def __init__(self, *args, **kwargs):

        # Build the help string
        params = kwargs.get("params")
        if "dtype" not in kwargs and params is not None:

            multiple_choice = getattr(params, "isMulti", False)
            if multiple_choice:
                self.dtype = "array-like"

            options = getattr(params, "options", None)
            if options is not None:
                self.valid_vals = [option["value"] for option in options]

        super().__init__(*args, **kwargs)

    def get_options(self, raw=False):
        return [opt if raw else opt["value"] for opt in self['inputField.params.options']]

    def _set_options(self, val):
        self.modify("inputField.params.options", val)

    options = property(fget=get_options, fset=_set_options)


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

    _type = "creatable options"
