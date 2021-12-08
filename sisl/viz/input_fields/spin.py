# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from .basic import OptionsInput
from sisl import Spin
from sisl._help import isiterable


class SpinSelect(OptionsInput):
    """ Input field that helps selecting and managing the desired spin.

    It has a method to update the options according to spin class.

    Parameters
    ------------
    only_if_polarized: bool, optional
        If set to `True`, the options can only be either [UP, DOWN] or [].

        That is, no extra options for non collinear and spin orbit calculations.

        Defaults to False.
    """

    _default = {
        "default": None,
        "params": {
            "placeholder": "Select spin...",
            "options": [],
            "isMulti": True,
            "isClearable": True,
            "isSearchable": True,
        },
        "style": {
            "width": 200
        }
    }

    _options = {
        Spin.UNPOLARIZED: [],
        Spin.POLARIZED: [{"label": "↑", "value": 0}, {"label": "↓", "value": 1}],
        Spin.NONCOLINEAR: [{"label": val, "value": val} for val in ("total", "x", "y", "z")],
        Spin.SPINORBIT: [{"label": val, "value": val}
                         for val in ("total", "x", "y", "z")]
    }

    def __init__(self, *args, only_if_polarized=False, **kwargs):

        super().__init__(*args, **kwargs)

        self._only_if_polarized = only_if_polarized

    def update_options(self, spin, only_if_polarized=None):
        """
        Updates the options of the spin selector.

        It does so according to the type of spin that the plot is handling.

        Parameters
        -----------
        spin: sisl.Spin, str or int
            It is used to indicate the kind of spin.
        only_if_polarized: bool, optional
            If set to `True`, the options can only be either [UP, DOWN] or [].

            That is, no extra options for non collinear and spin orbit calculations.

            If not provided the initialization value of `only_if_polarized` will be used.

        See also
        ---------
        sisl.physics.Spin
        """
        if not isinstance(spin, Spin):
            spin = Spin(spin)

        # Use the default for this input field if only_if_polarized is not provided.
        if only_if_polarized is None:
            only_if_polarized = self._only_if_polarized

        # Determine what are the new options
        if only_if_polarized:
            if spin.is_polarized:
                options = self._options[Spin.POLARIZED]
            else:
                options = self._options[Spin.UNPOLARIZED]
        else:
            options = self._options[spin.kind]

        # Update them
        self.modify("inputField.params.options", options)

        return self

    def parse(self, val):
        if val is None:
            return val

        if not isiterable(val):
            val = [val]

        return val
