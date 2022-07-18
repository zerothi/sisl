# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from dataclasses import dataclass, field
import typing
from .basic.options import OptionsInput, OptionsParams
from sisl import Spin
from sisl._help import isiterable

@dataclass
class SpinIndexSelectParams(OptionsParams):
    placeholder: str = "Select spin..."
    options: list = field(default_factory=list)
    multiple_choices: bool = True
    clearable: bool = True
    spin: Spin = field(default_factory=Spin)
    only_if_polarized: bool = False

class SpinIndexSelect(OptionsInput):
    """ Input field that helps selecting and managing the desired spin.

    It has a method to update the options according to spin class.

    Parameters
    ------------
    only_if_polarized: bool, optional
        If set to `True`, the options can only be either [UP, DOWN] or [].

        That is, no extra options for non collinear and spin orbit calculations.

        Defaults to False.
    """
    params: SpinIndexSelectParams

    _options = {
        Spin.UNPOLARIZED: [],
        Spin.POLARIZED: [{"label": "↑", "value": 0}, {"label": "↓", "value": 1}],
        Spin.NONCOLINEAR: [{"label": val, "value": val} for val in ("total", "x", "y", "z")],
        Spin.SPINORBIT: [{"label": val, "value": val}
                         for val in ("total", "x", "y", "z")]
    }

    @classmethod
    def from_typehint(cls, type_):
        return cls()

    @classmethod
    def get_spin_options(cls, spin: typing.Union[Spin, str]) -> typing.List[typing.Literal[0, 1, "total", "x", "y", "z"]]:
        """Returns the options for a given spin class."""
        spin = Spin(spin)
        return [option['value'] for option in cls._options[spin.kind]]

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

        self.params.spin = spin

        # Use the default for this input field if only_if_polarized is not provided.
        if only_if_polarized is None:
            only_if_polarized = self.params.only_if_polarized

        # Determine what are the new options
        if only_if_polarized:
            if spin.is_polarized:
                options = self._options[Spin.POLARIZED]
            else:
                options = self._options[Spin.UNPOLARIZED]
        else:
            options = self._options[spin.kind]

        # Update them
        self.set_options(options=options, infer_labels=False)

        return self

    def parse(self, val):
        if val is None:
            return val

        if not isiterable(val):
            val = [val]

        return val
