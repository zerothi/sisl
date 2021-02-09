import numpy as np

from sisl import Spin
from .._input_field import InputField


class DropdownInput(InputField):

    _type = 'dropdown'

    _default = {
        "width": "s100% m50% l33%",
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

    def _get_options(self):

        return [opt["value"] for opt in self['inputField.params.options']]

    def _set_options(self, val):

        self.modify("inputField.params.options", options)

    options = property(fget=_get_options, fset=_set_options)


class CreatableDropdown(DropdownInput):

    _type = "creatable dropdown"


class AtomSelect(DropdownInput):

    _default={
        "width": "s100% m50% l33%",
        "default": None,
        "params": {
            "placeholder": "Select atoms...",
            "options": [
            ],
            "isMulti": True,
            "isClearable": True,
            "isSearchable": True,
        }
    }

    def update_options(self, geom):

        self.modify("inputField.params.options",
            [{"label": f"{at} ({geom.atoms[at].symbol})", "value": at}
             for at in geom])

        return self


class SpeciesSelect(DropdownInput):

    _default = {
        "width": "s100% m50% l40%",
        "default": None,
        "params": {
            "placeholder": "Select species...",
            "options": [],
            "isMulti": True,
            "isClearable": True,
            "isSearchable": True,
        }
    }

    def update_options(self, geom):

        self.modify("inputField.params.options",
                    [{"label": unique_at.symbol, "value": unique_at.symbol}
                     for unique_at in geom.atoms.atom])

        return self


class OrbitalsNameSelect(DropdownInput):

    _default = {
        "width": "s100% m50% l50%",
        "default": None,
        "params": {
            "placeholder": "Select orbitals...",
            "options": [],
            "isMulti": True,
            "isClearable": True,
            "isSearchable": True,
        }
    }

    def update_options(self, geom):

        orbs = set([orb.name() for unique_at in geom.atoms.atom for orb in unique_at])

        self.modify("inputField.params.options",
                    [{"label": orb, "value": orb}
                     for orb in orbs])

        return self


class SpinSelect(DropdownInput):
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
        "width": "s100% m50% l25%",
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
        Spin.UNPOLARIZED: [{"label": "Total", "value": 0}],
        Spin.POLARIZED: [{"label": "↑", "value": 0}, {"label": "↓", "value": 1}],
        Spin.NONCOLINEAR: [{"label": val, "value": val} for val in ("sum", "x", "y", "z")],
        Spin.SPINORBIT: [{"label": val, "value": val} for val in ("sum", "x", "y", "z")]
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

        if not isinstance(val, (list, np.ndarray)):
            val = [val]

        return val


class GeomAxisSelect(DropdownInput):

    _default = {
        "width": "s100% m50% l33%",
        "params": {
            "placeholder": "Choose axis...",
            "options": [
                {'label': ax, 'value': ax} for ax in ["x", "y", "z", 0, 1, 2, "a", "b", "c"]
            ],
            "isMulti": True,
            "isClearable": False,
            "isSearchable": True,
        }
    }
