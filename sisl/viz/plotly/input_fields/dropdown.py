import numpy as np

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

class AtomSelect(DropdownInput):

    _default={
        "width": "s100% m50% l33%",
        "default": None,
        "params": {
            "placeholder": "Select some atoms...",
            "options": [
            ],
            "isMulti": True,
            "isClearable": True,
            "isSearchable": True,
        }
    }

    def update_options(self, geom):

        self.modify("inputField.params.options", 
            [{"label": f"{at} ({geom.atoms[at].symbol})", "value": at} for at in geom])

        return self

class SpeciesSelect(DropdownInput):

    _default = {
        "width": "s100% m50% l40%",
        "default": None,
        "params": {
            "placeholder": "Select the species...",
            "options": [],
            "isMulti": True,
            "isClearable": True,
            "isSearchable": True,
        }
    }

    def update_options(self, geom):

        self.modify("inputField.params.options",
            [{"label": unique_at.symbol, "value": unique_at.symbol} for unique_at in geom.atoms.atom])

        return self

class OrbitalsNameSelect(DropdownInput):

    _default = {
        "width": "s100% m50% l50%",
        "default": None,
        "params": {
            "placeholder": "Select the species...",
            "options": [],
            "isMulti": True,
            "isClearable": True,
            "isSearchable": True,
        }
    }

    def update_options(self, geom):

        orbs = [orb.name() for unique_at in geom.atoms.atom for orb in unique_at]

        self.modify("inputField.params.options",
                [{"label": orb, "value": orb} for orb in np.unique(orbs)])

        return self

class SpinSelect(DropdownInput):

    _default = {
        "width": "s100% m50% l25%",
        "default": None,
        "params": {
            "placeholder": "Select the species...",
            "options": [],
            "isMulti": True,
            "isClearable": True,
            "isSearchable": True,
        },
        "style": {
            "width": 200
        }
    }

    def update_options(self, polarized):

        self.modify("inputField.params.options",
                    [{"label": "↑", "value": 0}, {"label": "↓", "value": 1}] if polarized else [])

        return self

class GeomAxisSelect(DropdownInput):

    _default = {
        "width": "s100% m50% l33%",
        "params": {
            "placeholder": "Choose an option...",
            "options": [
                {'label': ax, 'value': ax} for ax in ["x", "y", "z", 0, 1, 2, "a", "b", "c"]
            ],
            "isMulti": True,
            "isClearable": False,
            "isSearchable": True,
        }
    }
