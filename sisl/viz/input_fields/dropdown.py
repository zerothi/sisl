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

class AtomSelect(DropdownInput):

    def update_options(self, geom):

        self.modify("inputField.params.options", 
            [{"label": f"{at+1} ({geom.atoms[at].symbol})", "value": at} for at in geom])

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
