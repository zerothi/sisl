from .._input_field import InputField


class RangeInput(InputField):

    dtype = "array-like of shape (2,)"

    _type = 'range'

    _default = {
        "width": "s100% m50% l33%",
        "params": {
            'step': 0.1
        }
    }


class ErangeInput(RangeInput):

    def __init__(self, key, name="Energy range", default=None, params={"step": 1}, help="The energy range that is displayed", **kwargs):

        super().__init__(key=key, name=name, default=default, params=params, help=help, **kwargs)
