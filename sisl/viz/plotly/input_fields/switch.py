from .._input_field import InputField


class SwitchInput(InputField):

    dtype = bool

    _type = 'switch'

    _default = {
        "width": "s50% m30% l15%",
        "params": {
            "offLabel": "Off",
            "onLabel": "On"
        }
    }
