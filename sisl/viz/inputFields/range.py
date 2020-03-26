from ..inputField import InputField

class RangeInput(InputField):

    _type = 'range'

    _default = {
        "width": "s100% m50% l33%",
        "params": {
            'step': 0.1
        }
    }