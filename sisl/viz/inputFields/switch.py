from ..inputField import InputField

class SwitchInput(InputField):

    _type = 'switch'

    _default = {
        "width": "s50% m30% l15%",
        "params": {
            "offLabel": "Off",
            "onLabel": "On"
        }
    }