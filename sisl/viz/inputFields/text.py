from ..inputField import InputField

class TextInput(InputField):

    _type = "textinput"

    _default = {
        "width": "s100%",
        "params": {
            "placeholder": "Write your value here...",
        }
    }