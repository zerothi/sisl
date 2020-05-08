from .._input_field import InputField

class TextInput(InputField):

    dtype = str

    _type = "textinput"

    _default = {
        "width": "s100%",
        "params": {
            "placeholder": "Write your value here...",
        }
    }

class FilePathInput(TextInput):

    _default = {
        "width": "s100%",
        "params": {
            "placeholder": "Write your path here...",
        }
    }