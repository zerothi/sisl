from sisl import BaseSile
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

    def _parse(self, val):

        if isinstance(val, BaseSile):
            return str(val.file)
        else:
            return val
