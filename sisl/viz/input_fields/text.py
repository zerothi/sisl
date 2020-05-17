from .._input_field import InputField
from types import MethodType

class TextInput(InputField):

    dtype = str

    _type = "textinput"

    _default = {
        "width": "s100%",
        "params": {
            "placeholder": "Write your value here...",
        }
    }

from sisl import BaseSile

if not hasattr(BaseSile, "to_json"):
    # Little patch so that Siles can be sent to the GUI
    def sile_to_json(self):
        return str(self.file)

    BaseSile.to_json = sile_to_json

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
