from pathlib import Path

from sisl import BaseSile

from .basic import TextInput

if not hasattr(BaseSile, "to_json"):
    # Little patch so that Siles can be sent to the GUI
    def sile_to_json(self):
        return str(self.file)

    BaseSile.to_json = sile_to_json


class FilePathInput(TextInput):

    _default = {
        "params": {
            "placeholder": "Write your path here...",
        }
    }

    def parse(self, val):

        if isinstance(val, BaseSile):
            val = val.file

        if isinstance(val, str):
            val = Path(val)

        return val
