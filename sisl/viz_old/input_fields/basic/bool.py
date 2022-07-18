# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ..._input_field import InputField


class BoolInput(InputField):
    """Simple input that controls a boolean variable.

    GUI indications
    ----------------
    It can be implemented as a switch or a checkbox, for example.
    """
    _false_strings = ("f", "false")
    _true_strings = ("t", "true")

    dtype = bool

    _type = 'bool'

    _default = {}

    def parse(self, val):
        if val is None:
            pass
        elif isinstance(val, str):
            val = val.lower()
            if val in self._true_strings:
                val = True
            elif val in self._false_strings:
                val = False
            else:
                raise ValueError(f"String '{val}' is not understood by {self.__class__.__name__}")
        elif not isinstance(val, bool):
            self._raise_type_error(val)

        return val
