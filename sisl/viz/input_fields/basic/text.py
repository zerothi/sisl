# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from dataclasses import dataclass
from typing import Any

from .._input_field import InputField, InputParams

@dataclass
class TextInputParams(InputParams):
    """These are the parameters that any implementation of TextInput should use.
    
    Parameters
    ----------
    placeholder: str
        Text that you may want to show to the user if the input field is empty.
    """
    placeholder: str = "Write your value here..."

class TextInput(InputField):
    """Simple input for text.

    GUI indications
    ----------------
    The implementation of this input should be a simple text field.
    """
    params: TextInputParams

    def parse(self, val: Any) -> str:
        return str(val)


