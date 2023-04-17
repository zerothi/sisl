# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ..._input_field import InputField


class TextInput(InputField):
    """Simple input for text.

    GUI indications
    ----------------
    The implementation of this input should be a simple text field.

    Optionally, you may use `param.inputField["params"]["placeholder"]`
    as the placeholder.
    """

    dtype = str

    _type = "textinput"

    _default = {
        "params": {
            "placeholder": "Write your value here...",
        }
    }
