# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ..._input_field import InputField


class ColorInput(InputField):
    """Input field to pick a color.

    GUI indications
    ----------------
    The best implementation for this input is probably a color picker
    that let's the user choose any color they like.

    The value returned by the input field should be a string representing
    a color in hex, rgb, rgba or any other named color supported in html.  
    """

    dtype = str

    _type = 'color'
