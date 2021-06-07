# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from .._input_field import InputField


class ColorPicker(InputField):

    dtype = str

    _type = 'color'

    _default = {
        "width": "s50% m30% l15%"
    }
