# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from .._input_field import InputField
from .text import TextInput


class ListInput(InputField):

    dtype = "array-like"

    _type = 'list'

    _default = {
        "width": "s100% l50%",
        "params": {"itemInput": TextInput("-", "-"), "sortable": True}
    }
