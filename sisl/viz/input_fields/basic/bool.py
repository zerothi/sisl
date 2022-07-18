# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations
from typing import overload

from .._input_field import InputField

class BoolInput(InputField):
    """Simple input that controls a boolean variable.

    GUI indications
    ----------------
    It can be implemented as a switch or a checkbox, for example.
    """

