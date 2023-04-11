# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from .basic import RangeInput


class ErangeInput(RangeInput):

    def __init__(self, key, name="Energy range", default=None,
        params={"step": 1}, help="The energy range that is displayed", **kwargs):

        super().__init__(key=key, name=name, default=default,
                         params=params, help=help, **kwargs)
