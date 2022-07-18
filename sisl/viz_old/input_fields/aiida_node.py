# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from .._input_field import InputField

try:
    from aiida import orm
    AIIDA_AVAILABLE = True
except ModuleNotFoundError:
    AIIDA_AVAILABLE = False


class AiidaNodeInput(InputField):

    dtype = orm.Node if AIIDA_AVAILABLE else None

    def parse(self, val):

        if AIIDA_AVAILABLE and val is not None and not isinstance(val, self.dtype):
            val = orm.load_node(val)

        return val
