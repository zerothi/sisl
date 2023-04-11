# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from sisl._help import isiterable

from ..._input_field import InputField
from .text import TextInput


class ListInput(InputField):
    """A versatile list input.

    GUI indications
    ---------------
    This input field is to be used to create and mutate lists. Therefore,
    it should implement some way of performing the following actions:
        - Create a new element
        - Delete an existing element
        - Reorganize the existing list.

    The input field of each element in the list is indicated in `param.inputField["params"]`
    under the `"itemInput"` key. The `"sortable"` key contains a boolean specifying whether
    the user should have the ability of sorting the list.
    """

    dtype = "array-like"

    _type = 'list'

    _default = {
        "params": {"itemInput": TextInput("-", "-"), "sortable": True}
    }

    def get_item_input(self):
        return self.inputField["params"]["itemInput"]

    def modify_item_input(self, *args, **kwargs):
        return self.get_item_input().modify(*args, **kwargs)

    def parse(self, val):
        if val is None:
            return val
        elif not isiterable(val):
            self._raise_type_error(val)

        return [self.get_item_input().parse(v) for v in val]
