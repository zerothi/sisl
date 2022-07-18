# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, overload
import typing
from sisl._help import isiterable

from .._input_field import InputField, InputParams
from .text import TextInput

@dataclass
class ListInputParams(InputParams):
    """These are the parameters that any implementation of ListInput should use.

    Parameters
    ----------
    item_input: InputField, optional
        The input field that will be used to represent each item in the list.
        Currently all items in the list must come from the same type of input
        field.
    sortable: bool, optional
        whether the user should have the ability to sort the list
    """
    item_input: InputField = field(default_factory=TextInput)
    sortable: bool = True

class ListInput(InputField):
    """A versatile list input.

    GUI indications
    ---------------
    This input field is to be used to create and mutate lists. Therefore,
    it should implement some way of performing the following actions:
        - Create a new element
        - Delete an existing element
        - Reorganize the existing list.
    """
    params: ListInputParams

    # Variables that help parsing from a generic sequence type to the
    # specific input field designed for it.
    _type_to_class: typing.Dict[type, type[ListInput]] = {}
    # If a subclass is not found through the type annotation, then ListInput
    # gets the corresponding input field for the type annotation and then searches
    # with that field. This makes the logic easier than searching by type.
    _field_to_class: typing.Dict[type[InputField], type[ListInput]] = {}

    # In a subclass of ListInput, one can define either the type annotation or the
    # input field that it uses for its items.
    _item_input_type: type | None = None
    _item_input_field: type[InputField] | None = None

    def __init__(self, *args, params: ListInputParams | None =None, **kwargs):
        if params is None:
            params = self._init_params_from_type(self._item_input_type)
        super().__init__(*args, params=params, **kwargs)
    
    @classmethod
    def from_typehint(cls, type_):
        from .._typetofield import get_field_from_type
        arg = typing.get_args(type_)[0]

        # If this is the base ListInput, find whether there is a more specific
        # type for sequences with the "arg" item type.
        kls = cls
        if cls is ListInput:
            kls = cls._type_to_class.get(arg)

        if kls is None:
            # Get the type of input from the field
            item_input = get_field_from_type(arg, init=False)
            # And try to search for a subclass that uses these input field.
            if item_input is not None:
                kls = cls._field_to_class.get(item_input)
        
        if kls is None:
            kls = cls
            
        params = kls._init_params_from_type(arg)
            
        return kls(params=params)

    @classmethod
    def _init_params_from_type(cls, type_arg) -> ListInputParams | None:
        from .._typetofield import get_field_from_type

        item_input = get_field_from_type(type_arg, init=True)
        if item_input is None:
            return None

        return cls._params_init(item_input=item_input)

    def __init_subclass__(cls) -> None:
        # Register the subclass with its item input.
        item_input_type = cls._item_input_type
        if item_input_type is not None:
            ListInput._type_to_class[item_input_type] = cls

        item_input_field = cls._item_input_field
        if item_input_field is not None:
            ListInput._field_to_class[item_input_field] = cls

        return super().__init_subclass__()

    def get_item_input(self):
        return self.params.item_input

    def modify_item_input(self, *args, **kwargs):
        return self.get_item_input().modify(*args, **kwargs)

    def parse(self, val):
        if val is None:
            return val

        parser = self.get_current_parser()

        if parser is not None:
            val = parser().parse(val)
            return [self.get_item_input().parse(v) for v in val]

        if not isiterable(val):
            self._raise_type_error(val)

        return val
