# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import typing

from .._input_field import InputField, InputParams

@dataclass
class DictInputParams(InputParams):
    """These are the parameters that any implementation of DictInput should use.
    
    Parameters
    ----------
    fields: sequence of InputField
        List of all the input fields that are contained in the dictionary.
        Each input field can be of any type.
    """
    fields: typing.Dict[str, InputField] = field(default_factory=dict)

class DictInput(InputField):
    """Input field for a dictionary.

    GUI indications
    ---------------
    This input field is just a container for key-value pairs
    of other inputs. Despite its simplicity, it's not trivial
    to implement. One must have all the other input fields implemented
    in a very modular way for `DictInput` to come up naturally. Otherwise
    it can get complicated.
    """
    params: DictInputParams

    # Function to be used to itialize the dictionary. It will receive key-value
    # pairs as keyword arguments.
    dict_init: typing.Callable

    _fields = {}

    # Variables that help parsing from a generic dict type to the
    # specific input field designed for it.
    _type_to_class: typing.Dict[type, type[DictInput]] = {}
    # In a subclass of DictInput, one must define the type of dictionary
    # that should be parsed into that particular input field.
    _dict_type: type | None = None

    def __init__(self, params: DictInputParams | None = None, dict_init: typing.Callable = dict):
        self.dict_init = dict_init

        super().__init__(params=params)

    @classmethod
    def from_typehint(cls, type_: type):
        # If this is the base DictInput, find whether there is a more specific
        # input for this type of dictionaries.
        kls = cls
        if cls is DictInput:
            kls = cls._type_to_class.get(type_)
            if kls is None:
                kls = cls
        
        # Now that we've found the exact input field class, initialize the parameters
        params = kls._init_params_from_type(type_)
        
        # And return the new input field.
        return kls(params=params, dict_init=type_)

    @classmethod
    def _init_params_from_type(cls, type_: type):
        from .._typetofield import get_field_from_type, get_function_fields

        if is_dataclass(type_):
            # If this is a dataclass, we get the information from the signature,
            # in this way fields have information about the defaults, etc...
            fields = get_function_fields(type_)
        else:
            field_hints = typing.get_type_hints(type_)

            fields = {
                k: get_field_from_type(v) for k, v in field_hints.items()
            }

        return cls._params_init(fields=fields)

    def __init_subclass__(cls) -> None:
        # Register the input field subclass with the type that is
        # specific to it.
        dict_type = cls._dict_type
        if dict_type is not None:
            DictInput._type_to_class[dict_type] = cls

        return super().__init_subclass__()

    @property
    def fields(self):
        return self.params.fields

    def complete_dict(self, query, as_dict: bool = False, **kwargs):
        """Completes a partially build dictionary with the missing fields.

        Parameters
        -----------
        query: dict
            the query to be completed.
        as_dict: bool, optional
            Whether we should ensure that the returned object is a dictionary.
            Basically, if the initialization of the dict returns a dataclass, it will be
            converted to a dictionary.
        **kwargs:
            other keys that need to be added to the query IN CASE THEY DON'T ALREADY EXIST
        """
        returns = self.dict_init(**{
            **{k: field.params.default for k, field in self.params.fields.items()},
            **kwargs,
            **query
        })

        if as_dict:
            if is_dataclass(returns):
                returns = asdict(returns)
        
        return returns
            
    def parse(self, val):
        parser = self.get_current_parser()

        if parser is not None:
            val = parser().parse(val)
            val = {**val}
            for key, field in self.params.fields.items():
                print(key, field)
                if key in val and field is not None:
                    val[key] = field.parse(val[key])

        if not isinstance(val, dict):
            self._raise_type_error(val)

        return val

    def __getitem__(self, key):
        if key in self.params.fields:
            return self.params.fields[key]

        return super().__getitem__(key)

    def __contains__(self, key):
        return key in self.params.fields


class CreatableDictInput(DictInput):
    """Input field for a dictionary for which entries can be created and removed.

    GUI indications
    ---------------
    This input is a bit trickier than `DictInput`. It should be possible to remove
    and add the fields as you wish.
    """

    _type = "creatable dict"
