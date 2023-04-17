# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ...configurable import Configurable
from ..._input_field import InputField


class DictInput(InputField):
    """Input field for a dictionary.

    GUI indications
    ---------------
    This input field is just a container for key-value pairs
    of other inputs. Despite its simplicity, it's not trivial
    to implement. One must have all the other input fields implemented
    in a very modular way for `DictInput` to come up naturally. Otherwise
    it can get complicated.


    `param.inputField["params"]["fields"]` contains a list of all the input
    fields that are contained in the dictionary. Each input field can be of
    any type.
    """

    dtype = dict

    _type = 'dict'

    _fields = []

    _default = {}

    @property
    def fields(self):
        return self.inputField["fields"]

    def __init__(self, *args, fields=[], help="", **kwargs):

        fields = self._sanitize_fields(fields)

        input_field_attrs = {
            **kwargs.pop("input_field_attrs", {}),
            "fields": fields,
        }

        def get_fields_help():
            return "\n\t".join([f"'{param.key}': {param.help}" for param in fields])

        help += "\n\n Structure of the dict: {\n\t" + get_fields_help() + "\n}"

        super().__init__(*args, **kwargs, help=help, input_field_attrs=input_field_attrs)

    def _sanitize_fields(self, fields):
        """Parses the fields, converting strings to the known input fields (under self._fields)."""
        sanitized_fields = []
        for i, field in enumerate(fields):
            if isinstance(field, str):
                if field not in self._fields:
                    raise KeyError(
                        f"{self.__class__.__name__} has no pre-built field for '{field}'")

                built_field = self._fields[field]['field'](
                    key=field, **{key: val for key, val in self._fields[field].items() if key != 'field'}
                )

                sanitized_fields.append(built_field)
            else:
                sanitized_fields.append(field)

        return sanitized_fields

    def get_param(self, key, **kwargs):
        """Gets a parameter from the fields of this dictionary."""
        return Configurable.get_param(
            self, key, params_extractor=lambda obj: obj.inputField["fields"], **kwargs
        )

    def modify_param(self, key, *args, **kwargs):
        """Modifies a parameter from the fields of this dictionary."""
        return Configurable.modify_param(self, key, *args, **kwargs)

    def complete_dict(self, query, **kwargs):
        """Completes a partially build dictionary with the missing fields.

        Parameters
        -----------
        query: dict
            the query to be completed.
        **kwargs:
            other keys that need to be added to the query IN CASE THEY DON'T ALREADY EXIST
        """
        return {
            **{param.key: param.default for param in self.fields},
            **kwargs,
            **query
        }

    def parse(self, val):
        if val is None:
            val = {}
        if not isinstance(val, dict):
            self._raise_type_error(val)

        val = {**val}
        for field in self.fields:
            if field.key in val:
                val[field.key] = field.parse(val[field.key])

        return val

    def __getitem__(self, key):
        for field in self.inputField['fields']:
            if field.key == key:
                return field

        return super().__getitem__(key)

    def __contains__(self, key):

        for field in self.inputField['fields']:
            if field.key == key:
                return True

        return False


class CreatableDictInput(DictInput):
    """Input field for a dictionary for which entries can be created and removed.

    GUI indications
    ---------------
    This input is a bit trickier than `DictInput`. It should be possible to remove
    and add the fields as you wish.
    """

    _type = "creatable dict"
