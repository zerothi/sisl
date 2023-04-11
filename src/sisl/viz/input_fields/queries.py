# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from .basic import ListInput, DictInput


class QueriesInput(ListInput):
    """
    Parameters
    ----------
    queryForm: list of InputField
        The list of input fields that conform a query.
    """

    dtype = "array-like of dict"

    _dict_input = DictInput

    _default = {}

    def __init__(self, *args, queryForm=[], help="", params={}, **kwargs):

        query_form = self._sanitize_queryform(queryForm)

        self._dict_param = self._dict_input(key="", name="", fields=query_form)

        params = {
            "sortable": True,
            "itemInput": self._dict_param,
            **params,
        }

        input_field_attrs = {
            **kwargs.get("input_field_attrs", {}),
        }

        help += f"\n\n Each item is a dict. {self._dict_param.help}"

        super().__init__(*args, **kwargs, help=help, params=params, input_field_attrs=input_field_attrs)

    def get_query_param(self, key, **kwargs):
        """Gets the parameter info for a given key."""
        return self._dict_param.get_param(key, **kwargs)

    def get_param(self, *args, **kwargs):
        """
        Just a clone of get_query_param.

        Because Configurable looks for this method when modifying parameters, but the other name is clearer.
        """
        return self.get_query_param(*args, **kwargs)

    def modify_query_param(self, key, *args, **kwargs):
        """
        Uses Configurable.modify_param to modify a parameter inside QueryForm
        """
        return self._dict_param.modify_param(self, key, *args, **kwargs)

    def complete_query(self, query, **kwargs):
        """
        Completes a partially build query with the default values

        Parameters
        -----------
        query: dict
            the query to be completed.
        **kwargs:
            other keys that need to be added to the query IN CASE THEY DON'T ALREADY EXIST
        """
        return {
            "active": True,
            **self._dict_param.complete_dict(query, **kwargs),
        }

    def filter_df(self, df, query, key_to_cols, raise_not_active=False):
        """
        Filters a dataframe according to a query

        Parameters
        -----------
        df: pd.DataFrame
            the dataframe to filter.
        query: dict
            the query to be used as a filter. Can be incomplete, it will be completed using
            `self.complete_query()`
        keys_to_cols: array-like of tuples
            An array of tuples that look like (key, col)
            where key is the key of the parameter in the query and col the corresponding
            column in the dataframe.
        """
        query = self.complete_query(query)

        if raise_not_active:
            if not query["active"]:
                raise ValueError(f"Query {query} is not active and you are trying to use it")

        query_str = []
        for key, val in query.items():
            key = key_to_cols.get(key, key)
            if key in df and val is not None:
                if isinstance(val, (np.ndarray, tuple)):
                    val = np.ravel(val).tolist()
                query_str.append(f'{key}=={repr(val)}')

        return df.query(" & ".join(query_str))

    def _sanitize_queryform(self, queryform):
        """
        Parses a query form to fields, converting strings
        to the known input fields (under self._fields). As an example,
        see OrbitalQueries.
        """
        sanitized_form = []
        for i, field in enumerate(queryform):
            if isinstance(field, str):
                if field not in self._fields:
                    raise KeyError(
                        f"{self.__class__.__name__} has no pre-built field for '{field}'")

                built_field = self._fields[field]['field'](
                    key=field, **{key: val for key, val in self._fields[field].items() if key != 'field'}
                )

                sanitized_form.append(built_field)
            else:
                sanitized_form.append(field)

        return sanitized_form

    def parse(self, val):
        if isinstance(val, dict):
            val = [val]

        return super().parse(val)

    def __getitem__(self, key):
        try:
            return self._dict_param.get_param(key)
        except KeyError:
            return super().__getitem__(key)

    def __contains__(self, key):
        return self._dict_param.__contains__(key)
