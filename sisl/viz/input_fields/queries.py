from collections import defaultdict

import pandas as pd

from .._input_field import InputField
from .dropdown import AtomSelect, SpeciesSelect, OrbitalsNameSelect, SpinSelect
from ..configurable import Configurable

class QueriesInput(InputField):

    '''
    Parameters
    ----------
    queryForm: list of InputField
        The list of input fields that conform a query.
    '''

    dtype = "array-like of dict"
    
    _type = 'queries'

    _default = {
        "width": "s100%",
        "queryForm": []
    }

    def __init__(self, queryForm = [], *args, **kwargs):

        inputFieldAttrs = {
            **kwargs.get("inputFieldAttrs", {}),
            "queryForm": self._sanitize_queryform(queryForm)
        }

        super().__init__(*args, **kwargs, inputFieldAttrs = inputFieldAttrs)
    
    def get_query_param(self, key, **kwargs):

        '''
        Gets the parameter info for a given key. It uses the Configurable.get_param method.
        '''

        return Configurable.get_param(self, key, paramsExtractor = lambda obj: obj.inputField["queryForm"], **kwargs)
    
    def get_param(self, *args, **kwargs):
        '''
        Just a clone of getQueryParam.

        Because Configurable looks for this method when modifying parameters, but the other name is clearer.
        '''

        return self.get_query_param(*args, **kwargs)

    def modify_query_param(self, key, *args, **kwargs):

        '''
        Uses Configurable.modify_param to modify a parameter inside QueryForm
        '''

        return Configurable.modify_param(self, key, *args, **kwargs)

    def complete_query(self, query, **kwargs):
        '''
        Completes a partially build query with the default values

        Parameters
        -----------
        query: dict
            the query to be completed.
        **kwargs:
            other keys that need to be added to the query IN CASE THEY DON'T ALREADY EXIST
        '''

        return {
            "active": True,
            **{param.key: param.default for param in self.inputField["queryForm"]},
            **kwargs,
            **query
        }
    
    def filter_df(self, df, query, key_to_cols, raise_not_active=False):
        '''
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
        '''

        query = self.complete_query(query)

        if raise_not_active:
            if not group["active"]:
                raise Exception(f"Query {query} is not active and you are trying to use it")
        
        query_df = df

        cond = None
        for key, col in key_to_cols:
            if query.get(key):
                query_df = query_df[query_df[col].isin(query[key])]

        return query_df

    def _sanitize_queryform(self, queryform):
        '''
        Parses a query form to fields, converting strings
        to the known input fields (under self._fields). As an example,
        see OrbitalQueries.
        '''

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

    def __getitem__(self, key):

        for field in self.inputField['queryForm']:
            if field.key == key:
                return field
        
        return super().__getitem__(key)

class OrbitalQueries(QueriesInput):
    '''
    This class implements an input field that allows you to select orbitals by atom, species, etc...
    '''

    _fields = {
        "species": {"field": SpeciesSelect, "name": "Species"},
        "atoms": {"field": AtomSelect, "name": "Atoms"},
        "orbitals": {"field": OrbitalsNameSelect, "name": "Orbitals"},
        "spin": {"field": SpinSelect, "name": "Spin"},
    }

    def _build_orb_filtering_df(self, geom):

        orb_props = defaultdict(list)
        #Loop over all orbitals of the basis
        for at, iorb in geom.iter_orbitals():

            atom = geom.atoms[at]
            orb = atom[iorb]

            orb_props["atom"].append(at)
            orb_props["species"].append(atom.symbol)
            orb_props["orbital name"].append(orb.name())
        
        self.orb_filtering_df = pd.DataFrame(orb_props)

    def update_options(self, geom, polarized=False):

        for key in ("species", "atoms", "orbitals"):
            try:
                self.get_query_param(key).update_options(geom)
            except KeyError:
                pass
        
        try:
            self.get_query_param('spin').update_options(polarized)
        except KeyError:
            pass

        self._build_orb_filtering_df(geom)

    def get_orbitals(self, request):

        filtered_df = self.filter_df(self.orb_filtering_df, request,
            [
                ("atoms", "atom"),
                ("species", "species"),
                ("orbitals", "orbital name"),
            ]
        )

        return filtered_df.index




        

