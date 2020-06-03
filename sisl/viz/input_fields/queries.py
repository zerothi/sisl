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
    
    def _generate_queries(self, on, only=None, exclude=None, clean=True, query_gen=None, **kwargs):
        '''
        Automatically generates queries based on the current options.

        Parameters
        --------
        on: str, {"species", "atoms", "orbitals", "spin"}
            the parameter to split along
        only: array-like, optional
            if desired, the only values that should be plotted out of
            all of the values that come from the splitting.
        exclude: array-like, optional
            values that should not be plotted
        query_gen: function, optional
            the request generator. It is a function that takes all the parameters for each
            request that this method has come up with and gets a chance to do some modifications.

            This may be useful, for example, to give each request a color, or a custom name.
        **kwargs:
            keyword arguments that go directly to each request.
            
            This is useful to add extra filters. For example:
            `plot._generate_requests(on="orbitals", species=["C"])`
            will split the PDOS on the different orbitals but will take
            only those that belong to carbon atoms.
        '''

        if exclude is None:
            exclude = []

        # First, we get all available values for the parameter we want to split
        options = self.get_param(on)["inputField.params.options"]

        # If the parameter is spin but the orbitals are not polarized we will not be providing
        # options to the user, but in fact there is one option: 0
        if on == "spin" and len(options) == 0:
            options = [{"label": 0, "value": 0}]

        # If no function to modify requests was provided we are just going to generate a 
        # dummy one that just returns the request as it gets it
        if query_gen is None:
            def query_gen(**kwargs):
                return kwargs

        # Build all the requests that will be passed to the settings of the plot
        requests = [
            query_gen(
                **{on: [option["value"]], "name": option["label"], **kwargs})
            for option in options if option["value"] not in exclude and (only is None or option["value"] in only)
        ]

        return requests




        

