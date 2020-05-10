from .._input_field import InputField
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
            "queryForm": queryForm 
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

