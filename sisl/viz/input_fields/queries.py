from .._input_field import InputField
from ..configurable import Configurable

class QueriesInput(InputField):

    '''
    Parameters
    ----------
    queryForm: list of InputField
        The list of input fields that conform a query.
    '''

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
