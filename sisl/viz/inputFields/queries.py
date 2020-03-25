from ..inputField import InputField
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

        super().__init__(*args, **kwargs, inputType = "queries", inputFieldAttrs = inputFieldAttrs)
    
    def getQueryParam(self, key, **kwargs):

        '''
        Gets the parameter info for a given key. It uses the Configurable.getParam method.
        '''

        return Configurable.getParam(self, key, paramsExtractor = lambda obj: obj.inputField["queryForm"], **kwargs)
    
    def getParam(self, *args, **kwargs):
        '''
        Just a clone of getQueryParam.

        Because Configurable looks for this method when modifying parameters, but the other name is clearer.
        '''

        return self.getQueryParam(*args, **kwargs)

    def modifyQueryParam(self, key, *args, **kwargs):

        '''
        Uses Configurable.modifyParam to modify a parameter inside QueryForm
        '''

        return Configurable.modifyParam(self, key, *args, **kwargs)