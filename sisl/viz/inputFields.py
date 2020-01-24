#This file defines all the currently available input fields so that it is easier to develop plots

from copy import deepcopy

from .configurable import Configurable
from .plotutils import modifyNestedDict

TEXT_INPUT = {
    "type": "textinput",
    "width": "s100%",
    "params": {
        "placeholder": "Write your value here...",
    }
}

SWITCH = {
    "type": "switch",
    "width": "s50% m30% l15%",
    "params": {
        "offLabel": "Off",
        "onLabel": "On"
    }
}

COLOR_PICKER = {
    "type": "color",
    "width": "s50% m30% l15%",
}

DROPDOWN = {
    "type": "dropdown",
    "width": "s100% m50% l33%",
    "params": {
        "placeholder": "Choose an option...",
        "options": [
        ],
        "isMulti": False,
        "isClearable": True,
        "isSearchable": True,
    }
}

NUMBER_INPUT = {
    "type": "number",
    "width": "s50% m30% l30%",
    "default": 0,
    "params": {
        "min": 0,
    }
}

INTEGER_INPUT = {
    **NUMBER_INPUT,
    "params": {
        **NUMBER_INPUT["params"],
        "step": 1
    }
}

FLOAT_INPUT = {
    **NUMBER_INPUT,
    "params": {
        **NUMBER_INPUT["params"],
        "step": 0.1
    }
}

RANGE_SLIDER = {
    "type": "rangeslider",
    "width": "s100%",
    "params": {
        "min": -10,
        "max": 10,
        "step": 0.1,
        "marks": { i: str(i) for i in range(-10,11) },
    }
}

QUERIES_INPUT = {
    "type": "queries",
    "width": "s100%",
    "queryForm": [
    ]
}

allInputs = {
    "text": TEXT_INPUT,
    "switch": SWITCH,
    "color": COLOR_PICKER,
    "dropdown": DROPDOWN,
    "integer": INTEGER_INPUT,
    "float": FLOAT_INPUT,
    "rangeslider": RANGE_SLIDER,
    "queries": QUERIES_INPUT
}

allowedTypes = allInputs.keys()

class InputField:

    '''
    Returns the input field that you ask for with the default values. 
    However, you can overwrite them.

    Parameters
    ----------
    key: str
        The key with which you will be able to access the value of the setting in your class
    name: str
        The name that you want to show for this setting in the GUI.
    default: optional (None)
        The default value for the setting. If it is not provided it will be None.
    inputType: str, optional {'text', 'switch', 'color', 'dropdown'}
        The type of input you want to retrieve. If you don't specify a type or it is an inexistent one,
        sisl will attempt to give it one according to the type of data of the default value
        If it doesn't succeed, no type will be asigned. Your parameter will still work but won't be showed in the GUI.

        This is how data types are mapped to inputFields: {
        
        }

        You may want to create settings without graphical interface for parameters that are too complex
        to modify by hand but can be useful for people that use your plot programatically.
    params: dict, optional
        A dictionary with parameters that you want to add to the params key of the input field.
        If a key is already in the defaults, your provided value will have preference.
    style: dict, optional
        A dictionary with parameters that you want to add to the style key of the input field.
        If a key is already in the defaults, your provided value will have preference.

        The keys inside style determine the aesthetical appearance of the input field. This is passed directly
        to the style property of the container of the input field. Therefore, one could pass any react CSS key.

        If you don't know what CSS is, don't worry, it's easy and cool. The only thing that you need to know is
        that the style dictionary contains keys that determine how something looks in the web. For example:

         {backgroundColor: "red", padding: 30}

        would render a container with red background color and a padding of 30px.

        This links provide info on:
            - What CSS is: https://www.youtube.com/watch?v=4BEyFVufmM8&list=PL4cUxeGkcC9gQeDH6xYhmO-db2mhoTSrT&index=2
            - React CSS examples (you can play with them): https://www.w3schools.com/react/react_css.asp
        
        Just remember that you want to pass REACT CSS keys, NOT CSS. Basically the difference is that "-" are replaced by
        capital letters:
            Normal CSS: {font-size: 10}         React CSS: {fontSize: 10}

        You probably won't need to style anything and the defaults are good enough, but we still give this option for more flexibility.
    width: str, optional
        A string that determines the width of the input field. Although this is a "style" property, it is set apart because it is special.

        It looks like this: s100% m50% l30%. In this way, you are specifying the width of your parameter if it is displayed in a small
        ("s", phones), medium ("m", tablets) or large ("l", computers) screen, so that your input doesn't look ugly in any screen. 

        Percentages are relative to the full width of the settings container (see it in the GUI).
    inputFieldAttrs: dict, optional
        A dictionary with additional keys that you want to add to the inputField dictionary.
    group: str, optional
        Group of parameters to which the parameter belongs
    subGroup: str, optional
        If the setting belongs to a group, the subgroup it is in (if any).
    help: str, optional
        Help message to guide the user on what the parameter does. They will appear as tooltips in the GUI.

        Supports html tags, so one can write <br> to generate a new line or <a>mylink</a> to display a link, for example.

        This parameter is optional but extremely adviseable.
    **kwargs:
        All keyword arguments passed will be added to the parameter, overwriting any existing value in case there is one.
    
    '''

    def __init__(self, key, name, default = None, inputType = False, params = {}, style = {}, width = "", inputFieldAttrs = {}, group = None, subGroup = None, **kwargs):

        setattr(self, "key", key)
        setattr(self, "name", name)
        setattr(self, "default", default)
        setattr(self, "group", group)
        setattr(self, "subGroup", subGroup)

        if inputType:

            defaultInp = deepcopy(allInputs.get(inputType, False))

            setattr(self, "inputField", {
                **defaultInp,
                "params": {
                    **defaultInp.get("params", {}),
                    **params
                },
                "style": {
                    **defaultInp.get("style", {}),
                    **style
                },
                "width": width or defaultInp.get("width"),
                **inputFieldAttrs
            })
        
        for key, value in kwargs.items():

            setattr(self, key, value)

    def modify(self, *args):
        '''
        Modifies the parameter.
        
        See *args to know how can it be used.

        This is a general schema of how an input field parameter looks internally, so that you
        can know what do you want to change:

        (Note that it is very easy to modify nested values, more on this in *args explanation)

        {
            "key": whatever,
            "name": whatever,
            "default": whatever,
            .
            . (keys that affect, let's say, the programmatic functionality of the parameter,
            . they can be modified with Configurable.modifyParam)
            .
            "inputField": {
                "type": whatever,
                "width": whatever,  (keys that affect the inputField control that is displayed
                "params": {          they can be modified with Configurable.modifyInputField)
                    whatever
                },
                "style": {
                    whatever
                }

            }
        }

        Arguments
        --------
        *args:
            Depending on what you pass the setting will be modified in different ways:
                - Two arguments:
                    the first argument will be interpreted as the attribute that you want to change,
                    and the second one as the value that you want to set.

                    Ex: obj.modifyParam("length", "default", 3)
                    will set the default attribute of the parameter with key "length" to 3

                    Modifying nested keys is possible using dot notation.

                    Ex: obj.modifyParam("length", "inputField.width", 3)
                    will modify the width key inside inputField on the schema above.

                    The last key, but only the last one, will be created if it does not exist.
                    
                    Ex: obj.modifyParam("length", "inputField.width.inWinter.duringDay", 3)
                    will only work if all the path before duringDay exists and the value of inWinter is a dictionary.

                    Otherwise you could go like this: obj.modifyParam("length", "inputField.width.inWinter", {"duringDay": 3})

                - One argument and it is a dictionary:
                    the keys will be interpreted as attributes that you want to change and the values
                    as the value that you want them to have.

                    Each key-value pair in the dictionary will be updated in exactly the same way as
                    it is in the previous case.
                
                - One argument and it is a function:

                    the function will recieve the parameter and can act on it in any way you like.
                    It doesn't need to return the parameter, just modify it.
                    In this function, you can call predefined methods of the parameter, for example.

                    Ex: obj.modifyParam("length", lambda param: param.incrementByOne() )

                    given that you know that this type of parameter has this method.

        Returns
        --------
        self:
            The configurable object.
        '''
                
        if len(args) == 2:
           
            modFunction = lambda obj: modifyNestedDict( obj.__dict__, *args)

        elif isinstance(args[0], dict):

            def modFunction(obj):
                for attr, val in args[0].items():
                    modifyNestedDict( obj.__dict__, attr, val)

        elif callable(args[0]):

            modFunction = args[0]

        modFunction(self)

        return self

    def to_json(self):

        return self.__dict__

class TextInput(InputField):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs, inputType = "text")

class SwitchInput(InputField):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs, inputType = "switch")

class ColorPicker(InputField):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs, inputType = "color")

class DropdownInput(InputField):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs, inputType = "dropdown")

class IntegerInput(InputField):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs, inputType = "integer")

class FloatInput(InputField):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs, inputType = "float")

class RangeSlider(InputField):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs, inputType = "rangeslider")

class QueriesInput(InputField):

    '''
    Parameters
    ----------
    queryForm: list of InputField
        The list of input fields that conform a query.
    '''

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