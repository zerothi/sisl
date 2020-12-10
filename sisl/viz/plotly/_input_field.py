#This file defines all the currently available input fields so that it is easier to develop plots
from copy import deepcopy, copy

import json
import numpy as np

from .plotutils import modify_nested_dict, get_nested_key

__all__ = ["InputField"]


class InputField:
    """ This class is meant to help a smooth interface between python and the GUI.

    A class that inherits from Configurable should have all its settings defined as `ÌnputField`s. In
    this way, the GUI will know how to display an input field to let the user interact with the 
    settings plot.

    This is just the base class of all input fields. Each type of field has its own class. The most
    simple one is `TextInput`, which just renders an input of type text in the GUI.

    Input fields also help documenting the class and parsing the user's input to normalize it (with
    the `parse` method.).

    Finally, since all input fields are copied to the instance, the classes that define input fields
    can have methods that help making your life easier. See the `OrbitalQueries` input field for an example
    of that. In that case, the input field can update its own options based on a geometry that is passed.

    Parameters
    ----------
    key: str
        The key with which you will be able to access the value of the setting in your class
    name: str
        The name that you want to show for this setting in the GUI.
    default: optional (None)
        The default value for the setting. If it is not provided it will be None.
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
    """

    dtype = None

    def __init__(self, key, name, default=None, params={}, style={}, width="", inputFieldAttrs={}, group=None, subGroup=None, dtype=None, help="", **kwargs):
        self.key = key
        self.name = name
        self.default = default
        self.group = group
        self.subGroup = subGroup
        self.help = help

        if dtype is not None:
            self.dtype = dtype

        default_input = deepcopy(getattr(self, "_default", {}))

        setattr(self, "inputField", {
            'type': copy(getattr(self, '_type', None)),
            **default_input,
            "params": {
                **default_input.get("params", {}),
                **params
            },
            "style": {
                **default_input.get("style", {}),
                **style
            },
            "width": width or default_input.get("width"),
            **inputFieldAttrs
        })

        for key, value in kwargs.items():

            setattr(self, key, value)

    def __getitem__(self, key):
        """ Gets a key from the input field, even if it is nested """
        if isinstance(key, str):
            return get_nested_key(self.__dict__, key)

        return None

    def __str__(self):
        """ String representation of the structure of the input field """
        return str(vars(self))

    def __repr__(self):
        """ String representation of the structure of the input field """
        return self.__str__()

    def modify(self, *args):
        """ Modifies the parameter

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
            . they can be modified with Configurable.modify_param)
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

                    Ex: obj.modify_param("length", "default", 3)
                    will set the default attribute of the parameter with key "length" to 3

                    Modifying nested keys is possible using dot notation.

                    Ex: obj.modify_param("length", "inputField.width", 3)
                    will modify the width key inside inputField on the schema above.

                    The last key, but only the last one, will be created if it does not exist.

                    Ex: obj.modify_param("length", "inputField.width.inWinter.duringDay", 3)
                    will only work if all the path before duringDay exists and the value of inWinter is a dictionary.

                    Otherwise you could go like this: obj.modify_param("length", "inputField.width.inWinter", {"duringDay": 3})

                - One argument and it is a dictionary:
                    the keys will be interpreted as attributes that you want to change and the values
                    as the value that you want them to have.

                    Each key-value pair in the dictionary will be updated in exactly the same way as
                    it is in the previous case.

                - One argument and it is a function:

                    the function will recieve the parameter and can act on it in any way you like.
                    It doesn't need to return the parameter, just modify it.
                    In this function, you can call predefined methods of the parameter, for example.

                    Ex: obj.modify_param("length", lambda param: param.incrementByOne() )

                    given that you know that this type of parameter has this method.

        Returns
        --------
        self:
            The configurable object.
        """
        if len(args) == 2:

            modFunction = lambda obj: modify_nested_dict(obj.__dict__, *args)

        elif isinstance(args[0], dict):

            def modFunction(obj):
                for attr, val in args[0].items():
                    modify_nested_dict(obj.__dict__, attr, val)

        elif callable(args[0]):

            modFunction = args[0]

        modFunction(self)

        return self

    def to_json(self):
        """ Helps converting the input field to json so that it can be sent to the GUI

        Returns
        ---------
        dict
            the dict ready to be jsonified.
        """
        def default(obj):

            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            else:
                return getattr(obj, '__dict__', str(obj))

        return json.loads(
            json.dumps(self, default=default)
        )

    def parse(self, val):
        """ Parses the user input to the actual values that will be used

        This method may be overwritten, but you probably need to still call
        it with `super().parse(val)`, because it implements the basic functionality
        for the `splot` commmand to understand the values that receives.

        Parameters
        -----------
        val: any
            the value to parse

        Returns
        -----------
        self.dtype
            the parsed value, which will be of the datatype specified by the input.
        """
        if val is None:
            return None

        dtypes = self.dtype

        if dtypes is None:
            return val

        if not isinstance(dtypes, tuple):
            dtypes = (dtypes, )

        for dtype in dtypes:
            try:
                if dtype == bool and isinstance(val, str):
                    val = val.lower() not in ('false', 'f', 'no', 'n')
                elif dtype in [list, int, float]:
                    val = dtype(val)
            except:
                continue

        return val

    def _get_docstring(self):
        """ Generates the docstring for this input field """
        import textwrap

        valid_vals = getattr(self, "valid_vals", None)

        if valid_vals is None:

            dtypes = getattr(self, "dtype", "")
            if dtypes is None:
                dtypes = ""

            if not isinstance(dtypes, tuple):
                dtypes = (dtypes,)

            vals_help = " or ".join([getattr(dtype, "__name__", str(dtype)) for dtype in dtypes])

        else:
            vals_help = '{' + ', '.join(valid_vals) + '}'

        help_message = getattr(self, "help", "")
        tw = textwrap.TextWrapper(width=70, initial_indent="\t", subsequent_indent="\t")
        help_message = tw.fill(help_message)

        doc = f'{self.key}: {vals_help}{"," if vals_help else ""} optional\n{help_message}'

        return doc
