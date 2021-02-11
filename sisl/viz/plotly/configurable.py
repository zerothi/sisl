from copy import copy, deepcopy
from functools import wraps
import inspect
from types import MethodType
from collections import deque, defaultdict
from collections.abc import Iterable
import sys

import numpy as np

from sisl.messages import info
from sisl._dispatcher import AbstractDispatch
from ._presets import get_preset
from .plotutils import get_configurable_docstring, get_configurable_kwargs, get_configurable_kwargs_to_pass

__all__ = ["Configurable", "vizplotly_settings"]


class NamedHistory:
    """ Useful for tracking and modifying the history of named parameters

    This is useful to keep track of how a dict changes, for example.

    Parameters
    ----------
    init_params: dict
        The initial values for the parameters.

        If defaults are not provided, this will be treated as "defaults" (only if keep_defaults is true!)
    defaults: dict, optional
       the default values for each parameter. In case some parameter is missing in `init_params`
       it will be initialized with the default value.

       This will also be used to restore settings to defaults.
    history_len: int, optional
        how much steps of history should be recorded
    keep_defaults: boolean, optional
        whether the defaults should be kept in case you want to restore them.

    Attributes
    ----------
    current : dict
       the current values for the parameters
    """

    def __init__(self, init_params, defaults=None, history_len=20, keep_defaults=True):
        self._defaults_kept = keep_defaults

        if defaults is not None:
            if keep_defaults:
                self._defaults = defaults

        # This makes it easier to restore the parameters
        if hasattr(self, "_defaults"):
            init_params = {**self._defaults, **init_params}

        # Vals will contain the unique values for each parameter
        self._vals = {key: [val] for key, val in init_params.items()}

        # And _hist will just hold params
        self._hist = {key: deque([0], maxlen=history_len) for key in init_params}

    def __str__(self):
        """ str of the object """
        return self.__class__.__name__ + f"{{history: {self._hist}, parameters={list(self._vals.keys())}}}"

    @property
    def current(self):
        """ The current state of the history """
        return self.step(-1)

    def step(self, i):
        """ Retrieves a given step of the history

        Parameters
        -----------
        i: int
            the index of the step that you want. It can be negative.

        Returns
        -----------
        dict
            The key-value pairs for the given step
        """
        return {key: self._vals[key][hist[i]] for key, hist in self._hist.items()}

    def __len__(self):
        """ Returns the number of steps stored in the history """
        # TODO is this really what you want?
        # Needs clarification, in any case len(self._hist.values()[0]) would
        # clarify that you don't really care...?
        for _, hist in self._hist.items():
            return len(hist)

    def __getitem__(self, item):
        if isinstance(item, int):
            # If an int is provided, we get that step of the history
            return self.step(item)
        elif isinstance(item, str):
            # If a string is provided, we get the full history of a given setting
            return [self._vals[item][i] for i in self._hist[item]]
        elif isinstance(item, Iterable):
            try:
                # If it's an array-like of strings, we will get the full history of each key
                return {key: np.array(self._vals[key])[self._hist[key]] for key in item}
            except:
                # Otherwise, we just map the array with __getitem__()
                return [self.__getitem__(i) for i in item]
        elif isinstance(item, slice):
            # Finally, if it's a slice, we will get the steps that are within that slice.
            return {key: np.array(self._vals[key])[hist][item] for key, hist in self._hist.items()}

    def __contains__(self, item):
        """ Check if we are storing that named item """
        return item in self._vals

    def update(self, **new_settings):
        """ Updates the history by appending a new step

        Parameters
        ----------
        **new_settings:
            all the settings that you want to change passed as keyword arguments.

            You don't need to provide the values for all parameters, only those that you
            need to change.

        Returns
        -------
        self
        """
        for key in self._vals:
            if key not in new_settings:
                new_index = self._hist[key][-1]

            else:
                # Check if we already have that value
                val = new_settings[key]
                is_nparray = isinstance(val, np.ndarray)

                # We have to do this because np.arrays don't like being compared :)
                # Otherwise we would just do if val in self._vals[key]
                for i, saved_val in enumerate(self._vals[key]):
                    if not isinstance(saved_val, np.ndarray) and not is_nparray:
                        try:
                            if val == saved_val:
                                new_index = i
                                break
                        except ValueError:
                            # It is possible that the value itself is not a numpy array
                            # but contains one. This is very hard to handle
                            pass
                else:
                    self._vals[key].append(val)
                    new_index = len(self._vals[key]) - 1

            # Append the index to the history
            self._hist[key].append(new_index)

        return self

    @property
    def last_updated(self):
        """ The names of the parameters that were changed in the last update """
        return self.updated_params(-1)

    def last_update_for(self, key):
        """ Returns the index of the last update for a given parameter

        Parameters
        -----------
        key: str
            the parameter we want the last update for.

        Returns
        -----------
        int or None
            the index of the last update. 
            If the parameter was never updated it returns None.
        """
        current = self._vals[key][self._hist[key][-1]]

        for i, val in enumerate(reversed(self._hist[key])):
            if val != current:
                return len(self._hist[key]) - (i+1)

    def updated_params(self, step):
        """ Gets the keys of the parameters that were updated in a given step

        Parameters
        -----------
        step: int
            the index of the step that you want to check.

        Returns
        -----------
        list of str
            the list of parameters that were updated at that step
        """
        return self.diff_keys(step, step - 1)

    def was_updated(self, key, step=-1):
        """ Checks whether a given step updated the parameter's value

        Parameters
        -----------
        key: str
            the name of the parameter that you want to check.
        step: int
            the index of the step we want to check

        Returns
        -----------
        bool
            whether the parameter was updated or not
        """
        return self.is_different(key, step1=step, step2=step-1)

    def is_different(self, key, step1, step2):
        """ Checks if a parameter has a different value between two steps

        The steps DO NOT need to be consecutive.

        Parameters
        -----------
        key:str
            the name of the parameter.
        step1 and step2: int
            the indices of the two steps that you want to check.

        Returns
        -----------
        bool
            whether the value of the parameter is different in these
            two steps
        """
        hist = self._hist[key]
        return hist[step1] != hist[step2]

    def diff_keys(self, step1, step2):
        """ Gets the keys that are different between two steps of the history

        The steps DO NOT need to be consecutive.

        Parameters
        -----------
        step1 and step2: int
            the indices of the two steps that you want to check.

        Returns
        -----------
        list of str
            the names of the parameters that are different between these two steps.
        """
        return [key for key in self._vals if self.is_different(key, step1, step2)]

    def delta(self, step_after=-1, step_before=None):
        """ Gets a dictionary with the diferences between two steps

        Parameters
        -----------
        step_after: int, optional
            the step that is considered as "after" in the delta log.
        step_before: int, optional
            the step that is considered as "before" in the delta log.

            If not provided, it will just be one step previous to `step_after`

        Returns
        -----------
        dict
            a dictionary containing, for each CHANGED key, the values before
            and after.

            The dict will not contain the parameters that were not changed.
        """
        if step_before is None:
            step_before = step_after -1

        keys = self.diff_keys(step_before, step_after)
        return {
            key: {
                "before": self[step_before][key],
                "after": self[step_after][key],
            } for key in keys
        }

    @property
    def last_delta(self):
        """ A log with the last changes

        See `delta` for more information.
        """
        return self.delta(-1, -2)

    def undo(self, steps=1):
        """ Takes the history back a number of steps

        Currently, this is irreversible, there is no "redo".

        Parameters
        -----------
        steps: int, optional
            the number of steps that you want to move the history back.

        Returns
        -----------
        self
        """
        # Decide which keys to undo.
        # This can not be done yet because all keys
        # are supposed to have the same number of steps.
        # if only is not None:
        #     keys = only
        # else:
        #     keys = self._vals.keys()
        #     if exclude is not None:
        #         keys = [ key for key in keys if key not in exclude]

        for hist in self._hist.values():
            for _ in range(steps):
                hist.pop()

        # Clear the unused values (for the moment we are setting them to None
        # so that we don't need to change the indices of the history)
        for key in self._vals:
            if self._defaults_kept:
                self._vals[key] = [val if i in hist or i==0 else None for i, val in enumerate(self._vals[key])]
            else:
                self._vals[key] = [val if i in hist else None for i, val in enumerate(self._vals[key])]

        return self

    def clear(self):
        """ Clears the history.

        It sets all settings to `None`. If you want to restore the defaults,
        use `restore_defaults` instead.

        """
        self.__init__(init_settings={key: None for key in self._vals})
        return self

    def restore_initial(self):
        """ Restores the history to its initial values (the first step) """
        self.__init__(init_settings=self.step(0))
        return self

    def restore_defaults(self):
        """ Restores the history to its defaults """
        if self._defaults_kept:
            if hasattr(self, "_defaults"):
                self.__init__({})
            else:
                self.restore_initial()
        else:
            raise RuntimeError("Defaults were not kept! You need to use keep_defaults=True on initialization")

        return self

    @property
    def defaults(self):
        """ The default values for this history """
        if self._defaults_kept:
            if hasattr(self, "_defaults"):
                return self._defaults
            else:
                return self.step(0)
        else:
            raise RuntimeError("Defaults were not kept! You need to use keep_defaults=True on initialization")

        return self

    def is_default(self, key):
        """ Checks if a parameter currently has the default value

        Parameters
        -----------
        key: str
            the parameter that you want to check.

        Returns
        -----------
        bool
            whether the parameter currently holds the default value.
        """
        return self[key][-1] == self.defaults[key]


class ConfigurableMeta(type):
    """ Metaclass used to build the Configurable class and its childs.

    This is used mainly for two reasons, and they both affect only subclasses of Configurable
    not Configurable itself.:
        - Make the class functions able to access settings through their arguments
        (see the `_populate_with_settings` function in this same file)
        - Set documentation to the `update_settings` method that is specific to the particular class
        so that the user can check what each parameter does exactly.
    """

    def __new__(cls, name, bases, attrs):
        """Prepares a subclass of Configurable, as explained in this class' docstring."""
        # If there are no bases, it is the Configurable class, and we don't need to modify its methods.
        if bases:
            # If this is a sub class
            class_params = attrs.get("_parameters", [])
            for base in bases:
                if "_parameters" in vars(base):
                    class_params = [*class_params, *base._parameters]
            for f_name, f in attrs.items():
                if callable(f) and not f_name.startswith("__"):
                    attrs[f_name] = _populate_with_settings(f, [param["key"] for param in class_params])
        new_cls = super().__new__(cls, name, bases, attrs)

        new_cls._create_update_maps()

        if bases:
            # Change the docs of the update_settings method to truly reflect
            # the available kwargs for the plot class and provide more help to the user
            def update_settings(self, *args, **kwargs):
                return self._update_settings(*args, **kwargs)

            update_settings.__doc__ = f"Updates the settings of this plot.\n\nDocs for {new_cls.__name__}:\n\n{get_configurable_docstring(new_cls)}"
            new_cls.update_settings = update_settings

        return new_cls


class Configurable(metaclass=ConfigurableMeta):

    def init_settings(self, presets=None, **kwargs):
        """
        Initializes the settings for the object.

        Parameters
        -----------
        presets: str or array-like of str
            all the presets that you want to use.
            Note that you can register new presets using `sisl.viz.plotly.add_preset`
        **kwargs:
            the values of the settings passed as keyword arguments.

            If a setting is not provided, the default value will be used.
        """
        # If the class needs to overwrite some defaults of settings that has inherited, do it
        overwrite_defaults = getattr(self, "_overwrite_defaults", {})
        for key, val in overwrite_defaults.items():
            if key not in kwargs:
                kwargs[key] = val

        #Get the parameters of all the classes the object belongs to
        self.params, self.param_groups = self._get_class_params()

        if presets is not None:
            if isinstance(presets, str):
                presets = [presets]

            for preset in presets:
                preset_settings = get_preset(preset)
                kwargs = {**preset_settings, **kwargs}

        # Define the settings dictionary, taking the value of each parameter from kwargs if it is there or from the defaults otherwise.
        # And initialize the settings history
        defaults = {param.key: deepcopy(param.default) for param in self.params}
        self.settings_history = NamedHistory(
            {key: kwargs.get(key, val) for key, val in defaults.items()},
            defaults=defaults, history_len=20, keep_defaults=True
        )

        return self

    @classmethod
    def _create_update_maps(cls):
        """ Generates a mapping from setting keys to functions that use them

        Therefore, this mapping (`cls._run_on_update`) contains information about 
        which functions need to be executed again when a setting is updated.

        The mapping generated here is used in `Configurable.run_updates`
        """
        #Initialize the object where we are going to store what each setting needs to rerun when it is updated
        if hasattr(cls, "_run_on_update"):
            updates_dict = copy(cls._run_on_update)
        else:
            updates_dict = defaultdict(list)

        cls._run_on_update = updates_dict

        for name, f in inspect.getmembers(cls, predicate=inspect.isfunction):
            for _, param in getattr(f, "_settings", []):
                cls._run_on_update[param].append(f.__name__)

    @property
    def settings(self):
        """ The current settings of the object """
        return self.settings_history.current

    @classmethod
    def _get_class_params(cls):
        """ Returns all the parameters that can be tweaked for that class

        These are obtained from the `_parameters` class variable.

        Note that parameters are inherited even if you overwrite the `_parameters`
        variable. 

        Probably there should be a variable `_exclude_params` to avoid some parameters.
        """
        params, param_groups = [], []
        for clss in type.mro(cls):
            if "_parameters" in vars(clss):
                params = [*params, *deepcopy(clss._parameters)]
            if "_param_groups" in vars(clss):
                param_groups = [*deepcopy(clss._param_groups), *param_groups]

        # Build an extra group for unclassified settings
        param_groups.append({
            "key": None,
            "name": "Other settings",
            "icon": "settings",
            "description": "Here are some unclassified settings. Even if they don't belong to any group, they might still be important. They may be here just because the developer was too lazy to categorize them or forgot to do so. <b>If you are the developer</b> and it's the first case, <b>shame on you<b>."
        })

        return params, param_groups

    def update_settings(self, *args, **kwargs):
        """ This method will be overwritten for each class. See `_update_settings` """
        return self._update_settings(*args, **kwargs)

    def _update_settings(self, run_updates=True, **kwargs):
        """ Updates the settings of the object

        Note that this is only private because we provide a public update_settings
        with the specific kwargs for each class so that users can quickly know which
        settings are available. You can see how we define this method in `__init_subclass__`

        Parameters
        ------------
        run_updates: bool, optional
            whether we should run updates after updating the settings. If not, the settings
            will be updated, but you won't see any change in the object.
        **kwargs:
            the values of the settings that we want to update passed as keyword arguments.
        """
        #Initialize the settings in case there are none yet
        if not hasattr(self, "settings_history"):
            return self.init_settings(**kwargs)

        # Otherwise, update them
        updates = {key: val for key, val in kwargs.items() if key in self.settings_history}
        if updates:
            self.settings_history.update(**updates)

            #Do things after updating the settings
            if len(self.settings_history.last_updated) > 0 and run_updates:
                self._run_updates(self.settings_history.last_updated)

        return self

    def _run_updates(self, for_keys):
        """ Runs the functions/methods that are supposed to be ran when given settings are updated

        It uses the `_run_on_update` dict, which contains what to run
        in case each setting is updated.

        Parameters
        -----------
        for_keys: array-like of str
            the keys of the settings that have been updated.
        """
        # Get the functions that need to be executed for each key that has been updated and
        # put them in a list
        func_names = [self._run_on_update.get(setting_key, None) for setting_key in for_keys]

        # Flatten that list (list comprehension) and take only the unique values (set)
        func_names = set([f_name for sublist in func_names for f_name in sublist])

        # Give the oportunity to parse the functions that need to be ran. See `Plot._parse_update_funcs`
        # for an example
        func_names = self._parse_update_funcs(func_names)

        # Execute the functions that we need to execute.
        for f_name in func_names:
            getattr(self, f_name)()

        return self

    def _parse_update_funcs(self, func_names):
        """ Called on _run_updates as a final oportunity to decide what functions to run

        May be overwritten in child classes.

        Parameters
        -----------
        func_names: set of str
            the unique functions names that are to be executed unless you modify them.

        Returns
        -----------
        array-like of str
            the final list of functions that will be executed.
        """
        return func_names

    def undo_settings(self, steps=1, run_updates=True):
        """ Brings the settings back a number of steps

        Parameters
        ------------
        steps: int, optional
            the number of steps you want to go back.
        run_updates: bool, optional
            whether we should run updates after updating the settings. If not, the settings
            will be updated, but you won't see any change in the object.
        """
        try:
            diff = self.settings_history.diff_keys(-1, -steps-1)
            self.settings_history.undo(steps=steps)
            if run_updates:
                self._run_updates(diff)
        except IndexError:
            info(f"This instance of {self.__class__.__name__} does not "
                 f"contain earlier settings as requested ({steps} step(s) back)")

        return self

    def undo_setting(self, key):
        """ Undoes only a particular setting and leaves the others unchanged

        At the moment it is a 'fake' undo function, since it actually updates the settings.

        Parameters
        -----------
        key: str
            the key of the setting that you want to undo.
        """
        i = self.settings_history.last_update_for(key)

        if i is None:
            info(f"key={key} was never changed; cannot undo nothing.")

        self.update_settings(key=self.settings_history[key][i])

        return self

    def undo_settings_group(self, group):
        """ Takes the desired group of settings one step back, but the rest of the settings remain unchanged

        At the moment it is a 'fake' undo function, since it actually updates the settings.

        Parameters
        -----------
        group: str
            the key of the settings group for which you want to undo its values.
        """
        #Get the actual settings for that group
        actualSettings = self.get_settings_group(group)

        #Try to find any different values for the settings
        for i in range(len(self.settings_history)):

            previousSettings = self.get_settings_group(group, steps_back = i)

            if previousSettings != actualSettings:

                return self.update_settings(previousSettings)
        else:
            info(f"group={group} was never changed; cannot undo nothing.")

        return self

    def get_param(self, key, as_dict=False, paramsExtractor=False):
        """ Gets the parameter for a given setting

        By default it returns its dictionary, so that one can check the information that it contains.
        You can ask for the parameter itself by setting as_dict to False. However, if you want to
        modify the parameter you should use the modify_param() method instead.

        Arguments
        ---------
        key: str
            The key of the desired parameter.
        as_dict: bool, optional
            If set to True, returns a dictionary instead of the actual parameter object.
        paramsExtractor: function, optional
            A function that accepts the object (self) and returns its params (NOT A COPY OF THEM!).
            This will only be used in case this method is used outside the class, where objects
            have a different structure (e.g. QueriesInput inputField) or if there is some nested params
            field that the class is not aware of (although this second case is probably not advisable).

        Returns
        -------
        param: dict or InputField
            The parameter in the form specified by as_dict.
        """
        for param in self.params if not paramsExtractor else paramsExtractor(self):
            if param.key == key:
                return param.__dict__ if as_dict else param
        else:
            raise KeyError(f"There is no parameter '{key}' in {self.__class__.__name__}")

    def modify_param(self, key, *args, **kwargs):
        """ Modifies a given parameter

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
        key: str
            The key of the parameter to be modified
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
        **kwargs: optional
            They are passed directly to the Configurable.get_param method to retrieve the parameter.

        Returns
        --------
        self:
            The configurable object.
        """
        self.get_param(key, as_dict = False, **kwargs).modify(*args)

        return self

    def get_setting(self, key, copy=True, parse=True):
        """ Gets the value for a given setting

        Parameters
        ------------
        key: str
            The key of the setting we want to get
        copy: boolean, optional
            Whether you want a copy of the object or the actual object
        parse: boolean, optional
            whether the setting should be parsed before returning it.
        """
        # Get the value of the setting and parse it using the parse method
        # defined for the parameter
        val = self.get_param(key).parse(self.settings[key])

        return deepcopy(val) if copy else val

    def get_settings_group(self, group, steps_back=0):
        """ Gets the subset of the settings that corresponds to a given group

        Arguments
        ---------
        group: str
            The key of the settings group that we desire.
        steps_back: optional, int
            If you don't want the actual settings, but some point of the settings history,
            use this argument to state how many steps back you want the settings' values.

        Returns
        -------
        settings_group: dict
            A subset of the settings with only those that belong to the asked group.
        """
        if steps_back:
            settings = self.settings_history[-steps_back]
        else:
            settings = self.settings

        return deepcopy({setting.key: settings[setting.key] for setting in self.params if getattr(setting, "group", None) == group})

    def has_these_settings(self, settings={}, **kwargs):
        """ Checks if the object settings match the provided settings

        Parameters
        ----------
        settings: dict
            dictionary containing the settings keys and values
        **kwargs:
            setting keys and values can also be passed as keyword arguments.

        You can use settings and **kwargs at the same time, they will be merged.
        """
        settings = {**settings, **kwargs}

        for key, val in settings.items():
            if self.get_setting(key) != val:
                return False
        else:
            return True


# DECORATOR TO USE WHEN DEFINING METHODS IN CLASSES THAT INHERIT FROM Configurable

def vizplotly_settings(when='before', init=False):
    """ Specifies how settings should be updated when running a method

    It can only decorate a method of a class that inherits from Configurable.

    Works by grabbing the kwargs from the method and taking the ones whose keys
    represent settings.

    Parameters
    ----------
    when: {'after', 'before'}
        specifies when should the settings be updated.

        'after': After the method has been ran.
        'before': Before running the method.

    init: boolean, optional
        whether the settings should be initialized (restored).

        If `False`, the settings are just updated.
    """
    extra_kwargs = {}
    if init:
        method_name = 'init_settings'
    else:
        method_name = '_update_settings'
        extra_kwargs = {'from_decorator': True, 'run_updates': True}

    def decorator(method):
        if when == 'before':
            @wraps(method)
            def func(obj, *args, **kwargs):
                getattr(obj, method_name)(**kwargs, **extra_kwargs)
                return method(obj, *args, **kwargs)

        elif when == 'after':
            @wraps(method)
            def func(obj, *args, **kwargs):
                ret = method(obj, *args, **kwargs)
                getattr(obj, method_name)(**kwargs, **extra_kwargs)
                return ret
        else:
            raise ValueError("Incorrect decorator usage")
        return func
    return decorator


def _populate_with_settings(f, class_params):
    """ Makes functions of a Configurable object able to access settings through arguments

    Parameters
    -----------
    f: function
        the function that you want to give this functionality
    class_params: array-like of str
        the keys of the parameters that this function will be able to access. Presumably these
        are the keys of the parameters of the class where the function is defined.

    Returns
    ------------
    function
        in case the function has some arguments named like parameters that are available to it,
        this will be a wrapped function that defaults the values of those arguments to the values
        of the settings.

        Otherwise, it returns the same function.

    Examples
    -----------

    >>> class MyPlot(Configurable):
    >>>     _parameters = (TextInput(key="my_param", name=...))
    >>>
    >>>     def some_method(self, my_param):
    >>>          return my_param

    After `some_method` has been correctly passed through `_populate_with_settings`:
    >>> plot = MyPlot(my_param=3)
    >>> plot.some_method() # Returns 3
    >>> plot.some_method(5) # Returns 5
    >>> plot.some_method() # Returns 3
    """
    try:
        # note that params takes `self` as argument
        # So first actual argument has index 1
        params = inspect.signature(f).parameters
        # Also, there is no need to use numpy if not needed
        # In this case it was just an overhead.
        idx_params = tuple(filter(lambda i_p: i_p[1] in class_params,
                                  enumerate(params)))
    except:
        return f

    if len(idx_params) == 0:
        # no need to wrap it
        return f

    # Tuples are immutable, so they should have a *slightly* lower overhead.
    # Also, get rid of zip below
    # The below gets called alot, I suspect.
    # So it should probably be *fast* :)
    f._settings = idx_params

    @wraps(f)
    def f_default_setting_args(self, *args, **kwargs):
        nargs = len(args)
        for i, param in f._settings:
            # nargs does not count `self` and then the above indices will fine
            if i > nargs and param not in kwargs:
                try:
                    kwargs[param] = self.get_setting(param, copy=False)
                except KeyError:
                    pass

        return f(self, *args, **kwargs)

    return f_default_setting_args
