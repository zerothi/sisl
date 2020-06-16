from copy import deepcopy
from functools import wraps
from types import MethodType
from collections import deque, Iterable
import sys

import numpy as np

from sisl._dispatcher import AbstractDispatch
from ._presets import get_preset
from .plotutils import get_configurable_docstring, get_configurable_kwargs, get_configurable_kwargs_to_pass

__all__ = ['Configurable', 'vizplotly_settings']

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
        """
        The current state of the history
        """
        return self.step(-1)
    
    def step(self, i):
        """
        Retrieves a given step of the history.

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
        """
        Returns the number of steps stored in the history.
        """
        for _, hist in self._hist.items():
            return len(hist)

    def __getitem__(self, item):

        # If an int is provided, we get that step of the history
        if isinstance(item, int):
            return self.step(item)
        # If a string is provided, we get the full history of a given setting
        elif isinstance(item, str):
            return [self._vals[item][i] for i in self._hist[item]]
        elif isinstance(item, Iterable):
            try:
                # If it's an array-like of strings, we will get the full history of each key
                return {key: np.array(self._vals[key])[self._hist[key]] for key in item}
            except:
                # Otherwise, we just map the array with __getitem__()
                return [self.__getitem__(i) for i in item]
        # Finally, if it's a slice, we will get the steps that are within that slice.
        elif isinstance(item, slice):
            return {key: np.array(self._vals[key])[hist][item] for key, hist in self._hist.items()}
    
    def __contains__(self, item):
        """
        Check if we are storing that named item.
        """
        return item in self._vals
    
    def update(self, **new_settings):
        """
        Updates the history by appending a new step.

        Parameters
        -----------
        **new_settings:
            all the settings that you want to change passed as keyword arguments.

            You don't need to provide the values for all parameters, only those that you
            need to change.

        Returns
        -----------
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
                for saved_val in self._vals[key]:
                    if not isinstance(saved_val, np.ndarray) and not is_nparray:
                        if val == saved_val:
                            new_index = self._vals[key].index(val)   
                else:
                    self._vals[key].append(val)
                    new_index = len(self._vals[key]) - 1
            
            # Append the index to the history
            self._hist[key].append(new_index)
        
        return self
    
    @property
    def last_updated(self):
        """
        The names of the parameters that were changed in the last update.
        """
        return self.updated_params(-1)

    def last_update_for(self, key):
        """
        Returns the index of the last update for a given parameter.

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
        """
        Gets the keys of the parameters that were updated in a given step.

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
        """
        Checks whether a given step updated the parameter's value.

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
        """
        Checks if a parameter has a different value between two steps.

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
        """
        Gets the keys that are different between two steps of the history.

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
        """
        Gets a dictionary with the diferences between two steps.

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
        """
        A log with the last changes.

        See `delta` for more information.
        """
        return self.delta(-1, -2)

    def undo(self, steps=1):
        """
        Takes the history back a number of steps.

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
        
        for key, hist in self._hist.items():
            
            for _ in range(steps):
                hist.pop()
        
        # Clear the unused values (for the moment we are setting them to None
        # so that we don't need to change the indices of the history)
        for key in self._vals:
            
            if self._defaults_kept:
                self._vals[key] = [ val if i in hist or i==0 else None for i, val in enumerate(self._vals[key]) ]
            else:
                self._vals[key] = [ val if i in hist else None for i, val in enumerate(self._vals[key]) ]
        
        return self
    
    def clear(self):
        """
        Clears the history.

        It sets all settings to `None`. If you want to restore the defaults,
        use `restore_defaults` instead.

        """
        
        self.__init__(init_settings={key: None for key in self._vals})
        
        return self
    
    def restore_initial(self):
        """
        Restores the history to its initial values (the first step).
        """

        self.__init__(init_settings=self.step(0))

        return self

    def restore_defaults(self):
        """
        Restores the history to its defaults.
        """
        
        if self._defaults_kept:
            if hasattr(self, "_defaults"):
                self.__init__({})
            else:
                self.restore_initial()
        else:
            raise Exception("Defaults were not kept! You need to use keep_defaults=True on initialization")
            
        return self
    
    @property
    def defaults(self):
        """
        The default values for this history.
        """
        if self._defaults_kept:
            if hasattr(self, "_defaults"):
                return self._defaults
            else:
                return self.step(0)
        else:
            raise Exception("Defaults were not kept! You need to use keep_defaults=True on initialization")
            
        return self
    
    def is_default(self, key):
        """
        Checks if a parameter currently has the default value.

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

class FakeSettingsDispatch(AbstractDispatch):
    """
    Provides a dispatch that executes methods using "fake" settings.

    This is mainly useful to use methods from other classes without having
    the settings that are required.

    AT THE MOMENT ATTRIBUTES ARE NOT SET TO THE PLOT WHEN THE METHOD RUNS, SO YOU 
    SHOULD PROBABLY NOT USE IT UNLESS YOU ARE AWARE OF THAT!
    """

    def __init__(self, obj, **settings):

        self.fake_settings = settings

        super().__init__(obj)
    
    def setting(self, *args, **kwargs):

        kwargs["fake_settings"] = { **kwargs.get("fake_settings", {}), **self.fake_settings}
        
        return Configurable.setting(self._obj, *args, **kwargs)

    def dispatch(self, method):

        @wraps(method)
        def run_with_fake_settings(*args, **kwargs):

            self._obj.setting = self.setting

            ret = method(*args, **kwargs)

            self._obj.setting = MethodType(Configurable.setting, self._obj)

            return ret
            
        return run_with_fake_settings

class Configurable:
    
    def with_settings(self, method=None, method_args=[], method_kwargs={}, **settings):
        """
        Gives you the possibility to run the methods of an object AS IF the object
        had certain settings.

        NOTE: The settings of the object WILL NOT BE UPDATED, it will just use the
        provided settings to run the methods that you want.

        Parameters
        ----------
        method: method or str, optional
            If provided, the method that will be run with those settings.
            If not, a `FakeSettingsDispatch` is returned. See returns for more info.
        method_args: array-like, optional
            the arguments to pass to the method on call
        method_kwargs: dict, optional
            the keyword arguments to pass to the method on call.
        **settings: 
            The settings that you want to make the object believe it has.
            Pass each setting as a keyword argument.

            For example: `obj.with_settings(color="green")`

            will provide the object with a fake "color" setting of value "green"
            
        Returns
        ---------
        any
            if a method is provided the returns of the method will be returned.
            If not, a `FakeSettingsDispatch` is returned. A `FakeSettingsDispatch`
            acts exactly as the object itself, but any method that you run on it 
            will use the fake settings.
        """

        disp = FakeSettingsDispatch(self, **settings)

        if method is not None:
            if isinstance(method, str):
                method = getattr(self, method)
            return method(*method_args, **method_kwargs)
        
        return disp

    def init_settings(self, presets=None, **kwargs):
        """
        Initializes the settings for the object.
        
        Parameters
        -----------
        presets: str or array-like of str
            all the presets that you want to use.
            Note that you can register new presets using `sisl.viz.add_preset`
        **kwargs:
            the values of the settings passed as keyword arguments.

            If a setting is not provided, the default value will be used.
        """

        if getattr(self, "AVOID_SETTINGS_INIT", False):
            delattr(self, "AVOID_SETTINGS_INIT")
            return

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
        defaults = { param.key: deepcopy(param.default) for param in self.params}
        self.settings_history = NamedHistory(
            {key: kwargs.get(key, val) for key, val in defaults.items()},
            defaults=defaults, history_len=20, keep_defaults=True
        )

        #Initialize the object where we are going to store what each setting needs to rerun when it is updated
        self._run_on_update = {}
        
        return self
    
    def __init_subclass__(cls):
        """
        When a configurable class is defined, this will be called.

        We will make use of it to:
            - Give a more specific docstring to update settings.
        """

        # Change the docs of the update_settings method to truly reflect
        # the available kwargs for the plot class and provide more help to the user
        def update_settings(self, *args, **kwargs):
            return self._update_settings(*args, **kwargs)

        update_settings.__doc__ = f"Updates the settings of this plot.\n\nDocs for {cls.__name__}:\n\n{get_configurable_docstring(cls)}" 
        cls.update_settings = update_settings

    @property
    def settings(self):
        """
        The current settings of the object.
        """
        return self.settings_history.current

    @classmethod
    def _get_class_params(cls):
        """
        Returns all the parameters that can be tweaked for that class.

        These are obtained from the `_parameters` class variable.

        Note that parameters are inherited even if you overwrite the `_parameters`
        variable. 
        
        Probably there should be a variable `_exclude_params` to avoid some parameters.
        """

        params = []; param_groups = []
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
            "description": "Here are some unclassified settings. Even if they don't belong to any group, they might still be important :) They may be here just because the developer was too lazy to categorize them or forgot to do so. <b>If you are the developer</b> and it's the first case, <b>shame on you<b>."
        })

        return params, param_groups

    def update_settings(self, *args, **kwargs):
        """
        This method will be overwritten for each class. See `_update_settings`.
        """
        return self._update_settings(*args, **kwargs)

    def _update_settings(self, run_updates=True, **kwargs):
        """
        Updates the settings of the object. 
        
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
        updates = {key:val for key, val in kwargs.items() if key in self.settings_history}
        if updates:
            self.settings_history.update(**updates)

            #Do things after updating the settings
            if len(self.settings_history.last_updated) > 0 and run_updates:
                self._run_updates(self.settings_history.last_updated)
        
        return self
        
    def _run_updates(self, for_keys):
        """
        Runs the functions/methods that are supposed to be ran when
        given settings are updated.

        It uses the `_run_on_update` dict, which contains what to run
        in case each setting is updated.

        Parameters
        -----------
        for_keys: array-like of str
            the keys of the settings that have been updated.
        """

        #Run the functions specified
        if hasattr(self, "_onSettingsUpdate"):
            
            #Get the unique names of the functions that should be executed
            noInfoKeys = [settingKey for settingKey in for_keys if settingKey not in self._run_on_update]
            if len(noInfoKeys) > 0 and len(noInfoKeys) == len(for_keys):
                print(f"We don't know (yet) what to do when the following settings are updated: {noInfoKeys}. Please run the corresponding methods yourself in order to update the plot")

            funcNames = set([self._run_on_update.get(settingKey, None) for settingKey in for_keys])

            for fName in self._onSettingsUpdate["functions"]:
                if fName in funcNames:
                    getattr(self, fName)()
                    
                    #If we need to execute all the functions keep going thorugh the loop, else stop here
                    if not self._onSettingsUpdate["config"].get("multipleFunc", False):
                        break
            
        return self
    
    def undo_settings(self, steps=1, run_updates=True):
        """
        Brings the settings back a number of steps.

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
            print("This instance of {} does not contain earlier settings as requested ({} step{} back)"
                 .format(self.__class__.__name__, steps, "" if steps == 1 else "s"))
            pass
            
        return self
    
    def undo_setting(self, key):
        """
        Undoes only a particular setting and leaves the others unchanged.

        At the moment it is a 'fake' undo function, since it actually updates the settings.

        Parameters
        -----------
        key: str
            the key of the setting that you want to undo.
        """

        i = self.settings_history.last_update_for(key)

        if i is None:
            print(f"There is no registry of the setting '{key}' having been changed. Sorry :(")

        self.update_settings(key=self.settings_history[key][i])

        return self

    def undo_settings_group(self, group_key):

        """
        Takes the desired group of settings one step back, but the rest of the settings remain unchanged.

        At the moment it is a 'fake' undo function, since it actually updates the settings.

        Parameters
        -----------
        group_key: str
            the key of the settings group for which you want to undo its values.
        """

        #Get the actual settings for that group
        actualSettings = self.get_settings_group(group_key)

        #Try to find any different values for the settings
        for i in range(len(self.settings_history)):

            previousSettings = self.get_settings_group(group_key, steps_back = i)

            if previousSettings != actualSettings:

                return self.update_settings(previousSettings)
        else:
            print("There is no registry of any setting of the group '{}' having been changed. Sorry :(".format(group_key))
            
            return self
    
    def get_param(self, key, as_dict=False, paramsExtractor=False):
        """
        Gets the parameter for a given setting. 
        
        By default it returns its dictionary, so that one can check the information that it contains.
        You can ask for the parameter itself by setting as_dict to False. However, if you want to
        modify the parameter you should use the modify_param() method instead.

        Arguments
        ----------
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
        ----------
        param: dict or InputField
            The parameter in the form specified by as_dict.
        """

        for param in self.params if not paramsExtractor else paramsExtractor(self):
            if param.key == key:
                return param.__dict__ if as_dict else param
        else:
            raise KeyError(f"There is no parameter '{key}' in {self.__class__.__name__}")
    
    def modify_param(self, key, *args, **kwargs):
        """
        Modifies a given parameter.
        
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
    
    def get_setting(self, key, copy=True, fake_settings=None, parse=True):
        """
        Gets the value for a given setting.

        Parameters
        ------------
        key: str
            The key of the setting we want to get
        copy: boolean, optional
            Whether you want a copy of the object or the actual object
        fake_settings: dict, optional
            These will be added to the real settings. It is intended to make it look
            as if the object had these settings.
            This is mainly to make it possible for classes to use other's classes methods even if
            they don't have the necessary settings.
            It is used by the `with_settings` method, which
        parse: boolean, optional
            whether the setting should be parsed before returning it.
        """
        
        settings = self.settings
        if fake_settings is not None:
            settings = {**settings, **fake_settings}

        # Get the value of the setting and parse it using the parse method
        # defined for the parameter
        val = self.get_param(key).parse(settings[key])

        return deepcopy(val) if copy else val
    
    def setting(self, key, **kwargs):

        """
        Gets the value for a given setting while logging where it has been required. 
        
        THIS METHOD MUST BE USED IN DEVELOPEMENT! (And should not be used by users, use get_setting() instead)
        
        It stores where the setting has been demanded so that the plot can be efficiently updated when it is modified.
        """

        #Get the last frame
        frame = sys._getframe().f_back
        #And then iterate over all the past frames to get their names
        while frame:

            #Get the function name
            funcName = frame.f_code.co_name

            #If it is in the list of functions provided on class definition (e.g. on Plot class) store it
            functions_list = self._onSettingsUpdate["functions"]
            if funcName in functions_list:

                prevFunc = self._run_on_update.get(key, None)

                if prevFunc is None or functions_list.index(funcName) < functions_list.index(prevFunc):
                    self._run_on_update[key] = funcName
                
                break
            
            frame = frame.f_back

        return self.get_setting(key, copy=False, **kwargs)
    
    def get_settings_group(self, group_key, steps_back = 0):
        """
        Gets the subset of the settings that corresponds to a given group

        Arguments
        -----------
        group_key: str
            The key of the settings group that we desire.
        steps_back: optional, int
            If you don't want the actual settings, but some point of the settings history,
            use this argument to state how many steps back you want the settings' values.

        Returns
        -----------
        settings_group: dict
            A subset of the settings with only those that belong to the asked group.
        """

        if steps_back:
            settings = self.settings_history[-steps_back]
        else:
            settings = self.settings

        return deepcopy({ setting.key: settings[setting.key] for setting in self.params if getattr(setting, "group", None) == group_key })
    
    def settings_group(self, group_key):
        """
        Gets the subset of the settings that corresponds to a given group and logs its use

        This method is to `get_settings_group` the same as `setting` is to `get_setting`.

        That is, the return is exactly the same but the use of the settings is logged to update
        the plot properly.

        Arguments
        -----------
        group_key: str
            The key of the settings group that we desire.
        """

        return deepcopy({ setting.key: self.setting(setting.key) for setting in self.params if getattr(setting, "group", None) == group_key })

    def has_these_settings(self, settings={}, **kwargs):
        """
        Checks if the object settings match the provided settings.

        Parameters
        -----------
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

# DECORATORS TO USE WHEN DEFINING METHODS IN CLASSES THAT INHERIT FROM Configurable

def vizplotly_settings(when='before', init=False):
    """
    Specifies how settings should be updated when running a method.

    It can only decorate a method of a class that inherits from Configurable.

    Works by grabbing the kwargs from the method and taking the ones whose keys
    represent settings.

    Parameters
    -----------
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
