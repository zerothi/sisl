from copy import deepcopy
from functools import wraps
from types import MethodType
import sys

import numpy as np

from sisl._dispatcher import AbstractDispatch
from ._presets import get_preset
from .plotutils import get_configurable_docstring, get_configurable_kwargs, get_configurable_kwargs_to_pass

class FakeSettingsDispatch(AbstractDispatch):
    '''
    Provides a dispatch that executes methods using "fake" settings.

    This is mainly useful to use methods from other classes without having
    the settings that are required.
    '''

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
        '''
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
        '''

        disp = FakeSettingsDispatch(self, **settings)

        if method is not None:
            if isinstance(method, str):
                method = getattr(self, method)
            return method(*method_args, **method_kwargs)
        
        return disp

    def init_settings(self, presets=None, **kwargs):

        if getattr(self, "AVOID_SETTINGS_INIT", False):
            delattr(self, "AVOID_SETTINGS_INIT")
            return
        
        self.settingsHistory = []

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

        #Define the settings dictionary, taking the value of each parameter from kwargs if it is there or from the defaults otherwise.
        self.settings = { param.key: kwargs.get( param.key, deepcopy(param.default) ) for param in self.params}
        
        self.settingsHistory.append(deepcopy(self.settings))

        #Initialize the object where we are going to store what each setting needs to rerun when it is updated
        self.whatToRunOnUpdate = {}

        # Update the docs of the update_settings method to truly reflect
        # the available kwargs for the object
        def update_settings(self, **kwargs):
            return Configurable.update_settings(self, **kwargs)

        update_settings.__doc__ = get_configurable_docstring(self)
        self.update_settings = MethodType(update_settings, self)
        
        return self
    
    @classmethod
    def _get_class_params(cls):

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

    def update_settings(self, from_decorator=False, update_fig=True, no_log=False , **kwargs):
        
        #Initialize the settings in case there are none yet
        if "settings" not in vars(self):
            return self.init_settings(**kwargs)
        
        #Otherwise, update them
        updated = []
        for paramKey, paramValue in kwargs.items():
            
            #It is important to check this, because kwargs may contain other parameters that are not settings
            if paramKey in self.settings.keys() and self.settings[paramKey] != paramValue:
                
                self.settings[paramKey] = paramValue
                updated.append(paramKey)
        
        #Do things after updating the settings
        if len(updated) > 0:
            
            if not no_log:
                #Record rhe change in the settings history
                self.settingsHistory.append(deepcopy(self.settings))
        
            #Run the functions specified
            if not from_decorator and hasattr(self, "_onSettingsUpdate") and update_fig:
                
                #Get the unique names of the functions that should be executed
                noInfoKeys = [settingKey for settingKey in updated if settingKey not in self.whatToRunOnUpdate]
                if len(noInfoKeys) > 0 and len(noInfoKeys) == len(updated):
                    print(f"We don't know (yet) what to do when the following settings are updated: {noInfoKeys}. Please run the corresponding methods yourself in order to update the plot")

                funcNames = set([self.whatToRunOnUpdate.get(settingKey, None) for settingKey in updated])

                for fName in self._onSettingsUpdate["functions"]:
                    if fName in funcNames:
                        getattr(self, fName)()
                        
                        #If we need to execute all the functions keep going thorugh the loop, else stop here
                        if not self._onSettingsUpdate["config"].get("multipleFunc", False):
                            break
            
        return self
    
    def undo_settings(self, nsteps = 1, **kwargs):
        
        try:
            self.settingsHistory = self.settingsHistory[0:-nsteps]         
            self.update_settings( **deepcopy(self.settingsHistory[-1]), **kwargs)
        except IndexError:
            print("This instance of {} does not contain earlier settings as requested ({} step{} back)"
                 .format(self.__class__.__name__, nsteps, "" if nsteps == 1 else "s"))
            pass
            
        return self
    
    def undo_setting(self, settingKey):
        '''
        Undoes only a particular setting and lives the others unchanged.

        At the moment it is a 'fake' undo function, since it actually updates the settings.

        '''

        #Get the actual settings for that group
        actualSetting = self.get_setting(settingKey)

        #Try to find any different values for the settings
        for pastValue in reversed(self.get_setting_history(settingKey)):

            if pastValue != actualSetting:

                return self.update_settings( **{settingKey: pastValue} )
        else:
            print("There is no registry of the setting '{}' having been changed. Sorry :(".format(settingKey))
            
            return self

    def undo_settings_group(self, groupKey):

        '''
        Takes the desired group of settings one step back, but the rest of the settings remain unchanged.

        At the moment it is a 'fake' undo function, since it actually updates the settings.

        '''

        #Get the actual settings for that group
        actualSettings = self.get_settings_group(groupKey)

        #Try to find any different values for the settings
        for i in range(len(self.settingsHistory)):

            previousSettings = self.get_settings_group(groupKey, stepsBack = i)

            if previousSettings != actualSettings:

                return self.update_settings(previousSettings)
        else:
            print("There is no registry of any setting of the group '{}' having been changed. Sorry :(".format(groupKey))
            
            return self
    
    def get_param(self, settingKey, justDict = False, paramsExtractor = False):
        '''
        Gets the parameter for a given setting. 
        
        By default it returns its dictionary, so that one can check the information that it contains.
        You can ask for the parameter itself by setting justDict to False. However, if you want to
        modify the parameter you should use the modify_param() method instead.

        Arguments
        ----------
        settingKey: str
            The key of the desired parameter.
        justDict: bool, optional (True)
            If set to False, returns the actual parameter object. By default it returns just the info.
        paramsExtractor: function, optional
            A function that accepts the object (self) and returns its params (NOT A COPY OF THEM!).
            This will only be used in case this method is used outside the class, where objects
            have a different structure (e.g. QueriesInput inputField) or if there is some nested params
            field that the class is not aware of (although this second case is probably not advisable).
        
        Returns
        ----------
        param: dict or InputField
            The parameter in the form specified by justDict.
        '''

        for param in self.params if not paramsExtractor else paramsExtractor(self):
            if param.key == settingKey:
                return param.__dict__ if justDict else param
        else:
            return None
    
    def modify_param(self, settingKey, *args, **kwargs):
        '''
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
        settingKey: str
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
        '''

        self.get_param(settingKey, justDict = False, **kwargs).modify(*args)

        return self
    
    def get_setting(self, setting_key, copy=True, fake_settings=None):
        '''
        Gets the value for a given setting.

        Parameters
        ------------
        setting_key: str
            The key of the setting we want to get
        copy: boolean, optional
            Whether you want a copy of the object or the actual object
        fake_settings: dict, optional
            These will be added to the real settings. It is intended to make it look
            as if the object had these settings.
            This is mainly to make it possible for classes to use other's classes methods even if
            they don't have the necessary settings.
            It is used by the `with_settings` method, which 
        '''
        
        settings = self.settings
        if fake_settings is not None:
            settings = {**settings, **fake_settings}
        val = settings[setting_key]

        return deepcopy(val) if copy else val
    
    def setting(self, settingKey, **kwargs):

        '''
        Gets the value for a given setting while logging where it has been required. 
        
        THIS METHOD MUST BE USED IN DEVELOPEMENT! (And should not be used by users, use get_setting() instead)
        
        It stores where the setting has been demanded so that the plot can be efficiently updated when it is modified.
        '''

        #Get the last frame
        frame = sys._getframe().f_back
        #And then iterate over all the past frames to get their names
        while frame:

            #Get the function name
            funcName = frame.f_code.co_name

            #If it is in the list of functions provided on class definition (e.g. on Plot class) store it
            functions_list = self._onSettingsUpdate["functions"]
            if funcName in functions_list:

                prevFunc = self.whatToRunOnUpdate.get(settingKey, None)

                if prevFunc is None or functions_list.index(funcName) < functions_list.index(prevFunc):
                    self.whatToRunOnUpdate[settingKey] = funcName
                    break
            
            frame = frame.f_back
        
        return self.get_setting(settingKey, copy=False, **kwargs)

    def get_setting_history(self, settingKey):
        
        return deepcopy([step[settingKey] for step in self.settingsHistory])
    
    def get_settings_group(self, groupKey, stepsBack = 0):
        '''
        Gets the subset of the settings that corresponds to a given group

        Arguments
        -----------
        groupKey: str
            The key of the settings group that we desire.
        stepsBack: optional, int
            If you don't want the actual settings, but some point of the settings history,
            use this argument to state how many steps back you want the settings' values.

        Returns
        -----------
        settings_group: dict
            A subset of the settings with only those that belong to the asked group.
        '''

        if stepsBack:
            settings = self.settingsHistory[-stepsBack]
        else:
            settings = self.settings

        return deepcopy({ setting.key: settings[setting.key] for setting in self.params if getattr(setting, "group", None) == groupKey })
    
    def settings_group(self, groupKey):
        '''
        Gets the subset of the settings that corresponds to a given group and logs its use

        This method is to `getSettingsGroup` the same as `setting` is to `get_setting`.

        That is, the return is exactly the same but the use of the settings is logged to update
        the plot properly.
        '''

        return deepcopy({ setting.key: self.setting(setting.key) for setting in self.params if getattr(setting, "group", None) == groupKey })

    def settings_updates_log(self, frame = -1):
        '''
        Returns a dictionary with a log of a given update in the settings (by default, the last one).

        Each key contains a dictionary with 'before' and 'after' values.

        Arguments
        --------
        frame: optional, int (-1)
            This is the settings history step for which we want the updates. By default it returns the last update.
        '''

        try:
            self.settingsHistory[frame]
            self.settingsHistory[frame - 1]
        except IndexError:
            return {}

        updatedDict = {
            key: {
                "before": self.settingsHistory[frame - 1][key],
                "after": postValue ,
            } for key, postValue in self.settingsHistory[frame].items() if self.settingsHistory[frame - 1][key] != postValue
        }

        return updatedDict

    def is_default(self, settingKey):

        '''
        Checks if the current value for a setting is the default one.
        
        DOESN'T WORK FOR VALUES THAT ARE FUNCTIONS!
        '''

        return self.settings[settingKey] == self.get_param(settingKey)["default"]
    
    def did_setting_update(self, setting_key, all_updates=False):
        '''
        Returns if a given setting did update in the last settings update

        Parameters
        --------
        all_updates: boolean, optional
            whether to return a list stating if the setting did update in each frame.
        '''

        if all_updates:
            history = self.get_setting_history(setting_key)
            return np.array([
                history[0] != self.get_param(setting_key)["default"], # Did the setting change on initialization
                *[value != history[iPrev] for iPrev, value in enumerate(history[1:])] # Was the setting updated (for each step)
            ])
        else:
            return setting_key in self.settings_updates_log(frame=-1)

    def has_these_settings(self, settings={}, **kwargs):
        '''
        Checks if the object settings match the provided settings.

        Parameters
        -----------
        settings: dict
            dictionary containing the settings keys and values
        **kwargs:
            setting keys and values can also be passed as keyword arguments.

        You can use settings and **kwargs at the same time, they will be merged.
        '''

        settings = {**settings, **kwargs}

        for key, val in settings.items():
            if self.get_setting(key) != val:
                return False
        else:
            return True

#DECORATORS TO USE WHEN DEFINING METHODS IN CLASSES THAT INHERIT FROM Configurable
#Run the method after having initialized the settings
def after_settings_init(method):
    
    def update_and_execute(obj, *args, **kwargs):
        
        obj.init_settings(**kwargs)
        
        return method(obj, *args, **kwargs)
    
    return update_and_execute

#Run the method and then initialize the settings
def before_settings_init(method):
    
    def update_and_execute(obj, *args, **kwargs):
        
        returns = method(obj, *args, **kwargs)
        
        obj.init_settings(**kwargs)
        
        return returns
    
    return update_and_execute

#Run the method after having updated the settings
def after_settings_update(method):

    def update_and_execute(obj, *args, **kwargs):
        
        obj.update_settings(**kwargs, from_decorator = True)
        
        return method(obj, *args, **kwargs)
    
    return update_and_execute

#Run the method and then update the settings
def before_settings_update(method):
    
    def execute_and_update(obj, *args, **kwargs):
        
        returns = method(obj, *args, **kwargs)
        
        obj.update_settings(**kwargs, from_decorator = True)
        
        return returns
    
    return execute_and_update 
