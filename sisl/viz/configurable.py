from copy import deepcopy
import sys

class Configurable:
    
    def initSettings(self, **kwargs):
        
        self.settingsHistory = []
        
        #Get the parameters of all the classes the object belongs to
        self.params = []
        for clss in type.mro(self.__class__):
            if "_parameters" in vars(clss):
                self.params = [*self.params, *clss._parameters]

        #Define the settings dictionary, taking the value of each parameter from kwargs if it is there or from the defaults otherwise.
        self.settings = { param.key: kwargs.get( param.key, deepcopy(param.default) ) for param in self.params}
        
        self.settingsHistory.append(deepcopy(self.settings))

        #Initialize the object where we are going to store what each setting needs to rerun when it is updated
        self.whatToRunOnUpdate = {}
        
        return self
    
    def updateSettings(self, exFromDecorator = False, updateFig = True, **kwargs):
        
        #Initialize the settings in case there are none yet
        if "settings" not in vars(self):
            return self.initSettings(**kwargs)
        
        #Otherwise, update them
        updated = []
        for paramKey, paramValue in kwargs.items():
            
            #It is important to check this, because kwargs may contain other parameters that are not settings
            if paramKey in self.settings.keys() and self.settings[paramKey] != paramValue:
                
                self.settings[paramKey] = paramValue
                updated.append(paramKey)
        
        #Do things after updating the settings
        if len(updated) > 0:
            
            #Record rhe change in the settings history
            self.settingsHistory.append(deepcopy(self.settings))
        
            #Run the functions specified
            if not exFromDecorator and hasattr(self, "_onSettingsUpdate") and updateFig:
                
                #Get the unique names of the functions that should be executed
                funcNames = set([self.whatToRunOnUpdate[settingKey] for settingKey in updated])

                for fName in self._onSettingsUpdate["functions"]:
                    if fName in funcNames:
                        getattr(self, fName)()
                        
                        #If we need to execute all the functions keep going thorugh the loop, else stop here
                        if not self._onSettingsUpdate["config"].get("multipleFunc", False):
                            break
            
        return self
    
    def undoSettings(self, nsteps = 1):
        
        try:
            self.settingsHistory = self.settingsHistory[0:-nsteps]         
            self.updateSettings( **deepcopy(self.settingsHistory[-1]) )
        except IndexError:
            print("This instance of {} does not contain earlier settings as requested ({} step{} back)"
                 .format(self.__class__.__name__, nsteps, "" if nsteps == 1 else "s"))
            pass
            
        return self
    
    def undoSetting(self, settingKey):
        '''
        Undoes only a particular setting and lives the others unchanged.

        At the moment it is a 'fake' undo function, since it actually updates the settings.

        '''

        #Get the actual settings for that group
        actualSetting = self.getSetting(settingKey)

        #Try to find any different values for the settings
        for pastValue in reversed(self.getSettingHistory(settingKey)):

            if pastValue != actualSetting:

                return self.updateSettings( **{settingKey: pastValue} )
        else:
            print("There is no registry of the setting '{}' having been changed. Sorry :(".format(settingKey))
            
            return self

    def undoSettingsGroup(self, groupKey):

        '''
        Takes the desired group of settings one step back, but the rest of the settings remain unchanged.

        At the moment it is a 'fake' undo function, since it actually updates the settings.

        '''

        #Get the actual settings for that group
        actualSettings = self.getSettingsGroup(groupKey)

        #Try to find any different values for the settings
        for i in range(len(self.settingsHistory)):

            previousSettings = self.getSettingsGroup(groupKey, stepsBack = i)

            if previousSettings != actualSettings:

                return self.updateSettings(previousSettings)
        else:
            print("There is no registry of any setting of the group '{}' having been changed. Sorry :(".format(groupKey))
            
            return self
    
    def getParam(self, settingKey, justDict = True, paramsExtractor = False):
        '''
        Gets the parameter for a given setting. 
        
        By default it returns its dictionary, so that one can check the information that it contains.
        You can ask for the parameter itself by setting justDict to False. However, if you want to
        modify the parameter you should use the modifyParam() method instead.

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
    
    def modifyParam(self, settingKey, *args, **kwargs):
        '''
        Modifies a parameter using a provided function

        Arguments
        --------
        settingKey: str
            The key of the parameter to be modified
        *args:
            Depending on what you pass the setting will be modified in different ways:
                - Two arguments:
                    the first argument will be interpreted as the attribute that you want to change,
                    and the second one as the value that you want to set.

                    Ex: obj.modifyParam("length", "default", 3)
                    will set the default attribute of the parameter with key "length" to 3
                
                - One argument and it is a dictionary:
                    the keys will be interpreted as attributes that you want to change and the values
                    as the value that you want them to have. 
                
                - One argument and it is a function:

                    the function will recieve the parameter and can act on it in any way you like.
                    It doesn't need to return the parameter, just modify it.
                    In this function, you can call predefined methods of the parameter, for example.

                    Ex: obj.modifyParam("length", lambda param: param.incrementByOne() )

                    given that you know that this type of parameter has this method
        **kwargs: optional
            They are passed directly to the Configurable.getParam method to retrieve the parameter.

        Returns
        --------
        self:
            The configurable object.
        '''

        if len(args) == 2:
           
            modFunction = lambda obj: setattr(obj, *args)

        elif isinstance(args[0], dict):

            def modFunction(obj):
                for attr, val in args[0].items():
                    setattr(obj, attr, val)

        elif callable(args[0]):

            modFunction = args[0]

        modFunction(self.getParam(settingKey, justDict = False, **kwargs))

        return self

    def getSetting(self, settingKey):

        '''
        Gets the value for a given setting .
        '''
        
        #A setting can be a function that returns the true value of the setting
        if callable( self.settings[settingKey] ):
            return self.settings[settingKey](self)

        return deepcopy(self.settings[settingKey])
    
    def setting(self, settingKey):

        '''
        Gets the value for a given setting while logging where it has been required. 
        
        THIS METHOD MUST BE USED IN DEVELOPEMENT! (And should not be used by users, use getSetting() instead)
        
        It stores where the setting has been demanded so that the plot can be efficiently updated when it is modified.
        '''

        #Get the last frame
        frame = sys._getframe().f_back
        #And then iterate over all the past frames to get their names
        while frame:

            #Get the function name
            funcName = frame.f_code.co_name
            
            #If it is in the list of functions provided on class definition (e.g. on Plot class) store it
            if funcName in self._onSettingsUpdate["functions"]:
                self.whatToRunOnUpdate[settingKey] = funcName
                break
            
            frame = frame.f_back
        
        return self.getSetting(settingKey)

    def getSettingHistory(self, settingKey):
        
        return deepcopy([step[settingKey] for step in self.settingsHistory])
    
    def getSettingsGroup(self, groupKey, stepsBack = 0):
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
        settingsGroup: dict
            A subset of the settings with only those that belong to the asked group.
        '''

        if stepsBack:
            settings = self.settingsHistory[-stepsBack]
        else:
            settings = self.settings

        return deepcopy({ setting.key: settings[setting.key] for setting in self.params if getattr(setting, "group", None) == groupKey })

    def settingsUpdatesLog(self, frame = -1):
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

    def isDefault(self, settingKey):

        '''
        Checks if the current value for a setting is the default one.
        
        DOESN'T WORK FOR VALUES THAT ARE FUNCTIONS!
        '''

        return self.settings[settingKey] == self.getParam(settingKey)["default"]

#DECORATORS TO USE WHEN DEFINING METHODS IN CLASSES THAT INHERIT FROM Configurable
#Run the method after having initialized the settings
def afterSettingsInit(method):
    
    def updateAndExecute(obj, *args, **kwargs):
        
        obj.initSettings(**kwargs)
        
        return method(obj, *args, **kwargs)
    
    return updateAndExecute

#Run the method and then initialize the settings
def beforeSettingsInit(method):
    
    def updateAndExecute(obj, *args, **kwargs):
        
        returns = method(obj, *args, **kwargs)
        
        obj.initSettings(**kwargs)
        
        return returns
    
    return updateAndExecute

#Run the method after having updated the settings
def afterSettingsUpdate(method):

    def updateAndExecute(obj, *args, **kwargs):
        
        obj.updateSettings(**kwargs, exFromDecorator = True)
        
        return method(obj, *args, **kwargs)
    
    return updateAndExecute

#Run the method and then update the settings
def beforeSettingsUpdate(method):
    
    def updateAndExecute(obj, *args, **kwargs):
        
        returns = method(obj, *args, **kwargs)
        
        obj.updateSettings(**kwargs, exFromDecorator = True)
        
        return returns
    
    return updateAndExecute  
