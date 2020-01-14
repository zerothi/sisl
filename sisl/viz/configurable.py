from copy import deepcopy

class Configurable:
    
    def initSettings(self, **kwargs):
        
        self.settingsHistory = []
        
        #Get the parameters of all the classes the object belongs to
        self.params = []
        for clss in type.mro(self.__class__):
            if "_parameters" in vars(clss):
                self.params = [*self.params, *clss._parameters]

        #Define the settings dictionary, taking the value of each parameter from kwargs if it is there or from the defaults otherwise.
        self.settings = { param["key"]: kwargs.get( param["key"], deepcopy(param["default"]) ) for param in self.params}
        
        self.settingsHistory.append(deepcopy(self.settings))
        
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
                funcNames = set([param.get("onUpdate", None) for param in self.params if param["key"] in updated])
                
                if self._onSettingsUpdate["__config"]["multipleFunc"]:
                    
                    #Execute all the functions
                    funcs = [self._onSettingsUpdate[fName](self) for fName in funcNames]
                    
                    for f in funcs:
                        f()
                
                else:
                    
                    #Execute only the most important function
                    for fName in self._onSettingsUpdate["__config"]["importanceOrder"]:
                        
                        if fName in funcNames:
                            #Execute this function, as it is the most important (presumably the most deep)
                            self._onSettingsUpdate[fName](self)()
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
    
    def getParam(self, settingKey):
        '''
        Gets the parameter info for a given setting
        '''

        for param in self.params:
            if param["key"] == settingKey:
                return param
        else:
            return None

    def getSetting(self, settingKey):

        '''
        Gets the value for a given setting.
        '''
        
        #A setting can be a function that returns the true value of the setting
        if callable( self.settings[settingKey] ):
            return self.settings[settingKey](self)

        return deepcopy(self.settings[settingKey])

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

        return deepcopy({ setting["key"]: settings[setting["key"]] for setting in self.params if setting.get("group", None) == groupKey })
    
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
