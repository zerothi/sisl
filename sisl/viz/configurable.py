from copy import deepcopy

class Configurable:
    
    def initSettings(self, **kwargs):
        
        self.settingsHistory = []
        
        #Get the parameters of all the classes the object belongs to
        self.params = []
        for clss in type.mro(self.__class__):
            if hasattr(clss, "_parameters"):
                self.params = [*self.params, *clss._parameters]

        #Define the settings dictionary, taking the value of each parameter from kwargs if it is there or from the defaults otherwise.
        self.settings = { param["key"]: kwargs.get( param["key"], deepcopy(param["default"]) ) for param in self.params}
        
        self.settingsHistory.append(deepcopy(self.settings))
        
        return self
    
    def updateSettings(self, exFromDecorator = False, **kwargs):
        
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
            if not exFromDecorator and hasattr(self, "_onSettingsUpdate"):
                
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
    
    def getSettingHistory(self, settingKey):
        
        return deepcopy([step[settingKey] for step in self.settingsHistory])
    
    def getSettingsGroup(self, groupKey):
        '''
        Gets the subset of the settings that corresponds to a given group
        '''

        return deepcopy({ setting["key"]: self.settings[setting["key"]] for setting in self.params if setting.get("group", None) == groupKey })


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
