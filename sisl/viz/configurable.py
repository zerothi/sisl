from copy import deepcopy

#Class that should be inherited by all classes that contain settings
class Configurable:
    
    def initSettings(self, **kwargs):

        #Get the parameters of all the classes the object belongs to
        self.params = []
        for clss in type.mro(self.__class__):
            if hasattr(clss, "_parameters"):
                self.params = [*self.params, *clss._parameters]

        #Define the settings dictionary, taking the value of each parameter from kwargs if it is there or from the defaults otherwise.
        self.settings = { param["key"]: kwargs.get( param["key"], deepcopy(param["default"]) ) for param in self.params}
            
        return self
    
    def updateSettings(self, **kwargs):
        
        #Initialize the settings in case there are none yet
        if "settings" not in vars(self):
            return self.initSettings(**kwargs)
        
        #Otherwise, update them
        for paramKey, paramValue in kwargs.items():
            
            if paramKey in self.settings.keys():
                #It is important to check this, because kwargs may contain other parameters that are not settings
                self.settings[paramKey] = paramValue
        
        return self


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
        
        obj.updateSettings(**kwargs)
        
        return method(obj, *args, **kwargs)
    
    return updateAndExecute

#Run the method and then update the settings
def beforeSettingsUpdate(method):
    
    def updateAndExecute(obj, *args, **kwargs):
        
        returns = method(obj, *args, **kwargs)
        
        obj.updateSettings(**kwargs)
        
        return returns
    
    return updateAndExecute  
