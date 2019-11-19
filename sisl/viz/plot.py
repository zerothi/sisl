'''
This file contains the Plot class, which should be inherited by all plot classes
'''
import time
import os

import sisl

from .configurable import *

PLOTS_CONSTANTS = {
    "spins": ["up", "down"],
    "readFuncs": {
        "fromH": lambda obj: obj._readfromH, 
        "siesOut": lambda obj: obj._readSiesOut
    }
}

class Plot(Configurable):
    
    #These are the possible ways of reading data
    _parameters = (
        
        {
            "key": "readingOrder",
            "name": "Output reading/generating order",
            "default": ("guiOut", "siesOut", "fromH")
        },
        
    )
    
    @afterSettingsInit
    def __init__(self, **kwargs):
        
        #Give an ID to the plot
        self.id = time.time()

        if "rootFdf" in kwargs.keys():
            
            #Set the other relevant files
            self.setFiles(kwargs["rootFdf"])

            #Try to read the hamiltonian
            if "readHamiltonian" in kwargs.keys() and kwargs["readHamiltonian"]:
                try:
                    self.setupHamiltonian()
                except Exception:
                    log.warning("Unable to find or read {}.HSX".format(self.struct))
                    pass
            
            #Process data in the required files, optimally to build a dataframe that can be queried afterwards
            dataRead = self.readData(**kwargs)
    
    def __str__(self):
        
        string = '''
    Plot class: {}    Plot type: {}
        
    Settings: 
    {}
        
        '''.format(
            self.__class__.__name__,
            self._plotType,
            "\n".join([ "\t- {}: {}".format(key,value) for key, value in self.settings.items()])
        )
        
        return string
    
    def setFiles(self, rootFdf):
        '''
        Checks if the required files are available and then builds a list with them
        '''
        #Set the fdfSile
        self.rootFdf = rootFdf
        self.rootDir, fdfFile = os.path.split( self.rootFdf )
        self.fdfSile = sisl.get_sile(self.rootFdf)
        self.struct = self.fdfSile.get("SystemLabel")
            
        #Check that the required files are there
        #if RequirementsFilter().check(self.rootFdf, self.__class__.__name__ ):
        if True:
            #If they are there, we can confidently build this list
            self.requiredFiles = [ os.path.join( self.rootDir, req.replace("$struct$", self.struct) ) for req in self.__class__._requirements["files"] ]
        else:
            log.error("\t the required files were not found, please check your file system.")
            raise Exception("The required files were not found, please check your file system.")

        return self



        return self
    
    def setupHamiltonian(self):
        '''
        Sets up the hamiltonian for calculations with sisl.
        '''
        
        self.geom = self.fdfSile.read_geometry(output = True)

        #Try to read the hamiltonian in two different ways
        try:
            #This one is favoured because it may read from TSHS file, which contains all the information of the geometry and basis already
            self.H = self.fdfSile.read_hamiltonian()
        except Exception:
            Hsile = sisl.get_sile(os.path.join(self.rootDir, self.struct + ".HSX"))
            self.H = Hsile.read_hamiltonian(geom = self.geom)

        self.fermi = self.H.fermi_level()
    
    def _readFromSources(self):
        
        '''
        Tries to read the data from the different possible sources in the order 
        determined by self.settings["readingOrder"].
        '''
        
        errors = []
        #Try to read in the order specified by the user
        for source in self.settings["readingOrder"]:
            try:
                #Get the reading function
                readingFunc = PLOTS_CONSTANTS["readFuncs"][source](self)
                #Execute it
                data = readingFunc()
                self.source = source
                return data
            except Exception as e:
                errors.append("\t- {}: {}.{}".format(source, type(e).__name__, e))
                
        else:
            raise Exception("Could not read or generate data for {} from any of the possible sources.\n\n Here are the errors for each source:\n\n {}  "
                            .format(self.__class__.__name__, "\n".join(errors)) )
        
