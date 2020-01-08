import os

from .session import Session

class BasicSession(Session):

    _sessionName = "Basic session"

    _description = "The most basic session one could have, really."

    def _afterInit(self):
        
        self.addTab("First tab")

        print( os.listdir( self.getSetting("rootDir") ) )
