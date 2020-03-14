import os

from .session import Session

class BlankSession(Session):

    _sessionName = "Blank session"

    _description = "The most basic session one could have, really."

    def _afterInit(self):
        
        self.addTab("First tab")