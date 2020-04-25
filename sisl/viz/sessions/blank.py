import os

from ..session import Session

class BlankSession(Session):

    _sessionName = "Blank session"

    _description = "The most basic session one could have, really."

    def _after_init(self):
        
        self.add_tab("First tab")