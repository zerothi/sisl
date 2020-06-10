'''
These tests check that all session subclasses fulfill at least the most basic stuff
More tests should be run on each session, but these are the most basic ones to
ensure that at least they do not break basic session functionality.
'''

from sisl.viz.plotly.tests.test_session import BaseSessionTester
from sisl.viz import Session
from sisl.viz.plotly.sessions import *

def get_basic_functionality_test(SessionSubClass):

    class BasicSubClassTest(BaseSessionTester):

        SessionClass = SessionSubClass

    return BasicSubClassTest 


for SessionSubClass in Session.__subclasses__():

    globals()[f'Test{SessionSubClass.__name__}'] = get_basic_functionality_test(SessionSubClass)