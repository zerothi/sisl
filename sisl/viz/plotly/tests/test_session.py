from copy import deepcopy

import numpy as np


from sisl.viz import Session, Plot

from sisl.viz.plotly.plots import *
from sisl.viz.plotly.tests.test_plot import BasePlotTester

# This file tests general session behavior

# ------------------------------------------------------------
# Checks that will be available to be used on any session class
# ------------------------------------------------------------

class BaseSessionTester:

    SessionClass = Session

    def test_session_settings(self):

        session = self.SessionClass()
        # Check that all the parameters have been passed to the settings
        assert np.all([param.key in session.settings for param in self.SessionClass._parameters])
        # Build some test settings
        new_settings = {'root_dir': 'Test', 'search_depth': [4,6]}
        # Update settings and check they have been succesfully updated
        old_settings = deepcopy(session.settings)
        session.update_settings(**new_settings, run_updates=False)
        assert np.all([session.settings[key] == val for key, val in new_settings.items()])
        # Undo settings and check if they go back to the previous ones
        session.undo_settings(run_updates=False)
        assert np.all([session.settings[key] ==
                    val for key, val in old_settings.items()])
        
        # Build a session directly with test settings and check if it works
        session = self.SessionClass(**new_settings)
        assert np.all([session.settings[key] == val for key, val in new_settings.items()])

    def test_session_connected(self):

        session = self.SessionClass()

        # Check that the session has a socketio attribute and that it can be changed
        # Seems dumb, but socketio is really a property that uses functions to set the
        # real attribute, so they may be broken by something
        assert hasattr(session, 'socketio')
        assert session.socketio is None
        session.socketio = 2
        assert session.socketio == 2

        # Check that if we add a plot to the session, their socketio will be tracked by
        # the session
        new_plot = Plot()
        session.add_tab('Test tab')
        session.add_plot(new_plot, 'Test tab')
        assert new_plot.socketio == 2, f'Socketio not transfered from {session.__class__} to plot on add_plot'
        # Fake a disconnection of the session and see if the plot follows
        session.socketio = None
        assert new_plot.socketio is None, f'Socketio change in {session.__class__} not transmitted to plots'

    def test_save_and_load(self):

        BasePlotTester.test_save_and_load(self, obj=self.SessionClass())

        
# ------------------------------------------------------------
#           Actual tests on the Session parent class
# ------------------------------------------------------------

class TestSession(BaseSessionTester):

    SessionClass = Session