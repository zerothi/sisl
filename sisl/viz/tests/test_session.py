# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from copy import deepcopy
import pytest

import numpy as np

from sisl.viz import Session
from sisl.viz.plots import *
from sisl.viz.tests.test_plot import _TestPlotClass

try:
    import dill
    skip_dill = pytest.mark.skipif(False, reason="dill not available")
except ImportError:
    skip_dill = pytest.mark.skipif(True, reason="dill not available")

# This file tests general session behavior

# ------------------------------------------------------------
# Checks that will be available to be used on any session class
# ------------------------------------------------------------

pytestmark = [pytest.mark.viz, pytest.mark.plotly]


class _TestSessionClass:

    _cls = Session

    def test_session_settings(self):

        session = self._cls()
        # Check that all the parameters have been passed to the settings
        assert np.all([param.key in session.settings for param in self._cls._parameters])
        # Build some test settings
        new_settings = {'root_dir': 'Test', 'search_depth': [4, 6]}
        # Update settings and check they have been succesfully updated
        old_settings = deepcopy(session.settings)
        session.update_settings(**new_settings, run_updates=False)
        assert np.all([session.settings[key] == val for key, val in new_settings.items()])
        # Undo settings and check if they go back to the previous ones
        session.undo_settings(run_updates=False)
        assert np.all([session.settings[key] ==
                    val for key, val in old_settings.items()])

        # Build a session directly with test settings and check if it works
        session = self._cls(**new_settings)
        assert np.all([session.settings[key] == val for key, val in new_settings.items()])

    @skip_dill
    def test_save_and_load(self):

        _TestPlotClass.test_save_and_load(self, obj=self._cls())


# ------------------------------------------------------------
#           Actual tests on the Session parent class
# ------------------------------------------------------------

class TestSession(_TestSessionClass):

    _cls = Session
