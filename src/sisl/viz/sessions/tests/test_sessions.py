# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
These tests check that all session subclasses fulfill at least the most basic stuff
More tests should be run on each session, but these are the most basic ones to
ensure that at least they do not break basic session functionality.
"""
import pytest

from sisl.viz.tests.test_session import _TestSessionClass
from sisl.viz import Session
from sisl.viz.sessions import *

pytestmark = [pytest.mark.viz, pytest.mark.plotly]


@pytest.fixture(autouse=True, scope="class", params=Session.__subclasses__())
def plot_class(request):
    request.cls._cls = request.param


class TestSessionSubClass(_TestSessionClass):
    pass
