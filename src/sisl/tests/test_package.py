# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest

import math as m
import numpy as np

import sisl

pytestmark = pytest.mark.version


def test_version():
    sisl.__bibtex__
    sisl.__version__


def test_import1():
    # The imports should only be visible in the io module
    s = sisl.BaseSile
    s = sisl.Sile
    s = sisl.SileCDF
    s = sisl.SileBin
    s = sisl.io.xyzSile


def test_import2():
    # The imports should only be visible in the io module
    with pytest.raises(AttributeError):
        sisl.xyzSile


def test_import3():
    # The imports should only be visible in the io module
    with pytest.raises(ImportError):
        from sisl import xyzSile
