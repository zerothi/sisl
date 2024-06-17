# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import math as m

import numpy as np
import pytest

import sisl

pytestmark = pytest.mark.version


def test_version():
    sisl.__bibtex__
    sisl.__version__


def test_import_simple():
    # The imports should only be visible in the io module
    sisl.BaseSile
    sisl.Sile
    sisl.SileCDF
    sisl.SileBin
    sisl.io.xyzSile


def test_import_in_io():
    # The imports should only be visible in the io module
    with pytest.raises(AttributeError):
        sisl.xyzSile


def test_import_in_io_from():
    # The imports should only be visible in the io module
    with pytest.raises(ImportError):
        from sisl import xyzSile


def test_dispatch_methods():
    # sisl exposes some dispatch methods via
    #  sisl._ufuncs and sisl._core._*_ufuncs.py
    # For instance tile is the first, true dispatch
    # method used.
    sisl.tile


def test_dispatch_methods_not_allowed():
    with pytest.raises(sisl.SislError):
        sisl.tile(2)
