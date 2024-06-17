# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import warnings as w

import pytest

import sisl.messages as sm

pytestmark = pytest.mark.messages


def test_deprecate():
    with pytest.warns(sm.SislDeprecation):
        sm.deprecate("Deprecation warning")


def test_deprecation():
    with pytest.warns(sm.SislDeprecation):
        w.warn(sm.SislDeprecation("Deprecation warning"))


def test_warn_method():
    with pytest.warns(sm.SislWarning):
        sm.warn("Warning")


def test_warn_specific():
    with pytest.warns(sm.SislWarning):
        sm.warn(sm.SislWarning("Warning"))


def test_warn_category():
    with pytest.warns(sm.SislWarning):
        sm.warn("Warning", sm.SislWarning)


def test_info_method():
    with pytest.warns(sm.SislInfo):
        sm.info("Information")


def test_info_specific():
    with pytest.warns(sm.SislInfo):
        sm.info(sm.SislInfo("Info"))


def test_info_category():
    with pytest.warns(sm.SislInfo):
        sm.info("Information", sm.SislInfo)


def test_error():
    with pytest.raises(sm.SislError):
        raise sm.SislError("This is an error")


def test_exception():
    with pytest.raises(sm.SislException):
        raise sm.SislException("This is an error")


def test_progressbar_true():
    eta = sm.progressbar(2, "Hello", "unit", True)
    eta.update()
    eta.update()
    eta.close()


def test_progressbar_false():
    eta = sm.progressbar(2, "Hello", "unit", False)
    eta.update()
    eta.update()
    eta.close()
