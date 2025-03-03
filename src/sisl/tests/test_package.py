# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

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


def test_submodule_attr_access():
    for mod in (
        "geom",
        "io",
        "physics",
        "linalg",
        "shape",
        "mixing",
        "utils",
        "unit",
        "C",
        "constant",
    ):
        getattr(sisl, mod)


def test_submodule_attr_access_viz():
    pytest.importorskip("plotly")
    sisl.viz


def test_import_in_io():
    # The imports should only be visible in the io module
    with pytest.raises(AttributeError):
        sisl.xyzSile


def test_import_in_io_from():
    # The imports should only be visible in the io module
    with pytest.raises(ImportError):
        from sisl import xyzSile  # noqa: F401


@pytest.mark.parametrize("obj", [dict(), 2])
def test_dispatch_methods_not_allowed(obj):
    with pytest.raises(sisl.SislError):
        sisl.tile(obj, 0, 2)
