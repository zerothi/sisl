# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

from sisl.viz.processors.spin import get_spin_options

pytestmark = [pytest.mark.viz, pytest.mark.processors]


def test_get_spin_options():
    # Unpolarized spin
    assert len(get_spin_options("unpolarized")) == 0

    # Polarized spin
    options = get_spin_options("polarized")
    assert len(options) == 4
    assert 0 in options
    assert 1 in options
    assert "total" in options
    assert "z" in options

    # Non colinear spin
    options = get_spin_options("noncolinear")
    assert len(options) == 4
    assert "total" in options
    assert "x" in options
    assert "y" in options
    assert "z" in options

    options = get_spin_options("noncolinear", only_if_polarized=True)
    assert len(options) == 0
