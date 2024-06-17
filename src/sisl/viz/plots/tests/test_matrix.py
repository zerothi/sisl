# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

import sisl
from sisl.viz.plots import atomic_matrix_plot


def test_atomic_matrix_plot():
    pytest.importorskip("skimage", reason="scikit-image not importable")

    graphene = sisl.geom.graphene()
    H = sisl.Hamiltonian(graphene)

    atomic_matrix_plot(H)
