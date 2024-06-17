# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl._namedindex import NamedIndex

pytestmark = pytest.mark.namedindex


def test_ni_init():
    ni = NamedIndex()
    str(ni)
    ni = NamedIndex("name", [1])
    str(ni)
    ni = NamedIndex(["name-1", "name-2"], [[1], [0]])


def test_ni_iter():
    ni = NamedIndex()
    assert len(ni) == 0
    ni.add_name("name-1", [0])
    assert len(ni) == 1
    ni.add_name("name-2", [1])
    assert len(ni) == 2
    for n in ni:
        assert n in ["name-1", "name-2"]
    assert "name-1" in ni
    assert "name-2" in ni


def test_ni_clear():
    ni = NamedIndex()
    assert len(ni) == 0
    ni.add_name("name-1", [0])
    assert len(ni) == 1
    ni.clear()
    assert len(ni) == 0


def test_ni_copy():
    ni = NamedIndex()
    ni.add_name("name-1", [0])
    ni.add_name("name-2", [1])
    n2 = ni.copy()
    assert ni._name == n2._name


def test_ni_delete():
    ni = NamedIndex()
    ni.add_name("name-1", [0])
    ni.add_name("name-2", [1])
    ni.delete_name("name-1")
    for n in ni:
        assert n in ["name-2"]


def test_ni_items():
    ni = NamedIndex()
    ni["Hello"] = [0]
    ni[[1, 2]] = "Hello-1"
    assert np.all(ni["Hello"] == [0])
    no = ni.remove_index(1)
    assert np.all(no["Hello-1"] == [2])


def test_ni_dict():
    ni = NamedIndex({"r1": [1, 2], "r2": [3, 4]})
    assert len(ni) == 2
    assert np.all(ni["r1"] == [1, 2])
    assert np.all(ni["r2"] == [3, 4])


def test_ni_sub_index():
    ni = NamedIndex({"r1": [1, 2], "r2": [3, 4]})
    no = ni.sub_index([2, 4])
    assert len(no) == 2
    assert len(no["r1"]) == 1
    assert len(no["r2"]) == 1

    no = ni.sub_name("r1")
    assert no.names[0] == "r1"
    assert len(no) == 1


def test_ni_merge():
    ni1 = NamedIndex(["r1", "r2"], [[1, 2], [3, 4]])
    ni2 = ni1.copy()
    ni2.add_name("r3", [5, 6])

    with pytest.raises(ValueError):
        ni1.merge(ni2)
    with pytest.raises(ValueError):
        ni1.merge(ni2, duplicate="raise")
    with pytest.raises(ValueError):
        ni1.merge(ni2, duplicate="not something viable")

    ni3 = ni1.merge(ni2, offset=10, duplicate="union")
    assert len(ni3) == 3
    assert len(ni3["r1"]) == 4
    assert len(ni3["r2"]) == 4
    assert len(ni3["r3"]) == 2

    ni3 = ni1.merge(ni2, offset=10, duplicate="omit")
    assert len(ni3) == 1

    ni3 = ni1.merge(ni2, offset=10, duplicate="left")
    assert len(ni3) == 3
    assert np.array_equal(ni3["r1"], ni1["r1"])
    assert np.array_equal(ni3["r2"], ni1["r2"])
    assert np.array_equal(ni3["r3"], ni2["r3"] + 10)

    ni3 = ni1.merge(ni2, offset=10, duplicate="right")
    assert len(ni3) == 3
    assert np.array_equal(ni3["r1"], ni2["r1"] + 10)
    assert np.array_equal(ni3["r2"], ni2["r2"] + 10)
    assert np.array_equal(ni3["r3"], ni2["r3"] + 10)
