from __future__ import print_function, division

import pytest

import numpy as np

from sisl._namedindex import NamedIndex


pytestmark = pytest.mark.namedindex


def test_ni_init():

    ni = NamedIndex()
    str(ni)
    ni = NamedIndex('name', [1])
    str(ni)
    ni = NamedIndex(['name-1', 'name-2'], [[1], [0]])


def test_ni_iter():
    ni = NamedIndex()
    assert len(ni) == 0
    ni.add_name('name-1', [0])
    assert len(ni) == 1
    ni.add_name('name-2', [1])
    assert len(ni) == 2
    for n in ni:
        assert n in ['name-1', 'name-2']
    assert 'name-1' in ni
    assert 'name-2' in ni


def test_ni_copy():
    ni = NamedIndex()
    ni.add_name('name-1', [0])
    ni.add_name('name-2', [1])
    n2 = ni.copy()
    assert ni._name == n2._name


def test_ni_delete():
    ni = NamedIndex()
    ni.add_name('name-1', [0])
    ni.add_name('name-2', [1])
    ni.delete_name('name-1')
    for n in ni:
        assert n in ['name-2']


def test_ni_items():
    ni = NamedIndex()
    ni['Hello'] = [0]
    ni[[1, 2]] = 'Hello-1'
    assert np.all(ni['Hello'] == [0])
    ni.remove(1)
    assert np.all(ni['Hello-1'] == [2])


def test_ni_add():
    ni1 = NamedIndex(["r1", "r2"], [[1, 2], [3, 4]])
    ni2 = ni1.copy()
    ni2.add_name("r3", [5, 6])

    with pytest.raises(ValueError):
        ni1.add(ni2)

    ni3 = ni1.add(ni2, offset=10, duplicates="union")
    assert len(ni3) == 3
    assert all(len(ni3[r]) == 4 for r in ("r1", "r2"))
    assert len(ni3["r3"]) == 2

    ni3 = ni1.add(ni2, offset=10, duplicates="omit")
    assert len(ni3) == 1

    ni3 = ni1.add(ni2, offset=10, duplicates="left")
    assert len(ni3) == 3
    assert all(np.array_equal(ni3[r], ni1[r]) for r in ("r1", "r2"))
    assert np.array_equal(ni3["r3"], ni2["r3"] + 10)

    ni3 = ni1.add(ni2, offset=10, duplicates="right")
    assert len(ni3) == 3
    assert all(np.array_equal(ni3[r], ni2[r] + 10) for r in ("r1", "r2"))
    assert np.array_equal(ni3["r3"], ni2["r3"] + 10)
