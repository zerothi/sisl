from __future__ import print_function, division

import pytest

import numpy as np

from sisl._namedindex import NamedIndex


def test_ni_init():

    ni = NamedIndex()
    ni = NamedIndex('name', [1])
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
