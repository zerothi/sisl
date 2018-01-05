from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl.eigensystem import EigenSystem

pytestmark = pytest.mark.eigen


def crt(n):
    e = np.random.rand(n)
    v = np.random.rand(n, n)
    return EigenSystem(e, v)


def test_print():
    es = crt(10)
    print(es)


def test_len_size():
    es = crt(10)
    assert len(es) == 10
    assert es.size() == 10


def test_get():
    es = crt(10)
    assert np.allclose(es.e[2], es[2][0])
    assert np.allclose(es.v[2, :], es[2][1])


def test_iter():
    ES = crt(10)
    E = ES.e[:]
    V = ES.v[:, :]
    for i, es in enumerate(ES):
        assert np.allclose(es.e, E[i])
        assert np.allclose(es.v, V[i])
        ess = ES.sub(i)
        assert np.allclose(es.e, ess.e)
        assert np.allclose(es.v, ess.v)
    for i, e in enumerate(ES.iter(only_e=True)):
        assert np.allclose(e, E[i])
    for i, v in enumerate(ES.iter(only_v=True)):
        assert np.allclose(v, V[i])
    for i, (e, v) in enumerate(ES.iter(True, True)):
        assert np.allclose(e, E[i])
        assert np.allclose(v, V[i])


def test_sort():
    es = crt(10)
    es_copy = es.copy()
    es_copy.sort()
    assert np.allclose(np.sort(es.e), es_copy.e)


def test_outer():
    es = crt(10)
    es.outer()
