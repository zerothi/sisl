# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import operator as op

import numpy as np
import pytest

from sisl.mixing import AndersonMixer, LinearMixer

pytestmark = pytest.mark.mixing


def test_linear_mixer():
    def scf(f):
        return np.cos(f)

    f = np.linspace(0, 7, 1000)
    mix = LinearMixer()
    s = str(mix)

    dmax = 1
    i = 0
    while dmax > 1e-7:
        i += 1
        df = scf(f) - f
        dmax = np.fabs(df).max()
        f = mix(f, df)


def test_anderson_mixer():
    def scf(f):
        return np.cos(f)

    f = np.linspace(0, 7, 1000)
    mix = AndersonMixer()
    s = str(mix)

    dmax = 1
    i = 0
    while dmax > 1e-7:
        i += 1
        df = scf(f) - f
        dmax = np.fabs(df).max()
        f = mix(f, df)


def test_composite_mixer():
    def scf(f):
        return np.cos(f)

    f = np.linspace(0, 7, 1000)
    mix = AndersonMixer() * 0.1 + 0.9 * LinearMixer()
    s = str(mix)

    dmax = 1
    i = 0
    while dmax > 1e-7:
        i += 1
        df = scf(f) - f
        dmax = np.fabs(df).max()
        f = mix(f, df)


@pytest.mark.parametrize("op", [op.add, op.sub, op.mul, op.truediv, op.pow])
def test_composite_mixer_init(op):
    mix1 = AndersonMixer()
    mix2 = LinearMixer()

    f = np.linspace(0, 7, 1000)
    df = f * 0.1

    _ = op(mix1, mix2)(f, df)
    _ = op(mix1, 1)(f, df)
    _ = op(1, mix1)(f, df)


def test_anderson_mixer_second_step():
    f1 = np.array([0.0, 1.0, 2.0])
    df1 = np.array([0.5, -0.2, 0.1])
    f2 = np.array([0.1, 0.9, 2.1])
    df2 = np.array([0.3, -0.1, 0.2])
    w = 0.1
    mix = AndersonMixer(w)
    assert mix.history.max_elements == 1
    assert np.allclose(mix(f1, df1), f1 + w * df1)

    ddf = df2 - df1
    beta = df2.dot(ddf) / ddf.dot(ddf)
    nin = (1 - beta) * f2 + beta * f1
    nout = (1 - beta) * (f2 + df2) + beta * (f1 + df1)
    out = mix(f2, df2)
    assert np.allclose(out, (1 - w) * nin + w * nout)
    assert not np.allclose(out, f2 + w * df2)


def test_anderson_mixer_same_delta():
    f = np.array([0.0, 1.0, 2.0])
    df = np.array([0.5, -0.2, 0.1])
    mix = AndersonMixer(0.1)
    mix(f, df)
    assert np.allclose(mix(f, df), f + 0.1 * df)
