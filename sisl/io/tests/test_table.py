from __future__ import print_function, division

import pytest

import numpy as np

from sisl.io.table import *

pytestmark = pytest.mark.io
_dir = 'sisl/io'


def test_tbl1(sisl_tmp):
    dat0 = np.arange(2)
    dat1 = np.arange(2) + 1

    io0 = tableSile(sisl_tmp('t0.dat', _dir), 'w')
    io1 = tableSile(sisl_tmp('t1.dat', _dir), 'w')
    io0.write_data(dat0, dat1)
    io1.write_data((dat0, dat1))

    F0 = open(io0.file).readlines()
    F1 = open(io1.file).readlines()
    assert all([l0 == l1 for l0, l1 in zip(F0, F1)])


def test_tbl2(sisl_tmp):
    dat0 = np.arange(8).reshape(2, 2, 2)
    dat1 = np.arange(8).reshape(2, 2, 2) + 1

    io0 = tableSile(sisl_tmp('t0.dat', _dir), 'w')
    io1 = tableSile(sisl_tmp('t1.dat', _dir), 'w')
    io0.write_data(dat0, dat1)
    io1.write_data((dat0, dat1))

    F0 = open(io0.file).readlines()
    F1 = open(io1.file).readlines()
    assert all([l0 == l1 for l0, l1 in zip(F0, F1)])


def test_tbl3(sisl_tmp):
    dat0 = np.arange(8).reshape(2, 2, 2)
    dat1 = np.arange(8).reshape(2, 2, 2) + 1
    DAT = np.stack([dat0, dat1])
    DAT.shape = (-1, 2, 2)

    io = tableSile(sisl_tmp('t.dat', _dir), 'w')
    io.write_data(dat0, dat1)
    dat = tableSile(io.file, 'r').read_data()
    assert np.allclose(dat, DAT)
    io = tableSile(io.file, 'w')
    io.write_data((dat0, dat1))
    dat = tableSile(io.file, 'r').read_data()
    assert np.allclose(dat, DAT)


def test_tbl4(sisl_tmp):
    dat0 = np.arange(8)
    dat1 = np.arange(8) + 1
    DAT = np.stack([dat0, dat1])

    io = tableSile(sisl_tmp('t.dat', _dir), 'w')
    io.write_data(dat0, dat1)
    dat = tableSile(io.file, 'r').read_data()
    assert np.allclose(dat, DAT)
    io = tableSile(io.file, 'w')
    io.write_data((dat0, dat1))
    dat = tableSile(io.file, 'r').read_data()
    assert np.allclose(dat, DAT)


def test_tbl_automatic_stack(sisl_tmp):
    dat0 = np.arange(4)
    dat1 = np.arange(8).reshape(2, 4) + 1
    DAT = np.vstack([dat0, dat1])

    io = tableSile(sisl_tmp('t.dat', _dir), 'w')
    io.write_data(dat0, dat1)
    dat = tableSile(io.file, 'r').read_data()
    assert np.allclose(dat, DAT)
    io = tableSile(io.file, 'w')
    io.write_data((dat0, dat1))
    dat = tableSile(io.file, 'r').read_data()
    assert np.allclose(dat, DAT)


@pytest.mark.parametrize("delimiter", ['\t', ' ', ',', ':', 'M'])
def test_tbl5(sisl_tmp, delimiter):
    dat0 = np.arange(8)
    dat1 = np.arange(8) + 1
    DAT = np.stack([dat0, dat1])

    io = tableSile(sisl_tmp('t.dat', _dir), 'w')
    io.write_data(dat0, dat1, delimiter=delimiter)
    if delimiter in ['\t', ' ', ',']:
        dat = tableSile(io.file, 'r').read_data()
        assert np.allclose(dat, DAT)
    dat = tableSile(io.file, 'r').read_data(delimiter=delimiter)
    assert np.allclose(dat, DAT)
