from __future__ import print_function, division

import pytest

import os
import numpy as np

from sisl.io.table import *


import common as tc

_C = type('Temporary', (object, ), {})

join = os.path.join


def setup_module(module):
    tc.setup(module._C)


def teardown_module(module):
    tc.teardown(module._C)


@pytest.mark.io
class TestTable(object):

    def test_tbl1(self):
        dat0 = np.arange(2)
        dat1 = np.arange(2) + 1

        io0 = TableSile(join(_C.d, 't0.dat'), 'w')
        io1 = TableSile(join(_C.d, 't1.dat'), 'w')
        io0.write_data(dat0, dat1)
        io1.write_data((dat0, dat1))

        F0 = open(io0.file).readlines()
        F1 = open(io1.file).readlines()
        assert all([l0 == l1 for l0, l1 in zip(F0, F1)])

        os.remove(io0.file)
        os.remove(io1.file)

    def test_tbl2(self):
        dat0 = np.arange(8).reshape(2, 2, 2)
        dat1 = np.arange(8).reshape(2, 2, 2) + 1

        io0 = TableSile(join(_C.d, 't0.dat'), 'w')
        io1 = TableSile(join(_C.d, 't1.dat'), 'w')
        io0.write_data(dat0, dat1)
        io1.write_data((dat0, dat1))

        F0 = open(io0.file).readlines()
        F1 = open(io1.file).readlines()
        assert all([l0 == l1 for l0, l1 in zip(F0, F1)])

        os.remove(io0.file)
        os.remove(io1.file)

    def test_tbl3(self):
        dat0 = np.arange(8).reshape(2, 2, 2)
        dat1 = np.arange(8).reshape(2, 2, 2) + 1
        DAT = np.stack([dat0, dat1])
        DAT.shape = (-1, 2, 2)

        io = TableSile(join(_C.d, 't0.dat'), 'w')
        io.write_data(dat0, dat1)
        dat = TableSile(io.file, 'r').read_data()
        assert np.allclose(dat, DAT)
        io = TableSile(io.file, 'w')
        io.write_data((dat0, dat1))
        dat = TableSile(io.file, 'r').read_data()
        assert np.allclose(dat, DAT)

        os.remove(io.file)

    def test_tbl4(self):
        dat0 = np.arange(8)
        dat1 = np.arange(8) + 1
        DAT = np.stack([dat0, dat1])

        io = TableSile(join(_C.d, 't0.dat'), 'w')
        io.write_data(dat0, dat1)
        dat = TableSile(io.file, 'r').read_data()
        assert np.allclose(dat, DAT)
        io = TableSile(io.file, 'w')
        io.write_data((dat0, dat1))
        dat = TableSile(io.file, 'r').read_data()
        assert np.allclose(dat, DAT)

        os.remove(io.file)

    @pytest.mark.parametrize("delimiter", ['\t', ' ', ',', ':', 'M'])
    def test_tbl5(self, delimiter):
        dat0 = np.arange(8)
        dat1 = np.arange(8) + 1
        DAT = np.stack([dat0, dat1])

        io = TableSile(join(_C.d, 't0.dat'), 'w')
        io.write_data(dat0, dat1, delimiter=delimiter)
        if delimiter in ['\t', ' ', ',']:
            dat = TableSile(io.file, 'r').read_data()
            assert np.allclose(dat, DAT)
        dat = TableSile(io.file, 'r').read_data(delimiter=delimiter)
        assert np.allclose(dat, DAT)

        os.remove(io.file)
