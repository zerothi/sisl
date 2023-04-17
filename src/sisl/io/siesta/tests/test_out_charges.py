# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from itertools import product
import sys
import pytest
import os.path as osp
import sisl
from sisl.io.siesta.out import *
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta', 'outs')

# tests here tests charge reads for output
#  voronoi + hirshfeld: test_vh_*
#  voronoi: test_v_*
#  hirshfeld: test_h_*
#  mulliken: test_m_*

Opt = sisl.Opt
SileError = sisl.SileError


def with_pandas():
    try:
        import pandas
        return True
    except ImportError:
        return False


@pytest.mark.parametrize('name', ["voronoi", "Hirshfeld"])
def test_vh_empty_file(name, sisl_files):
    f = sisl_files(_dir, "voronoi_hirshfeld_4.1_none.out")
    out = outSileSiesta(f)

    with pytest.raises(SileError, match="any charges"):
        out.read_charge(name)

    with pytest.raises(SileError, match="any charges"):
        out.read_charge(name, iscf=-1, imd=Opt.NONE)

    with pytest.raises(SileError, match="any charges"):
        out.read_charge(name, iscf=-1, imd=-1)

    with pytest.raises(SileError, match="any charges"):
        out.read_charge(name, iscf=None, imd=-1)


@pytest.mark.parametrize('name', ["voronoi", "Hirshfeld"])
def test_vh_final(name, sisl_files):
    f = sisl_files(_dir, "voronoi_hirshfeld.out")
    out = outSileSiesta(f)

    q = out.read_charge(name, iscf=None, imd=None)
    assert q.size > 0
    assert not all(v is None for v in q)

    q0 = out.read_charge(name, iscf=Opt.NONE, imd=Opt.NONE)
    assert np.allclose(q, q0)

    with pytest.raises(SileError, match="MD/SCF charges"):
        out.read_charge(name, iscf=-1, imd=Opt.NONE)

    with pytest.raises(SileError, match="MD/SCF charges"):
        out.read_charge(name, iscf=-1, imd=-1)

    with pytest.raises(SileError, match="MD/SCF charges"):
        out.read_charge(name, iscf=None, imd=-1)

    if with_pandas():
        df = out.read_charge(name, as_dataframe=True)
        assert np.allclose(df.values, q)


@pytest.mark.parametrize('fname', ["md", "4.1_pol_md", "nc_md"])
@pytest.mark.parametrize('name', ["voronoi", "Hirshfeld"])
def test_vh_md(name, fname, sisl_files):
    #  voronoi_hirshfeld_md.out
    f = sisl_files(_dir, f"voronoi_hirshfeld_{fname}.out")
    out = outSileSiesta(f)

    q = out.read_charge(name)

    q0 = out.read_charge(name, imd=Opt.ALL)
    assert np.allclose(q, q0)

    q0 = out.read_charge(name, imd=-1)
    assert np.allclose(q[-1], q0)

    with pytest.raises(SileError, match="final charges"):
        out.read_charge(name, iscf=None, imd=None)

    for iscf in [-1, Opt.ALL]:
        # space to not match MD/SCF charges
        with pytest.raises(SileError, match=" SCF charges"):
            out.read_charge(name, iscf=-1)

    if with_pandas():
        df = out.read_charge(name, imd=Opt.ALL, as_dataframe=True)
        assert np.allclose(q.ravel(), df.values.ravel())

        df = out.read_charge(name, imd=-1, as_dataframe=True)
        assert np.allclose(q[-1].ravel(), df.values.ravel())


@pytest.mark.parametrize('fname', ["md_scf", "nc_md_scf", "pol_md_scf", "soc_md_scf"])
@pytest.mark.parametrize('name', ["voronoi", "Hirshfeld"])
def test_vh_md_scf(name, fname, sisl_files):
    f = sisl_files(_dir, f"voronoi_hirshfeld_{fname}.out")
    out = outSileSiesta(f)

    q = out.read_charge(name)

    with pytest.raises(SileError, match="final charges"):
        out.read_charge(name, iscf=None, imd=Opt.NONE)

    q0 = out.read_charge(name, iscf=-1, imd=-1)
    assert np.allclose(q0, q[-1][-1])

    q0 = out.read_charge(name, iscf=Opt.ANY, imd=-1)
    assert np.allclose(q0, q[-1])

    q0 = out.read_charge(name, iscf=-1, imd=Opt.ALL)
    assert np.allclose(q0, np.stack((qq[-1] for qq in q)))

    # iscf is
    q0 = out.read_charge(name, iscf=1, imd=Opt.ALL)
    assert np.allclose(q0, np.stack((qq[1] for qq in q)))

    q0 = out.read_charge(name, iscf=None, imd=-1)
    assert np.allclose(q0, q[-1][-1])

    if with_pandas():
        df = out.read_charge(name, iscf=-1, imd=-1, as_dataframe=True)
        assert np.allclose(df.values, q[-1][-1])
