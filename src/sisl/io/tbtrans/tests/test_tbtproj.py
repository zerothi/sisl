# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

""" pytest test configures """


import numpy as np
import pytest

import sisl

pytestmark = [pytest.mark.io, pytest.mark.tbtrans]

netCDF4 = pytest.importorskip("netCDF4")


@pytest.mark.slow
def test_2_projection_content(sisl_files):
    tbt = sisl.get_sile(
        sisl_files("siesta", "tbtrans", "c60_projection", "projection.TBT.nc")
    )
    tbtp = sisl.get_sile(
        sisl_files("siesta", "tbtrans", "c60_projection", "projection.TBT.Proj.nc")
    )

    assert np.allclose(tbt.E, tbtp.E)
    assert np.allclose(tbt.kpt, tbtp.kpt)

    # Get geometry
    assert tbt.read_geometry() == tbtp.read_geometry()

    assert np.allclose(tbt.n_btd(), tbtp.n_btd())
    assert np.allclose(tbt.pivot(), tbtp.pivot())

    # Loop molecules
    left = tbt.elecs[0]
    right = tbt.elecs[1]
    for mol in tbtp.molecules:
        for proj in tbtp.projections(mol):
            t1 = tbtp.transmission((left, mol, proj), (right, mol, proj))
            t2 = tbtp.transmission(
                ".".join((left, mol, proj)), ".".join((right, mol, proj))
            )
            assert np.allclose(t1, t2)

            te1 = tbtp.transmission_eig((left, mol, proj), (right, mol, proj))
            te2 = tbtp.transmission_eig(
                ".".join((left, mol, proj)), ".".join((right, mol, proj))
            )
            assert np.allclose(te1, te2)
            assert np.allclose(t1, te1.sum(-1))
            assert np.allclose(t2, te2.sum(-1))

    # Check eigenstate
    es = tbtp.eigenstate("C60")
    assert len(es) == 3  # 1-HOMO, 2-LUMO
    assert (es.eig < 0.0).nonzero()[0].size == 1
    assert (es.eig > 0.0).nonzero()[0].size == 2
    assert np.allclose(es.norm2(), 1)


@pytest.mark.slow
def test_2_projection_tbtav(sisl_files, sisl_tmp):
    tbtp = sisl.get_sile(
        sisl_files("siesta", "tbtrans", "c60_projection", "projection.TBT.Proj.nc")
    )
    f = sisl_tmp("2_projection.TBT.Proj.AV.nc")
    tbtp.write_tbtav(f)


def test_2_projection_ArgumentParser(sisl_files, sisl_tmp):
    pytest.importorskip("matplotlib", reason="matplotlib not available")
    import argparse

    # Local routine to run the collected actions
    def run(ns):
        ns._actions_run = True
        # Run all so-far collected actions
        for A, Aargs, Akwargs in ns._actions:
            A(*Aargs, **Akwargs)
        ns._actions_run = False
        ns._actions = []

    tbtp = sisl.get_sile(
        sisl_files("siesta", "tbtrans", "c60_projection", "projection.TBT.Proj.nc")
    )

    p, ns = tbtp.ArgumentParser(argparse.ArgumentParser(conflict_handler="resolve"))
    p.parse_args([], namespace=ns)

    p, ns = tbtp.ArgumentParser(argparse.ArgumentParser(conflict_handler="resolve"))
    out = p.parse_args(["--energy", " -1.995:1.995"], namespace=ns)
    assert not out._actions_run
    run(out)

    p, ns = tbtp.ArgumentParser(argparse.ArgumentParser(conflict_handler="resolve"))
    out = p.parse_args(["--norm", "orbital"], namespace=ns)
    run(out)
    assert out._norm == "orbital"

    p, ns = tbtp.ArgumentParser(argparse.ArgumentParser(conflict_handler="resolve"))
    out = p.parse_args(["--norm", "atom"], namespace=ns)
    run(out)
    assert out._norm == "atom"

    p, ns = tbtp.ArgumentParser(argparse.ArgumentParser(conflict_handler="resolve"))
    out = p.parse_args(["--atom", "10:11,14"], namespace=ns)
    run(out)
    assert out._Ovalue == "10:11,14"
    # Only atom 14 is in the device region
    assert np.all(out._Orng + 1 == [14])

    p, ns = tbtp.ArgumentParser(argparse.ArgumentParser(conflict_handler="resolve"))
    out = p.parse_args(["--atom", "10:11,12,14:20"], namespace=ns)
    run(out)
    assert out._Ovalue == "10:11,12,14:20"
    # Only 13-72 is in the device
    assert np.all(out._Orng + 1 == [14, 15, 16, 17, 18, 19, 20])

    p, ns = tbtp.ArgumentParser(argparse.ArgumentParser(conflict_handler="resolve"))
    out = p.parse_args(
        ["--transmission", "Left.C60.HOMO", "Right.C60.HOMO"], namespace=ns
    )
    run(out)
    assert len(out._data) == 2
    assert out._data_header[0][0] == "E"
    assert out._data_header[1][0] == "T"

    p, ns = tbtp.ArgumentParser(argparse.ArgumentParser(conflict_handler="resolve"))
    out = p.parse_args(["--molecules", "-P", "C60"], namespace=ns)
    run(out)

    p, ns = tbtp.ArgumentParser(argparse.ArgumentParser(conflict_handler="resolve"))
    out = p.parse_args(
        [
            "--transmission",
            "Left",
            "Right.C60.LUMO",
            "--transmission",
            "Left.C60.LUMO",
            "Right",
        ],
        namespace=ns,
    )
    run(out)
    assert len(out._data) == 3
    assert out._data_header[0][0] == "E"
    assert out._data_header[1][0] == "T"
    assert out._data_header[2][0] == "T"

    p, ns = tbtp.ArgumentParser(argparse.ArgumentParser(conflict_handler="resolve"))
    out = p.parse_args(
        ["--ados", "Left.C60.HOMO", "--ados", "Left.C60.LUMO"], namespace=ns
    )
    run(out)
    assert len(out._data) == 3
    assert out._data_header[0][0] == "E"
    assert out._data_header[1][:2] == "AD"
    assert out._data_header[2][:2] == "AD"

    p, ns = tbtp.ArgumentParser(argparse.ArgumentParser(conflict_handler="resolve"))
    out = p.parse_args(
        ["--transmission-eig", "Left.C60.HOMO", "Right.C60.LUMO"], namespace=ns
    )
    run(out)
    assert out._data_header[0][0] == "E"
    for i in range(1, len(out._data)):
        assert out._data_header[i][:4] == "Teig"

    p, ns = tbtp.ArgumentParser(argparse.ArgumentParser(conflict_handler="resolve"))
    out = p.parse_args(["--info"], namespace=ns)

    # Test output
    f = sisl_tmp("projection.dat")
    p, ns = tbtp.ArgumentParser(argparse.ArgumentParser(conflict_handler="resolve"))
    out = p.parse_args(
        ["--transmission-eig", "Left", "Right.C60.HOMO", "--out", f], namespace=ns
    )
    assert len(out._data) == 0

    f1 = sisl_tmp("projection_1.dat")
    f2 = sisl_tmp("projection_2.dat")
    p, ns = tbtp.ArgumentParser(argparse.ArgumentParser(conflict_handler="resolve"))
    out = p.parse_args(
        [
            "--transmission",
            "Left",
            "Right.C60.HOMO",
            "--out",
            f1,
            "--ados",
            "Left.C60.HOMO",
            "--atom",
            "13:2:72",
            "--ados",
            "Left.C60.HOMO",
            "--atom",
            "14:2:72",
            "--ados",
            "Left.C60.HOMO",
            "--out",
            f2,
        ],
        namespace=ns,
    )

    d = sisl.io.tableSile(f1).read_data()
    assert len(d) == 2
    d = sisl.io.tableSile(f2).read_data()
    assert len(d) == 4
    assert np.allclose(d[1, :], d[2, :] + d[3, :])

    f = sisl_tmp("projection_T.png")
    p, ns = tbtp.ArgumentParser(argparse.ArgumentParser(conflict_handler="resolve"))
    out = p.parse_args(
        [
            "--transmission",
            "Left",
            "Right.C60.HOMO",
            "--transmission",
            "Left.C60.HOMO",
            "Right.C60.HOMO",
            "--plot",
            f,
        ],
        namespace=ns,
    )
