# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

""" pytest test configures """

import warnings

import numpy as np
import pytest

import sisl

pytestmark = [pytest.mark.io, pytest.mark.tbtrans]

netCDF4 = pytest.importorskip("netCDF4")


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore", message="*.*.o2p")
def test_1_graphene_all_content(sisl_files):
    """This tests manifolds itself as:

    sisl.geom.graphene(orthogonal=False).tile(2, 0).tile(3, 0)

    All output is enabled:

    ### FDF ###
    # Transmission related quantities
    TBT.T.All T
    TBT.T.Out T
    TBT.T.Eig 2
    TBT.T.Orbital T

    # Density of states
    TBT.DOS.Elecs T
    TBT.DOS.Gf T
    TBT.DOS.A T
    TBT.DOS.A.All T

    # Orbital transmissions and Crystal-Orbital investigations.
    TBT.Symmetry.TimeReversal F
    TBT.Current.Orb T
    TBT.COOP.Gf T
    TBT.COOP.A T
    TBT.COHP.Gf T
    TBT.COHP.A T

    TBT.k [1 11 1]
    ### FDF ###
    """
    tbt = sisl.get_sile(sisl_files("siesta", "tbtrans", "graphene", "graphene.TBT.nc"))
    assert tbt.E.min() > -1.0
    assert tbt.E.max() < 1.0
    # We have 400 energy-points
    ne = len(tbt.E)
    assert ne == 4
    assert tbt.ne == ne

    # We have 100 k-points
    nk = len(tbt.kpt)
    assert nk == 11
    assert tbt.nk == nk
    assert tbt.wk.sum() == pytest.approx(1.0)

    for i in range(ne):
        assert tbt.Eindex(tbt.E[i]) == i

    # Check raises
    with pytest.warns(sisl.SislWarning):
        tbt.Eindex(tbt.E.min() - 1.0)
    with pytest.warns(sisl.SislInfo):
        tbt.Eindex(tbt.E.min() - 2e-3)
    with pytest.warns(sisl.SislWarning):
        tbt.kindex([0, 0, 0.5])

    for i in range(nk):
        assert tbt.kindex(i) == i
        assert tbt.kindex(tbt.kpt[i]) == i

    # Get geometry
    geom = tbt.geometry
    geom_c1 = tbt.read_geometry(atoms=sisl.Atoms(sisl.Atom[6], geom.na))
    geom_c2 = tbt.read_geometry(atoms=sisl.Atoms(sisl.Atom(6, orbs=2), geom.na))
    assert geom_c1 == geom_c2

    norbs = 4

    # Check read is the same as the direct query
    assert tbt.na == geom.na
    assert tbt.no == geom.no
    assert tbt.no != geom.na
    assert tbt.na == 2 * 3 * 4
    assert np.allclose(tbt.cell, geom.cell)

    # Check device atoms (1-orbital system)
    assert tbt.na_d != tbt.no_d
    assert tbt.na_d * norbs == tbt.no_d
    assert tbt.na_d == 2 * 3 * 2  # device has not two ends
    assert len(tbt.pivot()) == 2 * 3 * 2 * norbs
    assert len(tbt.pivot(in_device=True)) == len(tbt.pivot())
    assert np.all(tbt.pivot(in_device=True, sort=True) == np.arange(tbt.no_d))
    assert np.all(tbt.pivot(sort=True) == np.sort(tbt.pivot()))

    # Just check they are there
    assert tbt.n_btd() == len(tbt.btd())

    # Check electrodes
    assert len(tbt.elecs) == 2
    elecs = tbt.elecs[:]
    assert elecs == ["Left", "Right"]
    for i, elec in enumerate(elecs):
        assert tbt._elec(i) == elec

    # Check the chemical potentials
    assert tbt.eta() == pytest.approx(0, abs=1e-7)
    for elec in elecs:
        assert tbt.n_btd(elec) == len(tbt.btd(elec))
        assert tbt.chemical_potential(elec) == pytest.approx(0.0)
        assert tbt.electron_temperature(elec) == pytest.approx(300.0, abs=1)
        assert tbt.eta(elec) == pytest.approx(1e-6, abs=1e-8)

    # Check electrode relevant stuff
    left = elecs[0]
    right = elecs[1]

    # Assert we have transmission symmetry
    assert np.allclose(tbt.transmission(left, right), tbt.transmission(right, left))
    assert np.allclose(
        tbt.transmission_eig(left, right), tbt.transmission_eig(right, left)
    )
    # Check that the total transmission is larger than the sum of transmission eigenvalues
    assert np.all(
        tbt.transmission(left, right) + 1e-7
        >= tbt.transmission_eig(left, right).sum(-1)
    )
    assert np.all(
        tbt.transmission(right, left) + 1e-7
        >= tbt.transmission_eig(right, left).sum(-1)
    )

    # Check that we can't retrieve from same to same electrode
    with pytest.raises(ValueError):
        tbt.transmission(left, left)
    with pytest.raises(ValueError):
        tbt.transmission_eig(left, left)

    assert np.allclose(
        tbt.transmission(left, right, kavg=False),
        tbt.transmission(right, left, kavg=False),
    )

    # Both methods should be identical for simple bulk systems
    # However, when dealing with DFT and a relatively large eta,
    # they tend to be too different
    assert np.allclose(
        tbt.reflection(left), tbt.reflection(left, from_single=True), atol=1e-5
    )

    # Also check for each k
    for ik in range(nk):
        assert np.allclose(
            tbt.transmission(left, right, ik), tbt.transmission(right, left, ik)
        )
        assert np.allclose(
            tbt.transmission_eig(left, right, ik), tbt.transmission_eig(right, left, ik)
        )
        assert np.all(
            tbt.transmission(left, right, ik) + 1e-7
            >= tbt.transmission_eig(left, right, ik).sum(-1)
        )
        assert np.all(
            tbt.transmission(right, left, ik) + 1e-7
            >= tbt.transmission_eig(right, left, ik).sum(-1)
        )
        assert np.allclose(
            tbt.DOS(kavg=ik),
            tbt.ADOS(left, kavg=ik) + tbt.ADOS(right, kavg=ik),
            atol=1e-5,
        )
        assert np.allclose(
            tbt.DOS(E=0.195, kavg=ik),
            tbt.ADOS(left, E=0.195, kavg=ik) + tbt.ADOS(right, E=0.195, kavg=ik),
            atol=1e-5,
        )

    # Check that norm returns correct values
    assert tbt.norm() == 1
    assert tbt.norm(norm="all") == tbt.no_d
    assert tbt.norm(norm="atom") == tbt.norm(norm="orbital")

    # Check atom is equivalent to orbital
    for norm in ("atom", "orbital"):
        assert tbt.norm(0, norm=norm) == 0.0
        assert tbt.norm(3 * 2, norm=norm) == norbs
        assert tbt.norm(range(3 * 2, 3 * 3), norm=norm) == 3 * norbs

    # Assert sum(ADOS) == DOS
    assert np.allclose(tbt.DOS(), tbt.ADOS(left) + tbt.ADOS(right), atol=1e-5)
    assert np.allclose(
        tbt.DOS(sum=False),
        tbt.ADOS(left, sum=False) + tbt.ADOS(right, sum=False),
        atol=1e-5,
    )

    # Now check orbital resolved DOS
    assert np.allclose(
        tbt.DOS(sum=False),
        tbt.ADOS(left, sum=False) + tbt.ADOS(right, sum=False),
        atol=1e-5,
    )

    # Current must be 0 when the chemical potentials are equal
    assert tbt.current(left, right) == pytest.approx(0.0)
    assert tbt.current(right, left) == pytest.approx(0.0)

    high_low = tbt.current_parameter(left, 0.5, 0.0025, right, -0.5, 0.0025)
    low_high = tbt.current_parameter(left, -0.5, 0.0025, right, 0.5, 0.0025)
    assert high_low > 0.0
    assert low_high < 0.0
    assert -high_low == pytest.approx(low_high)
    with pytest.warns(sisl.SislWarning):
        tbt.current_parameter(left, -10.0, 0.0025, right, 10.0, 0.0025)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Since this is a perfect system there should be *no* QM shot-noise
        # Also, the shot-noise is related to the applied bias, so NO shot-noise
        assert np.allclose(
            (tbt.shot_noise(left, right, kavg=False) * tbt.wkpt.reshape(-1, 1)).sum(0),
            0.0,
        )
        assert np.allclose(tbt.shot_noise(left, right), 0.0)
        assert np.allclose(tbt.shot_noise(right, left), 0.0)
        assert np.allclose(tbt.shot_noise(left, right, kavg=1), 0.0)

        # Since the data-file does not contain all T-eigs (only the first two)
        # we can't correctly calculate the fano factors
        # This is a pristine system, hence all fano-factors should be more or less zero
        # All transmissions are step-functions, however close to band-edges there is some
        # smearing.
        # When calculating the Fano factor it is extremely important that the zero_T is *sufficient*
        # I don't now which value is *good*
        assert np.all(
            (tbt.fano(left, right, kavg=False) * tbt.wkpt.reshape(-1, 1)).sum(0) <= 1
        )
        assert np.all(tbt.fano(left, right) <= 1)
        assert np.all(tbt.fano(right, left) <= 1)
        assert np.all(tbt.fano(left, right, kavg=0) <= 1)

        # Neither should the noise_power exist
        assert (
            tbt.noise_power(right, left, kavg=False) * tbt.wkpt
        ).sum() == pytest.approx(0.0)
        assert tbt.noise_power(right, left) == pytest.approx(0.0)
        assert tbt.noise_power(right, left, kavg=0) == pytest.approx(0.0)

    # Check specific DOS queries
    DOS = tbt.DOS
    ADOS = tbt.ADOS

    assert DOS(2.0, atoms=True, sum=False).size == geom.names["Device"].size
    assert np.allclose(
        DOS(2.0, atoms="Device", sum=False), DOS(2.0, atoms=True, sum=False)
    )
    assert DOS(2.0, orbitals=True, sum=False).size == geom.a2o("Device", all=True).size
    assert ADOS(left, 2.0, atoms=True, sum=False).size == geom.names["Device"].size
    assert (
        ADOS(left, 2.0, orbitals=True, sum=False).size
        == geom.a2o("Device", all=True).size
    )
    assert np.allclose(
        ADOS(left, 2.0, atoms="Device", sum=False),
        ADOS(left, 2.0, atoms=True, sum=False),
    )

    for E in (None, 2.0, 3.0):
        assert np.allclose(DOS(E), ADOS(left, E) + ADOS(right, E), atol=1e-5)

        for o, indices in (("atoms", range(8, 12)), ("orbitals", range(50, 70))):
            opt = {o: indices}

            for sum in (False, True):
                opt["sum"] = sum
                assert np.allclose(
                    DOS(E, **opt),
                    ADOS(left, E, **opt) + ADOS(right, E, **opt),
                    atol=1e-5,
                )

            opt["norm"] = o[:-1]
            assert np.allclose(
                DOS(E, **opt), ADOS(left, E, **opt) + ADOS(right, E, **opt), atol=1e-5
            )

    # Check orbital currents
    E = 0.75
    # Sum of orbital current should be 0 (in == out)
    # However, when eta is *big*, this tends to get blurred.
    orb_left = tbt.orbital_transmission(E, left)
    orb_right = tbt.orbital_transmission(E, right)
    assert orb_left.sum() == pytest.approx(0.0, abs=1e-5)
    assert orb_right.sum() == pytest.approx(0.0, abs=1e-5)

    d1 = geom.a2o(range(6, 12), all=True).reshape(-1, 1)
    d2 = geom.a2o(range(12, 18), all=True).reshape(-1, 1)
    # TODO, this does not hold here... why?
    # assert orb_left[d1, d2.T].sum() == pytest.approx(tbt.transmission(left, right)[E])
    # assert orb_left[d1, d2.T].sum() == pytest.approx(-orb_left[d2, d1.T].sum())
    # assert orb_right[d2, d1.T].sum() == pytest.approx(tbt.transmission(right, left)[E])
    # assert orb_right[d2, d1.T].sum() == pytest.approx(-orb_right[d1, d2.T].sum())

    orb_left.sort_indices()
    atom_left = tbt.bond_transmission(E, left, what="all")
    atom_left.sort_indices()
    assert np.allclose(atom_left.data, tbt.sparse_orbital_to_atom(orb_left).data)
    orb_right.sort_indices()
    atom_right = tbt.bond_transmission(E, right, what="all")
    atom_right.sort_indices()
    assert np.allclose(atom_right.data, tbt.sparse_orbital_to_atom(orb_right).data)

    # Calculate the atom current
    # For 1-orbital systems the activity and non-activity are equivalent
    assert np.allclose(
        tbt.atom_transmission(E, left), tbt.atom_transmission(E, left, activity=False)
    )
    tbt.vector_transmission(E, left)
    assert np.allclose(
        tbt.sparse_atom_to_vector(atom_left) / 2,
        tbt.vector_transmission(E, left, what="all"),
    )

    # Check COOP curves
    coop = tbt.orbital_COOP(E)
    coop_l = tbt.orbital_ACOOP(E, left)
    coop_r = tbt.orbital_ACOOP(E, right)
    coop_lr = coop_l + coop_r

    # Ensure aligment
    coop.eliminate_zeros()
    coop.sorted_indices()
    coop_lr.eliminate_zeros()
    coop_lr.sorted_indices()
    assert np.allclose(coop.data, coop_lr.data)

    coop = tbt.orbital_COOP(E, isc=[0, 0, 0])
    coop_l = tbt.orbital_ACOOP(E, left, isc=[0, 0, 0])
    coop_r = tbt.orbital_ACOOP(E, right, isc=[0, 0, 0])
    coop_lr = coop_l + coop_r

    coop.eliminate_zeros()
    coop.sorted_indices()
    coop_lr.eliminate_zeros()
    coop_lr.sorted_indices()
    assert np.allclose(coop.data, coop_lr.data)

    coop = tbt.atom_COOP(E)
    coop_l = tbt.atom_ACOOP(E, left)
    coop_r = tbt.atom_ACOOP(E, right)
    coop_lr = coop_l + coop_r

    coop.eliminate_zeros()
    coop.sorted_indices()
    coop_lr.eliminate_zeros()
    coop_lr.sorted_indices()
    assert np.allclose(coop.data, coop_lr.data)

    coop = tbt.atom_COOP(E, isc=[0, 0, 0])
    coop_l = tbt.atom_ACOOP(E, left, isc=[0, 0, 0])
    coop_r = tbt.atom_ACOOP(E, right, isc=[0, 0, 0])
    coop_lr = coop_l + coop_r

    coop.eliminate_zeros()
    coop.sorted_indices()
    coop_lr.eliminate_zeros()
    coop_lr.sorted_indices()
    assert np.allclose(coop.data, coop_lr.data)

    # Check COHP curves
    coop = tbt.orbital_COHP(E)
    coop_l = tbt.orbital_ACOHP(E, left)
    coop_r = tbt.orbital_ACOHP(E, right)
    coop_lr = coop_l + coop_r

    coop.eliminate_zeros()
    coop.sorted_indices()
    coop_lr.eliminate_zeros()
    coop_lr.sorted_indices()
    assert np.allclose(coop.data, coop_lr.data)

    coop = tbt.atom_COHP(E)
    coop_l = tbt.atom_ACOHP(E, left)
    coop_r = tbt.atom_ACOHP(E, right)
    coop_lr = coop_l + coop_r

    coop.eliminate_zeros()
    coop.sorted_indices()
    coop_lr.eliminate_zeros()
    coop_lr.sorted_indices()
    assert np.allclose(coop.data, coop_lr.data)

    # Simply print out information
    tbt.info()
    for elec in elecs:
        tbt.info(elec)


@pytest.mark.filterwarnings("ignore", message="closest energy")
def test_1_graphene_Eindex(sisl_files):
    # gh-829 (Eindex -> only energy resolutions)
    tbt = sisl.get_sile(sisl_files("siesta", "tbtrans", "graphene", "graphene.TBT.nc"))
    idx = tbt.Eindex(-0.7)
    assert idx == 0
    assert tbt.E[idx] == pytest.approx(-0.75)
    idx = tbt.Eindex(-0.7, method="below")
    assert idx == 0
    assert tbt.E[idx] == pytest.approx(-0.75)
    idx = tbt.Eindex(-0.7, method="above")
    assert idx == 1
    assert tbt.E[idx] == pytest.approx(-0.25)

    with pytest.raises(ValueError):
        tbt.Eindex(-0.76, method="below")
    with pytest.raises(ValueError):
        tbt.Eindex(0.76, method="above")


@pytest.mark.slow
def test_1_graphene_all_tbtav(sisl_files, sisl_tmp):
    tbt = sisl.get_sile(sisl_files("siesta", "tbtrans", "graphene", "graphene.TBT.nc"))
    f = sisl_tmp("graphene.TBT.AV.nc")
    tbt.write_tbtav(f)


def test_1_graphene_all_fail_kavg(sisl_files, sisl_tmp):
    tbt = sisl.get_sile(sisl_files("siesta", "tbtrans", "graphene", "graphene.TBT.nc"))
    with pytest.raises(ValueError):
        tbt.transmission(kavg=[0, 1])


def test_1_graphene_sparse_current(sisl_files, sisl_tmp):
    tbt = sisl.get_sile(sisl_files("siesta", "tbtrans", "graphene", "graphene.TBT.nc"))

    J = tbt.orbital_current()
    assert np.allclose(J.data, 0)

    # convert to bond_current
    j = tbt.sparse_orbital_to_atom(J)
    assert np.allclose(j.data, 0)

    j = tbt.sparse_atom_to_vector(j)
    assert np.allclose(j.data, 0)

    j = tbt.sparse_orbital_to_scalar(J, activity=True)
    assert np.allclose(j.data, 0)

    J = tbt.orbital_current(what="+")
    assert np.allclose(J.data, 0)


@pytest.mark.filterwarnings("ignore:.*requesting energy")
def test_1_graphene_all_fail_kavg_E(sisl_files, sisl_tmp):
    tbt = sisl.get_sile(sisl_files("siesta", "tbtrans", "graphene", "graphene.TBT.nc"))
    with pytest.raises(ValueError):
        tbt.orbital_COOP(kavg=[0, 1], E=0.1)


def test_1_graphene_all_ArgumentParser(sisl_files, sisl_tmp):
    pytest.importorskip("matplotlib", reason="matplotlib not available")
    import matplotlib as mpl

    mpl.rcParams["text.usetex"] = False

    # Local routine to run the collected actions
    def run(ns):
        ns._actions_run = True
        # Run all so-far collected actions
        for A, Aargs, Akwargs in ns._actions:
            A(*Aargs, **Akwargs)
        ns._actions_run = False
        ns._actions = []

    tbt = sisl.get_sile(sisl_files("siesta", "tbtrans", "graphene", "graphene.TBT.nc"))
    geom = tbt.read_geometry()

    p, ns = tbt.ArgumentParser()
    p.parse_args([], namespace=ns)

    p, ns = tbt.ArgumentParser()
    out = p.parse_args(["--energy", " -0.75:0.75"], namespace=ns)
    assert not out._actions_run
    run(out)

    p, ns = tbt.ArgumentParser()
    out = p.parse_args(["--kpoint", "1"], namespace=ns)
    assert out._krng
    run(out)
    assert out._krng == 1

    p, ns = tbt.ArgumentParser()
    out = p.parse_args(["--norm", "orbital"], namespace=ns)
    run(out)
    assert out._norm == "orbital"

    p, ns = tbt.ArgumentParser()
    out = p.parse_args(["--norm", "atom"], namespace=ns)
    run(out)
    assert out._norm == "atom"

    p, ns = tbt.ArgumentParser()
    out = p.parse_args(["--kpoint", "1", "--norm", "orbital"], namespace=ns)
    run(out)
    assert out._krng == 1
    assert out._norm == "orbital"

    #
    # Atoms 7-12 are in the device region
    #

    p, ns = tbt.ArgumentParser()
    atom = "10:11,24"
    out = p.parse_args(["--atom", atom], namespace=ns)
    run(out)
    assert out._Ovalue == atom
    assert np.all(out._Orng == geom.a2o([9, 10], all=True))

    p, ns = tbt.ArgumentParser()
    atom = "10:11,12,24:30"
    out = p.parse_args(["--atom", atom], namespace=ns)
    run(out)
    assert out._Ovalue == atom
    # Only 13-48 is in the device
    assert np.all(out._Orng == geom.a2o([9, 10, 11], all=True))

    p, ns = tbt.ArgumentParser()
    out = p.parse_args(["--transmission", "Left", "Right"], namespace=ns)
    run(out)
    assert len(out._data) == 2
    assert out._data_header[0][0] == "E"
    assert out._data_header[1][0] == "T"

    p, ns = tbt.ArgumentParser()
    out = p.parse_args(
        ["--transmission", "Left", "Right", "--transmission-bulk", "Left"], namespace=ns
    )
    run(out)
    assert len(out._data) == 3
    assert out._data_header[0][0] == "E"
    assert out._data_header[1][0] == "T"
    assert out._data_header[2][:2] == "BT"

    p, ns = tbt.ArgumentParser()
    out = p.parse_args(["--dos", "--dos", "Left", "--ados", "Right"], namespace=ns)
    run(out)
    assert len(out._data) == 4
    assert out._data_header[0][0] == "E"
    assert out._data_header[1][0] == "D"
    assert out._data_header[2][:2] == "AD"
    assert out._data_header[3][:2] == "AD"

    p, ns = tbt.ArgumentParser()
    out = p.parse_args(["--bulk-dos", "Left", "--ados", "Right"], namespace=ns)
    run(out)
    assert len(out._data) == 3
    assert out._data_header[0][0] == "E"
    assert out._data_header[1][:2] == "BD"
    assert out._data_header[2][:2] == "AD"

    p, ns = tbt.ArgumentParser()
    out = p.parse_args(["--transmission-eig", "Left", "Right"], namespace=ns)
    run(out)
    assert out._data_header[0][0] == "E"
    for i in range(1, len(out._data)):
        assert out._data_header[i][:4] == "Teig"

    p, ns = tbt.ArgumentParser()
    out = p.parse_args(["--info"], namespace=ns)

    # Test output
    f = sisl_tmp("graphene.dat")
    p, ns = tbt.ArgumentParser()
    out = p.parse_args(
        ["--transmission-eig", "Left", "Right", "--out", f], namespace=ns
    )
    assert len(out._data) == 0

    f1 = sisl_tmp("graphene_1.dat")
    f2 = sisl_tmp("graphene_2.dat")
    p, ns = tbt.ArgumentParser()
    out = p.parse_args(
        [
            "--transmission",
            "Left",
            "Right",
            "--out",
            f1,
            "--dos",
            "--atom",
            "8:2:48",
            "--dos",
            "Right",
            "--ados",
            "Left",
            "--out",
            f2,
        ],
        namespace=ns,
    )

    d = sisl.io.tableSile(f1).read_data()
    assert len(d) == 2
    d = sisl.io.tableSile(f2).read_data()
    assert len(d) == 4
    assert np.allclose(d[1, :], (d[2, :] + d[3, :]) * 2, atol=1e-3)
    assert np.allclose(d[2, :], d[3, :], atol=1e-3)

    f = sisl_tmp("graphene_T.png")
    p, ns = tbt.ArgumentParser()
    out = p.parse_args(
        ["--transmission", "Left", "Right", "--transmission-bulk", "Left", "--plot", f],
        namespace=ns,
    )


# Requesting an orbital outside of the device region
def test_1_graphene_all_warn_orbital(sisl_files):
    tbt = sisl.get_sile(sisl_files("siesta", "tbtrans", "graphene", "graphene.TBT.nc"))
    with pytest.warns(sisl.SislWarning):
        tbt.o2p(1)


# Requesting an atom outside of the device region
def test_1_graphene_all_warn_atom(sisl_files):
    tbt = sisl.get_sile(sisl_files("siesta", "tbtrans", "graphene", "graphene.TBT.nc"))
    with pytest.warns(sisl.SislWarning):
        tbt.a2p(1)


def test_1_graphene_all_sparse_data_isc_request(sisl_files):
    tbt = sisl.get_sile(sisl_files("siesta", "tbtrans", "graphene", "graphene.TBT.nc"))

    # get supercell with isc
    sc = tbt.read_lattice()

    # request the full matrix
    for elec in [0, 1]:
        J_all = tbt.orbital_transmission(0.75, elec)
        J_all.eliminate_zeros()

        # Ensure we actually have something
        assert J_all.nnz > 0

        # partial summed isc
        # Test that the full matrix and individual access is the same
        J_sum = sum(tbt.orbital_transmission(0.75, elec, isc=isc) for isc in sc.sc_off)
        assert J_sum.nnz == J_all.nnz
        assert (J_sum - J_all).nnz == 0


def test_1_graphene_all_sparse_data_orbitals(sisl_files):
    tbt = sisl.get_sile(sisl_files("siesta", "tbtrans", "graphene", "graphene.TBT.nc"))

    # request the full matrix
    J_all = tbt.orbital_transmission(0.75, 0)
    J_12 = tbt.orbital_transmission(0.75, 0, orbitals=[2, 3])

    assert J_12.nnz < J_all.nnz // 2


def test_projection_raise(sisl_files):
    tbt = sisl.get_sile(
        sisl_files("siesta", "tbtrans", "c60_projection", "projection.TBT.nc")
    )

    with pytest.raises(sisl.io.tbtrans.MissingFDFTBtransError):
        tbt.orbital_transmission(1.995, 0)
