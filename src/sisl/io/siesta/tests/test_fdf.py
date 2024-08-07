# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sisl
from sisl import geom
from sisl.io import SileError, fdfSileSiesta
from sisl.messages import SislWarning
from sisl.unit.siesta import unit_convert

pytestmark = [
    pytest.mark.io,
    pytest.mark.siesta,
    pytest.mark.fdf,
    pytest.mark.filterwarnings("ignore", message="*number of supercells"),
]


def test_fdf1(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.fdf")
    sisl_system.g.write(fdfSileSiesta(f, "w"))

    fdf = fdfSileSiesta(f)
    str(fdf)
    with fdf:
        fdf.readline()

        # Be sure that we can read it in a loop
        assert fdf.get("LatticeConstant") > 0.0
        assert fdf.get("LatticeConstant") > 0.0
        assert fdf.get("LatticeConstant") > 0.0

        fdf.read_lattice()
        fdf.read_geometry()


def test_fdf2(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.fdf")
    sisl_system.g.write(fdfSileSiesta(f, "w"))
    g = fdfSileSiesta(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, sisl_system.g.cell)
    assert np.allclose(g.xyz, sisl_system.g.xyz)
    for ia in g:
        assert g.atoms[ia].Z == sisl_system.g.atoms[ia].Z
        assert g.atoms[ia].tag == sisl_system.g.atoms[ia].tag


def test_fdf_units(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.fdf")
    fdf = fdfSileSiesta(f, "w")
    g = sisl_system.g

    for unit in ["bohr", "ang", "fractional", "frac"]:
        fdf.write_geometry(g, unit=unit)
        g2 = fdfSileSiesta(f).read_geometry()
        assert np.allclose(g.cell, g2.cell)
        assert np.allclose(g.xyz, g2.xyz)
        for ia in g:
            assert g.atoms[ia].Z == g2.atoms[ia].Z
            assert g.atoms[ia].tag == g2.atoms[ia].tag


def test_lattice(sisl_tmp):
    f = sisl_tmp("file.fdf")
    lines = [
        "Latticeconstant 1. Ang",
        "%block Latticevectors",
        " 1. 1. 1.",
        " 0. 0. 1.",
        " 1. 0. 1.",
        "%endblock",
    ]
    with open(f, "w") as fh:
        fh.write("\n".join(lines))

    cell = np.array([[1.0] * 3, [0, 0, 1], [1, 0, 1]])
    lattice = fdfSileSiesta(f).read_lattice()
    assert np.allclose(lattice.cell, cell)

    lines = [
        "Latticeconstant 1. Bohr",
        "%block Latticevectors",
        " 1. 1. 1.",
        " 0. 0. 1.",
        " 1. 0. 1.",
        "%endblock",
    ]
    with open(f, "w") as fh:
        fh.write("\n".join(lines))

    lattice = fdfSileSiesta(f).read_lattice()
    assert np.allclose(lattice.cell, cell * unit_convert("Bohr", "Ang"))

    cell = np.diag([2.0] * 3)
    lines = [
        "Latticeconstant 2. Ang",
        "%block Latticeparameters",
        " 1. 1. 1. 90. 90. 90.",
        "%endblock",
    ]
    with open(f, "w") as fh:
        fh.write("\n".join(lines))

    lattice = fdfSileSiesta(f).read_lattice()
    assert np.allclose(lattice.cell, cell)


def test_lattice_fail(sisl_tmp):
    f = sisl_tmp("file.fdf")
    lines = [
        "%block Latticevectors",
        " 1. 1. 1.",
        " 0. 0. 1.",
        " 1. 0. 1.",
        "%endblock",
    ]
    with open(f, "w") as fh:
        fh.write("\n".join(lines))
    with pytest.raises(SileError):
        fdfSileSiesta(f).read_lattice()


def test_geometry(sisl_tmp):
    f = sisl_tmp("file.fdf")
    sc_lines = [
        "Latticeconstant 1. Ang",
        "%block latticeparameters",
        " 1. 1. 1. 90. 90. 90.",
        "%endblock",
    ]
    lines = [
        "NumberOfAtoms 2",
        "%block chemicalSpeciesLabel",
        " 1 6 C",
        " 2 12 H",
        "%endblock",
        "AtomicCoordinatesFormat Ang",
        "%block atomiccoordinatesandatomicspecies",
        " 1. 1. 1. 1",
        " 0. 0. 1. 1",
        " 1. 0. 1. 2",
        "%endblock",
    ]

    with open(f, "w") as fh:
        fh.write("\n".join(sc_lines) + "\n")
        fh.write("\n".join(lines))

    fdf = fdfSileSiesta(f, base=sisl_tmp.getbase())
    g = fdf.read_geometry()
    assert g.na == 2
    assert np.allclose(g.xyz, [[1.0] * 3, [0, 0, 1]])
    assert np.allclose(g.atoms.Z, [6, 6])
    assert g.atoms.nspecies == 2

    # default read # of atoms from list
    with open(f, "w") as fh:
        fh.write("\n".join(sc_lines) + "\n")
        fh.write("\n".join(lines[1:]))

    fdf = fdfSileSiesta(f, base=sisl_tmp.getbase())
    g = fdf.read_geometry()
    assert g.na == 3
    assert np.allclose(g.xyz, [[1.0] * 3, [0, 0, 1], [1, 0, 1]])
    assert g.atoms[0].Z == 6
    assert g.atoms[1].Z == 6
    assert g.atoms[2].Z == 12
    assert g.atoms.nspecies == 2


def test_re_read(sisl_tmp):
    f = sisl_tmp("file.fdf")
    with open(f, "w") as fh:
        fh.write("Flag1 date\n")
        fh.write("Flag1 not-date\n")
        fh.write("Flag1 not-date-2\n")
        fh.write("Flag3 true\n")

    fdf = fdfSileSiesta(f)
    for i in range(10):
        assert fdf.get("Flag1") == "date"
    assert fdf.get("Flag3")


def test_get_set(sisl_tmp):
    f = sisl_tmp("file.fdf")
    with open(f, "w") as fh:
        fh.write("Flag1 date\n")

    fdf = fdfSileSiesta(f)
    assert fdf.get("Flag1") == "date"
    fdf.set("Flag1", "not-date")
    assert fdf.get("Flag1") == "not-date"
    fdf.set("Flag1", "date")
    assert fdf.get("Flag1") == "date"
    fdf.set("Flag1", "date-date")
    assert fdf.get("Flag1") == "date-date"
    fdf.set("Flag1", "date-date", keep=False)


def test_get_block(sisl_tmp):
    f = sisl_tmp("file.fdf")
    with open(f, "w") as fh:
        fh.write("%block MyBlock\n  date\n%endblock\n")

    fdf = fdfSileSiesta(f)

    assert isinstance(fdf.get("MyBlock"), list)
    assert fdf.get("MyBlock")[0] == "date"
    assert "block" in fdf.print("MyBlock", fdf.get("MyBlock"))


def test_include(sisl_tmp):
    f = sisl_tmp("file.fdf")
    with open(f, "w") as fh:
        fh.write("Flag1 date\n")
        fh.write("# Flag2 comment\n")
        fh.write("Flag2 date2\n")
        fh.write("# Flag3 is read through < from file hello\n")
        fh.write("Flag3 Sub < hello\n")
        fh.write("FakeInt 1\n")
        fh.write("Test 1. eV\n")
        fh.write(" %INCLUDE file2.fdf\n")
        fh.write("TestRy 1. Ry\n")
        fh.write("%block Hello < hello\n")
        fh.write("\n")
        fh.write("TestLast 1. eV\n")

    hello = sisl_tmp("hello")
    with open(hello, "w") as fh:
        fh.write("Flag4 hello\n")
        fh.write("# Comments should be discarded\n")
        fh.write("Flag3 test\n")
        fh.write("Sub sub-test\n")

    file2 = sisl_tmp("file2.fdf")
    with open(file2, "w") as fh:
        fh.write("Flag4 non\n")
        fh.write("\n")
        fh.write("FakeReal 2.\n")
        fh.write("  %incLude file3.fdf")

    file3 = sisl_tmp("file3.fdf")
    with open(file3, "w") as fh:
        fh.write("Sub level\n")
        fh.write("Third level\n")
        fh.write("MyList [1 , 2 , 3]\n")

    fdf = fdfSileSiesta(f, base=sisl_tmp.getbase())
    assert fdf.includes() == [Path(hello), Path(file2), Path(file3)]
    assert fdf.get("Flag1") == "date"
    assert fdf.get("Flag2") == "date2"
    assert fdf.get("Flag3") == "test"
    assert fdf.get("Flag4") == "non"
    assert fdf.get("FLAG4") == "non"
    assert fdf.get("Fakeint") == 1
    assert fdf.get("Fakeint", "0") == "1"
    assert fdf.get("Fakereal") == 2.0
    assert fdf.get("Fakereal", 0.0) == 2.0
    assert fdf.get("test", "eV") == pytest.approx(1.0)
    assert fdf.get("test", with_unit=True)[0] == pytest.approx(1.0)
    assert fdf.get("test", with_unit=True)[1] == "eV"
    assert fdf.get("test", unit="Ry") == pytest.approx(unit_convert("eV", "Ry"))
    assert fdf.get("testRy") == pytest.approx(unit_convert("Ry", "eV"))
    assert fdf.get("testRy", with_unit=True)[0] == pytest.approx(1.0)
    assert fdf.get("testRy", with_unit=True)[1] == "Ry"
    assert fdf.get("testRy", unit="Ry") == pytest.approx(1.0)
    assert fdf.get("Sub") == "sub-test"
    assert fdf.get("Third") == "level"
    assert fdf.get("test-last", with_unit=True)[0] == pytest.approx(1.0)
    assert fdf.get("test-last", with_unit=True)[1] == "eV"

    # Currently lists are not implemented
    # assert np.allclose(fdf.get('MyList'), np.arange(3) + 1)
    # assert np.allclose(fdf.get('MyList', []), np.arange(3) + 1)

    # Read a block
    ll = open(sisl_tmp("hello")).readlines()
    ll.pop(1)
    assert fdf.get("Hello") == [l.replace("\n", "").strip() for l in ll]


def test_xv_preference(sisl_tmp):
    g = geom.graphene()
    g.write(sisl_tmp("file.fdf"))
    g.xyz[0, 0] += 1.0
    g.write(sisl_tmp("siesta.XV"))

    lattice = fdfSileSiesta(sisl_tmp("file.fdf")).read_lattice(True)
    g2 = fdfSileSiesta(sisl_tmp("file.fdf")).read_geometry(True)
    assert np.allclose(lattice.cell, g.cell)
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)

    g2 = fdfSileSiesta(sisl_tmp("file.fdf")).read_geometry(order=["fdf"])
    assert np.allclose(g.cell, g2.cell)
    g2.xyz[0, 0] += 1.0
    assert np.allclose(g.xyz, g2.xyz)


def test_geom_order(sisl_tmp):
    pytest.importorskip("netCDF4")
    gfdf = geom.graphene()
    gxv = gfdf.copy()
    gxv.xyz[0, 0] += 0.5
    gnc = gfdf.copy()
    gnc.xyz[0, 0] += 0.5

    gfdf.write(sisl_tmp("siesta.fdf"))

    # Create fdf-file
    fdf = fdfSileSiesta(sisl_tmp("siesta.fdf"))
    assert fdf.read_geometry(order=["nc"]) is None
    gxv.write(sisl_tmp("siesta.XV"))
    gnc.write(sisl_tmp("siesta.nc"))

    # Should read from XV
    g = fdf.read_geometry(True)
    assert np.allclose(g.xyz, gxv.xyz)
    g = fdf.read_geometry(order=["nc", "fdf"])
    assert np.allclose(g.xyz, gnc.xyz)
    g = fdf.read_geometry(order=["fdf", "nc"])
    assert np.allclose(g.xyz, gfdf.xyz)
    g = fdf.read_geometry(True, order="^fdf")
    assert np.allclose(g.xyz, gxv.xyz)


def test_geom_constraints(sisl_tmp):
    gfdf = geom.graphene().tile(2, 0).tile(2, 1)
    gfdf["CONSTRAIN"] = 0
    gfdf["CONSTRAIN-x"] = 2
    gfdf["CONSTRAIN-y"] = [1, 3, 4, 5]
    gfdf["CONSTRAIN-z"] = range(len(gfdf))

    gfdf.write(sisl_tmp("siesta.fdf"))


def test_h2_dynamical_matrix(sisl_files):
    si = fdfSileSiesta(sisl_files("siesta", "H2_hessian", "h2_hessian.fdf"))

    TF = [True, False]

    eV2cm = 8065.54429
    hw_true = [
        -7.11492125e-06,
        1.71779189e-05,
        2.02363231e-05,
        1.25035599e03,
        1.25035599e03,
        3.16727461e03,
    ]

    from itertools import product

    for ti, s0, herm in product(TF, TF, TF):

        dyn = si.read_dynamical_matrix(trans_inv=ti, sum0=s0, hermitian=herm)

        hw = dyn.eigenvalue().hw
        if ti and s0 and herm:
            assert np.allclose(hw * eV2cm, hw_true, atol=1e-4)


def test_dry_read(sisl_tmp):
    # This test runs the read-functions. They aren't expected to actually read anything,
    # it is only a dry-run.
    file = sisl_tmp("siesta.fdf")
    geom.graphene().write(file)
    fdf = fdfSileSiesta(file)

    read_methods = set(m for m in dir(fdf) if m.startswith("read_"))
    output = dict(output=True)
    kwargs = {
        "lattice": output,
        "geometry": output,
        "grid": dict(name="rho"),
    }

    with pytest.warns(SislWarning):
        assert np.allclose(fdf.read_lattice_nsc(), (1, 1, 1))
    read_methods.remove("read_lattice_nsc")

    geom_methods = set(f"read_{x}" for x in ("basis", "lattice", "geometry"))
    read_methods -= geom_methods

    for methodname in read_methods:
        kwarg = kwargs.get(methodname[5:], dict())
        assert getattr(fdf, methodname)(**kwarg) is None

    for methodname in geom_methods:
        # Also run these, but dont assert None due to the graphene values being present
        # in the fdf. The read functions will still go dry-running through eg. nc-files.
        kwarg = kwargs.get(methodname[5:], dict())
        getattr(fdf, methodname)(**kwarg)


def test_fdf_argumentparser(sisl_tmp):
    f = sisl_tmp("file.fdf")
    with open(f, "w") as fh:
        fh.write("Flag1 date\n")
        fh.write("Flag1 not-date\n")
        fh.write("Flag1 not-date-2\n")
        fh.write("Flag3 true\n")

    fdfSileSiesta(f).ArgumentParser()


def test_fdf_fe_basis(sisl_files):
    geom = fdfSileSiesta(
        sisl_files("siesta", "fe_bcc_simple", "fe.fdf")
    ).read_geometry()
    assert geom.na == 1
    # The fe.fdf does not contain basis information, so it will default to 1.
    assert geom.no == 1


def test_fdf_pao_basis():
    fdf = fdfSileSiesta

    block = """
Mg                    1                    # Species label, number of l-shells
 n=3   0   1                         # n, l, Nzeta
   6.620
   1.000
C                     2                    # Species label, number of l-shells
 n=2   0   1                         # n, l, Nzeta
   4.192
   1.000
 n=2   1   1                         # n, l, Nzeta
   4.870
   1.000
O                     2                    # Species label, number of l-shells
 n=2   0   1                         # n, l, Nzeta
   3.305
   1.000
 n=2   1   1                         # n, l, Nzeta
   3.937
   1.000
    """.splitlines()

    atom_orbs = fdf._parse_pao_basis(block)
    assert len(atom_orbs) == 3
    assert len(atom_orbs["Mg"]) == 1
    assert len(atom_orbs["C"]) == 4
    assert len(atom_orbs["O"]) == 4
    for i, (tag, orbs) in enumerate(atom_orbs.items()):
        specie_orbs = fdf._parse_pao_basis(block, species=tag)
        assert specie_orbs == orbs

    block = """
Fe_SOC                2                    # Species label, number of l-shells
 n=4   0   2 P   1                   # n, l, Nzeta, Polarization, NzetaPol
   7.329      6.153
   1.000      1.000
 n=3   2   2                         # n, l, Nzeta
   4.336      2.207
   1.000      1.000
Pt_SOC                2                    # Species label, number of l-shells
 n=6   0   2 P   1                   # n, l, Nzeta, Polarization, NzetaPol
   7.158      6.009
   1.000      1.000
 n=5   2   2                         # n, l, Nzeta
   5.044      3.022
   1.000      1.000
"""
    atom_orbs = fdf._parse_pao_basis(block)
    assert len(atom_orbs) == 2
    assert len(atom_orbs["Fe_SOC"]) == 5 + 10
    assert len(atom_orbs["Pt_SOC"]) == 5 + 10
    for i, (tag, orbs) in enumerate(atom_orbs.items()):
        specie_orbs = fdf._parse_pao_basis(block, species=tag)
        assert specie_orbs == orbs


def test_fdf_gz(sisl_files):
    f = sisl_files("siesta", "fdf", "main.fdf.gz")
    fdf = fdfSileSiesta(f)

    # read from gzipped file
    assert fdf.get("Main.Foo") == "hello"
    assert fdf.get("Main.Bar") == "world"

    # read from included non-gzipped file
    assert fdf.get("Lvl2.Foo") == "world"
    assert fdf.get("Lvl2.Bar") == "hello"

    # read from nested included gzipped file
    assert fdf.get("Lvl3.Foo") == "world3"
    assert fdf.get("Lvl3.Bar") == "hello3"

    f = sisl_files("siesta", "fdf", "level2.fdf")
    fdf = fdfSileSiesta(f)

    # read from non-gzipped file
    assert fdf.get("Lvl2.Foo") == "world"
    assert fdf.get("Lvl2.Bar") == "hello"

    # read from included gzipped file
    assert fdf.get("Lvl3.Foo") == "world3"
    assert fdf.get("Lvl3.Bar") == "hello3"


def test_fdf_block_write_print(sisl_tmp):
    f = sisl_tmp("block_write_print.fdf")
    fdf = fdfSileSiesta(f, "w")

    block_value = ["this is my life"]
    fdf.set("hello", block_value)
    fdf.set("goodbye", "hello")

    fdf = fdfSileSiesta(f)

    assert "hello" == fdf.get("goodbye")
    assert block_value == fdf.get("hello")
    assert f"""%block hello
 {block_value[0]}
%endblock hello
""" == fdf.print(
        "hello", block_value
    )


def test_fdf_write_bandstructure(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.fdf")

    bs = sisl.BandStructure(
        sisl_system.g,
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.5, 0]],
        200,
        names=["Gamma", "Edge", "L"],
    )

    with fdfSileSiesta(f, "w") as fdf:
        fdf.write_brillouinzone(bs)

    with fdfSileSiesta(f) as fdf:
        block = fdf.get("BandLines")
    assert len(block) == 3


def test_fdf_read_from_xv(sisl_tmp):
    # test for #778
    f_fdf = sisl_tmp("read_from_xv.fdf")
    sc_lines = [
        "Latticeconstant 1. Ang",
        "%block latticeparameters",
        " 1. 1. 1. 90. 90. 90.",
        "%endblock",
    ]
    lines = [
        "%block chemicalSpeciesLabel",
        " 2 6 C",
        " 1 2 He",
        " 4 3 Li",
        " 3 1 H",  # not present in the geometry
        "%endblock",
        "AtomicCoordinatesFormat Ang",
        "%block atomiccoordinatesandatomicspecies",
        " 1. 1. 1. 2",
        " 0. 0. 1. 1",
        " 1. 0. 1. 4",
        " 1. 1. 1. 2",
        " 1. 0. 1. 4",
        " 0. 0. 1. 1",
        "%endblock",
    ]
    with open(f_fdf, "w") as fh:
        fh.write("\n".join(sc_lines) + "\n")
        fh.write("\n".join(lines))
        fh.write("\nSystemLabel read_from_xv")

    f_xv = sisl_tmp("read_from_xv.XV")
    with open(f_xv, "w") as fh:
        fh.write(
            """\
1. 0. 0.  0. 0. 0.
0. 1. 0.  0. 0. 0.
0. 0. 2.  0. 0. 0.
6
2 6 0. 1. 0.  0. 0. 0.
1 2 0. 1. 1.  0. 0. 0.
4 3 1. 1. 1.  0. 0. 0.
2 6 0. 1. 0.  0. 0. 0.
4 3 1. 1. 1.  0. 0. 0.
1 2 0. 1. 1.  0. 0. 0.
"""
        )

    fdf = fdfSileSiesta(f_fdf, track=True, base=sisl_tmp.getbase())
    geom_fdf = fdf.read_geometry(order="fdf")

    assert len(geom_fdf) == 6
    assert len(geom_fdf.atoms.atom) == 4
    assert np.allclose(geom_fdf.atoms.Z, [6, 2, 3, 6, 3, 2])
    assert np.allclose(geom_fdf.atoms.species, [1, 0, 3, 1, 3, 0])
    assert np.allclose(geom_fdf.xyz[0], [1, 1, 1])
    assert np.allclose(geom_fdf.xyz[1], [0, 0, 1])
    assert np.allclose(geom_fdf.xyz[2], [1, 0, 1])

    geom_xv = fdf.read_geometry(order="xv")
    assert len(geom_xv) == 6
    assert len(geom_xv.atoms.atom) == 4
    assert np.allclose(geom_xv.atoms.Z, [6, 2, 3, 6, 3, 2])
    assert np.allclose(geom_xv.atoms.species, [1, 0, 3, 1, 3, 0])
    xyz = geom_xv.xyz * unit_convert("Ang", "Bohr")
    assert np.allclose(xyz[0], [0, 1, 0])
    assert np.allclose(xyz[1], [0, 1, 1])
    assert np.allclose(xyz[2], [1, 1, 1])


def test_fdf_multiple_atoms_scrambeld(sisl_tmp):
    # test for #778
    f_fdf = sisl_tmp("multiple_atoms_scrambled.fdf")
    sc_lines = [
        "Latticeconstant 1. Ang",
        "%block latticeparameters",
        " 1. 1. 1. 90. 90. 90.",
        "%endblock",
    ]
    lines = [
        "%block chemicalSpeciesLabel",
        " 1 1 H.opt88",
        " 4 6 C.blyp",
        " 2 79 Au.blyp",
        " 3 79 Aus.blyp",
        "%endblock",
        "AtomicCoordinatesFormat Ang",
        "%block atomiccoordinatesandatomicspecies",
        " 1. 1. 1. 1",
        " 0. 0. 1. 2",
        " 1. 0. 1. 3",
        " 1. 1. 1. 4",
        " 1. 0. 1. 1",
        "%endblock",
    ]
    with open(f_fdf, "w") as fh:
        fh.write("\n".join(sc_lines) + "\n")
        fh.write("\n".join(lines))

    geom = fdfSileSiesta(f_fdf).read_geometry()

    assert len(geom) == 5
    assert len(geom.atoms.atom) == 4
    assert np.allclose(geom.atoms.Z, [1, 79, 79, 6, 1])
    assert np.allclose(geom.atoms.species, [0, 1, 2, 3, 0])


def test_fdf_multiple_atoms_linear(sisl_tmp):
    # test for #778
    f = sisl_tmp("multiple_atoms_linear.fdf")
    sc_lines = [
        "Latticeconstant 1. Ang",
        "%block latticeparameters",
        " 1. 1. 1. 90. 90. 90.",
        "%endblock",
    ]
    lines = [
        "%block chemicalSpeciesLabel",
        " 1 1 H.opt88",
        " 2 79 Au.blyp",
        " 3 79 Aus.blyp",
        " 4 6 C.blyp",
        "%endblock",
        "AtomicCoordinatesFormat Ang",
        "%block atomiccoordinatesandatomicspecies",
        " 1. 1. 1. 1",
        " 0. 0. 1. 2",
        " 1. 0. 1. 3",
        " 1. 1. 1. 4",
        " 1. 0. 1. 1",
        "%endblock",
    ]
    with open(f, "w") as fh:
        fh.write("\n".join(sc_lines) + "\n")
        fh.write("\n".join(lines))

    geom = fdfSileSiesta(f).read_geometry()

    assert len(geom) == 5
    assert len(geom.atoms.atom) == 4
    assert np.allclose(geom.atoms.Z, [1, 79, 79, 6, 1])
    assert np.allclose(geom.atoms.species, [0, 1, 2, 3, 0])


def test_fdf_multiple_simple(sisl_tmp):
    # test for #778
    f = sisl_tmp("multiple_simple.fdf")

    with open(f, "w") as fh:
        fh.write(
            """
%block ChemicalSpeciesLabel
    1   79  Au
    2    8   O
%endblock ChemicalSpeciesLabel

LatticeConstant    1.000 Ang
%block LatticeVectors
    4  0  0
    0  10 0
    0  0  10
%endblock LatticeVectors
AtomicCoordinatesFormat Ang
%block AtomicCoordinatesAndAtomicSpecies
    0 0 0  2
    2 0 0  1
%endblock AtomicCoordinatesAndAtomicSpecies
                """
        )

    geom = fdfSileSiesta(f).read_geometry()
    assert geom.na == 2
    assert np.allclose(geom.atoms.Z, [8, 79])
    assert np.allclose(geom.atoms.species, [1, 0])

    with open(f, "w") as fh:
        fh.write(
            """
%block ChemicalSpeciesLabel
    1   79  Au
    2    8   O
%endblock ChemicalSpeciesLabel

LatticeConstant    1.000 Ang
%block LatticeVectors
    4  0  0
    0  10 0
    0  0  10
%endblock LatticeVectors
AtomicCoordinatesFormat Ang
%block AtomicCoordinatesAndAtomicSpecies
    0 0 0  2
    2 0 0  1
    0 0 0  2
%endblock AtomicCoordinatesAndAtomicSpecies
                """
        )

    geom = fdfSileSiesta(f).read_geometry()
    assert geom.na == 3
    assert np.allclose(geom.atoms.Z, [8, 79, 8])
    assert np.allclose(geom.atoms.species, [1, 0, 1])
