# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Atom, Atoms
from sisl.io.siesta.orb_indx import *

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def test_si_pdos_kgrid_orb_indx(sisl_files):
    f = sisl_files("siesta", "Si_pdos_k", "Si_pdos.ORB_INDX")
    nsc = orbindxSileSiesta(f).read_lattice_nsc()
    assert np.all(nsc > 1)
    atoms = orbindxSileSiesta(f).read_basis()

    assert len(atoms) == 2
    assert atoms.nspecies == 1
    assert atoms.atom[0].Z == 14
    assert len(atoms[0]) == 9
    assert len(atoms[1]) == 9


def test_h2o_dipole_orb_indx(sisl_files):
    f = sisl_files("siesta", "H2O_dipole", "h2o_dipole.ORB_INDX")

    nsc = orbindxSileSiesta(f).read_lattice_nsc()
    assert np.all(nsc == 1)
    atoms = orbindxSileSiesta(f).read_basis()

    assert len(atoms) == 6
    assert atoms.nspecies == 2
    Z = np.zeros(len(atoms))
    Z[:] = 1
    Z[[0, 3]] = 8

    assert np.allclose(atoms.Z, Z)
    # number of orbitals on each atom
    assert len(atoms[0]) == 9
    assert len(atoms[-1]) == 4


def test_orb_indx_order(sisl_tmp):
    f = sisl_tmp("test.ORD_INDX")

    with open(f, "w") as fh:
        fh.write(
            """\
     29  29 = orbitals in unit cell and supercell. See end of file.

    io    ia is   spec iao  n  l  m  z  p          sym      rc    isc     iuo
     1     1  2 Fe_SOC   1  4  0  0  1  F            s   7.329  0  0  0     1
     2     1  2 Fe_SOC   2  4  0  0  2  F            s   6.153  0  0  0     2
     3     1  2 Fe_SOC   3  3  2 -2  1  F          dxy   4.336  0  0  0     3
     4     1  2 Fe_SOC   4  3  2 -1  1  F          dyz   4.336  0  0  0     4
     5     1  2 Fe_SOC   5  3  2  0  1  F          dz2   4.336  0  0  0     5
     6     1  2 Fe_SOC   6  3  2  1  1  F          dxz   4.336  0  0  0     6
     7     1  2 Fe_SOC   7  3  2  2  1  F       dx2-y2   4.336  0  0  0     7
     8     1  2 Fe_SOC   8  3  2 -2  2  F          dxy   2.207  0  0  0     8
     9     1  2 Fe_SOC   9  3  2 -1  2  F          dyz   2.207  0  0  0     9
    10     1  2 Fe_SOC  10  3  2  0  2  F          dz2   2.207  0  0  0    10
    11     1  2 Fe_SOC  11  3  2  1  2  F          dxz   2.207  0  0  0    11
    12     1  2 Fe_SOC  12  3  2  2  2  F       dx2-y2   2.207  0  0  0    12
    13     1  2 Fe_SOC  13  4  1 -1  1  T          Ppy   7.329  0  0  0    13
    14     1  2 Fe_SOC  14  4  1  0  1  T          Ppz   7.329  0  0  0    14
    15     1  2 Fe_SOC  15  4  1  1  1  T          Ppx   7.329  0  0  0    15
    16     2  1     Pt   1  6  0  0  1  F            s   7.158  0  0  0    16
    17     2  1     Pt   2  6  0  0  2  F            s   6.009  0  0  0    17
    18     2  1     Pt   3  5  2 -2  1  F          dxy   5.044  0  0  0    18
    19     2  1     Pt   4  5  2 -1  1  F          dyz   5.044  0  0  0    19
    20     2  1     Pt   5  5  2  0  1  F          dz2   5.044  0  0  0    20
    21     2  1     Pt   6  5  2  1  1  F          dxz   5.044  0  0  0    21
    22     2  1     Pt   7  5  2  2  1  F       dx2-y2   5.044  0  0  0    22
    23     2  1     Pt   8  5  2 -2  2  F          dxy   3.022  0  0  0    23
    24     2  1     Pt   9  5  2 -1  2  F          dyz   3.022  0  0  0    24
    25     2  1     Pt  10  5  2  0  2  F          dz2   3.022  0  0  0    25
    26     2  1     Pt  11  5  2  1  2  F          dxz   3.022  0  0  0    26
    27     2  1     Pt  12  5  2  2  2  F       dx2-y2   3.022  0  0  0    27
    28     2  1     Pt  13  6  1 -1  1  T          Ppy   7.158  0  0  0    28
    29     2  1     Pt  14  6  1  0  1  T          Ppz   7.158  0  0  0    29
"""
        )

    atoms = orbindxSileSiesta(f).read_basis()

    assert len(atoms) == 2
    assert atoms.nspecies == 2
    Z = [1001, 78]
    assert np.allclose(atoms.Z, Z)
    # number of orbitals on each atom
    assert len(atoms[0]) == 15
    assert len(atoms[1]) == 14
    assert atoms.atom[0].tag == "Pt"
    assert len(atoms.atom[0]) == 14
    assert atoms.atom[1].tag == "Fe_SOC"
    assert len(atoms.atom[1]) == 15

    atoms = orbindxSileSiesta(f).read_basis(atoms=Atoms([Atom("Pt"), Atom("Fe")]))

    assert len(atoms) == 2
    assert atoms.nspecies == 2
    Z = [26, 78]
    assert np.allclose(atoms.Z, Z)
    # number of orbitals on each atom
    assert len(atoms[0]) == 15
    assert len(atoms[1]) == 14
    assert atoms.atom[0].tag == "Pt"
    assert len(atoms.atom[0]) == 14
    assert atoms.atom[1].tag == "Fe_SOC"
    assert len(atoms.atom[1]) == 15
