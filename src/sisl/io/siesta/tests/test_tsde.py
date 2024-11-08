# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

""" pytest test configures """


import numpy as np
import pytest

import sisl

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def test_si_pdos_kgrid_tsde_dm(sisl_files):
    fdf = sisl.get_sile(
        sisl_files("siesta", "Si_pdos_k", "Si_pdos.fdf"),
        base=sisl_files("siesta", "Si_pdos_k"),
    )

    si = sisl.get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.TSDE"))

    DM1 = si.read_density_matrix(geometry=fdf.read_geometry())
    DM2 = fdf.read_density_matrix(order=["TSDE"])

    Ef1 = si.read_fermi_level()
    Ef2 = fdf.read_fermi_level()

    assert Ef1 == Ef2

    assert DM1._csr.spsame(DM2._csr)
    assert np.allclose(DM1._csr._D[:, :-1], DM2._csr._D[:, :-1])


def test_si_pdos_kgrid_tsde_edm(sisl_files):
    fdf = sisl.get_sile(
        sisl_files("siesta", "Si_pdos_k", "Si_pdos.fdf"),
        base=sisl_files("siesta", "Si_pdos_k"),
    )
    si = sisl.get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.TSDE"))

    EDM1 = si.read_energy_density_matrix(geometry=fdf.read_geometry())
    EDM2 = fdf.read_energy_density_matrix(order=["TSDE"])

    assert EDM1._csr.spsame(EDM2._csr)
    assert np.allclose(EDM1._csr._D[:, :-1], EDM2._csr._D[:, :-1])


@pytest.mark.filterwarnings("ignore", message="*Casting complex values")
@pytest.mark.parametrize(("matrix"), ["density", "energy_density"])
def test_si_pdos_kgrid_tsde_edm_dtypes(sisl_files, sisl_tmp, matrix):
    fdf = sisl.get_sile(
        sisl_files("siesta", "Si_pdos_k", "Si_pdos.fdf"),
        base=sisl_files("siesta", "Si_pdos_k"),
    )
    data = []
    mull = None

    for dtype in (np.float32, np.float64, np.complex64, np.complex128):
        M = getattr(fdf, f"read_{matrix}_matrix")(dtype=dtype)
        data.append(M)
        assert M.dtype == dtype

        if mull is None:
            mull = M.mulliken()
        else:
            assert np.allclose(mull, M.mulliken(), atol=1e-5)

    fnc = sisl_tmp("tmp.nc")
    for M in data:
        M.write(fnc)
        # The overlap should be here...
        M1 = M.read(fnc)
        assert np.allclose(mull, M1.mulliken(), atol=1e-5)


@pytest.mark.filterwarnings("ignore", message="*wrong sparse pattern")
def test_si_pdos_kgrid_tsde_dm_edm_rw(sisl_files, sisl_tmp):
    fdf = sisl.get_sile(
        sisl_files("siesta", "Si_pdos_k", "Si_pdos.fdf"),
        base=sisl_files("siesta", "Si_pdos_k"),
    )
    geom = fdf.read_geometry()
    f1 = sisl.get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.TSDE"))

    DM1 = f1.read_density_matrix(geometry=geom)
    EDM1 = f1.read_energy_density_matrix(geometry=geom)

    f2 = sisl.get_sile(sisl_tmp("noEf.TSDE"))
    # by default everything gets sorted...
    f2.write_density_matrices(DM1, EDM1, sort=False)
    DM2 = f2.read_density_matrix(geometry=geom)
    EDM2 = f2.read_energy_density_matrix(geometry=geom)
    assert DM1._csr.spsame(DM2._csr)
    assert np.allclose(DM1._csr._D[:, :-1], DM2._csr._D[:, :-1])
    assert EDM1._csr.spsame(EDM2._csr)
    assert np.allclose(EDM1._csr._D[:, :-1], EDM2._csr._D[:, :-1])

    # Now the matrices ARE finalized, we don't have to do anything again
    EDM2 = EDM1.copy()
    EDM2.shift(-2.0, DM1)
    f3 = sisl.get_sile(sisl_tmp("Ef.TSDE"))
    f3.write_density_matrices(DM1, EDM2, Ef=-2.0, sort=False)
    DM3 = f3.read_density_matrix(geometry=geom)
    EDM3 = f3.read_energy_density_matrix(geometry=geom)
    assert DM1._csr.spsame(DM3._csr)
    assert np.allclose(DM1._csr._D[:, :-1], DM3._csr._D[:, :-1])
    assert EDM1._csr.spsame(EDM3._csr)
    assert np.allclose(EDM1._csr._D[:, :-1], EDM3._csr._D[:, :-1])

    f3.write_density_matrices(DM1, EDM2, Ef=-2.0, sort=True)
    DM3 = f3.read_density_matrix(geometry=geom, sort=False)
    EDM3 = f3.read_energy_density_matrix(geometry=geom, sort=False)
    assert DM1._csr.spsame(DM3._csr)
    assert not np.allclose(DM1._csr._D[:, :-1], DM3._csr._D[:, :-1])
    DM3.finalize()
    assert np.allclose(DM1._csr._D[:, :-1], DM3._csr._D[:, :-1])
    assert EDM1._csr.spsame(EDM3._csr)
    assert not np.allclose(EDM1._csr._D[:, :-1], EDM3._csr._D[:, :-1])
