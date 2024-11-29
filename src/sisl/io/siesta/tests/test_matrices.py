# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl

pytestmark = [pytest.mark.io, pytest.mark.siesta]

listify = sisl.utils.listify


@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize(
    "matrix,ext",
    (map(lambda x: ("Hamiltonian", x), ["nc", "TSHS"]) | listify)
    + (map(lambda x: ("DensityMatrix", x), ["nc", "DM"]) | listify)
    + (map(lambda x: ("EnergyDensityMatrix", x), ["nc"]) | listify),
)
@pytest.mark.parametrize("read_dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
def test_non_colinear(sisl_tmp, sort, matrix, ext, dtype, read_dtype):
    if ext == "nc":
        pytest.importorskip("netCDF4")

    M = getattr(sisl, matrix)(sisl.geom.graphene(), spin=sisl.Spin("NC"), dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        onsite = [0.1 + 0j, 0.2 + 0j, 0.3 + 0.4j]
        nn = [0.2, 0.3, 0.4 + 0.5j]
    else:
        onsite = [0.1, 0.2, 0.3, 0.4]
        nn = [0.2, 0.3, 0.4, 0.5]
    M.construct(([0.1, 1.44], [onsite, nn]))

    f1 = sisl_tmp(f"M1.{ext}")
    f2 = sisl_tmp(f"M2.{ext}")
    M.write(f1, sort=sort)
    M.finalize()
    with sisl.get_sile(f1) as sile:
        M2 = M.read(sile, dtype=read_dtype)
    M2.write(f2, sort=sort)
    with sisl.get_sile(f2) as sile:
        M3 = M2.read(sile, dtype=read_dtype)

    if sort:
        M.finalize(sort=sort)
    assert M._csr.spsame(M2._csr)
    assert M._csr.spsame(M3._csr)

    # Convert to the same dtype
    M2 = M2.astype(dtype)
    M3 = M3.astype(dtype)
    if M.orthogonal and not M2.orthogonal:
        assert np.allclose(M._csr._D, M2._csr._D[..., :-1])
    else:
        assert np.allclose(M._csr._D, M2._csr._D)
    if M.orthogonal and not M3.orthogonal:
        assert np.allclose(M._csr._D, M3._csr._D[..., :-1])
    else:
        assert np.allclose(M._csr._D, M3._csr._D)


@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize(
    "matrix,ext",
    (map(lambda x: ("Hamiltonian", x), ["nc", "TSHS"]) | listify)
    + (map(lambda x: ("DensityMatrix", x), ["nc", "DM"]) | listify)
    + (map(lambda x: ("EnergyDensityMatrix", x), ["nc"]) | listify),
)
@pytest.mark.parametrize("read_dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
def test_spin_orbit(sisl_tmp, sort, matrix, ext, dtype, read_dtype):
    if ext == "nc":
        pytest.importorskip("netCDF4")

    M = getattr(sisl, matrix)(sisl.geom.graphene(), spin=sisl.Spin("SO"), dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        onsite = [0.1 + 0j, 0.2 + 0j, 0.3 + 0.4j, 0.3 - 0.4j]
        nn = [0.2 + 0.1j, 0.3 + 0.3j, 0.4 + 0.5j, 0.4 - 0.5j]
    else:
        onsite = [0.1, 0.2, 0.3, 0.4, 0, 0, 0.3, -0.4]
        nn = [0.2, 0.3, 0.4, 0.5, 0.1, 0.3, 0.4, -0.5]
    M.construct(([0.1, 1.44], [onsite, nn]))

    f1 = sisl_tmp(f"M1.{ext}")
    f2 = sisl_tmp(f"M2.{ext}")
    M.write(f1, sort=sort)
    M.finalize()
    with sisl.get_sile(f1) as sile:
        M2 = M.read(sile, dtype=read_dtype)
    M2.write(f2, sort=sort)
    with sisl.get_sile(f2) as sile:
        M3 = M2.read(sile, dtype=read_dtype)

    if sort:
        M.finalize(sort=sort)
    assert M._csr.spsame(M2._csr)
    assert M._csr.spsame(M3._csr)

    # Convert to the same dtype
    M2 = M2.astype(dtype)
    M3 = M3.astype(dtype)
    if M.orthogonal and not M2.orthogonal:
        assert np.allclose(M._csr._D, M2._csr._D[..., :-1])
    else:
        assert np.allclose(M._csr._D, M2._csr._D)
    if M.orthogonal and not M3.orthogonal:
        assert np.allclose(M._csr._D, M3._csr._D[..., :-1])
    else:
        assert np.allclose(M._csr._D, M3._csr._D)
