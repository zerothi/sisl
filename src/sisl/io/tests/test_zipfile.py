# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import sys
import tempfile
import zipfile

import numpy as np
import pytest

import sisl
from sisl.io._zipfile import ZipPath

# We will xfail all tests if the python version is <=3.9,
# this is because sisl is expected to raise a RuntimeError
# explaining that zipfile functionality requires Python 3.10 or newer.
_is_old_python = sys.version_info < (3, 10)
# 3.9 also prints very ugly errors when deleting a zipfile which make
# it difficult to read the test output. So we patch the __del__ method.
if _is_old_python:
    zipfile.ZipFile.__del__ = lambda self: None


@pytest.mark.xfail(
    _is_old_python,
    reason="Zip file functionality requires Python 3.10 or newer",
    raises=RuntimeError,
)
def test_zipfile_preserved():
    """Test that the zipfile is preserved through the sile framework

    This is VERY important, because the lookup for paths will only be
    fast if the index is already built. If each time a new sile is
    created we need to create a new zipfile, it will be very slow.
    """
    # Write a temporary zipfile
    _, tempfile_path = tempfile.mkstemp(suffix=".zip")
    with zipfile.ZipFile(tempfile_path, "a") as f:
        ...

    f = zipfile.ZipFile(tempfile_path)

    root_path = zipfile.Path(f, "")

    fdf = sisl.get_sile(root_path / "test.fdf")

    assert isinstance(fdf.file, ZipPath)
    assert fdf.file.root is f


@pytest.mark.xfail(
    _is_old_python,
    reason="Zip file functionality requires Python 3.10 or newer",
    raises=RuntimeError,
)
@pytest.mark.parametrize("specify_class", [True, False])
@pytest.mark.parametrize("external_zip", [True, False])
def test_zipfile_write_read(external_zip: bool, specify_class: bool):
    """Test that we can write and read within a zipfile.

    If the user provides the zipfile path, instead of
    letting the sile framework build it, the sile should not
    close the zipfile.

    The user is then responsible for closing the zipfile. This
    is important because if the sile closes the zipfile you can't
    write to it anymore.
    """

    geometry = sisl.geom.graphene()

    # Write a temporary zipfile
    _, tempfile_path = tempfile.mkstemp(suffix=".zip")
    zip_file = zipfile.ZipFile(tempfile_path, "a")

    filename = "graphene.dat{xyzSile}" if specify_class else "graphene.xyz"

    if external_zip:
        graphene_xyz = zipfile.Path(zip_file, filename)
    else:
        graphene_xyz = tempfile_path + "/" + filename

    geometry.write(graphene_xyz)

    if external_zip:
        # Try to write another file
        zipfile.Path(zip_file, "other_file.txt").open("w").write("Hello world!")

        # Close the zipfile
        zip_file.close()

    # Make sisl read from the written zipfile
    graphene_xyz = str(graphene_xyz)
    read_geometry = sisl.get_sile(graphene_xyz).read_geometry()

    assert np.allclose(read_geometry.xyz, geometry.xyz)
    assert np.allclose(read_geometry.cell, geometry.cell)


@pytest.mark.xfail(
    _is_old_python,
    reason="Zip file functionality requires Python 3.10 or newer",
    raises=RuntimeError,
)
@pytest.mark.parametrize("external_zip", [True, False])
@pytest.mark.parametrize("from_fdf", [True, False])
def test_zipfile_write_read_binary(external_zip: bool, from_fdf: bool):
    """Test that we can write and read binary files within a zipfile"""

    geometry = sisl.geom.graphene()
    H = sisl.Hamiltonian(geometry)
    H[0, 0] = 1.3
    H[1, 1] = 1.5

    # Write a temporary zipfile
    _, tempfile_path = tempfile.mkstemp(suffix=".zip")
    zip_file = zipfile.ZipFile(tempfile_path, "a")

    H_filename = "siesta.HSX"

    if external_zip:
        H_path = zipfile.Path(zip_file, H_filename)
        fdf_path = zipfile.Path(zip_file, "test.fdf")
    else:
        H_path = tempfile_path + "/" + H_filename
        fdf_path = tempfile_path + "/test.fdf"

    if from_fdf:
        # Write the geometry to the fdf file
        geometry.write(fdf_path)

    H.write(H_path)

    if external_zip:
        # Try to write another file
        zipfile.Path(zip_file, "other_file.txt").open("w").write("Hello world!")

        # Close the zipfile
        zip_file.close()

    # Make sisl read from the written zipfile
    path = str(fdf_path) if from_fdf else str(H_path)
    read_H = sisl.get_sile(path).read_hamiltonian()

    assert np.allclose(H.tocsr().toarray(), read_H.tocsr().toarray())


@pytest.mark.xfail(
    _is_old_python,
    reason="Zip file functionality requires Python 3.10 or newer",
    raises=RuntimeError,
)
@pytest.mark.parametrize("external_zip", [True, False])
@pytest.mark.parametrize("from_fdf", [True, False])
def test_zipfile_write_read_cdf(external_zip: bool, from_fdf: bool):
    """Test that we can write and read CDF files within a zipfile"""
    pytest.importorskip("netCDF4")

    geometry = sisl.geom.graphene()
    grid = sisl.Grid((2, 2, 2), geometry=geometry)
    grid[:] = np.random.random(grid.shape)

    # Write a temporary zipfile
    _, tempfile_path = tempfile.mkstemp(suffix=".zip")
    zip_file = zipfile.ZipFile(tempfile_path, "a")

    grid_filename = "Rho.grid.nc"

    if external_zip:
        grid_path = zipfile.Path(zip_file, grid_filename)
        fdf_path = zipfile.Path(zip_file, "test.fdf")
    else:
        grid_path = tempfile_path + "/" + grid_filename
        fdf_path = tempfile_path + "/test.fdf"

    if from_fdf:
        # Write the geometry to the fdf file
        geometry.write(fdf_path)

    grid.write(grid_path)

    if external_zip:
        # Try to write another file
        zipfile.Path(zip_file, "other_file.txt").open("w").write("Hello world!")

        # Close the zipfile
        zip_file.close()

    # Make sisl read from the written zipfile
    if from_fdf:
        read_grid = sisl.get_sile(str(fdf_path)).read_grid("RHO")
    else:
        # Read the grid from the grid.nc file
        read_grid = sisl.get_sile(str(grid_path)).read_grid()

    assert np.allclose(grid.grid, read_grid.grid)
