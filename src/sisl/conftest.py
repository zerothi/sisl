# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

""" Global sisl fixtures """
import logging
import os
import re
import warnings
from functools import cache, partial
from importlib import import_module
from itertools import count
from pathlib import Path

import numpy as np
import pytest

from sisl import Atom, Geometry, Hamiltonian, Lattice, _environ
from sisl._help import has_module
from sisl.io.sile import MissingFortranSileError

# Here we create the necessary methods and fixtures to enabled/disable
# tests depending on whether a sisl-files directory is present.

_log = logging.getLogger(__name__)

sisl_files_tests = _environ.get_environ_variable("SISL_FILES_TESTS")
if (sisl_files_tests / "tests").is_dir():
    sisl_files_tests = sisl_files_tests / "tests"


# Modify items based on whether the env is correct or not
def pytest_collection_modifyitems(config, items):
    global sisl_files_tests
    if sisl_files_tests.is_dir():
        return

    xfail_sisl_files = pytest.mark.xfail(
        run=False,
        reason="requires env(SISL_FILES_TESTS) pointing to clone of: https://github.com/zerothi/sisl-files",
    )
    for item in items:
        # Only skip those that have the sisl_files fixture
        # GLOBAL skipping of ALL tests that don't have this fixture
        if "sisl_files" in item.fixturenames:
            item.add_marker(xfail_sisl_files)


# Tests requiring the Fortran sources (sisl.io.siesta._siesta) cannot be
# detected up-front (they are spread across the test-suite, and only some
# tests in a file may use them). Instead we let them run, and convert the
# raised error into an xfail. This makes builds without Fortran sources
# (-DWITH_FORTRAN=OFF) report xfail, rather than a failure.
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is not None and isinstance(
        call.excinfo.value, MissingFortranSileError
    ):
        report.outcome = "skipped"
        report.wasxfail = "requires sisl built with the Fortran sources"


@pytest.fixture(scope="function")
def sisl_tmp(request, tmp_path_factory):
    """sisl specific temporary file and directory creator.

        sisl_tmp(file, dir_name='sisl')
        sisl_tmp.file(file, dir_name='sisl')
        sisl_tmp.dir('sisl')

    Each test has its own folder, ``<basetemp>/<package>/<test-name><N>``.
    Another `dir_name` is a sub-folder of it.
    The scope of the `sisl_tmp` fixture is at a function level to
    clean up after each function.
    A path that cannot be removed (e.g. a file that is still open on Windows)
    emits a `UserWarning`.
    """
    global __file__
    # The test file is in a tests/test_<name>.py,
    # so step back twice
    path = Path(request.node.path).parent.parent
    # This file is in sisl/conftest.py
    # And we wish to retain the sisl path, so go two levels up as well
    path = str(path.relative_to(Path(__file__).parent.parent))

    # Now path should be something like:
    # sisl/io/siesta

    class FileFactory:
        def __init__(self):
            self.package = tmp_path_factory.getbasetemp() / path.replace(
                os.path.sep, "-"
            )
            self.package.mkdir(exist_ok=True)
            name = re.sub(r"\W", "_", request.node.name)[:30]
            for n in count():
                self.root = self.package / f"{name}{n}"
                try:
                    self.root.mkdir()
                    break
                except FileExistsError:
                    pass
            self.dirs = []

        def dir(self, name=path):
            if name == path:
                D = self.root
            else:
                D = self.root / name.replace(os.path.sep, "-")
            if D not in self.dirs:
                D.mkdir(exist_ok=True)
                self.dirs.append(D)
            return D

        def file(self, name, dir_name=path):
            return str(self.dir(dir_name) / name)

        def getbase(self):
            return self.dirs[-1] if self.dirs else self.dir()

        def __call__(self, name, dir_name=path):
            """Shorthand for self.file"""
            return self.file(name, dir_name)

        def teardown(self):
            failed = []
            # reverse order removes children before their parent
            for p in sorted(self.root.rglob("*"), reverse=True) + [self.root]:
                try:
                    if p.is_dir() and not p.is_symlink():
                        p.rmdir()
                    else:
                        p.unlink()
                except OSError:
                    failed.append(p)
            try:
                # other tests can still have folders here
                self.package.rmdir()
            except OSError:
                pass
            # a folder fails when a file in it fails, so report only the files
            failed = [p for p in failed if not p.is_dir()] or failed
            if failed:
                warnings.warn(
                    f"sisl_tmp could not remove (still open?): {', '.join(map(str, failed))}"
                )

    ff = FileFactory()
    request.addfinalizer(ff.teardown)
    return ff


@pytest.fixture(scope="session")
def sisl_files():
    """Environment catcher for the large files hosted in a different repository.

    If SISL_FILES_TESTS has been defined in the environment variable the directory
    will be used for the tests with this as a fixture.

    If the environment variable is empty and a test has this fixture, it will
    be skipped.
    """
    global sisl_files_tests

    if sisl_files_tests.is_dir():

        def _path(*files):
            p = sisl_files_tests.joinpath(*files)
            if p.is_file():
                return p
            _log.info("sisl_files: test requested non-existing ' {p!s}' -> xfail")

            # I expect this test to fail due to the wrong environment.
            # But it isn't an actual fail since it hasn't runned...
            pytest.xfail(
                reason=f"Environment SISL_FILES_TESTS may point to a wrong path(?); file {p} not found",
            )

    else:
        _log.info(
            "sisl_files SISL_FILES_TESTS={sisl_files_tests!s} does not exist, xfail dependencies"
        )

        def _path(*files):
            pytest.xfail(
                reason=f"Environment SISL_FILES_TESTS not pointing to a valid directory.",
            )

    return _path


@pytest.fixture(scope="session")
def sisl_tolerance():
    r32 = (1e-6, 1e-11)
    r64 = (1e-9, 1e-15)
    return {
        np.float32: r32,
        np.float32(0).dtype: r32,
        np.float64: r64,
        np.float64(0).dtype: r64,
        np.complex64: r32,
        np.complex64(0).dtype: r32,
        np.complex128: r64,
        np.complex128(0).dtype: r64,
        None: r64,
    }


@pytest.fixture(scope="session")
def sisl_allclose(sisl_tolerance):
    def factory(dtype):
        atol, rtol = sisl_tolerance[dtype]
        return partial(np.allclose, atol=atol, rtol=rtol)

    return {key: factory(key) for key in sisl_tolerance.keys()}


@pytest.fixture(scope="session")
def sisl_isclose(sisl_tolerance):
    def factory(dtype):
        atol, rtol = sisl_tolerance[dtype]
        return partial(np.isclose, atol=atol, rtol=rtol)

    return {key: factory(key) for key in sisl_tolerance.keys()}


@pytest.fixture(scope="session", params=[np.float32, np.float64])
def sisl_float(request, sisl_tolerance):
    yield (request.param,) + sisl_tolerance[request.param]


@pytest.fixture(scope="session", params=[np.complex64, np.complex128])
def sisl_complex(request, sisl_tolerance):
    yield (request.param,) + sisl_tolerance[request.param]


@pytest.fixture(scope="session")
def sisl_system():
    """A preset list of geometries/Hamiltonians."""

    class System:
        pass

    d = System()

    alat = 1.42
    sq3h = 3.0**0.5 * 0.5
    C = Atom(Z=6, R=1.42)
    lattice = Lattice(
        np.array([[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64)
        * alat,
        nsc=[3, 3, 1],
    )
    d.g = Geometry(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * alat,
        atoms=C,
        lattice=lattice,
    )

    d.R = np.array([0.1, 1.5])
    d.t = np.array([0.0, 2.7])
    d.tS = np.array([(0.0, 1.0), (2.7, 0.0)])
    d.C = Atom(Z=6, R=max(d.R))
    d.lattice = Lattice(
        np.array([[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64)
        * alat,
        nsc=[3, 3, 1],
    )
    d.gtb = Geometry(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * alat,
        atoms=C,
        lattice=lattice,
    )

    d.ham = Hamiltonian(d.gtb)
    d.ham.construct([(0.1, 1.5), (0.1, 2.7)])
    return d


@cache
def _importable(module: str) -> bool:
    try:
        import_module(module)
    except ImportError:
        return False
    return True


def sisl_requires(*modules: str):
    """Decorator that skips a test, or a test class, if a module cannot be imported

    Contrary to `pytest.importorskip` this only skips the decorated test,
    and not the full test module.
    It may also be assigned to ``pytestmark`` to skip all tests in a module.

    Parameters
    ----------
    *modules :
        fully qualified module names that the test requires

    Examples
    --------
    >>> from sisl.conftest import sisl_requires
    >>> @sisl_requires("sisl.io.siesta._siesta")
    ... def test_fortran():
    ...     ...
    """
    missing = [module for module in modules if not _importable(module)]
    return pytest.mark.skipif(
        bool(missing), reason=f"requires module(s): {', '.join(missing)}"
    )


collect_ignore = []
collect_ignore_glob = []

# We are ignoring stuff in sisl.viz if nodify cannot be imported
# skip paths
_skip_paths = []
# We have to manually skip this as it's trying to load it @ import time
if not has_module("nodify"):
    _skip_paths.append(os.path.join("sisl", "viz"))


def pytest_ignore_collect(collection_path, config):
    # ensure we only compare against final *sisl* stuff
    global _skip_paths
    parts = list(Path(collection_path).parts)
    parts.reverse()
    sisl_parts = parts[: parts.index("sisl")]
    sisl_parts.reverse()
    sisl_path = str(Path("sisl").joinpath(*sisl_parts))

    for skip_path in _skip_paths:
        if skip_path in sisl_path:
            return True
    return False


def pytest_report_header(config, start_path):
    global sisl_files_tests
    global _skip_paths
    s = []
    s.append(f"sisl-test: found FILES_TESTS: {sisl_files_tests!s}")
    if _skip_paths:
        s.append("sisl-test: skip test-discovery in these folders:")
        for skip in _skip_paths:
            s.append(f" - {skip}")
    return s


def pytest_configure(config):
    pytest.sisl_travis_skip = pytest.mark.skipif(
        os.environ.get("SISL_TRAVIS_CI", "false").lower() == "true",
        reason="running on TRAVIS",
    )

    # Locally manage pytest.ini input
    for mark in [
        "io",
        "generic",
        "bloch",
        "hamiltonian",
        "geometry",
        "geom",
        "neighbor",
        "shape",
        "state",
        "electron",
        "phonon",
        "utils",
        "unit",
        "distribution",
        "spin",
        "self_energy",
        "help",
        "messages",
        "namedindex",
        "sparse",
        "lattice",
        "supercell",
        "sc",
        "quaternion",
        "sparse_geometry",
        "sparse_orbital",
        "ranges",
        "physics",
        "physics_feature",
        "orbital",
        "oplist",
        "grid",
        "atoms",
        "atom",
        "periodictable",
        "sgrid",
        "sdata",
        "sgeom",
        "version",
        "bz",
        "brillouinzone",
        "monkhorstpack",
        "bandstructure",
        "inv",
        "eig",
        "linalg",
        "densitymatrix",
        "dynamicalmatrix",
        "energydensitymatrix",
        "siesta",
        "tbtrans",
        "dftb",
        "vasp",
        "w90",
        "wannier90",
        "gulp",
        "fdf",
        "fhiaims",
        "aims",
        "orca",
        "collection",
        "category",
        "geom_category",
        "plot",
        "slow",
        "overlap",
        "mixing",
        "typing",
        "only",
        "viz",
        "processors",
        "data",
        "plots",
        "plotters",
        "mpi",
    ]:
        config.addinivalue_line(
            "markers", f"{mark}: mark test to run only on named environment"
        )
