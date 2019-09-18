""" Global sisl fixtures """
from __future__ import print_function

import contextlib
import os
import numpy as np

import pytest
from sisl import Atom, Geometry, SuperCell, Hamiltonian


# Here we create the necessary methods and fixtures to enabled/disable
# tests depending on whether a sisl-files directory is present.
__env = 'SISL_FILES_TESTS'


# Modify items based on whether the env is correct or not
def pytest_collection_modifyitems(config, items):
    sisl_files_tests = os.environ.get(__env, '_THIS_DIRECTORY_DOES_NOT_EXIST_')
    if os.path.isdir(sisl_files_tests):
        if os.path.isdir(os.path.join(sisl_files_tests, 'sisl')):
            return
        print('pytest-sisl: Could not locate sisl directory in: {}'.format(sisl_files_tests))

    skip_sisl_files = pytest.mark.skip(reason="requires env(SISL_FILES_TESTS) pointing to clone of: https://github.com/zerothi/sisl-files")
    for item in items:
        # Only skip those that have the sisl_files fixture
        # GLOBAL skipping of ALL tests that don't have this fixture
        if 'sisl_files' in item.fixturenames:
            item.add_marker(skip_sisl_files)


@pytest.fixture(scope='function')
def sisl_tmp(request, tmpdir_factory):
    """ sisl specific temporary file and directory creator.

        sisl_tmp(file, dir_name='sisl')
        sisl_tmp.file(file, dir_name='sisl')
        sisl_tmp.dir('sisl')

    The scope of the `sisl_tmp` fixture is at a function level to
    clean up after each function.
    """
    class FileFactory(object):
        def __init__(self):
            self.base = tmpdir_factory.getbasetemp()
            self.dirs = []
            self.files = []

        def dir(self, name='sisl'):
            D = ''
            for d in name.split(os.path.sep):
                # Collect tree...
                D = os.path.join(D, d)
                baseD = self.base.join(D)
                if baseD not in self.dirs:
                    if os.path.isdir(str(baseD)):
                        self.dirs.append(baseD)
                    else:
                        self.dirs.append(tmpdir_factory.mktemp(D, numbered=False))
            return self.dirs[-1]

        def file(self, name, dir_name='sisl'):
            D = self.base.join(dir_name)
            if D in self.dirs:
                i = self.dirs.index(D)
            else:
                self.dir(dir_name)
                i = -1
            self.files.append(self.dirs[i].join(name))
            return str(self.files[-1])

        def getbase(self):
            return str(self.dirs[-1])

        def __call__(self, name, dir_name='sisl'):
            """ Shorthand for self.file """
            return self.file(name, dir_name)

        def teardown(self):
            while len(self.files) > 0:
                # Do each removal separately
                f = str(self.files.pop())
                if os.path.isfile(f):
                    try:
                        os.remove(f)
                    except:
                        pass
            while len(self.dirs) > 0:
                # Do each removal separately (from back of directory)
                d = str(self.dirs.pop())
                if os.path.isdir(d):
                    try:
                        os.rmdir(d)
                    except:
                        pass
    ff = FileFactory()
    request.addfinalizer(ff.teardown)
    return ff


@pytest.fixture(scope='session')
def sisl_files():
    """ Environment catcher for the large files hosted in a different repository.

    If SISL_FILES_TESTS has been defined in the environment variable the directory
    will be used for the tests with this as a fixture.

    If the environment variable is empty and a test has this fixture, it will
    be skipped.
    """
    env = os.environ[__env]
    def _path(path, *files):
        if len(files) == 0:
            return os.path.join(os.environ[__env], path)
        return os.path.join(os.environ[__env], path, *files)
    return _path


@pytest.fixture(scope='session')
def sisl_system():
    """ A preset list of geometries/Hamiltonians. """
    class System(object):
        pass

    d = System()

    alat = 1.42
    sq3h = 3.**.5 * 0.5
    C = Atom(Z=6, R=1.42)
    sc = SuperCell(np.array([[1.5, sq3h, 0.],
                             [1.5, -sq3h, 0.],
                             [0., 0., 10.]], np.float64) * alat,
                   nsc=[3, 3, 1])
    d.g = Geometry(np.array([[0., 0., 0.],
                             [1., 0., 0.]], np.float64) * alat,
                   atom=C, sc=sc)

    d.R = np.array([0.1, 1.5])
    d.t = np.array([0., 2.7])
    d.tS = np.array([(0., 1.0),
                     (2.7, 0.)])
    d.C = Atom(Z=6, R=max(d.R))
    d.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                               [1.5, -sq3h, 0.],
                               [0., 0., 10.]], np.float64) * alat,
                     nsc=[3, 3, 1])
    d.gtb = Geometry(np.array([[0., 0., 0.],
                               [1., 0., 0.]], np.float64) * alat,
                     atom=C, sc=sc)

    d.ham = Hamiltonian(d.gtb)
    d.ham.construct([(0.1, 1.5), (0.1, 2.7)])
    return d


def pytest_configure(config):

    # Locally manage pytest.ini input
    for mark in ['io', 'bloch', 'hamiltonian', 'geometry', 'geom', 'shape',
                 'state', 'electron', 'phonon', 'utils', 'unit', 'distribution',
                 'spin', 'self_energy', 'help', 'messages', 'namedindex', 'sparse',
                 'supercell', 'sc', 'quaternion', 'sparse_geometry', 'ranges',
                 'orbital', 'oplist', 'grid', 'atoms', 'atom', 'sgrid', 'sdata', 'sgeom',
                 'version', 'bz', 'brillouinzone', 'inv', 'eig', 'linalg',
                 'density_matrix', 'dynamicalmatrix', 'energydensity_matrix',
                 'siesta', 'tbtrans', 'ham', 'vasp', 'w90', 'wannier90', 'gulp', 'fdf',
                 'table', 'cube', 'slow', 'selector']:
        config.addinivalue_line(
            "markers", "{}: mark test to run only on named environment".format(mark)
        )
