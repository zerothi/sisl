""" pytest test configures """
from __future__ import print_function

import pytest
from tempfile import mkstemp, mkdtemp
import os

# Here we create the necessary methods and fixtures to enabled/disable
# tests depending on whether a sisl-files directory is present.
__env = 'SISL_FILES_TESTS'


# Modify items based on whether the env is correct or not
def pytest_collection_modifyitems(config, items):
    sisl_files_tests = os.environ.get(__env, '_THIS_DIRECTORY_DOES_NOT_EXIST_')
    if os.path.isdir(sisl_files_tests):
        return

    skip_sisl_files = pytest.mark.skip(reason="requires env(SISL_FILES_TESTS) pointing to clone of: https://github.com/zerothi/sisl-files")
    for item in items:
        item.add_marker(skip_sisl_files)


# Create fixture for environment variable
@pytest.fixture(scope='session')
def files():
    return os.path.join(os.environ[__env], 'sisl/io/tbtrans')


@pytest.fixture(scope='session')
def dir_test(request, tmpdir_factory):
    class FileFactory(object):
        def __init__(self):
            self.dirs = []
            self.files = []
        def dir(self, name='sisl-io'):
            self.dirs.append(tmpdir_factory.mktemp(name))
            return self.dirs[-1]
        def file(self, name):
            if len(self.dirs) == 0:
                self.dir()
            self.files.append(self.dirs[-1].join(name))
            return str(self.files[-1])
        def teardown(self):
            while len(self.files) > 0:
                # Do each removal separately
                f = str(self.files.pop())
                if os.path.isfile(f):
                    os.remove(f)
            while len(self.files) > 0:
                # Do each removal separately
                f = str(self.files.pop())
                if os.path.isfile(f):
                    os.remove(f)
    ff = FileFactory()
    request.addfinalizer(ff.teardown)
    return ff
