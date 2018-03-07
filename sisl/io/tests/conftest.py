""" pytest test configures """
from __future__ import print_function

import pytest
from tempfile import mkstemp, mkdtemp
import os


@pytest.fixture(scope='module')
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
