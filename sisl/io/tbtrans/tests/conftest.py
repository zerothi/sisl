""" pytest test configures """
from __future__ import print_function

import pytest
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
