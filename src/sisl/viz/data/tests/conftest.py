import os.path as osp

import pytest


@pytest.fixture(scope="session")
def siesta_test_files(sisl_files):

    def _siesta_test_files(path):
        return sisl_files(osp.join('sisl', 'io', 'siesta', path))

    return _siesta_test_files