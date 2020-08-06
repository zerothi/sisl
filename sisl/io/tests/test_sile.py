from pathlib import Path
import pytest
import tempfile

import sisl
from sisl.io.sile import __siles

@pytest.fixture(params=["pickle", "dill"])
def pickle_module(request):
    return request.param

@pytest.fixture(params=__siles)
def sile(request):
    sile_cls = request.param

    return sile_cls("test", _open=False)


def test_pickling(sile, pickle_module):

    pickle_module = pytest.importorskip(pickle_module)
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    error = None

    try:
        with open(tmp_file.name, "wb") as fh:
            pickle_module.dump(sile, fh)

        with open(tmp_file.name, "rb") as fh:
            loaded_sile = pickle_module.load(fh)
    except Exception as e:
        error = e

    Path(tmp_file.name).unlink()

    if error is not None:
        raise error

    
