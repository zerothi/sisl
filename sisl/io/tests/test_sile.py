from pathlib import Path
import pickle
import pytest
import tempfile

import sisl
from sisl.io.sile import __siles


@pytest.fixture(params=__siles)
def sile(request):
    sile_cls = request.param

    return sile_cls("test")

def test_pickling(sile):

    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    error = None

    try:
        with open(tmp_file.name, "wb") as fh:
            pickle.dump(sile, fh)

        with open(tmp_file.name, "rb") as fh:
            loaded_sile = pickle.load(fh)
    except Exception as e:
        error = e

    Path(tmp_file.name).unlink()

    if error is not None:
        raise error

    
