import pytest

import math as m
import numpy as np

import sisl

pytestmark = pytest.mark.version


def test_version():
    sisl.__bibtex__
    sisl.__git_revision__
    sisl.__version__
    sisl.__major__
    sisl.__minor__
    sisl.__micro__
    # Currently we only do bibtex citation
    assert sisl.__bibtex__ == sisl.cite()
    sisl.info.bibtex
    sisl.info.git_revision
    sisl.info.version
    sisl.info.major
    sisl.info.minor
    sisl.info.micro
    sisl.info.release
    sisl.info.git_revision
    sisl.info.git_revision_short


def test_import1():
    # The imports should only be visible in the io module
    s = sisl.BaseSile
    s = sisl.Sile
    s = sisl.SileCDF
    s = sisl.SileBin
    s = sisl.io.xyzSile


def test_import2():
    # The imports should only be visible in the io module
    with pytest.raises(AttributeError):
        sisl.xyzSile


def test_import3():
    # The imports should only be visible in the io module
    with pytest.raises(ImportError):
        from sisl import xyzSile
