""" pytest test configures """

import pytest
import os.path as osp
import sisl


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')


def test_fe(sisl_files):
    si = sisl.get_sile(sisl_files(_dir, 'fe.bands'))
    labels, k, eig = si.read_data()
    assert k.shape == (131, )
    assert eig.shape == (131, 2, 15)
    assert len(labels[0]) == 5


def test_fe_ArgumentParser(sisl_files, sisl_tmp):
    pytest.importorskip("matplotlib", reason="matplotlib not available")
    png = sisl_tmp('fe.bands.png', _dir)
    si = sisl.get_sile(sisl_files(_dir, 'fe.bands'))
    p, ns = si.ArgumentParser()
    p.parse_args([], namespace=ns)
    p.parse_args(['--energy', ' -2:2'], namespace=ns)
    p.parse_args(['--energy', ' -2:2', '--plot', png], namespace=ns)


def test_fe_xarray(sisl_files, sisl_tmp):
    pytest.importorskip("xarray", reason="xarray not available")
    si = sisl.get_sile(sisl_files(_dir, 'fe.bands'))

    bands = si.read_data(as_dataarray=True)
    assert len(bands['k']) == 131
    assert len(bands['spin']) == 2
    assert len(bands['band']) == 15
    assert len(bands.ticks) == len(bands.ticklabels) == 5
