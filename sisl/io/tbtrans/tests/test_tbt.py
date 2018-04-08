""" pytest test configures """
from __future__ import print_function

import pytest
import os.path as osp
import numpy as np

import sisl


pytestmark = pytest.mark.tbtrans


@pytest.mark.slow
def test_1_graphene_all(files):
    """ This tests manifolds itself as:

    sisl.geom.graphene(orthogonal=True).tile(3, 0).tile(5, 1)

    All output is enabled:

    ### FDF ###
    # Transmission related quantities
    TBT.T.All T
    TBT.T.Out T
    TBT.T.Eig 2

    # Density of states
    TBT.DOS.Elecs T
    TBT.DOS.Gf T
    TBT.DOS.A T
    TBT.DOS.A.All T

    # Orbital currents and Crystal-Orbital investigations.
    TBT.Symmetry.TimeReversal F
    TBT.Current.Orb T
    TBT.COOP.Gf T
    TBT.COOP.A T
    TBT.COHP.Gf T
    TBT.COHP.A T

    TBT.k [100 1 1]
    ### FDF ###
    """
    tbt = sisl.get_sile(osp.join(files, '1_graphene_all.TBT.nc'))
    assert tbt.E.min() > -2.
    assert tbt.E.max() < 2.
    # We have 400 energy-points
    ne = len(tbt.E)
    assert ne == 400

    # We have 100 k-points
    nk = len(tbt.kpt)
    assert nk == 100

    # Get geometry
    geom = tbt.geometry

    # Check read is the same as the direct query
    assert tbt.na == geom.na
    assert tbt.no == geom.no
    assert tbt.no == geom.na
    assert tbt.na == 3 * 5 * 4
    assert np.allclose(tbt.cell, geom.cell)

    # Check device atoms (1-orbital system)
    assert tbt.na_d == tbt.no_d
    assert tbt.na_d == 36 # 3 * 5 * 4 (and device is without electrodes, so 3 * 3 * 4)
    assert len(tbt.pivot()) == 3 * 3 * 4 # 3 * 5 * 4 (and device is without electrodes, so 3 * 3 * 4)

    # Check electrodes
    assert len(tbt.elecs) == 2
    elecs = tbt.elecs[:]
    assert elecs == ['Left', 'Right']

    # Check the chemical potentials
    for elec in elecs:
        assert tbt.chemical_potential(elec) == pytest.approx(0.)
        assert tbt.electronic_temperature(elec) == pytest.approx(300., abs=1)

    # Check electrode relevant stuff
    left = elecs[0]
    right = elecs[1]

    # Assert we have transmission symmetry
    assert np.allclose(tbt.transmission(left, right),
                       tbt.transmission(right, left))
    assert np.allclose(tbt.transmission_eig(left, right),
                       tbt.transmission_eig(right, left))
    # Check that the total transmission is larger than the sum of transmission eigenvalues
    assert np.all(tbt.transmission(left, right) + 1e-7 >= tbt.transmission_eig(left, right).sum(-1))
    assert np.all(tbt.transmission(right, left) + 1e-7 >= tbt.transmission_eig(right, left).sum(-1))

    # Also check for each k
    for ik in range(nk):
        assert np.allclose(tbt.transmission(left, right, ik),
                           tbt.transmission(right, left, ik))
        assert np.allclose(tbt.transmission_eig(left, right, ik),
                           tbt.transmission_eig(right, left, ik))
        assert np.all(tbt.transmission(left, right, ik) + 1e-7 >= tbt.transmission_eig(left, right, ik).sum(-1))
        assert np.all(tbt.transmission(right, left, ik) + 1e-7 >= tbt.transmission_eig(right, left, ik).sum(-1))
        assert np.allclose(tbt.DOS(kavg=ik), tbt.ADOS(left, kavg=ik) + tbt.ADOS(right, kavg=ik))

    # Check that norm returns correct values
    assert tbt.norm() == 1
    assert tbt.norm(norm='all') == tbt.no_d
    assert tbt.norm(norm='atom') == tbt.norm(norm='orbital')

    # Check atom is equivalent to orbital
    for norm in ['atom', 'orbital']:
        assert tbt.norm(0, norm=norm) == 0.
        assert tbt.norm(3*4, norm=norm) == 1
        assert tbt.norm(range(3*4, 3*5), norm=norm) == 3

    # Assert sum(ADOS) == DOS
    assert np.allclose(tbt.DOS(), tbt.ADOS(left) + tbt.ADOS(right))
    assert np.allclose(tbt.DOS(sum=False), tbt.ADOS(left, sum=False) + tbt.ADOS(right, sum=False))

    # Current must be 0 when the chemical potentials are equal
    assert tbt.current(left, right) == pytest.approx(0.)
    assert tbt.current(right, left) == pytest.approx(0.)

    high_low = tbt.current_parameter(left, 0.5, 0.0025, right, -0.5, 0.0025)
    low_high = tbt.current_parameter(left, -0.5, 0.0025, right, 0.5, 0.0025)
    assert high_low > 0.
    assert low_high < 0.
    assert - high_low == pytest.approx(low_high)

    # Since this is a perfect system there should be *no* QM shot-noise
    # Also, the shot-noise is related to the applied bias, so NO shot-noise
    assert np.allclose(tbt.shot_noise(left, right), 0.)
    assert np.allclose(tbt.shot_noise(right, left), 0.)
    # Since the data-file does not contain all T-eigs (only the first two)
    # we can't correctly calculate the fano factors
    assert np.all(tbt.fano(left, right) > 0.)
    assert np.all(tbt.fano(right, left) > 0.)
