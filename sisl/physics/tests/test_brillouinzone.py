import pytest

from itertools import product
import math as m
import numpy as np

from sisl import SislError, geom
from sisl import Geometry, Atom, SuperCell, SuperCellChild
from sisl import BrillouinZone, BandStructure
from sisl import MonkhorstPack


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            self.s1 = SuperCell(1, nsc=[3, 3, 1])
            self.s2 = SuperCell([2, 2, 10, 90, 90, 60], [5, 5, 1])
    return t()


@pytest.mark.brillouinzone
@pytest.mark.bz
class TestBrillouinZone:

    def setUp(self, setup):
        setup.s1 = SuperCell(1, nsc=[3, 3, 1])
        setup.s2 = SuperCell([2, 2, 10, 90, 90, 60], [5, 5, 1])

    def test_bz1(self, setup):
        bz = BrillouinZone(1.)
        str(bz)
        bz.weight
        bz = BrillouinZone(setup.s1)
        assert len(bz) == 1
        assert np.allclose(bz.tocartesian([0, 0, 0]), [0] * 3)
        assert np.allclose(bz.tocartesian([0.5, 0, 0]), [m.pi, 0, 0])
        assert np.allclose(bz.toreduced([0, 0, 0]), [0] * 3)
        assert np.allclose([0.5, 0, 0], bz.tocartesian(bz.toreduced([0.5, 0, 0])))
        for k in bz:
            assert np.allclose(k, np.zeros(3))

        w = 0.
        for k, wk in bz.iter(True):
            assert np.allclose(k, np.zeros(3))
            w += wk
        assert w == pytest.approx(1.)

        bz = BrillouinZone(setup.s1, [[0]*3, [0.5]*3], [.5]*2)
        assert len(bz) == 2
        assert len(bz.copy()) == 2

    @pytest.mark.xfail(raises=ValueError)
    def test_bz_fail(self, setup):
        BrillouinZone(setup.s1, [0] * 3, [.5] * 2)

    def test_to_reduced(self, setup):
        bz = BrillouinZone(setup.s2)
        for k in [[0.1] * 3, [0.2] * 3]:
            cart = bz.tocartesian(k)
            rec = bz.toreduced(cart)
            assert np.allclose(rec, k)

    def test_class1(self, setup):
        class Test(SuperCellChild):
            def __init__(self, sc):
                self.set_supercell(sc)
            def eigh(self, k, *args, **kwargs):
                return np.arange(3)
            def eig(self, k, *args, **kwargs):
                return np.arange(3) - 1
        bz = BrillouinZone(Test(setup.s1))
        bz_arr = bz.apply.array
        str(bz)
        assert np.allclose(bz_arr.eigh(), np.arange(3))
        assert np.allclose(bz_arr.eig(), np.arange(3)-1)

    def test_class2(self, setup):
        class Test(SuperCellChild):
            def __init__(self, sc):
                self.set_supercell(sc)
            def eigh(self, k, *args, **kwargs):
                return np.arange(3)
            def eig(self, k, *args, **kwargs):
                return np.arange(3) - 1
        bz = BrillouinZone(Test(setup.s1))
        # Try the list/yield method
        for val in bz.apply.list.eigh():
            assert np.allclose(val, np.arange(3))
        for val in bz.apply.iter.eigh():
            assert np.allclose(val, np.arange(3))
        for val in bz.apply.iter.eig():
            assert np.allclose(val, np.arange(3) - 1)
        for val in bz.apply.oplist.eigh():
            assert np.allclose(val, np.arange(3))
        # Average
        bz_average = bz.apply.average
        assert np.allclose(bz_average.eigh(), np.arange(3))
        assert np.allclose(bz_average.eigh(eta=True), np.arange(3))
        assert np.allclose(bz.apply.eigh.average(eta=True), np.arange(3))

    def test_class3(self, setup):
        class Test(SuperCellChild):
            def __init__(self, sc):
                self.set_supercell(sc)
            def eigh(self, k, *args, **kwargs):
                return np.arange(3)
            def eig(self, k, *args, **kwargs):
                return np.arange(3) - 1
        bz = MonkhorstPack(Test(setup.s1), [2] * 3)
        # Try the yield method
        bz_yield = bz.apply.iter
        for val in bz_yield.eigh():
            assert np.allclose(val, np.arange(3))
        for val in bz_yield.eig():
            assert np.allclose(val, np.arange(3) - 1)
        # Average
        assert np.allclose(bz.apply.average.eigh(), np.arange(3))

    @pytest.mark.parametrize("N", [2, 3, 4, 5, 7])
    @pytest.mark.parametrize("centered", [True, False])
    def test_mp_asgrid(self, setup, N, centered):
        class Test(SuperCellChild):
            def __init__(self, sc):
                self.set_supercell(sc)
            def eigh(self, k, *args, **kwargs):
                return np.arange(3)
        bz = MonkhorstPack(Test(setup.s1), [2] * 3).asgrid()

        # Check the shape
        grid = bz.eigh(wrap=lambda eig: eig[0])
        assert np.allclose(grid.shape, [2] * 3)

        # Check the grids are different
        grid2 = bz.eigh(grid_unit='Bohr', wrap=lambda eig: eig[0])
        assert not np.allclose(grid.cell, grid2.cell)

        assert np.allclose(grid.grid, grid2.grid)
        for i in range(3):
            grid = bz.eigh(data_axis=i)
            shape = [2] * 3
            shape[i] = 3
            assert np.allclose(grid.shape, shape)

    @pytest.mark.xfail(raises=SislError)
    def test_mp_asgrid_fail(self, setup):
        class Test(SuperCellChild):
            def __init__(self, sc):
                self.set_supercell(sc)
            def eigh(self, k, *args, **kwargs):
                return np.arange(3)
        bz = MonkhorstPack(Test(setup.s1), [2] * 3, displacement=[0.1] * 3).asgrid()
        bz.eigh(wrap=lambda eig: eig[0])

    def test_mp1(self, setup):
        bz = MonkhorstPack(setup.s1, [2] * 3, trs=False)
        assert len(bz) == 8
        assert bz.weight[0] == 1. / 8

    def test_mp2(self, setup):
        bz1 = MonkhorstPack(setup.s1, [2] * 3, centered=False, trs=False)
        assert len(bz1) == 8
        bz2 = MonkhorstPack(setup.s1, [2] * 3, displacement=[.5] * 3, trs=False)
        assert len(bz2) == 8
        assert np.allclose(bz1.k, bz2.k)

    def test_mp_uneven(self, setup):
        bz1 = MonkhorstPack(setup.s1, [3] * 3, trs=False)
        bz2 = MonkhorstPack(setup.s1, [3] * 3, displacement=[.5] * 3, trs=False)
        assert not np.allclose(bz1.k, bz2.k)

    def test_mp3(self, setup):
        bz1 = MonkhorstPack(setup.s1, [2] * 3, size=0.5, trs=False)
        assert len(bz1) == 8
        assert np.all(bz1.k <= 0.25)
        assert bz1.weight.sum() == pytest.approx(0.5 ** 3)

    def test_trs(self, setup):
        size = [0.05, 0.5, 0.9]
        for x, y, z in product(np.arange(10) + 1, np.arange(20) + 1, np.arange(6) + 1):
            bz = MonkhorstPack(setup.s1, [x, y, z])
            assert bz.weight.sum() == pytest.approx(1.)
            bz = MonkhorstPack(setup.s1, [x, y, z], size=size)
            assert bz.weight.sum() == pytest.approx(np.prod(size))

    def test_mp_gamma_centered(self, setup):
        for x, y, z in product(np.arange(10) + 1, np.arange(20) + 1, np.arange(6) + 1):
            bz = MonkhorstPack(setup.s1, [x, y, z], trs=False)
            assert len(bz) == x * y * z
            assert ((bz.k == 0.).sum(1).astype(np.int32) == 3).sum() == 1

    def test_mp_gamma_non_centered(self, setup):
        for x, y, z in product(np.arange(10) + 1, np.arange(20) + 1, np.arange(6) + 1):
            bz = MonkhorstPack(setup.s1, [x, y, z], centered=False, trs=False)
            assert len(bz) == x * y * z
            # The gamma point will also be in the unit-cell for
            # non-centered
            has_gamma = x % 2 == 1
            has_gamma &= y % 2 == 1
            has_gamma &= z % 2 == 1
            if has_gamma:
                assert ((bz.k == 0.).sum(1).astype(np.int32) == 3).sum() == 1
            else:
                assert ((bz.k == 0.).sum(1).astype(np.int32) == 3).sum() == 0

    def test_mp_gamma_centered_displ(self, setup):
        for x, y, z in product(np.arange(10) + 1, np.arange(20) + 1, np.arange(6) + 1):
            bz = MonkhorstPack(setup.s1, [x, y, z], displacement=[0.2, 0, 0], trs=False)
            k = bz.k.copy()
            k[:, 0] -= 0.2
            assert len(bz) == x * y * z
            if x % 2 == 1:
                assert ((k == 0.).sum(1).astype(np.int32) == 3).sum() == 1
            else:
                assert ((k == 0.).sum(1).astype(np.int32) == 3).sum() == 0

    def test_pbz1(self, setup):
        bz = BandStructure(setup.s1, [[0]*3, [.5]*3], 300)
        assert len(bz) == 300

        bz2 = BandStructure(setup.s1, [[0]*2, [.5]*2], 300, ['A', 'C'])
        assert len(bz) == 300

        bz3 = BandStructure(setup.s1, [[0]*2, [.5]*2], [150] * 2)
        assert len(bz) == 300
        bz.lineartick()
        bz.lineark()
        bz.lineark(True)

    def test_pbz2(self, setup):
        bz = BandStructure(setup.s1, [[0]*3, [.25]*3, [.5]*3], 300)
        assert len(bz) == 300

    def test_as_simple(self):
        from sisl import geom, Hamiltonian
        g = geom.graphene()
        H = Hamiltonian(g)
        H.construct([[0.1, 1.44], [0, -2.7]])

        bz = MonkhorstPack(H, [2, 2, 2], trs=False)
        assert len(bz) == 2 ** 3

        # Assert that as* all does the same
        apply = bz.apply
        asarray = apply.array.eigh()
        aslist = np.array(apply.list.eigh())
        asyield = np.array([a for a in apply.iter.eigh()])
        asaverage = apply.average.eigh()
        assert np.allclose(asarray, aslist)
        assert np.allclose(asarray, asyield)
        # Average needs to be performed
        assert np.allclose((asarray / len(bz)).sum(0), asaverage)
        apply.none.eigh()

    def test_as_dataarray(self):
        try:
            import xarray
        except ImportError:
            pytest.skip('xarray not available')

        from sisl import geom, Hamiltonian
        g = geom.graphene()
        H = Hamiltonian(g)
        H.construct([[0.1, 1.44], [0, -2.7]])

        bz = MonkhorstPack(H, [2, 2, 2], trs=False)

        # Assert that as* all does the same
        asarray = bz.apply.array.eigh()
        bz_da = bz.apply.dataarray
        asdarray = bz_da.eigh()
        assert np.allclose(asarray, asdarray.values)
        assert isinstance(asdarray.bz, MonkhorstPack)
        assert isinstance(asdarray.parent, Hamiltonian)
        assert asdarray.dims == ('k', 'v1')

        asdarray = bz_da.eigh(coords=['orb'])
        assert asdarray.dims == ('k', 'orb')

    def test_as_single(self):
        from sisl import geom, Hamiltonian
        g = geom.graphene()
        H = Hamiltonian(g)
        H.construct([[0.1, 1.44], [0, -2.7]])

        def wrap(eig):
            return eig[0]

        bz = MonkhorstPack(H, [2, 2, 2], trs=False)
        assert len(bz) == 2 ** 3

        # Assert that as* all does the same
        asarray = bz.apply.array.eigh(wrap=wrap)
        aslist = np.array(bz.apply.list.eigh(wrap=wrap))
        asyield = np.array([a for a in bz.apply.iter.eigh(wrap=wrap)])
        assert np.allclose(asarray, aslist)
        assert np.allclose(asarray, asyield)

    def test_as_wrap(self):
        from sisl import geom, Hamiltonian
        g = geom.graphene()
        H = Hamiltonian(g)
        H.construct([[0.1, 1.44], [0, -2.7]])

        bz = MonkhorstPack(H, [2, 2, 2], trs=False)
        assert len(bz) == 2 ** 3

        # Check with a wrap function
        def wrap(arg):
            return arg[::-1]

        # Assert that as* all does the same
        asarray = bz.apply.array.eigh(wrap=wrap)
        aslist = np.array(bz.apply.list.eigh(wrap=wrap))
        asyield = np.array([a for a in bz.apply.iter.eigh(wrap=wrap)])
        asaverage = bz.apply.average.eigh(wrap=wrap)
        assert np.allclose(asarray, aslist)
        assert np.allclose(asarray, asyield)
        assert np.allclose((asarray / len(bz)).sum(0), asaverage)

        # Now we should check whether the reverse is doing its magic!
        mylist = [wrap(H.eigh(k=k)) for k in bz]
        assert np.allclose(aslist, mylist)

    def test_as_wrap_default_oplist(self):
        from sisl import geom, Hamiltonian
        g = geom.graphene()
        H = Hamiltonian(g)
        H.construct([[0.1, 1.44], [0, -2.7]])

        bz = MonkhorstPack(H, [2, 2, 2], trs=False)
        assert len(bz) == 2 ** 3

        # Check with a wrap function
        E = np.linspace(-2, 2, 100)
        def wrap_sum(es, weight):
            PDOS = es.PDOS(E) * weight
            return PDOS.sum(0), PDOS

        DOS, PDOS = bz.apply.sum.eigenstate(wrap=wrap_sum)
        bz_arr = bz.apply.array
        assert np.allclose(bz_arr.DOS(E), DOS)
        assert np.allclose(bz_arr.PDOS(E), PDOS)

    # Check with a wrap function and the weight argument
    def test_wrap_kwargs(arg):
        from sisl import geom, Hamiltonian
        g = geom.graphene()
        H = Hamiltonian(g)
        H.construct([[0.1, 1.44], [0, -2.7]])

        bz = MonkhorstPack(H, [2, 2, 2], trs=False)
        assert len(bz) == 2 ** 3

        def wrap_none(arg):
            return arg
        def wrap_kwargs(arg, parent, k, weight):
            return arg * weight

        E = np.linspace(-2, 2, 100)
        bz_array = bz.apply.array
        asarray1 = (bz_array.DOS(E, wrap=wrap_none) * bz.weight.reshape(-1, 1)).sum(0)
        asarray2 = bz_array.DOS(E, wrap=wrap_kwargs).sum(0)
        bz_list = bz.apply.list
        aslist1 = (np.array(bz_list.DOS(E, wrap=wrap_none)) * bz.weight.reshape(-1, 1)).sum(0)
        aslist2 = np.array(bz_list.DOS(E, wrap=wrap_kwargs)).sum(0)
        bz_yield = bz.apply.iter
        asyield1 = (np.array([a for a in bz_yield.DOS(E, wrap=wrap_none)]) * bz.weight.reshape(-1, 1)).sum(0)
        asyield2 = np.array([a for a in bz_yield.DOS(E, wrap=wrap_kwargs)]).sum(0)

        asaverage = bz.apply.average.DOS(E, wrap=wrap_none)
        assum = bz.apply.sum.DOS(E, wrap=wrap_kwargs)

        assert np.allclose(asarray1, asaverage)
        assert np.allclose(asarray2, asaverage)
        assert np.allclose(aslist1, asaverage)
        assert np.allclose(aslist2, asaverage)
        assert np.allclose(asyield1, asaverage)
        assert np.allclose(asyield2, asaverage)
        assert np.allclose(assum, asaverage)

    def test_replace_gamma(self):
        g = geom.graphene()
        bz = MonkhorstPack(g, 2, trs=False)
        bz_gamma = MonkhorstPack(g, [2, 2, 2], size=[0.5] * 3, trs=False)
        assert len(bz) == 2 ** 3
        bz.replace([0] * 3, bz_gamma)
        assert len(bz) == 2 ** 3 + 2 ** 3 - 1
        assert bz.weight.sum() == pytest.approx(1.)
        assert np.allclose(bz.copy().k, bz.k)
        assert np.allclose(bz.copy().weight, bz.weight)

    def test_in_primitive(self):
        assert np.allclose(MonkhorstPack.in_primitive([[1.] * 3, [-1.] * 3]), 0)

    def test_default_weight(self):
        bz1 = BrillouinZone(geom.graphene(), [[0] * 3, [0.25] * 3], [1/2] * 2)
        bz2 = BrillouinZone(geom.graphene(), [[0] * 3, [0.25] * 3])
        assert np.allclose(bz1.k, bz2.k)
        assert np.allclose(bz1.weight, bz2.weight)

    def test_brillouinzone_pickle(self, setup):
        import pickle as p
        bz1 = BrillouinZone(geom.graphene(), [[0] * 3, [0.25] * 3], [1/2] * 2)
        n = p.dumps(bz1)
        bz2 = p.loads(n)
        assert np.allclose(bz1.k, bz2.k)
        assert np.allclose(bz1.weight, bz2.weight)
        assert bz1.parent == bz2.parent

    def test_monkhorstpack_pickle(self, setup):
        import pickle as p
        bz1 = MonkhorstPack(geom.graphene(), [10, 11, 1], centered=False)
        n = p.dumps(bz1)
        bz2 = p.loads(n)
        assert np.allclose(bz1.k, bz2.k)
        assert np.allclose(bz1.weight, bz2.weight)
        assert bz1.parent == bz2.parent
        assert bz1._centered == bz2._centered

    def test_bandstructure_pickle(self, setup):
        import pickle as p
        bz1 = BandStructure(setup.s1, [[0]*2, [.5]*2], 300, ['A', 'C'])
        n = p.dumps(bz1)
        bz2 = p.loads(n)
        assert np.allclose(bz1.k, bz2.k)
        assert np.allclose(bz1.weight, bz2.weight)
        assert bz1.parent == bz2.parent
        assert np.allclose(bz1.point, bz2.point)
        assert np.allclose(bz1.division, bz2.division)
        assert bz1.name == bz2.name

    @pytest.mark.parametrize("n", [[0, 0, 1], [0.5] * 3])
    def test_param_circle(self, n):
        bz = BrillouinZone.param_circle(1, 10, 0.1, n, [1/2] * 3)
        assert len(bz) == 10
        sc = SuperCell(1)
        bz_loop = BrillouinZone.param_circle(sc, 10, 0.1, n, [1/2] * 3, True)
        assert len(bz_loop) == 10
        assert not np.allclose(bz.k, bz_loop.k)
        assert np.allclose(bz_loop.k[0, :], bz_loop.k[-1, :])

    def test_replace_gamma_trs(self):
        g = geom.graphene()
        bz = MonkhorstPack(g, [2, 2, 2], trs=False)
        N_bz = len(bz)
        bz_gamma = MonkhorstPack(g, [3, 3, 3], size=[0.5] * 3, trs=True)
        N_bz_gamma = len(bz_gamma)
        bz.replace([0] * 3, bz_gamma)
        assert len(bz) == N_bz + N_bz_gamma - 1
        assert bz.weight.sum() == pytest.approx(1.)
