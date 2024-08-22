# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import math as m
from itertools import product

import numpy as np
import pytest

from sisl import (
    BandStructure,
    BrillouinZone,
    Lattice,
    LatticeChild,
    MonkhorstPack,
    SislError,
    geom,
)

pytestmark = [pytest.mark.physics, pytest.mark.brillouinzone, pytest.mark.bz]


@pytest.fixture
def setup():
    class t:
        def __init__(self):
            self.s1 = Lattice(1, nsc=[3, 3, 1])
            self.s2 = Lattice([2, 2, 10, 90, 90, 60], [5, 5, 1])
            self.s3 = Lattice(1, nsc=[3, 1, 3])

    return t()


class TestBrillouinZone:
    def test_bz1(self, setup):
        bz = BrillouinZone(1.0)
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

        w = 0.0
        for k, wk in bz.iter(True):
            assert np.allclose(k, np.zeros(3))
            w += wk
        assert w == pytest.approx(1.0)

        bz = BrillouinZone(setup.s1, [[0] * 3, [0.5] * 3], [0.5] * 2)
        assert len(bz) == 2
        assert len(bz.copy()) == 2

    def test_weight_automatic(self, setup):
        bz = BrillouinZone(1.0)
        assert bz.weight[0] == 1.0

        bz = BrillouinZone(setup.s1, np.random.rand(3, 3))
        assert bz.weight.sum() == pytest.approx(1)

        bz = BrillouinZone(setup.s1, np.random.rand(3, 3), 0.5)
        assert bz.weight.sum() == pytest.approx(1.5)

    def test_volume_self(self):
        bz = BrillouinZone(1.0)
        bz.parent.pbc = (False, False, False)
        v, dim = bz.volume(True)
        assert dim == 0
        assert v == pytest.approx(0)
        bz.parent.pbc = (True, False, False)
        assert bz.volume(True)[1] == 1
        bz.parent.pbc = (True, True, False)
        assert bz.volume(True)[1] == 2
        bz.parent.pbc = (True, True, True)
        v, dim = bz.volume(True)
        assert dim == 3
        assert v == pytest.approx((2 * np.pi) ** 3)

    def test_volume_direct(self):
        bz = BrillouinZone(1.0)
        assert bz.volume(True, [0, 1])[1] == 2
        assert bz.volume(True, [1])[1] == 1
        assert bz.volume(True, [2, 1])[1] == 2
        assert bz.volume(True, [2, 1, 0])[1] == 3
        assert bz.volume(True, [])[1] == 0

    def test_fail(self, setup):
        with pytest.raises(ValueError):
            BrillouinZone(setup.s1, [0] * 3, [0.5] * 2)

    def test_to_reduced(self, setup):
        bz = BrillouinZone(setup.s2)
        for k in [[0.1] * 3, [0.2] * 3]:
            cart = bz.tocartesian(k)
            rec = bz.toreduced(cart)
            assert np.allclose(rec, k)

    def test_class1(self, setup):
        class Test(LatticeChild):
            def __init__(self, lattice):
                self.set_lattice(lattice)

            def eigh(self, k, *args, **kwargs):
                return np.arange(3)

            def eig(self, k, *args, **kwargs):
                return np.arange(3) - 1

        bz = BrillouinZone(Test(setup.s1))
        bz_arr = bz.apply.array
        str(bz)
        assert np.allclose(bz_arr.eigh(), np.arange(3))
        assert np.allclose(bz_arr.eig(), np.arange(3) - 1)

    def test_class2(self, setup):
        class Test(LatticeChild):
            def __init__(self, lattice):
                self.set_lattice(lattice)

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

    def test_parametrize_integer(self, setup):
        # parametrize for single integers
        def func(parent, N, i):
            return [i / N, 0, 0]

        bz = BrillouinZone.parametrize(setup.s1, func, 10)
        assert len(bz) == 10
        assert np.allclose(bz.k[-1], [9 / 10, 0, 0])

    def test_parametrize_list(self, setup):
        # parametrize for single integers
        def func(parent, N, i):
            return [i[0] / N[0], i[1] / N[1], 0]

        bz = BrillouinZone.parametrize(setup.s1, func, [10, 2])
        assert len(bz) == 20
        assert np.allclose(bz.k[-1], [9 / 10, 1 / 2, 0])
        assert np.allclose(bz.k[-2], [9 / 10, 0 / 2, 0])

    def test_default_weight(self):
        bz1 = BrillouinZone(geom.graphene(), [[0] * 3, [0.25] * 3], [1 / 2] * 2)
        bz2 = BrillouinZone(geom.graphene(), [[0] * 3, [0.25] * 3])
        assert np.allclose(bz1.k, bz2.k)
        assert np.allclose(bz1.weight, bz2.weight)

    def test_pickle(self, setup):
        import pickle as p

        bz1 = BrillouinZone(geom.graphene(), [[0] * 3, [0.25] * 3], [1 / 2] * 2)
        n = p.dumps(bz1)
        bz2 = p.loads(n)
        assert np.allclose(bz1.k, bz2.k)
        assert np.allclose(bz1.weight, bz2.weight)
        assert bz1.parent == bz2.parent

    @pytest.mark.parametrize("n", [[0, 0, 1], [0.5] * 3])
    def test_param_circle(self, n):
        bz = BrillouinZone.param_circle(1, 10, 0.1, n, [1 / 2] * 3)
        assert len(bz) == 10
        sc = Lattice(1)
        bz_loop = BrillouinZone.param_circle(sc, 10, 0.1, n, [1 / 2] * 3, True)
        assert len(bz_loop) == 10
        assert not np.allclose(bz.k, bz_loop.k)
        assert np.allclose(bz_loop.k[0, :], bz_loop.k[-1, :])

    def test_merge_simple(self):
        normal = [0] * 3
        origin = [1 / 2] * 3

        bzs = [
            BrillouinZone.param_circle(1, 10, 0.1, normal, origin),
            BrillouinZone.param_circle(1, 10, 0.2, normal, origin),
            BrillouinZone.param_circle(1, 10, 0.3, normal, origin),
        ]
        bz = BrillouinZone.merge(bzs)
        assert len(bz) == 30
        assert bz.weight.sum() == pytest.approx(
            bzs[0].weight.sum() + bzs[1].weight.sum() + bzs[2].weight.sum()
        )

    def test_merge_scales(self):
        normal = [0] * 3
        origin = [1 / 2] * 3

        bzs = [
            BrillouinZone.param_circle(1, 10, 0.1, normal, origin),
            BrillouinZone.param_circle(1, 10, 0.2, normal, origin),
            BrillouinZone.param_circle(1, 10, 0.3, normal, origin),
        ]
        bz = BrillouinZone.merge(bzs, [1, 2, 3])
        assert len(bz) == 30
        assert bz.weight.sum() == pytest.approx(
            bzs[0].weight.sum() + bzs[1].weight.sum() * 2 + bzs[2].weight.sum() * 3
        )

    def test_merge_scales_short(self):
        normal = [0] * 3
        origin = [1 / 2] * 3

        bzs = [
            BrillouinZone.param_circle(1, 10, 0.1, normal, [1 / 2] * 3),
            BrillouinZone.param_circle(1, 10, 0.2, normal, [1 / 2] * 3),
            BrillouinZone.param_circle(1, 10, 0.3, normal, [1 / 2] * 3),
        ]
        bz = BrillouinZone.merge(bzs, [1, 2])
        assert len(bz) == 30
        assert bz.weight.sum() == pytest.approx(
            bzs[0].weight.sum() + bzs[1].weight.sum() * 2 + bzs[2].weight.sum() * 2
        )

    def test_merge_scales_scalar(self):
        normal = [0] * 3
        origin = [1 / 2] * 3

        bzs = [
            BrillouinZone.param_circle(1, 10, 0.1, normal, [1 / 2] * 3),
            BrillouinZone.param_circle(1, 10, 0.3, normal, [1 / 2] * 3),
        ]
        bz = BrillouinZone.merge(bzs, 1)
        assert len(bz) == 20
        assert bz.weight.sum() == pytest.approx(
            bzs[0].weight.sum() + bzs[1].weight.sum()
        )


@pytest.mark.monkhorstpack
class TestMonkhorstPack:
    def test_class(self, setup):
        class Test(LatticeChild):
            def __init__(self, lattice):
                self.set_lattice(lattice)

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

    def test_pickle(self, setup):
        import pickle as p

        bz1 = MonkhorstPack(geom.graphene(), [10, 11, 1], centered=False)
        n = p.dumps(bz1)
        bz2 = p.loads(n)
        assert np.allclose(bz1.k, bz2.k)
        assert np.allclose(bz1.weight, bz2.weight)
        assert bz1.parent == bz2.parent
        assert bz1._centered == bz2._centered

    @pytest.mark.parametrize("N", [2, 3, 4, 5, 7])
    @pytest.mark.parametrize("centered", [True, False])
    def test_asgrid(self, setup, N, centered):
        class Test(LatticeChild):
            def __init__(self, lattice):
                self.set_lattice(lattice)

            def eigh(self, k, *args, **kwargs):
                return np.arange(3)

        bz = MonkhorstPack(Test(setup.s1), [2] * 3).apply.grid

        # Check the shape
        grid = bz.eigh(wrap=lambda eig: eig[0])
        assert np.allclose(grid.shape, [2] * 3)

        # Check the grids are different
        grid2 = bz.eigh(grid_unit="Bohr", wrap=lambda eig: eig[0])
        assert not np.allclose(grid.cell, grid2.cell)

        assert np.allclose(grid.grid, grid2.grid)
        for i in range(3):
            grid = bz.eigh(data_axis=i)
            shape = [2] * 3
            shape[i] = 3
            assert np.allclose(grid.shape, shape)

    def test_asgrid_fail(self, setup):
        class Test(LatticeChild):
            def __init__(self, lattice):
                self.set_lattice(lattice)

            def eigh(self, k, *args, **kwargs):
                return np.arange(3)

        bz = MonkhorstPack(Test(setup.s1), [2] * 3, displacement=[0.1] * 3).apply.grid
        with pytest.raises(SislError):
            bz.eigh(wrap=lambda eig: eig[0])

    def test_init_simple(self, setup):
        bz = MonkhorstPack(setup.s1, [2] * 3, trs=False)
        assert len(bz) == 8
        assert bz.weight[0] == 1.0 / 8

    def test_displaced(self, setup):
        bz1 = MonkhorstPack(setup.s1, [2] * 3, centered=False, trs=False)
        assert len(bz1) == 8
        bz2 = MonkhorstPack(setup.s1, [2] * 3, displacement=[0.5] * 3, trs=False)
        assert len(bz2) == 8
        assert np.allclose(bz1.k, bz2.k)

    def test_uneven(self, setup):
        bz1 = MonkhorstPack(setup.s1, [3] * 3, trs=False)
        bz2 = MonkhorstPack(setup.s1, [3] * 3, displacement=[0.5] * 3, trs=False)
        assert not np.allclose(bz1.k, bz2.k)

    def test_size_half(self, setup):
        bz1 = MonkhorstPack(setup.s1, [2] * 3, size=0.5, trs=False)
        assert len(bz1) == 8
        assert np.all(bz1.k <= 0.25)
        assert bz1.weight.sum() == pytest.approx(0.5**3)

    def test_as_dataarray(self):
        pytest.importorskip("xarray", reason="xarray not available")

        from sisl import Hamiltonian, geom

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
        assert asdarray.dims == ("k", "v1")

        asdarray = bz_da.eigh(coords=["orb"])
        assert asdarray.dims == ("k", "orb")

    def test_trs(self, setup):
        size = [0.05, 0.5, 0.9]
        for x, y, z in product(np.arange(10) + 1, np.arange(20) + 1, np.arange(6) + 1):
            bz = MonkhorstPack(setup.s1, [x, y, z])
            assert bz.weight.sum() == pytest.approx(1.0)
            bz = MonkhorstPack(setup.s1, [x, y, z], size=size)
            assert bz.weight.sum() == pytest.approx(np.prod(size))

    def test_gamma_centered(self, setup):
        for x, y, z in product(np.arange(10) + 1, np.arange(20) + 1, np.arange(6) + 1):
            bz = MonkhorstPack(setup.s1, [x, y, z], trs=False)
            assert len(bz) == x * y * z
            assert ((bz.k == 0.0).sum(1).astype(np.int32) == 3).sum() == 1

    def test_gamma_non_centered(self, setup):
        for x, y, z in product(np.arange(10) + 1, np.arange(20) + 1, np.arange(6) + 1):
            bz = MonkhorstPack(setup.s1, [x, y, z], centered=False, trs=False)
            assert len(bz) == x * y * z
            # The gamma point will also be in the unit-cell for
            # non-centered
            has_gamma = x % 2 == 1
            has_gamma &= y % 2 == 1
            has_gamma &= z % 2 == 1
            if has_gamma:
                assert ((bz.k == 0.0).sum(1).astype(np.int32) == 3).sum() == 1
            else:
                assert ((bz.k == 0.0).sum(1).astype(np.int32) == 3).sum() == 0

    def test_gamma_centered_displ(self, setup):
        for x, y, z in product(np.arange(10) + 1, np.arange(20) + 1, np.arange(6) + 1):
            bz = MonkhorstPack(setup.s1, [x, y, z], displacement=[0.2, 0, 0], trs=False)
            k = bz.k.copy()
            k[:, 0] -= 0.2
            assert len(bz) == x * y * z
            if x % 2 == 1:
                assert ((k == 0.0).sum(1).astype(np.int32) == 3).sum() == 1
            else:
                assert ((k == 0.0).sum(1).astype(np.int32) == 3).sum() == 0

    def test_as_simple(self):
        from sisl import Hamiltonian, geom

        g = geom.graphene()
        H = Hamiltonian(g)
        H.construct([[0.1, 1.44], [0, -2.7]])

        bz = MonkhorstPack(H, [2, 2, 2], trs=False)
        assert len(bz) == 2**3

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

    def test_as_dataarray_zip(self):
        pytest.importorskip("xarray", reason="xarray not available")

        from sisl import Hamiltonian, geom

        g = geom.graphene()
        H = Hamiltonian(g)
        H.construct([[0.1, 1.44], [0, -2.7]])

        E = np.linspace(-2, 2, 20)
        bz = MonkhorstPack(H, [2, 2, 1], trs=False)

        def wrap(es):
            return es.eig, es.DOS(E), es.PDOS(E)

        with bz.apply.renew(zip=True) as unzip:
            eig, DOS, PDOS = unzip.ndarray.eigenstate(wrap=wrap)
            ds0 = unzip.dataarray.eigenstate(wrap=wrap, name=["eig", "DOS", "PDOS"])
            # explicitly create dimensions
            ds1 = unzip.dataarray.eigenstate(
                wrap=wrap,
                coords=[
                    {"orb": np.arange(len(H))},
                    {"E": E},
                    {"spin": [0], "orb": np.arange(len(H)), "E": E},
                ],
                dims=(["orb"], ["E"], ["spin", "orb", "E"]),
                name=["eig", "DOS", "PDOS"],
            )

        for var, data in zip(["eig", "DOS", "PDOS"], [eig, DOS, PDOS]):
            assert np.allclose(ds0.data_vars[var].values, data)
            assert np.allclose(ds1.data_vars[var].values, data)
        assert len(ds1.coords) < len(ds0.coords)

    def test_bz_parallel_pathos(self):
        pytest.importorskip("pathos", reason="pathos not available")

        from sisl import Hamiltonian, geom

        g = geom.graphene()
        H = Hamiltonian(g)
        H.construct([[0.1, 1.44], [0, -2.7]])

        bz = MonkhorstPack(H, [2, 2, 2], trs=False)

        # try and determine a sensible
        import os

        nprocs = 2

        try:
            nprocs = len(os.sched_getaffinity(0)) // 2
        except Exception:
            pass

        try:
            import psutil

            nprocs = len(psutil.Process().cpu_affinity()) // 2
        except Exception:
            pass

        if nprocs == 1:
            pytest.skip("not in a parallel environment")

        nprocs = max(2, nprocs)
        omp_num_threads = os.environ.get("OMP_NUM_THREADS")

        # Check that the ObjectDispatcher works
        apply = bz.apply

        papply = bz.apply.renew(pool=nprocs)
        assert str(apply) != str(papply)

        for method in ("iter", "average", "sum", "array", "list", "oplist"):
            # TODO One should be careful with zip
            # zip will stop when it hits the final element in the first
            # list.
            # So if a generator has some clean-up code one has to use zip_longest
            # regardless of method
            os.environ["OMP_NUM_THREADS"] = "1"
            V1 = papply[method].eigh()
            if omp_num_threads is None:
                del os.environ["OMP_NUM_THREADS"]
            else:
                os.environ["OMP_NUM_THREADS"] = omp_num_threads
            V2 = apply[method].eigh()

            for v1, v2 in zip(V1, V2):
                assert np.allclose(v1, v2)

        # Check that the MethodDispatcher works
        apply = bz.apply.eigh
        papply = bz.apply.renew(pool=True).eigh
        assert str(apply) != str(papply)

        for method in ("iter", "average", "sum", "array", "list", "oplist"):
            # TODO One should be careful with zip
            # zip will stop when it hits the final element in the first
            # list.
            # So if a generator has some clean-up code one has to use zip_longest
            # regardless of method
            for v1, v2 in zip(papply[method](), apply[method]()):
                assert np.allclose(v1, v2)

    def test_as_single(self):
        from sisl import Hamiltonian, geom

        g = geom.graphene()
        H = Hamiltonian(g)
        H.construct([[0.1, 1.44], [0, -2.7]])

        def wrap(eig):
            return eig[0]

        bz = MonkhorstPack(H, [2, 2, 2], trs=False)
        assert len(bz) == 2**3

        # Assert that as* all does the same
        asarray = bz.apply.array.eigh(wrap=wrap)
        aslist = np.array(bz.apply.list.eigh(wrap=wrap))
        asyield = np.array([a for a in bz.apply.iter.eigh(wrap=wrap)])
        assert np.allclose(asarray, aslist)
        assert np.allclose(asarray, asyield)

    def test_as_wrap(self):
        from sisl import Hamiltonian, geom

        g = geom.graphene()
        H = Hamiltonian(g)
        H.construct([[0.1, 1.44], [0, -2.7]])

        bz = MonkhorstPack(H, [2, 2, 2], trs=False)
        assert len(bz) == 2**3

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
        from sisl import Hamiltonian, geom

        g = geom.graphene()
        H = Hamiltonian(g)
        H.construct([[0.1, 1.44], [0, -2.7]])

        bz = MonkhorstPack(H, [2, 2, 2], trs=False)
        assert len(bz) == 2**3

        # Check with a wrap function
        E = np.linspace(-2, 2, 20)

        def wrap_sum(es, weight):
            PDOS = es.PDOS(E)[0] * weight
            return PDOS.sum(0), PDOS

        DOS, PDOS = bz.apply.sum.eigenstate(wrap=wrap_sum)
        bz_arr = bz.apply.array
        assert np.allclose(bz_arr.eigenstate(wrap=lambda es: es.DOS(E)), DOS)
        assert np.allclose(bz_arr.eigenstate(wrap=lambda es: es.PDOS(E)[0]), PDOS)

    def test_wrap_unzip(self):
        from sisl import Hamiltonian, geom

        g = geom.graphene()
        H = Hamiltonian(g)
        H.construct([[0.1, 1.44], [0, -2.7]])

        bz = MonkhorstPack(H, [2, 2, 2], trs=False)

        # Check with a wrap function
        E = np.linspace(-2, 2, 20)

        def wrap(es):
            return es.eig, es.DOS(E)

        eig0, DOS0 = zip(*bz.apply.list.eigenstate(wrap=wrap))
        with bz.apply.renew(zip=True) as k_unzip:
            eig1, DOS1 = k_unzip.list.eigenstate(wrap=wrap)
            eig2, DOS2 = k_unzip.array.eigenstate(wrap=wrap)

        # eig0 and DOS0 are generators, and not list's
        # eig1 and DOS1 are generators, and not list's
        assert isinstance(eig2, np.ndarray)
        assert isinstance(DOS2, np.ndarray)

        assert np.allclose(eig0, eig1)
        assert np.allclose(DOS0, DOS1)
        assert np.allclose(eig0, eig2)
        assert np.allclose(DOS0, DOS2)

    # Check with a wrap function and the weight argument
    def test_wrap_kwargs(arg):
        from sisl import Hamiltonian, geom

        g = geom.graphene()
        H = Hamiltonian(g)
        H.construct([[0.1, 1.44], [0, -2.7]])

        bz = MonkhorstPack(H, [2, 2, 2], trs=False)
        assert len(bz) == 2**3

        def wrap_none(arg):
            return arg

        def wrap_kwargs(arg, parent, k, weight):
            return arg * weight

        E = np.linspace(-2, 2, 20)
        bz_array = bz.apply.array
        asarray1 = (
            bz_array.eigenstate(wrap=lambda es: es.DOS(E)) * bz.weight.reshape(-1, 1)
        ).sum(0)
        asarray2 = bz_array.eigenstate(
            wrap=lambda es, parent, k, weight: es.DOS(E)
        ).sum(0)
        bz_list = bz.apply.list
        aslist1 = (
            np.array(bz_list.eigenstate(wrap=lambda es: es.DOS(E)))
            * bz.weight.reshape(-1, 1)
        ).sum(0)
        aslist2 = np.array(
            bz_list.eigenstate(wrap=lambda es, parent, k, weight: es.DOS(E))
        ).sum(0)
        bz_yield = bz.apply.iter
        asyield1 = (
            np.array([a for a in bz_yield.eigenstate(wrap=lambda es: es.DOS(E))])
            * bz.weight.reshape(-1, 1)
        ).sum(0)
        asyield2 = np.array(
            [
                a
                for a in bz_yield.eigenstate(
                    wrap=lambda es, parent, k, weight: es.DOS(E)
                )
            ]
        ).sum(0)

        asaverage = bz.apply.average.eigenstate(wrap=lambda es: es.DOS(E))
        assum = bz.apply.sum.eigenstate(wrap=lambda es: es.DOS(E))

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
        assert len(bz) == 2**3
        bz.replace([0] * 3, bz_gamma)
        assert len(bz) == 2**3 + 2**3 - 1
        assert bz.weight.sum() == pytest.approx(1.0)
        assert np.allclose(bz.copy().k, bz.k)
        assert np.allclose(bz.copy().weight, bz.weight)

    def test_replace_gamma_trs(self):
        g = geom.graphene()
        bz = MonkhorstPack(g, [2, 2, 2], trs=False)
        N_bz = len(bz)
        bz_gamma = MonkhorstPack(g, [3, 3, 3], size=[0.5] * 3, trs=True)
        N_bz_gamma = len(bz_gamma)
        bz.replace([0] * 3, bz_gamma)
        assert len(bz) == N_bz + N_bz_gamma - 1
        assert bz.weight.sum() == pytest.approx(1.0)

    def test_replace_trs_neg(self):
        g = geom.graphene()
        bz_big = MonkhorstPack(g, [6, 6, 1], trs=True)
        N_bz_big = len(bz_big)
        bz_small = MonkhorstPack(
            g,
            [3, 3, 3],
            size=[1 / 6, 1 / 6, 1],
            displacement=[2 / 3, 1 / 3, 0],
            trs=True,
        )
        N_bz_small = len(bz_small)

        # it should be the same for both negative|positive displ
        bz_pos = bz_big.copy()
        bz_neg = bz_big.copy()

        bz_pos.replace(bz_small.displacement, bz_small)
        bz_neg.replace(-bz_small.displacement, bz_small)
        for bz in [bz_pos, bz_neg]:
            assert len(bz) == N_bz_big + N_bz_small - 1
            assert bz.weight.sum() == pytest.approx(1.0)

    def test_in_primitive(self):
        assert np.allclose(MonkhorstPack.in_primitive([[1.0] * 3, [-1.0] * 3]), 0)


@pytest.mark.bandstructure
class TestBandStructure:
    def test_pbz1(self, setup):
        bz = BandStructure(setup.s1, [[0] * 3, [0.5] * 3], 300)
        assert len(bz) == 300

        bz2 = BandStructure(setup.s1, [[0] * 2, [0.5] * 2], 300, ["A", "C"])
        assert len(bz) == 300

        bz3 = BandStructure(setup.s1, [[0] * 2, [0.5] * 2], [150])
        assert len(bz) == 300
        bz.lineartick()
        bz.lineark()
        bz.lineark(True)

    @pytest.mark.parametrize("n", range(3, 100, 10))
    def test_pbz2(self, setup, n):
        bz = BandStructure(setup.s1, [[0] * 3, [0.25] * 3, [0.5] * 3], n)
        assert len(bz) == n

    def test_pbs_divisions(self, setup):
        bz = BandStructure(setup.s1, [[0] * 3, [0.25] * 3, [0.5] * 3], [10, 10])
        assert len(bz) == 21

    def test_pbc_fill(self, setup):
        bz = BandStructure(setup.s3, [[0] * 2, [0.25] * 2, [0.5] * 2], [10, 10])
        assert np.allclose(bz.k[:, 1], 0)
        assert len(bz) == 21

    def test_pbs_missing_arguments(self, setup):
        with pytest.raises(ValueError):
            bz = BandStructure(setup.s1, divisions=[10, 10])

    def test_pbs_fail(self, setup):
        with pytest.raises(ValueError):
            BandStructure(setup.s1, [[0] * 3, [0.5] * 3, [0.25] * 3], 1)
        with pytest.raises(ValueError):
            BandStructure(setup.s1, [[0] * 3, [0.5] * 3, [0.25] * 3], [1, 1, 1, 1])
        with pytest.raises(ValueError):
            BandStructure(setup.s1, [[0] * 3, [0.5] * 3, [0.25] * 3], [1, 1, 1])

    def test_pickle(self, setup):
        import pickle as p

        bz1 = BandStructure(setup.s1, [[0] * 2, [0.5] * 2], 300, ["A", "C"])
        n = p.dumps(bz1)
        bz2 = p.loads(n)
        assert np.allclose(bz1.k, bz2.k)
        assert np.allclose(bz1.weight, bz2.weight)
        assert bz1.parent == bz2.parent
        assert np.allclose(bz1.points, bz2.points)
        assert np.allclose(bz1.divisions, bz2.divisions)
        assert bz1.names == bz2.names

    def test_jump(self):
        g = geom.graphene()
        bs = BandStructure(
            g,
            [[0] * 3, [0.5, 0, 0], None, [0] * 3, [0.0, 0.5, 0]],
            30,
            ["A", "B", "C", "D"],
        )
        assert len(bs) == 30

    def test_jump_skipping_none(self):
        g = geom.graphene()
        bs1 = BandStructure(
            g,
            [[0] * 3, [0.5, 0, 0], None, [0] * 3, [0.0, 0.5, 0]],
            30,
            ["A", "B", "C", "D"],
        )
        bs2 = BandStructure(
            g,
            [[0] * 3, [0.5, 0, 0], None, [0] * 3, [0.0, 0.5, 0], None],
            30,
            ["A", "B", "C", "D"],
        )
        assert np.allclose(bs1.k, bs2.k)

    def test_insert_jump(self):
        g = geom.graphene()
        nk = 10
        bs = BandStructure(
            g,
            [[0] * 3, [0.5, 0, 0], None, [0] * 3, None, [0.0, 0.5, 0]],
            nk,
            ["A", "B", "C", "D"],
        )
        d = np.empty([nk])
        d_jump = bs.insert_jump(d)
        assert d_jump.shape == (nk + 2,)

        d = np.empty([nk, 5])
        d_jump = bs.insert_jump(d)
        assert d_jump.shape == (nk + 2, 5)
        assert np.isnan(d_jump).sum() == 10

        d_jump = bs.insert_jump(d.T, value=np.inf)
        assert d_jump.shape == (5, nk + 2)
        assert np.isinf(d_jump).sum() == 10

    def test_insert_jump_fail(self):
        g = geom.graphene()
        nk = 10
        bs = BandStructure(
            g,
            [[0] * 3, [0.5, 0, 0], None, [0] * 3, [0.0, 0.5, 0]],
            nk,
            ["A", "B", "C", "D"],
        )
        d = np.empty([nk + 1])
        with pytest.raises(ValueError):
            bs.insert_jump(d)
