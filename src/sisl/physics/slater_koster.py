# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from numbers import Real

import numpy as np

from sisl._internal import set_module

from .spin import Spin

__all__ = ["SlaterKoster"]


_MU = ("sigma", "pi", "delta")

# Real harmonics of sisl (Condon-Shortley phase), m = -l..l.
# p: (-y, z, -x) as unit vectors
_P = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float64)
# d: (xy, -yz, z2, -xz, x2-y2) as Frobenius-normalized symmetric traceless tensors
_s2 = 1 / np.sqrt(2)
_D = np.array(
    [
        [[0, _s2, 0], [_s2, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, -_s2], [0, -_s2, 0]],
        np.diag([-1, -1, 2]) / np.sqrt(6),
        [[0, 0, -_s2], [0, 0, 0], [-_s2, 0, 0]],
        np.diag([1, -1, 0]) * _s2,
    ]
)
del _s2


def _sk_factors(l1: int, l2: int, d: np.ndarray) -> np.ndarray:
    r"""Angular factors of the Slater-Koster block, shape ``(n, 2l1+1, 2l2+1, min(l1, l2)+1)``

    ``block = factors @ [V_sigma, V_pi, V_delta][:min(l1, l2)+1]``, for the orbital
    `l1` at the origin and `l2` at the unit vectors `d`.
    """
    if l1 > l2:
        return _sk_factors(l2, l1, d).transpose(0, 2, 1, 3)
    n = len(d)
    if l1 == 0:
        if l2 == 0:
            return np.ones((n, 1, 1, 1))
        if l2 == 1:
            return (d @ _P.T)[:, None, :, None]
        q = np.einsum("ni,bij,nj->nb", d, _D, d)
        return np.sqrt(1.5) * q[:, None, :, None]

    c = d @ _P.T
    if l2 == 1:
        sigma = c[:, :, None] * c[:, None, :]
        return np.stack([sigma, np.eye(3) - sigma], axis=-1)

    v = np.einsum("bij,nj->nbi", _D, d)
    q = np.einsum("nbi,ni->nb", v, d)
    if l1 == 1:
        w = v @ _P.T
        sigma = np.sqrt(1.5) * c[:, :, None] * q[:, None, :]
        pi = np.sqrt(2) * (w.transpose(0, 2, 1) - c[:, :, None] * q[:, None, :])
        return np.stack([sigma, pi], axis=-1)

    qq = q[:, :, None] * q[:, None, :]
    sigma = 1.5 * qq
    pi = 2 * (np.einsum("nai,nbi->nab", v, v) - qq)
    return np.stack([sigma, pi, np.eye(5) - sigma - pi], axis=-1)


_SHELL = re.compile(r"(\d+)([spdfghij])(?:Z(\d+))?(P)?")


def _shell(label: str) -> tuple[int, int, int, bool]:
    m = _SHELL.fullmatch(label)
    if m is None:
        raise ValueError(
            f"{SlaterKoster.__name__} shell label {label!r} is not <n><l>[Z<zeta>][P]"
        )
    n, l, zeta, P = m.groups()
    l = "spdfghij".index(l)
    if l > 2:
        raise NotImplementedError(
            f"{SlaterKoster.__name__} supports shells up to l=2, got {label!r}"
        )
    return int(n), l, int(zeta or 1), P is not None


def _integral(V, R: float | None) -> tuple[Callable, float]:
    """Convert one integral to ``(func(r), range)``; `func` is 0 beyond the range"""
    if isinstance(V, Real):
        if R is None:
            raise ValueError(
                f"{SlaterKoster.__name__} needs R for a constant integral ({V})"
            )
        V = float(V)
        return (lambda r: np.where(r <= R, V, 0.0)), R
    if callable(V):
        if R is None:
            raise ValueError(f"{SlaterKoster.__name__} needs R for a function integral")
        return (lambda r: np.where(r <= R, V(r), 0.0)), R
    Rs, Vs = (np.asarray(x, dtype=np.float64) for x in V)
    Vs = np.append(Vs, 0.0)
    return (lambda r: Vs[np.searchsorted(Rs, r)]), Rs[-1]


@set_module("sisl.physics")
class SlaterKoster:
    r"""Two-center Slater-Koster parameterization of a tight-binding model

    Use it as the function of `~sisl.physics.Hamiltonian.construct`:

    >>> H.construct(sk)

    Atoms are keyed by `~sisl.Atom.tag`, and shells by the label ``<n><l>[Z<zeta>][P]``
    of `~sisl.AtomicOrbital` (``Z1`` is the default, e.g. ``"2p"`` is ``"2pZ1"``).
    Shells up to :math:`l=2` are supported.

    An integral :math:`V_{ab\mu}` has shell ``a`` of atom ``A`` at the origin and shell
    ``b`` of atom ``B`` along :math:`+z`, as in :cite:`Slater1954` (thus :math:`V_{sp\sigma}>0`
    for the usual models).
    The key ``(B, A)``, ``(b, a)`` gets :math:`(-1)^{l_a+l_b}V_{ab\mu}`; do not give both.

    Parameters that are missing give no matrix element.

    Parameters
    ----------
    hopping :
        the integrals, ``hopping[(A, B)][(a, b)][mu]`` with ``mu`` one of
        ``"sigma"``, ``"pi"``, ``"delta"``.
        An integral is a number (constant up to `R`), a tuple ``(R_list, values)``
        (``values[i]`` for ``R_list[i-1] < r <= R_list[i]``), or a function ``V(r)``
        (up to `R`).
    onsite :
        on-site energies, ``onsite[A][a]``.
    overlap :
        overlap integrals, in the same form as `hopping`.
        The on-site overlap is 1. If None, the model is orthogonal.
    R :
        the range of the number and function integrals.

    Examples
    --------
    The :math:`p_z` model of graphene:

    >>> C = sisl.Atom(6, sisl.AtomicOrbital("2pzZ1", R=1.5), tag="C")
    >>> sk = sisl.SlaterKoster({("C", "C"): {("2p", "2p"): {"pi": -2.7}}}, R=1.5)
    >>> H = sisl.Hamiltonian(sisl.geom.graphene(atoms=C))
    >>> H.construct(sk)

    Notes
    -----
    Only pairs inside the supercell of `H` (`~sisl.Geometry.nsc`) get matrix elements.

    The first call creates the neighbor list and all matrix elements of `H`.
    Each call writes the rows of the atom `ia`. The other rows are kept until all atoms
    are written.
    """

    def __init__(
        self,
        hopping: Mapping,
        onsite: Mapping | None = None,
        overlap: Mapping | None = None,
        R: float | None = None,
    ):
        self._onsite = {
            A: {_shell(a): E for a, E in shells.items()}
            for A, shells in (onsite or {}).items()
        }
        self._R = 0.0
        self._hopping = self._parse(hopping, R)
        self._overlap = None if overlap is None else self._parse(overlap, R)
        self._tags = set(self._onsite).union(*self._hopping, *(self._overlap or {}))
        self._cache = None

    def _parse(self, integrals: Mapping, R: float | None) -> dict:
        """Expand both key directions: ``out[(A, B)][(a, b)] = [(imu, func), ...]``"""
        out = {}
        for (A, B), shells in integrals.items():
            for (a, b), Vmu in shells.items():
                sa, sb = _shell(a), _shell(b)
                funcs = []
                for mu, V in Vmu.items():
                    imu = _MU.index(mu)
                    if imu > min(sa[1], sb[1]):
                        raise ValueError(
                            f"{self.__class__.__name__} has no {mu} integral for ({a}, {b})"
                        )
                    f, r = _integral(V, R)
                    self._R = max(self._R, r)
                    funcs.append((imu, f))
                entries = {((A, B), (sa, sb)): funcs}
                if (sa[1] + sb[1]) % 2:
                    funcs = [(imu, lambda r, f=f: -f(r)) for imu, f in funcs]
                entries.setdefault(((B, A), (sb, sa)), funcs)
                for key, s in entries:
                    if s in out.setdefault(key, {}):
                        raise ValueError(
                            f"{self.__class__.__name__} got the integral ({A}, {B}), "
                            f"({a}, {b}) in both directions"
                        )
                    out[key][s] = entries[key, s]
        return out

    @property
    def R(self) -> float:
        """Largest range of the integrals"""
        return self._R

    @property
    def orthogonal(self) -> bool:
        """Whether the model has no overlap integrals"""
        return self._overlap is None

    def __call__(self, H, ia: int, atoms=None, atoms_xyz=None):
        """Set the rows of atom `ia` in `H` (signature of `construct`)"""
        if self._cache is None or self._cache[0] is not H:
            self._cache = (H, *self._elements(H), set())
        _, ptr, col, D, done = self._cache
        geometry = H.geometry
        for io in range(geometry.firsto[ia], geometry.lasto[ia] + 1):
            s = slice(ptr[io], ptr[io + 1])
            if s.start < s.stop:
                H[io, col[s]] = D[s]
        done.add(int(ia))
        if len(done) == geometry.na:
            self._cache = None

    def _elements(self, H) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All matrix elements of `H` as a CSR (pointer, column, data)"""
        from sisl.geom import NeighborFinder

        if H.spin.kind == Spin.NAMBU:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support Nambu spin"
            )
        if H.orthogonal and not self.orthogonal:
            raise ValueError(
                f"{self.__class__.__name__} has overlap integrals, but H is orthogonal"
            )
        geometry = H.geometry
        atoms = geometry.atoms
        tags = [atom.tag for atom in atoms.atom]
        species = atoms.species
        firsto = geometry.firsto

        # orbitals of each shell: (n, l, zeta, P) -> (orbital offsets, m + l)
        shells = []
        for atom in atoms.atom:
            sh = {}
            if atom.tag in self._tags:
                for io, orb in enumerate(atom):
                    o, m = sh.setdefault((orb.n, orb.l, orb.zeta, orb.P), ([], []))
                    o.append(io)
                    m.append(orb.m + orb.l)
            shells.append({k: (np.array(o), np.array(m)) for k, (o, m) in sh.items()})

        rows, cols = [np.empty(0, np.int64)], [np.empty(0, np.int64)]
        Hv, Sv = [np.empty(0)], [np.empty(0)]

        for isp, sh in enumerate(shells):
            onsite = self._onsite.get(tags[isp], {})
            ia = (species == isp).nonzero()[0]
            for key, (oi, _) in sh.items():
                E = onsite.get(key)
                if E is None and H.orthogonal:
                    continue
                io = (firsto[ia, None] + oi).ravel()
                rows.append(io)
                cols.append(io)
                Hv.append(np.full(io.shape, 0.0 if E is None else E))
                Sv.append(np.ones(io.shape))

        # NeighborFinder needs the atoms in the unit cell
        shift = np.floor(geometry.fxyz)
        pbc = geometry.lattice.pbc
        shift[:, ~pbc] = shift[:, ~pbc].min(axis=0)
        wrapped = geometry.copy()
        wrapped.xyz = geometry.xyz - shift @ geometry.cell
        neighs = NeighborFinder(wrapped, R=self.R).find_neighbors()
        I, J = neighs.I, neighs.J
        isc = neighs.isc + (shift[I] - shift[J]).astype(np.int64)
        keep = (np.abs(isc) <= geometry.nsc // 2).all(axis=1)
        I, J, isc = I[keep], J[keep], isc[keep]
        d = geometry.xyz[J] + isc @ geometry.cell - geometry.xyz[I]
        r = np.linalg.norm(d, axis=1)
        d /= r[:, None]
        Ioff = firsto[I]
        Joff = firsto[J] + geometry.lattice.sc_index(isc) * geometry.no
        pair = species[I] * atoms.nspecies + species[J]
        for ip in np.unique(pair):
            isp, jsp = divmod(ip, atoms.nspecies)
            key = (tags[isp], tags[jsp])
            hop = self._hopping.get(key, {})
            ovl = {} if self._overlap is None else self._overlap.get(key, {})
            idx = (pair == ip).nonzero()[0]
            for sa, sb in hop.keys() | ovl.keys():
                if sa not in shells[isp] or sb not in shells[jsp]:
                    continue
                (oi, mi), (oj, mj) = shells[isp][sa], shells[jsp][sb]
                G = _sk_factors(sa[1], sb[1], d[idx])[:, mi][:, :, mj]
                h = np.zeros(G.shape[:3])
                s = np.zeros(G.shape[:3])
                for out, ints in ((h, hop), (s, ovl)):
                    for imu, f in ints.get((sa, sb), ()):
                        out += G[..., imu] * f(r[idx])[:, None, None]
                nz = (h != 0) | (s != 0)
                rows.append(
                    np.broadcast_to((Ioff[idx, None] + oi)[:, :, None], G.shape[:3])[nz]
                )
                cols.append(
                    np.broadcast_to((Joff[idx, None] + oj)[:, None, :], G.shape[:3])[nz]
                )
                Hv.append(h[nz])
                Sv.append(s[nz])

        row = np.concatenate(rows)
        order = np.argsort(row, kind="stable")
        D = np.zeros((len(row), H.dim), dtype=H.dtype)
        D[:, : 1 if H.spin.kind == Spin.UNPOLARIZED else 2] = np.concatenate(Hv)[
            :, None
        ]
        if not H.orthogonal:
            D[:, H.S_idx] = np.concatenate(Sv)
        ptr = np.searchsorted(row[order], np.arange(geometry.no + 1))
        return ptr, np.concatenate(cols)[order], D[order]
