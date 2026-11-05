# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import re
from collections.abc import Callable, Mapping

import numpy as np

from sisl._core.geometry import Geometry
from sisl._core.lattice import Lattice
from sisl._core.orbital import AtomicOrbital
from sisl._internal import set_module
from sisl.messages import warn

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


def _atom_shells(atom) -> dict:
    """Orbitals of each shell ``(n, l, zeta, P)`` of `atom`: (orbital offsets, m + l)"""
    sh = {}
    for io, orb in enumerate(atom):
        if not isinstance(orb, AtomicOrbital):
            raise TypeError(
                f"{SlaterKoster.__name__} needs AtomicOrbital orbitals, "
                f"the atom with tag {atom.tag!r} has {orb.__class__.__name__}"
            )
        o, m = sh.setdefault((orb.n, orb.l, orb.zeta, orb.P), ([], []))
        o.append(io)
        m.append(orb.m + orb.l)
    return {k: (np.array(o), np.array(m)) for k, (o, m) in sh.items()}


def _label(shell: tuple[int, int, int, bool]) -> str:
    n, l, zeta, P = shell
    return f"{n}{'spd'[l]}Z{zeta}{'P' if P else ''}"


_SPIN_SHAPES = ((), (2,), (2, 2))


def _spin_check(V: np.ndarray, shape: tuple) -> np.ndarray:
    if V.shape[len(shape) :] not in _SPIN_SHAPES:
        raise ValueError(
            f"{SlaterKoster.__name__} values are a scalar, [up, down] or a 2x2 spin "
            f"matrix, got the shape {V.shape[len(shape):]}"
        )
    return V


def _integral(V, R: float | None) -> tuple[Callable, float]:
    """Convert one integral to ``(func(r), range)``; `func` is 0 beyond the range"""
    if isinstance(V, tuple):
        Rs = np.asarray(V[0], dtype=np.float64)
        Vs = [V[1]] if Rs.ndim == 0 else list(V[1])
        Rs = np.atleast_1d(Rs)
        if Rs.ndim != 1 or len(Rs) != len(Vs):
            raise ValueError(
                f"{SlaterKoster.__name__} needs (R_list, values) of equal length, got {V}"
            )
        # each range has its own spin form; expand all to the largest
        Vs = [_spin_check(np.asarray(v), ()) for v in Vs]
        ndim = max(v.ndim for v in Vs)
        Vs = np.concatenate(
            [_spin_value(v[None], ndim) for v in Vs] + [np.zeros((1,) + (2,) * ndim)]
        )
        return (lambda r: Vs[np.searchsorted(Rs, r)]), Rs[-1]
    if R is None:
        raise ValueError(f"{SlaterKoster.__name__} needs R for {V}")
    if callable(V):
        func = V
    else:
        V = _spin_check(np.asarray(V), ())
        func = lambda r: np.broadcast_to(V, r.shape + V.shape)

    def f(r):
        v = np.asarray(func(r))
        return np.where((r <= R).reshape(r.shape + (1,) * (v.ndim - 1)), v, 0)

    return f, R


def _dagger(v: np.ndarray) -> np.ndarray:
    """Hermitian conjugate of spin values ``(n, ...)``"""
    v = np.conj(v)
    return v.swapaxes(1, 2) if v.ndim == 3 else v


def _spin_value(v: np.ndarray, ndim: int) -> np.ndarray:
    """Spin values ``(n, ...)`` as ``(n,)``, ``(n, 2)`` or ``(n, 2, 2)`` for `ndim` 0, 1, 2"""
    v = _spin_check(np.asarray(v), v.shape[:1])
    if v.ndim - 1 > ndim:
        raise ValueError(
            f"{SlaterKoster.__name__} got spin values of shape {v.shape[1:]} "
            "for a matrix with fewer spin components"
        )
    if v.ndim == 1 and ndim > 0:
        v = np.stack([v, v], axis=1)
    if v.ndim == 2 and ndim == 2:
        v = v[:, :, None] * np.eye(2)
    return v


def _pack(v: np.ndarray, spin: Spin, dtype) -> np.ndarray:
    """Spin values ``(n, ...)`` as the ``(n, K)`` components that sisl stores"""
    cmplx = np.iscomplexobj(np.empty(0, dtype))
    if spin.kind == Spin.UNPOLARIZED:
        out = v[:, None]
    elif spin.kind == Spin.POLARIZED:
        out = v
    else:
        uu, ud, du, dd = v[:, 0, 0], v[:, 0, 1], v[:, 1, 0], v[:, 1, 1]
        if spin.kind == Spin.NONCOLINEAR:
            if np.any(du != np.conj(ud)):
                raise ValueError(
                    f"{SlaterKoster.__name__}: a non-colinear matrix stores du = conj(ud), "
                    "use spin-orbit for other spin matrices"
                )
            out = [uu, dd, ud] if cmplx else [uu, dd, ud.real, ud.imag]
        elif cmplx:
            out = [uu, dd, ud, du]
        else:
            out = [
                uu.real,
                dd.real,
                ud.real,
                ud.imag,
                uu.imag,
                dd.imag,
                du.real,
                du.imag,
            ]
        out = np.stack(out, axis=1)
    if not cmplx and np.iscomplexobj(out):
        if np.any(out.imag != 0):
            raise ValueError(
                f"{SlaterKoster.__name__} got complex values for a real matrix, "
                "use a complex dtype"
            )
        out = out.real
    return out


def _neighbor_pairs(geometry: Geometry, R: float):
    """Atom pairs ``(I, J, isc)`` within `R`, inside the supercell of `geometry`"""
    from sisl.geom import NeighborFinder

    # NeighborFinder needs all atoms in the unit cell: move them into the cell along
    # the periodic axes, and make the other lattice vectors long enough for all atoms
    pbc = geometry.lattice.pbc
    fxyz = geometry.fxyz
    shift = np.floor(fxyz)
    lo = fxyz[:, ~pbc].min(axis=0)
    shift[:, ~pbc] = lo
    cell = geometry.cell.copy()
    cell[~pbc] *= (np.floor(fxyz[:, ~pbc].max(axis=0) - lo) + 1)[:, None]
    lattice = Lattice(
        cell, nsc=geometry.nsc, boundary_condition=geometry.lattice.boundary_condition
    )
    wrapped = Geometry(
        geometry.xyz - shift @ geometry.cell, geometry.atoms, lattice=lattice
    )
    neighs = NeighborFinder(wrapped, R=R).find_neighbors()
    I, J = neighs.I, neighs.J
    isc = neighs.isc + np.rint(shift[I] - shift[J]).astype(np.int64)
    keep = (np.abs(isc) <= geometry.nsc // 2).all(axis=1)
    return I[keep], J[keep], isc[keep]


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

    Parameters that are missing give no matrix element, and a warning.
    Give an integral as 0 to set no matrix element without a warning.

    Parameters
    ----------
    hopping :
        the integrals, ``hopping[(A, B)][(a, b)][mu]`` with ``mu`` one of
        ``"sigma"``, ``"pi"``, ``"delta"``.
        An integral is a value (constant up to `R`), a tuple ``(R_list, values)``
        (``values[i]`` for ``R_list[i-1] < r <= R_list[i]``, each a value;
        ``(R, value)`` for one range),
        or a function ``V(r)``
        (up to `R`, it returns a value for each ``r``).
        A value is a scalar, ``[up, down]``, or a 2x2 spin matrix
        ``[[uu, ud], [du, dd]]``, for the spin of `H` (a scalar is the same for both spins).
        The key ``(B, A)``, ``(b, a)`` gets the Hermitian conjugate of a spin matrix.
    onsite :
        on-site energies, ``onsite[A][a]``, a value as in `hopping` (Hermitian).
    overlap :
        overlap integrals, in the same form as `hopping`, with scalar values.
        The on-site overlap is 1. If None, the model is orthogonal.
    R :
        the range of the constant and function integrals.

    Examples
    --------
    The :math:`p_z` model of graphene:

    >>> C = sisl.Atom(6, sisl.AtomicOrbital("2pzZ1", R=1.5), tag="C")
    >>> sk = sisl.SlaterKoster(
    ...     {("C", "C"): {("2p", "2p"): {"pi": (1.5, -2.7)}}}, onsite={"C": {"2p": 0}}
    ... )
    >>> H = sisl.Hamiltonian(sisl.geom.graphene(atoms=C))
    >>> H.construct(sk)

    Notes
    -----
    Only pairs inside the supercell of `H` (`~sisl.Geometry.nsc`) get matrix elements.
    Create `H` with an ``nnzpr`` that holds all elements of a row: a row that grows is slow.

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
            A: {
                _shell(a): self._hermitian(np.asarray(E)[None])[0]
                for a, E in shells.items()
            }
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
                    if mu not in _MU:
                        raise ValueError(
                            f"{self.__class__.__name__} integrals are {_MU}, got {mu!r}"
                        )
                    imu = _MU.index(mu)
                    if imu > min(sa[1], sb[1]):
                        raise ValueError(
                            f"{self.__class__.__name__} has no {mu} integral for ({a}, {b})"
                        )
                    f, r = _integral(V, R)
                    self._R = max(self._R, r)
                    funcs.append((imu, f))
                if ((A, B), sa) == ((B, A), sb):
                    # the pair is its own reverse
                    funcs = [
                        (imu, lambda r, f=f: self._hermitian(f(r))) for imu, f in funcs
                    ]
                entries = {((A, B), (sa, sb)): funcs}
                sign = (-1) ** (sa[1] + sb[1])
                funcs = [
                    (imu, lambda r, f=f, sign=sign: sign * _dagger(f(r)))
                    for imu, f in funcs
                ]
                entries.setdefault(((B, A), (sb, sa)), funcs)
                for key, s in entries:
                    if s in out.setdefault(key, {}):
                        raise ValueError(
                            f"{self.__class__.__name__} got the integral ({A}, {B}), "
                            f"({a}, {b}) in both directions"
                        )
                    out[key][s] = entries[key, s]
        return out

    def _hermitian(self, v: np.ndarray) -> np.ndarray:
        v = _spin_check(np.asarray(v), np.shape(v)[:1])
        if np.any(v != _dagger(v)):
            raise ValueError(
                f"{self.__class__.__name__} needs Hermitian on-site values, and Hermitian "
                "integrals for a shell with itself"
            )
        return v

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

        shells = [
            _atom_shells(atom) if atom.tag in self._tags else {} for atom in atoms.atom
        ]

        # spin values are (n,), (n, 2) or (n, 2, 2)
        ndim = {Spin.UNPOLARIZED: 0, Spin.POLARIZED: 1}.get(H.spin.kind, 2)
        rows, cols = [np.empty(0, np.int64)], [np.empty(0, np.int64)]
        Hv, Sv = [np.zeros((0,) + (2,) * ndim)], [np.empty(0)]

        name = self.__class__.__name__
        missing = [atom.tag for atom in atoms.atom if atom.tag not in self._tags]
        if missing:
            warn(
                f"{name} has no parameters for the atoms with tags {missing}"
                + ("" if H.orthogonal else "; their on-site overlap is 1")
            )
        missing = [
            f"{tags[isp]}: {_label(key)}"
            for isp, sh in enumerate(shells)
            for key in sh
            if key not in self._onsite.get(tags[isp], {})
        ]
        if missing:
            warn(f"{name} has no on-site energy for {'; '.join(missing)}")
        for isp, sh in enumerate(shells):
            onsite = self._onsite.get(tags[isp], {})
            ia = (species == isp).nonzero()[0]
            if tags[isp] not in self._tags and not H.orthogonal:
                sh = {None: (np.arange(atoms.atom[isp].no), None)}
            for key, (oi, _) in sh.items():
                E = onsite.get(key)
                if E is None and H.orthogonal:
                    continue
                io = (firsto[ia, None] + oi).ravel()
                rows.append(io)
                cols.append(io)
                E = np.zeros(()) if E is None else E
                Hv.append(_spin_value(np.broadcast_to(E, io.shape + E.shape), ndim))
                Sv.append(np.ones(io.shape))

        I, J, isc = _neighbor_pairs(geometry, self.R)
        d = geometry.xyz[J] + isc @ geometry.cell - geometry.xyz[I]
        r = np.linalg.norm(d, axis=1)
        d /= r[:, None]
        Ioff = firsto[I]
        Joff = firsto[J] + geometry.lattice.sc_index(isc) * geometry.no
        pair = species[I] * atoms.nspecies + species[J]
        missing = {"hopping": set(), "overlap": set()}
        for ip in np.unique(pair):
            isp, jsp = divmod(ip, atoms.nspecies)
            key = (tags[isp], tags[jsp])
            hop = self._hopping.get(key, {})
            ovl = {} if self._overlap is None else self._overlap.get(key, {})
            for what, ints in (("hopping", hop), ("overlap", ovl)):
                if what == "overlap" and self._overlap is None:
                    continue
                missing[what].update(
                    min((key, sa, sb), (key[::-1], sb, sa))
                    for sa in shells[isp]
                    for sb in shells[jsp]
                    if (sa, sb) not in ints
                )
            idx = (pair == ip).nonzero()[0]
            for sa, sb in hop.keys() | ovl.keys():
                if sa not in shells[isp] or sb not in shells[jsp]:
                    continue
                (oi, mi), (oj, mj) = shells[isp][sa], shells[jsp][sb]
                G = _sk_factors(sa[1], sb[1], d[idx])[:, mi][:, :, mj]
                shape = G.shape[:3]
                h = s = 0
                for imu, f in hop.get((sa, sb), ()):
                    v = _spin_value(f(r[idx]), ndim)[:, None, None]
                    h = h + G[..., imu].reshape(shape + (1,) * ndim) * v
                for imu, f in ovl.get((sa, sb), ()):
                    s = s + G[..., imu] * _spin_value(f(r[idx]), 0)[:, None, None]
                h = np.broadcast_to(h, shape + (2,) * ndim)
                s = np.broadcast_to(s, shape)
                nz = (h != 0).reshape(shape + (-1,)).any(-1) | (s != 0)
                rows.append(
                    np.broadcast_to((Ioff[idx, None] + oi)[:, :, None], G.shape[:3])[nz]
                )
                cols.append(
                    np.broadcast_to((Joff[idx, None] + oj)[:, None, :], G.shape[:3])[nz]
                )
                Hv.append(h[nz])
                Sv.append(s[nz])

        for what, m in missing.items():
            if m:
                m = "; ".join(
                    f"({A}, {B}): ({_label(a)}, {_label(b)})"
                    for (A, B), a, b in sorted(m)
                )
                warn(f"{name} has no {what} integrals for {m}")

        row = np.concatenate(rows)
        order = np.argsort(row, kind="stable")
        D = np.zeros((len(row), H.dim), dtype=H.dtype)
        Hv = _pack(np.concatenate(Hv), H.spin, H.dtype)
        D[:, : Hv.shape[1]] = Hv
        if not H.orthogonal:
            D[:, H.S_idx] = np.concatenate(Sv)
        ptr = np.searchsorted(row[order], np.arange(geometry.no + 1))
        return ptr, np.concatenate(cols)[order], D[order]
