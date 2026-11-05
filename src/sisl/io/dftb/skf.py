# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from sisl._internal import set_module
from sisl.messages import warn
from sisl.physics import SlaterKoster
from sisl.physics.slater_koster import _MU, _shell
from sisl.unit import units

from ..sile import add_sile, sile_fh_open
from .sile import SileDFTB

__all__ = ["skfSileDFTB"]

_Bohr = units("Bohr", "Ang")
_Ha = units("Ha", "eV")
# columns of (l of the first atom, l of the second atom) in each half of a table row
_COLS = {
    (2, 2): (0, 1, 2),
    (1, 2): (3, 4),
    (1, 1): (5, 6),
    (0, 2): (7,),
    (0, 1): (8,),
    (0, 0): (9,),
}
# interpolation of DFTB+: 8 grid points, and a tail of 1 Bohr
_NPOL = 8
_TAIL = 1.0
_NODES = np.arange(_NPOL)
_DENOM = np.array([np.prod(j - np.delete(_NODES, j)) for j in _NODES])
# value, 1st and 2nd derivative of each Lagrange polynomial at the last node
_END = np.array(
    [
        [
            np.polynomial.Polynomial.fromroots(np.delete(_NODES, j)).deriv(d)(_NPOL - 1)
            / _DENOM[j]
            for j in _NODES
        ]
        for d in range(3)
    ]
)


def _floats(line: str, n: int) -> list[float]:
    """The first `n` values of `line`; DFTB+ ignores the other fields"""
    out = []
    for t in line.replace(",", " ").split():
        if len(out) >= n:
            break
        k, _, v = t.rpartition("*")
        out.extend([float(v)] * int(k or 1))
    return out[:n]


def _interpolate(table: np.ndarray, dist: float):
    """Interpolation of DFTB+ of `table` (first row at `dist` Bohr) as a function of r (Ang)"""
    n = len(table)
    rmax = n * dist + _TAIL
    y, dy, ddy = _END @ table[-_NPOL:]
    dx1 = -dy * _TAIL / dist
    dx2 = ddy * (_TAIL / dist) ** 2
    c3 = 10 * y - 4 * dx1 + 0.5 * dx2
    c4 = -15 * y + 7 * dx1 - dx2
    c5 = 6 * y - 3 * dx1 + 0.5 * dx2

    def f(r):
        r = np.asarray(r) / _Bohr
        x = r / dist
        last = np.clip(np.floor(x).astype(np.int64) + _NPOL // 2, _NPOL, n)
        t = (x - last + _NPOL - 1)[..., None] - _NODES
        w = np.stack(
            [np.prod(np.delete(t, j, axis=-1), axis=-1) for j in _NODES], axis=-1
        )
        V = np.einsum(
            "...j,...jc->...c", w / _DENOM, table[last[..., None] - _NPOL + _NODES]
        )
        xt = ((rmax - r) / _TAIL)[..., None]
        V = np.where((x >= n)[..., None], ((c5 * xt + c4) * xt + c3) * xt**3, V)
        return np.where((r >= rmax)[..., None], 0, V)

    return f, rmax * _Bohr


@set_module("sisl.io.dftb")
class skfSileDFTB(SileDFTB):
    """DFTB Slater-Koster file of the elements ``A`` and ``B``, e.g. ``A-B.skf``"""

    @sile_fh_open(True)
    def read_documentation(self) -> str:
        """The XML section ``<Documentation>`` of the file (author, elements, basis, ...)

        An empty string if the file has no documentation.
        """
        text = self.fh.read()
        i = text.find("<Documentation>")
        j = text.find("</Documentation>", i)
        return "" if i < 0 or j < 0 else text[i : j + len("</Documentation>")]

    def _r_pair(self) -> tuple[str, str, str]:
        """Tags ``A``, ``B`` in the order of the file name, and the file name of ``B-A``

        The documentation gives the elements, the file name gives their order.
        Without them, the file name must be ``A-B.skf``.
        """
        file, stem = Path(self.file), Path(self.file).stem
        doc = self.read_documentation()
        general = ET.fromstring(doc).find("General") if doc else None
        elements = []
        if general is not None:
            elements = [
                general.findtext(t, "").strip() for t in ("Element1", "Element2")
            ]
            elements = [e for e in elements if e]
        if elements:
            for A, B in ((elements[0], elements[-1]), (elements[-1], elements[0])):
                j = stem.find(B, len(A))
                if stem.startswith(A) and j >= 0:
                    other = f"{B}{stem[len(A) : j]}{A}{stem[j + len(B) :]}"
                    return A, B, other + file.suffix
            warn(
                f"{self.__class__.__name__} found the elements {elements} of the "
                f"documentation not in the name {file.name}"
            )
        tags = stem.split("-")
        if len(tags) != 2:
            raise ValueError(
                f"{self.__class__.__name__} needs <Element1> in the documentation, "
                f"or a file name A-B.skf, got {file.name}"
            )
        A, B = tags
        return A, B, f"{B}-{A}{file.suffix}"

    @sile_fh_open(True)
    def _r_table(self, homo: bool):
        """Grid distance, ``[H, S]`` table (Bohr, Ha) and on-site energies ``[s, p, d]``"""
        line = self.readline()
        if line.startswith("@"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not read the extended format (f shells) of {self.file}"
            )
        dist, n = _floats(line, 2)
        n = int(n) - 1
        if n <= _NPOL:
            raise ValueError(
                f"{self.__class__.__name__} needs more than {_NPOL} rows for the interpolation in {self.file}"
            )
        onsite = _floats(self.readline(), 3)[::-1] if homo else None
        # mass and polynomial repulsive
        self.readline()
        table = []
        for _ in range(n):
            row = _floats(self.readline(), 20)
            if len(row) < 20:
                raise ValueError(
                    f"{self.__class__.__name__} needs {n} rows of 20 integrals in {self.file}"
                )
            table.append(row)
        return dist, np.array(table), onsite

    def read_slater_koster(self, shells: Mapping[str, Sequence[str]]) -> SlaterKoster:
        """Slater-Koster model of the species pair of this file, and of ``B-A.skf`` next to it

        The elements come from the documentation of the file (`read_documentation`),
        and their order from the file name. The name of ``B-A.skf`` swaps only the
        two elements, e.g. ``N-H_short.skf`` and ``H-N_short.skf``.
        A file holds the integrals with the lower :math:`l` on its first atom, thus a pair
        of two species needs both files. Integrals of equal :math:`l` must agree in both files.
        A homonuclear file also gives the on-site energies.
        The model has the overlap integrals, thus it is non-orthogonal.
        The interpolation is that of DFTB+: a polynomial through 8 grid points,
        and a polynomial of 5th order to 0 in 1 Bohr after the last grid point.

        Use `~sisl.SlaterKoster.merge` to join the models of all species pairs.

        Parameters
        ----------
        shells :
            the shell labels (``<n><l>[Z<zeta>][P]``) of each tag, one for each :math:`l`,
            e.g. ``{"O": ["2s", "2p"], "H": ["1s"]}``. Only these shells get integrals.

        Examples
        --------
        >>> shells = {"O": ["2s", "2p"], "H": ["1s"]}
        >>> sk = sisl.SlaterKoster.merge(
        ...     *(sisl.get_sile(f).read_slater_koster(shells) for f in ["O-O.skf", "O-H.skf", "H-H.skf"])
        ... )
        """
        file = Path(self.file)
        A, B, other = self._r_pair()
        other = file.with_name(other)
        ls = {}
        for X in (A, B):
            if X not in shells:
                raise ValueError(
                    f"{self.__class__.__name__} needs the shells of {X} for {file.name}"
                )
            ls[X] = {}
            for label in shells[X]:
                l = _shell(label)[1]
                if l in ls[X]:
                    raise ValueError(
                        f"{self.__class__.__name__} needs one shell for each l of {X}, got {shells[X]}"
                    )
                ls[X][l] = label

        files = {(A, B): self}
        if A != B:
            if other.is_file():
                files[(B, A)] = self.__class__(other)
            else:
                warn(
                    f"{self.__class__.__name__} found no {other.name}: the model has no "
                    f"integrals with the lower l on {B}"
                )
        tables = {XY: sile._r_table(A == B) for XY, sile in files.items()}

        dist, table, onsite = tables[A, B]
        if (B, A) in tables:
            dist2, table2, _ = tables[B, A]
            if dist2 != dist or table2.shape != table.shape:
                raise ValueError(
                    f"{self.__class__.__name__} needs the same grid in {file.name} and {other.name}"
                )
            cols = [
                i + h
                for (la, lb), c in _COLS.items()
                if la == lb and la in ls[A] and la in ls[B]
                for i in c
                for h in (0, 10)
            ]
            if not np.allclose(table[:, cols], table2[:, cols], rtol=1e-10, atol=0):
                raise ValueError(
                    f"{self.__class__.__name__} found different integrals of equal l "
                    f"in {file.name} and {other.name}"
                )

        hopping, overlap = {}, {}
        for (X, Y), (dist, table, _) in tables.items():
            fH, R = _interpolate(table[:, :10] * _Ha, dist)
            fS, _ = _interpolate(table[:, 10:], dist)
            for (la, lb), c in _COLS.items():
                if la not in ls[X] or lb not in ls[Y] or (la == lb and X != A):
                    continue
                key = (ls[X][la], ls[Y][lb])
                for out, f in ((hopping, fH), (overlap, fS)):
                    out.setdefault((X, Y), {})[key] = {
                        mu: lambda r, f=f, i=i: f(r)[..., i] for mu, i in zip(_MU, c)
                    }
        if onsite is not None:
            onsite = {A: {label: onsite[l] * _Ha for l, label in ls[A].items()}}
        return SlaterKoster(hopping, onsite=onsite, overlap=overlap, R=R)


add_sile("skf", skfSileDFTB)
