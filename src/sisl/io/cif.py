# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""CIF file reader for sisl."""
from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np

from sisl import Atom, Atoms, Geometry, Lattice
from sisl._internal import set_module
from sisl.io import add_sile
from sisl.io.sile import Sile, SileError, sile_fh_open

__all__ = ["cifSile"]


# CIF tags that carry Jones-faithful symmetry operation strings, in priority order
_SYMOP_TAGS = (
    "_space_group_symop_operation_xyz",  # CIF 2 / newer dictionaries
    "_symmetry_equiv_pos_as_xyz",  # CIF 1.1 legacy
)


def _parse_jones(op: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Jones-faithful symmetry string into ``(W, w)``.

    Returns
    -------
    W : ndarray, shape (3, 3)
        Rotation / improper-rotation matrix (integer entries).
    w : ndarray, shape (3,)
        Translation vector (rational, in fractional coordinates).

    Examples
    --------
    >>> W, w = _parse_jones("-x+1/2, y, -z+3/4")
    """
    W = np.zeros((3, 3), dtype=np.float64)
    w = np.zeros(3, dtype=np.float64)

    # Tokenise: signed axis tokens  ±[xyz]  and rational offsets  ±N/M or ±N
    _TOKEN = re.compile(r"([+-]?)(\d+/\d+|\d+\.?\d*|[xyz])")

    for row, expr in enumerate(op.lower().split(",")):
        expr = expr.strip()
        pending_sign = 1
        for m in _TOKEN.finditer(expr):
            sign_str, val = m.group(1), m.group(2)
            sign = {"-": -1}.get(sign_str, 1)
            # A bare '+' or '-' before the first token has no leading digit
            if val in ("x", "y", "z"):
                col = ord(val) - ord("x")
                W[row, col] = sign * pending_sign if sign_str == "" else sign
                pending_sign = 1
            else:
                # numeric — could be int, float, or fraction
                # fraction parses real numbers and str, e.g. '1/2'
                w[row] += sign * float(Fraction(val))
            pending_sign = sign if sign_str == "" else 1

    return W, w


def _symops_from_loop(loop: dict) -> Optional[list[tuple[np.ndarray, np.ndarray]]]:
    """Extract ``[(W, w), …]`` from a parsed CIF loop dict, or *None*."""
    for tag in _SYMOP_TAGS:
        if tag in loop:
            return [_parse_jones(s) for s in loop[tag]]
    return None


def _apply_symops(
    frac: np.ndarray,
    species: list[str],
    symmetry_ops: list[tuple[np.ndarray, np.ndarray]],
    symmetry_atol: float = 1e-4,
) -> tuple[np.ndarray, list]:
    """Expand asymmetric-unit sites by all symmetry operations.

    Coordinates are mapped back into [0, 1) and duplicate sites
    (within `symmetry_atol`) are discarded.

    The unfolding will be done *per atom*, starting with the first atom.

    Parameters
    ----------
    frac : ndarray, shape (N, 3)
        Fractional coordinates of asymmetric-unit atoms.
    species : list of str
        Element symbols, length N.
    symops : list of (W, w) pairs
    symmetry_atol: float
        The absolute tolerance of the fraction accuracy.

    Returns
    -------
    frac_full : ndarray, shape (M, 3)
    species_full : list of str, length M
    """
    full_frac: list = []
    full_species: list = []

    for abc, sym in zip(frac, species):
        for W, w in symmetry_ops:
            new = (W @ abc + w) % 1.0
            # deduplicate: skip if already present within tolerance
            if any(
                np.isclose(new, f, atol=symmetry_atol, rtol=0).all() for f in full_frac
            ):
                continue
            full_frac.append(new)
            full_species.append(sym)

    return np.array(full_frac), full_species


def _jones_str(
    W: np.ndarray,
    w: np.ndarray,
    symmetry_atol: float = 1e-4,
) -> str:
    """Serialize ``(W, w)`` back to a Jones-faithful CIF string.

    Example output: ``'-x+1/2,y,-z+3/4'``

    Parameters
    ----------
    symmetry_atol: float
        The absolute tolerance of the fraction accuracy.
    """
    axes = ("x", "y", "z")
    parts = []
    for row in range(3):
        expr = ""
        for col in range(3):
            v = int(round(W[row, col]))
            if v == 0:
                continue
            sign = "+" if v > 0 and expr else ("-" if v < 0 else "")
            expr += f"{sign}{axes[col]}"

        # translation — express as exact fraction
        t = w[row] % 1.0
        if t > symmetry_atol:
            # The precision is limited to integer fractions of 24
            frac = Fraction(t).limit_denominator(24)
            expr += f"+{frac.numerator}/{frac.denominator}"
        parts.append(expr if expr else "0")
    return ",".join(parts)


@set_module("sisl.io")
class cifSile(Sile):
    """CIF (Crystallographic Information File) reader.

    Supports CIF 1.1 and a common subset of CIF 2.0.
    Currently implements :meth:`read_geometry` (and the implied
    :meth:`read_lattice`).
    Basic crystallographic symmetry operations (``_symmetry_equiv_pos_as_xyz``
    / ``_space_group_symop_operation_xyz``) are obeyed.

    When reading, symmetry operations are parsed and applied to the
    asymmetric-unit sites to produce the full unit-cell geometry.
    When writing, an optional list of ``(W, w)`` pairs may be supplied
    via the ``symmetry_ops`` keyword; if omitted the identity operation (P 1)
    is written.

    Examples
    --------
    >>> geom = cifSile("crystal.cif").read_geometry()

    Write with explicit symmetry operations:

    >>> symops = [("x,y,z", "-x,-y,-z")]
    >>> cifSile("out.cif", "w").write_geometry(geom, symmetry_ops=symops)
    """

    @staticmethod
    def _parse_value(raw: str) -> str:
        """Strip CIF value decorators (quotes, uncertainty parentheses)."""
        v = raw.strip().strip("'\"")
        # remove standard uncertainty, e.g. 3.456(7) -> 3.456
        return re.sub(r"\(\d+\)$", "", v)

    @classmethod
    def _float(cls, raw: str) -> float:
        return float(cls._parse_value(raw))

    # ------------------------------------------------------------------ #
    #  Low-level CIF tokeniser                                            #
    # ------------------------------------------------------------------ #

    @sile_fh_open(from_closed=True)
    def _iter_blocks(self):
        """Yield (block_name, data_dict, loop_list) for every data block.

        *data_dict* maps ``_tag -> value`` for scalar entries.
        *loop_list* is a list of dicts ``{col_name: [values …]}``.
        """
        block_name: Optional[str] = None
        data: dict = {}
        loops: list = []

        in_loop = False
        loop_cols: list = []
        loop_rows: list = []

        multiline_buf: list = []
        multiline_tag: Optional[str] = None

        def _flush_loop():
            if loop_cols:
                n_cols = len(loop_cols)
                col_data = {c: [] for c in loop_cols}
                for i, v in enumerate(loop_rows):
                    col_data[loop_cols[i % n_cols]].append(v)
                loops.append(col_data)
            loop_cols.clear()
            loop_rows.clear()

        def _flush_block():
            if block_name is not None:
                _flush_loop()
                yield block_name, data.copy(), list(loops)
                data.clear()
                loops.clear()
            # to make it work and not return too soon...
            yield ""

        for line in self.readlines():
            line = line.strip()

            # ---- multiline string (semicolon-delimited) -----------
            if multiline_tag is not None:
                if line == ";":
                    data[multiline_tag] = "\n".join(multiline_buf)
                    multiline_tag = None
                    multiline_buf = []
                else:
                    multiline_buf.append(line)
                continue

            if not line or line.startswith("#"):
                continue

            # ---- data block header --------------------------------
            if line.lower().startswith("data_"):
                ret = next(_flush_block())
                if ret:
                    yield ret
                block_name = line[5:]
                in_loop = False
                continue

            # ---- loop_ keyword -----------------------------------
            if line.lower() in ("loop_", "stop_"):
                _flush_loop()
                in_loop = True
                loop_cols = []
                loop_rows = []
                continue

            # ---- tag inside or outside loop ----------------------
            if line.startswith("_"):
                parts = line.split(None, 1)
                tag = parts[0].lower()

                if in_loop and not loop_rows:
                    # still collecting column headers
                    loop_cols.append(tag)
                    if len(parts) > 1:
                        # value on same line as tag (non-loop context leaked)
                        loop_rows.append(self._parse_value(parts[1]))
                    continue

                # scalar tag
                in_loop = False
                _flush_loop()
                if len(parts) == 1:
                    # value follows on the next non-blank line
                    for next_raw in self.fh:
                        next_line = next_raw.strip()
                        if next_line and not next_line.startswith("#"):
                            if next_line.startswith(";"):
                                multiline_tag = tag
                            else:
                                data[tag] = self._parse_value(next_line)
                            break
                else:
                    val = parts[1].strip()
                    if val.startswith(";"):
                        multiline_tag = tag
                    else:
                        data[tag] = self._parse_value(val)
                continue

            # ---- data value (loop body) --------------------------
            if in_loop:
                # handle quoted strings with spaces
                for tok in re.findall(r"'[^']*'|\"[^\"]*\"|\S+", line):
                    loop_rows.append(self._parse_value(tok))
                continue

        # end of file
        if block_name is not None:
            _flush_loop()
            yield block_name, data, loops

    # ------------------------------------------------------------------ #
    #  Public read methods                                                 #
    # ------------------------------------------------------------------ #

    @sile_fh_open()
    def read_lattice(self) -> Lattice:
        """Read the unit cell as a :class:`~sisl.Lattice`."""
        for _, data, _ in self._iter_blocks():
            try:
                a = self._float(data["_cell_length_a"])
                b = self._float(data["_cell_length_b"])
                c = self._float(data["_cell_length_c"])
                alpha = self._float(data["_cell_angle_alpha"])
                beta = self._float(data["_cell_angle_beta"])
                gamma = self._float(data["_cell_angle_gamma"])
            except KeyError as exc:
                raise SileError(f"{self!s}: missing cell parameter {exc}") from exc

            return Lattice([a, b, c, alpha, beta, gamma])

        raise SileError(f"{self!s}: no valid data block found")

    @sile_fh_open()
    def write_lattice(self, lattice: Lattice, fmt: str = ".8f", **kwargs) -> None:
        """Write only the unit-cell block to the CIF file.

        Parameters
        ----------
        lattice :
            Unit cell to write.
        fmt :
           used format for the precision of the data
        """
        from datetime import datetime

        abc, abg = lattice.parameters(rad=False)
        now = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"_refine_date {now}\n")
        self._write(f"_refine_method 'generated from sisl'\n")
        self._write(f"data_{Path(self.file).stem}\n\n")
        cell_line = f"_cell_length_{{0}}   {{1:{fmt}}}\n"
        for d, v in zip("abc", abc):
            self._write(cell_line.format(d, v))
        cell_line = f"_cell_angle_{{0}}   {{1:{fmt}}}\n"
        for d, v in zip(["alpha", "beta", "gamma"], abg):
            self._write(cell_line.format(d, v))

    @sile_fh_open()
    def read_geometry(self) -> Geometry:
        """Read the full crystal structure as a :class:`~sisl.Geometry`.

        Fractional coordinates are used when available
        (``_atom_site_fract_*``), otherwise Cartesian coordinates
        (``_atom_site_Cartn_*``) are used directly.

        The symmetry operations ``_symmetry_equiv_pos_as_xyz`` and
        ``_space_group_symop_operation_xyz`` are also parsed.

        Returns
        -------
        Geometry
        """

        # ---- lattice ------------------------------------------
        lattice = self.read_lattice()
        ATOM_SITE_KEYS = set(["_atom_site_type_symbol", "_atom_site_label"])
        ATOM_FRAC_KEYS = [f"_atom_site_fract_{f}" for f in "xyz"]
        ATOM_CART_KEYS = [f"_atom_site_cartn_{f}" for f in "xyz"]

        atoms = None

        for block_name, data, loops in self._iter_blocks():

            # TODO need to check about symmetry operations
            # I.e. currently we silently pass and read them, but do not
            # execute them.

            self._log("block_name", block_name)
            self._log("data", data)
            self._log("loops", loops)

            # ---- find atom_site loop ------------------------------
            atom_loop = None
            for lp in loops:
                keys = set(lp.keys())
                if keys & ATOM_SITE_KEYS:
                    atom_loop = lp
                    break

            if atom_loop is None:
                # No atom site is found... continue to next split
                continue

            # ---- species ------------------------------------------
            if "_atom_site_type_symbol" in atom_loop:
                raw_species = atom_loop["_atom_site_type_symbol"]
            else:
                raw_species = atom_loop["_atom_site_label"]

            # strip trailing digits/signs
            # Some programs also put this into the _type_symbol (gosh)!
            raw_species = [
                re.sub(r"[^A-Za-z].*$", "", s) for s in atom_loop["_atom_site_label"]
            ]

            atoms = Atoms([Atom(s) for s in raw_species])

            # ---- coordinates --------------------------------------

            if all(k in atom_loop for k in ATOM_FRAC_KEYS):
                frac = np.array(
                    [
                        [self._float(atom_loop[k][i]) for k in ATOM_FRAC_KEYS]
                        for i in range(len(raw_species))
                    ]
                )
            elif all(k in atom_loop for k in ATOM_CART_KEYS):
                cart = np.array(
                    [
                        [self._float(atom_loop[k][i]) for k in ATOM_CART_KEYS]
                        for i in range(len(raw_species))
                    ]
                )
                frac = cart @ lattice.icell.T
            else:
                raise SileError(
                    f"{self!s}: neither fractional nor Cartesian coordinates found"
                )

            # ---- symmetry expansion ------------------------------
            symops = next(
                (ops for lp in loops if (ops := _symops_from_loop(lp)) is not None),
                None,
            )
            if symops:
                frac, raw_species = _apply_symops(frac, raw_species, symops)

            xyz = frac @ lattice.cell
            atoms = Atoms([Atom(s) for s in raw_species])
            return Geometry(xyz, atoms=atoms, lattice=lattice)

        if atoms is None:
            raise SileError(f"{self!s}: no _atom_site loop found")

    @sile_fh_open()
    def write_geometry(
        self,
        geometry: Geometry,
        symmetry_ops: None | str | list[str | tuple[np.ndarray, np.ndarray]] = None,
        fmt: str = ".8f",
        **kwargs,
    ) -> None:
        """Write the crystal structure in CIF 1.1 format.

        Atom positions are stored as fractional coordinates
        (``_atom_site_fract_*``).  A sequential ``_atom_site_label`` is
        generated as ``<symbol><index>``, and ``_atom_site_type_symbol``
        holds the element symbol.

        Parameters
        ----------
        geometry :
            Structure to write.
        symmetry_ops :
            Jones symmetry operations, this option might change in the future.
        fmt :
           used format for the precision of the data
        """
        w = self._write  # shorthand

        # ---- header & cell -----------------------------------------
        self.write_lattice(geometry.lattice)

        # ---- symmetry block ----------------------------------------
        if symmetry_ops is None:
            self._log("writing geometry with P 1 symmetry")
            # P 1 — identity only
            w("_symmetry_space_group_name_H-M  'P 1'\n")
            w("_symmetry_Int_Tables_number      1\n\n")
            w("loop_\n")
            w("  _symmetry_equiv_pos_as_xyz\n")
            w("  'x,y,z'\n\n")
        else:
            self._log("writing geometry with UD symmetry")
            if isinstance(symmetry_ops, str):
                symmetry_ops = [symmetry_ops]

            def op2jones(op: str | tuple) -> tuple[np.ndarray, np.ndarray]:
                if isinstance(op, str):
                    return _parse_jones(op)
                return op

            w("loop_\n")
            w("  _symmetry_equiv_pos_as_xyz\n")
            for W, t in map(op2jones, symmetry_ops):
                w(f"  '{_jones_str(W, t)}'\n")
            w("\n")

        # ---- fractional coordinates ---------------------------------
        frac = geometry.fxyz

        w("loop_\n")
        w("  _atom_site_label\n")
        w("  _atom_site_type_symbol\n")
        w("  _atom_site_fract_x\n")
        w("  _atom_site_fract_y\n")
        w("  _atom_site_fract_z\n")

        # count per-species to build unique labels (e.g. Fe1, Fe2, …)
        atom_fmt = (
            f"  {{label:<8s}} {{sym:<4s}} "
            f"{{f[0]:{fmt}}} {{f[1]:{fmt}}} {{f[2]:{fmt}}}\n"
        )
        species_count: Counter = Counter()
        for i, atom in enumerate(geometry.atoms):
            sym = atom.symbol
            species_count[sym] += 1
            label = f"{sym}{species_count[sym]}"
            w(atom_fmt.format(label=label, sym=sym, f=frac[i]))


# Register so sisl.io.get_sile("file.cif") works automatically
add_sile("cif", cifSile, gzip=True)
