# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""CIF file reader for sisl."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from sisl import Atom, Atoms, Geometry, Lattice
from sisl._internal import set_module
from sisl.io import add_sile
from sisl.io.sile import Sile, SileError, sile_fh_open

__all__ = ["cifSile"]


@set_module("sisl.io")
class cifSile(Sile):
    """CIF (Crystallographic Information File) reader.

    Supports CIF 1.1 and a common subset of CIF 2.0.
    Currently implements :meth:`read_geometry` (and the implied
    :meth:`read_lattice`).

    Notes
    -----
    There is lacking symmetry support.

    Examples
    --------
    >>> geom = cifSile("crystal.cif").read_geometry()
    """

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

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

        multiline_buf: list = []
        multiline_tag: Optional[str] = None

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

        Returns
        -------
        Geometry
        """

        error_keywords = (
            (
                "A (currently) unsupported keyword {} was found in the cif file. "
                "The resulting geometry cannot be guaranteed."
            ),
            (
                "_space_group_symop_operation_xyz",
                "_symmetry_equiv_pos_as_xyz",
            ),
        )

        warning_keywords = (
            (
                "A (currently) unsupported keyword {} was found in the cif file. "
                "The resulting geometry cannot be guaranteed."
            ),
            (),
        )

        # ---- lattice ------------------------------------------
        lattice = self.read_lattice()
        ATOM_SITE_KEYS = set(["_atom_site_type_symbol", "_atom_site_label"])
        ATOM_FRAC_KEYS = [f"_atom_site_fract_{f}" for f in "xyz"]
        ATOM_CART_KEYS = [f"_atom_site_cartn_{f}" for f in "xyz"]

        atoms = None

        for _, data, loops in self._iter_blocks():

            # TODO need to check about symmetry operations
            # I.e. currently we silently pass and read them, but do not
            # execute them.

            print("block_name", _)
            print("data", data)
            print("loops", loops)

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
                # fall back to label, strip trailing digits/signs
                raw_species = [
                    re.sub(r"[^A-Za-z].*$", "", s)
                    for s in atom_loop["_atom_site_label"]
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
                xyz = frac @ lattice.cell  # fractional → Cartesian
            elif all(k in atom_loop for k in ATOM_CART_KEYS):
                xyz = np.array(
                    [
                        [self._float(atom_loop[k][i]) for k in ATOM_CART_KEYS]
                        for i in range(len(raw_species))
                    ]
                )
            else:
                raise SileError(
                    f"{self!s}: neither fractional nor Cartesian coordinates found"
                )

            return Geometry(xyz, atoms=atoms, lattice=lattice)

        if atoms is None:
            raise SileError(f"{self!s}: no _atom_site loop found")

    @sile_fh_open()
    def write_geometry(self, geometry: Geometry, fmt: str = ".8f", **kwargs) -> None:
        """Write the crystal structure in CIF 1.1 format.

        Atom positions are stored as fractional coordinates
        (``_atom_site_fract_*``).  A sequential ``_atom_site_label`` is
        generated as ``<symbol><index>``, and ``_atom_site_type_symbol``
        holds the element symbol.

        Parameters
        ----------
        geometry :
            Structure to write.
        fmt :
           used format for the precision of the data
        """
        w = self._write  # shorthand

        # ---- header & cell -----------------------------------------
        self.write_lattice(geometry.lattice)

        # ---- symmetry (P1 — no symmetry reduction performed) --------
        w("_symmetry_space_group_name_H-M   'P 1'\n")
        w("_symmetry_Int_Tables_number       1\n\n")

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
