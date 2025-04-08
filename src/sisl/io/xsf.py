# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import os.path as osp
from typing import Optional

import numpy as np

import sisl._array as _a
from sisl import AtomUnknown, Geometry, Grid, Lattice, PeriodicTable
from sisl._internal import set_module
from sisl.messages import deprecate_argument
from sisl.utils import str_spec

from ._multiple import SileBinder, postprocess_tuple

# Import sile objects
from .sile import *

__all__ = ["xsfSile"]


def _get_kw_index(key: str):
    # Get the integer in a line like 'ATOMS 2', converted to 0-indexing, and with -1 if no int is there
    kl = key.split()
    if len(kl) == 1:
        return -1
    return int(kl[1]) - 1


def reset_values(*names_values, animsteps: bool = False):
    if animsteps:

        def reset(self: xsfSile):
            nonlocal names_values
            self._write_animsteps()
            for name, value in names_values:
                setattr(self, name, value)

    else:

        def reset(self: xsfSile):
            nonlocal names_values
            for name, value in names_values:
                setattr(self, name, value)

    return reset


# The XSF files are compatible with Vesta, but ONLY
# if there are no empty lines!
@set_module("sisl.io")
class xsfSile(Sile):
    """XSF file for XCrySDen

    When creating an XSF file one must denote how many geometries to write out.
    It is also necessary to use the xsf in a context manager, otherwise it will
    overwrite itself repeatedly.

    >>> with xsfSile('file.xsf', 'w', steps=100) as xsf:
    ...     for i in range(100):
    ...         xsf.write_geometry(geom)

    Parameters
    ----------
    steps : int, optional
        number of steps the xsf file contains. Defaults to 1
    """

    def _setup(self, *args, **kwargs):
        """Setup the `xsfSile` after initialization"""
        super()._setup(*args, **kwargs)
        self._comment = ["#"]
        if "w" in self._mode:
            self._geometry_max = kwargs.get("steps", 1)
        else:
            self._geometry_max = kwargs.get("steps", -1)
        self._geometry_write = 0
        self._r_type = None
        self._r_cell = None

    def _write_key(self, key: str):
        self._write(f"{key}\n")

    def _write_key_index(self, key: str):
        # index is 1-based in file
        if self._geometry_max > 1:
            self._write(f"{key} {self._geometry_write + 1}\n")
        else:
            self._write(f"{key}\n")

    def _write_once(self, string: str):
        if self._geometry_write <= 0:
            self._write(string)

    def _write_animsteps(self):
        if self._geometry_max > 1:
            self._write(f"ANIMSTEPS {self._geometry_max}\n")

    @sile_fh_open(reset=reset_values(("_geometry_write", 0), animsteps=True))
    @deprecate_argument("sc", "lattice", "use lattice= instead of sc=", "0.15", "0.17")
    def write_lattice(self, lattice: Lattice, fmt: str = ".8f"):
        """Writes the supercell to the contained file

        Parameters
        ----------
        lattice :
           the supercell to be written
        fmt :
           used format for the precision of the data
        """
        # Check that we can write to the file
        sile_raise_write(self)

        # Write out top-header stuff
        from time import gmtime, strftime

        self._write_once(
            "# File created by: sisl {}\n#\n".format(strftime("%Y-%m-%d", gmtime()))
        )

        pbc = lattice.pbc
        if pbc.sum() == 0:
            self._write_once("MOLECULE\n#\n")
        elif all(pbc == (True, True, False)):
            self._write_once("SLAB\n#\n")
        elif all(pbc == (True, False, False)):
            self._write_once("POLYMER\n#\n")
        else:
            self._write_once("CRYSTAL\n#\n")

        self._write_once("# Primitive lattice vectors:\n#\n")
        self._write_key_index("PRIMVEC")

        # We write the cell coordinates as the cell coordinates
        fmt_str = f"{{:{fmt}}} " * 3 + "\n"
        for i in (0, 1, 2):
            self._write(fmt_str.format(*lattice.cell[i, :]))

        # Convert the unit cell to a conventional cell (90-90-90))
        # It seems this simply allows to store both formats in
        # the same file. However the below stuff is not correct.
        # self._write_once('#\n# Conventional lattice vectors:\n#\n')
        # self._write_key_index('CONVVEC')
        # convcell = lattice.to.Cuboid(orthogonal=True)._v
        # for i in [0, 1, 2]:
        #    self._write(fmt_str.format(*convcell[i, :]))

    @sile_fh_open(reset=reset_values(("_geometry_write", 0), animsteps=True))
    def write_geometry(self, geometry: Geometry, fmt: str = ".8f", data=None):
        """Writes the geometry to the contained file

        Parameters
        ----------
        geometry :
           the geometry to be written
        fmt :
           used format for the precision of the data
        data : (geometry.na, 3), optional
           auxiliary data associated with the geometry to be saved
           along side. Internally in XCrySDen this data is named *Forces*
        """
        self.write_lattice(geometry.lattice, fmt)

        has_data = data is not None
        if has_data:
            data.shape = (-1, 3)

        self._write_once("#\n# Atomic coordinates (in primitive coordinates)\n#\n")
        self._geometry_write += 1
        self._write_key_index("PRIMCOORD")
        self._write(f"{len(geometry)} 1\n")

        non_valid_Z = (geometry.atoms.Z <= 0).nonzero()[0]
        if len(non_valid_Z) > 0:
            geometry = geometry.remove(non_valid_Z)

        if has_data:
            fmt_str = (
                "{{0:3d}}  {{1:{0}}}  {{2:{0}}}  {{3:{0}}}   {{4:{0}}}  {{5:{0}}}  {{6:{0}}}\n"
            ).format(fmt)
            for ia in geometry:
                tmp = np.append(geometry.xyz[ia, :], data[ia, :])
                self._write(fmt_str.format(geometry.atoms[ia].Z, *tmp))
        else:
            fmt_str = "{{0:3d}}  {{1:{0}}}  {{2:{0}}}  {{3:{0}}}\n".format(fmt)
            for ia in geometry:
                self._write(fmt_str.format(geometry.atoms[ia].Z, *geometry.xyz[ia, :]))

    @sile_fh_open()
    def _r_geometry_next(
        self,
        lattice: Optional[Lattice] = None,
        atoms=None,
        ret_data: bool = False,
        only_lattice: bool = False,
    ) -> Geometry:
        if lattice is None:
            # fetch the prior read cell value
            lattice = self._r_cell

        # initialize all things
        cell = None
        convvec = None
        primvec = None
        atoms_r = []
        xyz = None
        data = None

        while line := self.readline():

            if line.isspace():
                continue

            if line.startswith("CONVVEC"):
                convvec = _a.emptyd([3, 3])
                for i in range(3):
                    convvec[i] = self.readline().split()

            elif line.startswith("PRIMVEC"):
                primvec = _a.emptyd([3, 3])
                for i in range(3):
                    primvec[i] = self.readline().split()

            elif line.startswith("PRIMCOORD"):
                # get number of atoms
                line = self.readline().split()
                na = int(line[0])
                xyz = _a.emptyd([na, 3])
                atoms_r, data = [], []
                for i in range(na):
                    line = self.readline().split()
                    atoms_r.append(line[0])
                    xyz[i] = line[1:4]
                    if ret_data and len(line) > 4:
                        data.append([float(x) for x in line[4:]])

                # the primcoord should always come after conv/prim-vec
                break

            elif line.startswith("ATOMS"):
                # molecule specification
                def next():
                    point = self.fh.tell()
                    line = self.readline().split()
                    try:
                        int(line[0])
                    except ValueError:
                        self.fh.seek(point)
                        return []
                    return line

                xyz, atoms_r, data = [], [], []
                lines = next()
                while lines:
                    atoms_r.append(line[0])
                    xyz.append(line[1:4])
                    if ret_data and len(line) > 4:
                        data.append(line[4:])
                    lines = next()
                xyz = _a.arrayd(xyz)
                if ret_data:
                    data = _a.arrayd(data)

                # the ATOMS key should always come after conv/prim-vec
                break

            elif line.startswith("CONVCOORD"):
                raise NotImplementedError(
                    f"{self.__class__.__name__} does not implement reading CONVCOORD"
                )

            elif line.startswith("CRYSTAL"):
                self._r_type = "CRYSTAL"
            elif line.startswith("SLAB"):
                self._r_type = "SLAB"
            elif line.startswith("POLYMER"):
                self._r_type = "POLYMER"
            elif line.startswith("MOLECULE"):
                self._r_type = "MOLECULE"

        typ = self._r_type
        if typ == "CRYSTAL":
            bc = ["periodic", "periodic", "periodic"]
        elif typ == "SLAB":
            bc = ["periodic", "periodic", "unknown"]
        elif typ == "POLYMER":
            bc = ["periodic", "unknown", "unknown"]
        elif typ == "MOLECULE":
            bc = ["unknown", "unknown", "unknown"]
        else:
            bc = ["unknown", "unknown", "unknown"]

        cell = None

        if primvec is not None:
            cell = Lattice(primvec, boundary_condition=bc)
        elif lattice is not None:
            cell = lattice

        elif typ == "MOLECULE":
            cell = Lattice(
                np.diag(xyz.max(0) - xyz.min(0) + 10.0), boundary_condition=bc
            )

        if cell is None:
            raise ValueError(
                f"{self.__class__.__name__} could not find lattice parameters."
            )

        # overwrite the currently read cell
        self._r_cell = cell

        if atoms is None:
            # this ensures that we will not parse atoms unless required
            pt = PeriodicTable()
            atoms = [pt.Z(Z) for Z in atoms_r]

        if only_lattice:
            return cell

        if xyz is None:
            if ret_data:
                return None, None
            return None

        geom = Geometry(xyz, atoms=atoms, lattice=cell)
        if ret_data:
            return geom, _a.arrayd(data)
        return geom

    @SileBinder(postprocess=postprocess_tuple(list))
    def read_basis(self) -> Atoms:
        """Basis set (`Atoms`) contained in file"""
        ret = self._r_geometry_next()
        if ret is None:
            return ret
        return ret.atoms

    @SileBinder(postprocess=postprocess_tuple(list))
    def read_lattice(self) -> Lattice:
        """Lattice contained in file"""
        ret = self._r_geometry_next(only_lattice=True)
        return ret

    @SileBinder(postprocess=postprocess_tuple(list))
    @deprecate_argument("sc", "lattice", "use lattice= instead of sc=", "0.15", "0.17")
    def read_geometry(
        self, lattice: Optional[Lattice] = None, atoms=None, ret_data: bool = False
    ) -> Geometry:
        """Geometry contained in file, and optionally the associated data

        Parameters
        ----------
        lattice :
            the supercell in case the lattice vectors are not present in the current
            block.
        atoms : Atoms, optional
            atomic species used regardless of the contained atomic species
        ret_data :
           in case the the file has auxiliary data, return that as well.
        """
        return self._r_geometry_next(lattice=lattice, atoms=atoms, ret_data=ret_data)

    @sile_fh_open()
    def write_grid(self, *args, **kwargs):
        """Store grid(s) data to an XSF file

        Examples
        --------
        >>> g1 = Grid(0.1, lattice=2.)
        >>> g2 = Grid(0.1, lattice=2.)
        >>> get_sile('output.xsf', 'w').write_grid(g1, g2)

        Parameters
        ----------
        *args : Grid
            a list of data-grids to be written to the XSF file.
            Each argument gets the field name ``?grid_<>`` where <> starts
            with the integer 0, and *?* is ``real_``/``imag_`` for complex
            valued grids.
        geometry : Geometry, optional
            the geometry stored in the file, defaults to ``args[0].geometry``
        fmt : str, optional
            floating point format for data (.5e)
        buffersize : int, optional
            size of the buffer while writing the data, (6144)
        """
        sile_raise_write(self)
        # for now we do not allow an animation with grid data... should this
        # even work?
        if self._geometry_max > 1:
            raise NotImplementedError(
                f"{self.__class__.__name__}.write_grid not allowed in an animation file."
            )

        geom = kwargs.get("geometry", args[0].geometry)
        if geom is None:
            geom = Geometry([0, 0, 0], AtomUnknown(999), lattice=args[0].lattice)
        self.write_geometry(geom)

        # Buffer size for writing
        buffersize = min(kwargs.get("buffersize", 6144), args[0].grid.size)

        # Format for precision
        fmt = kwargs.get("fmt", ".5e")

        self._write("BEGIN_BLOCK_DATAGRID_3D\n")
        name = kwargs.get("name", "sisl_{}".format(len(args)))
        # Transfer all spaces to underscores (no spaces allowed)
        self._write(" " + name.replace(" ", "_") + "\n")
        _v3 = (("{:" + fmt + "} ") * 3).strip() + "\n"

        def write_cell(grid):
            # Now write the grid
            self._write("  {} {} {}\n".format(*grid.shape))
            self._write("  " + _v3.format(*grid.origin))
            self._write("  " + _v3.format(*grid.cell[0, :]))
            self._write("  " + _v3.format(*grid.cell[1, :]))
            self._write("  " + _v3.format(*grid.cell[2, :]))

        for i, grid in enumerate(args):
            if isinstance(grid, Grid):
                name = kwargs.get(f"grid{i}", str(i))
            else:
                # it must be a tuple
                name, grid = grid
                name = kwargs.get(f"grid{i}", name)

            is_complex = np.iscomplexobj(grid.grid)

            if is_complex:
                self._write(f" BEGIN_DATAGRID_3D_real_{name}\n")
            else:
                self._write(f" BEGIN_DATAGRID_3D_{name}\n")

            write_cell(grid)

            # for z
            #   for y
            #     for x
            #       write...
            _fmt = "{:" + fmt + "}\n"
            for x in np.nditer(
                np.asarray(grid.grid.real.T, order="C").reshape(-1),
                flags=["external_loop", "buffered"],
                op_flags=[["readonly"]],
                order="C",
                buffersize=buffersize,
            ):
                self._write((_fmt * x.shape[0]).format(*x.tolist()))

            self._write(" END_DATAGRID_3D\n")

            # Skip if not complex
            if not is_complex:
                continue
            self._write(f" BEGIN_DATAGRID_3D_imag_{name}\n")
            write_cell(grid)
            for x in np.nditer(
                np.asarray(grid.grid.imag.T, order="C").reshape(-1),
                flags=["external_loop", "buffered"],
                op_flags=[["readonly"]],
                order="C",
                buffersize=buffersize,
            ):
                self._write((_fmt * x.shape[0]).format(*x.tolist()))

            self._write(" END_DATAGRID_3D\n")

        self._write("END_BLOCK_DATAGRID_3D\n")

    def ArgumentParser(self, p=None, *args, **kwargs):
        """Returns the arguments that is available for this Sile"""
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)

    def ArgumentParser_out(self, p, *args, **kwargs):
        """Adds arguments only if this file is an output file

        Parameters
        ----------
        p : `argparse.ArgumentParser`
           the parser which gets amended the additional output options.
        """
        import argparse

        ns = kwargs.get("namespace", None)
        if ns is None:

            class _:
                pass

            ns = _()

        # We will add the vector data
        class VectorNoScale(argparse.Action):
            def __call__(self, parser, ns, no_value, option_string=None):
                setattr(ns, "_vector_scale", False)

        p.add_argument(
            "--no-vector-scale",
            "-nsv",
            nargs=0,
            action=VectorNoScale,
            help="""Do not modify vector components (same as --vector-scale 1.)""",
        )
        # Default to scale the vectors
        setattr(ns, "_vector_scale", True)

        # We will add the vector data
        class VectorScale(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                setattr(ns, "_vector_scale", float(value))

        p.add_argument(
            "--vector-scale",
            "-sv",
            metavar="SCALE",
            action=VectorScale,
            help="""Scale vector components by this factor.""",
        )

        # We will add the vector data
        class Vectors(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                routine = values.pop(0)

                # Default input file
                input_file = getattr(ns, "_input_file", None)

                # Figure out which of the segments are a file
                for i, val in enumerate(values):
                    if osp.isfile(str_spec(val)[0]):
                        input_file = values.pop(i)
                        break

                # Quick return if there is no input-file...
                if input_file is None:
                    return

                # Try and read the vector
                from sisl.io import get_sile

                input_sile = get_sile(input_file, mode="r")

                vector = None
                if hasattr(input_sile, f"read_{routine}"):
                    vector = getattr(input_sile, f"read_{routine}")(*values)

                if vector is None:
                    # Try the read_data function
                    d = {routine: True}
                    vector = input_sile.read_data(*values, **d)

                if vector is None and len(values) > 1:
                    # try and see if the first argument is a str, if
                    # so use that as a keyword
                    if isinstance(values[0], str):
                        d = {values[0]: True}
                        vector = input_sile.read_data(*values[1:], **d)

                # Clean the sile
                del input_sile

                if vector is None:
                    # Use title to capitalize
                    raise ValueError(
                        "{} could not be read from file: {}.".format(
                            routine.title(), input_file
                        )
                    )

                if len(vector) != len(ns._geometry):
                    raise ValueError(
                        f"read_{routine} could read from file: {input_file}, sizes does not conform to geometry."
                    )
                setattr(ns, "_vector", vector)

        p.add_argument(
            "--vector",
            "-v",
            metavar=("DATA", "*ARGS[, FILE]"),
            nargs="+",
            action=Vectors,
            help="""Adds vector arrows for each atom, first argument is type (force, moment, ...).
If the current input file contains the vectors no second argument is necessary, else
the file containing the data is required as the last input.

Any arguments inbetween are passed to the `read_data` function (in order).

By default the vectors scaled by 1 / max(|V|) such that the longest vector has length 1.
                       """,
        )


add_sile("xsf", xsfSile, case=False, gzip=True)
add_sile("axsf", xsfSile, case=False, gzip=True)
