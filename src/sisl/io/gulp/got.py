# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Sile object for reading/writing GULP in/output
"""
import numpy as np

from sisl import Atom, Geometry, Lattice, Orbital, constant, units
from sisl._internal import set_module
from sisl.messages import deprecation, info, warn
from sisl.physics import DynamicalMatrix

from .._help import parse_order
from ..sile import add_sile, sile_fh_open
from .fc import fcSileGULP
from .sile import SileGULP

__all__ = ["gotSileGULP"]


@set_module("sisl.io.gulp")
class gotSileGULP(SileGULP):
    """GULP output file object

    Parameters
    ----------
    filename : str
        filename of the file
    mode : str, optional
        opening mode of file, default to read-only
    base : str, optional
        base directory of the file
    """

    def _setup(self, *args, **kwargs):
        """Setup `gotSileGULP` after initialization"""
        super()._setup(*args, **kwargs)
        self._keys = dict()
        self.set_lattice_key("Cartesian lattice vectors")
        self.set_geometry_key("Final fractional coordinates")
        self.set_dynamical_matrix_key("Real Dynamical matrix")

    def set_key(self, segment, key):
        """Sets the segment lookup key"""
        if key is not None:
            self._keys[segment] = key

    def set_lattice_key(self, key):
        """Overwrites internal key lookup value for the cell vectors"""
        self.set_key("lattice", key)

    set_supercell_key = deprecation(
        "set_supercell_key is deprecated in favor of set_lattice_key", "0.15", "0.17"
    )(set_lattice_key)

    @sile_fh_open()
    def read_lattice_nsc(self, key=None):
        """Reads the dimensions of the supercell"""

        f, l = self.step_to("Supercell dimensions")
        if not f:
            return np.array([1, 1, 1], np.int32)

        # Read off the supercell dimensions
        xyz = l.split("=")[1:]

        # Now read off the quantities...
        nsc = [int(i.split()[0]) for i in xyz]

        return np.array(nsc[:3], np.int32)

    @sile_fh_open()
    def read_lattice(self, key=None, **kwargs) -> Lattice:
        """Reads a `Lattice` and creates the GULP cell"""
        self.set_lattice_key(key)

        f, _ = self.step_to(self._keys["lattice"])
        if not f:
            raise ValueError(
                "SileGULP tries to lookup the Lattice vectors "
                'using key "' + self._keys["lattice"] + '". \n'
                'Use ".set_lattice_key(...)" to search for different name.\n'
                'This could not be found found in file: "' + self.file + '".'
            )

        # skip 1 line
        self.readline()
        cell = np.empty([3, 3], np.float64)
        for i in [0, 1, 2]:
            l = self.readline().split()
            cell[i, :] = [float(x) for x in l[:3]]

        return Lattice(cell)

    def set_geometry_key(self, key):
        """Overwrites internal key lookup value for the geometry vectors"""
        self.set_key("geometry", key)

    @sile_fh_open()
    def read_geometry(self, **kwargs) -> Geometry:
        """Reads a geometry and creates the GULP dynamical geometry"""
        # create default supercell
        lattice = Lattice([1, 1, 1])

        for _ in [0, 1]:
            # Step to either the geometry or
            f, _, ki = self.step_to(
                [self._keys["lattice"], self._keys["geometry"]], ret_index=True
            )
            if not f and ki == 0:
                raise ValueError(
                    "SileGULP tries to lookup the Lattice vectors "
                    'using key "' + self._keys["lattice"] + '". \n'
                    'Use ".set_lattice_key(...)" to search for different name.\n'
                    'This could not be found found in file: "' + self.file + '".'
                )
            elif f and ki == 0:
                # supercell
                self.readline()
                cell = np.empty([3, 3], np.float64)
                for i in [0, 1, 2]:
                    l = self.readline().split()
                    cell[i, 0] = float(l[0])
                    cell[i, 1] = float(l[1])
                    cell[i, 2] = float(l[2])
                lattice = Lattice(cell)

            elif not f and ki == 1:
                raise ValueError(
                    "SileGULP tries to lookup the Geometry coordinates "
                    'using key "' + self._keys["geometry"] + '". \n'
                    'Use ".set_geom_key(...)" to search for different name.\n'
                    'This could not be found found in file: "' + self.file + '".'
                )
            elif f and ki == 1:
                orbs = [Orbital(-1, tag=tag) for tag in "xyz"]

                # We skip 5 lines
                for _ in [0] * 5:
                    self.readline()

                Z = []
                xyz = []
                while True:
                    l = self.readline()
                    if l[0] == "-":
                        break

                    ls = l.split()
                    Z.append(Atom(ls[1], orbitals=orbs))
                    xyz.append([float(x) for x in ls[3:6]])

                # Convert to array and correct size
                xyz = np.array(xyz, np.float64)
                xyz.shape = (-1, 3)

                if len(Z) == 0 or len(xyz) == 0:
                    raise ValueError(
                        "Could not read in cell information and/or coordinates"
                    )

            elif not f:
                # could not find either cell or geometry
                raise ValueError(
                    "SileGULP tries to lookup the Lattice or Geometry.\n"
                    "None succeeded, ensure file has correct format.\n"
                    'This could not be found found in file: "{}".'.format(self.file)
                )

        # as the cell may be read in after the geometry we have
        # to wait until here to convert from fractional
        if "fractional" in self._keys["geometry"].lower():
            # Correct for fractional coordinates
            xyz = np.dot(xyz, lattice.cell)

        # Return the geometry
        return Geometry(xyz, Z, lattice=lattice)

    def set_dynamical_matrix_key(self, key):
        """Overwrites internal key lookup value for the dynamical matrix vectors"""
        self.set_key("dyn", key)

    set_dyn_key = set_dynamical_matrix_key

    def read_dynamical_matrix(self, **kwargs) -> DynamicalMatrix:
        """Returns a GULP dynamical matrix model for the output of GULP

        Parameters
        ----------
        cutoff: float, optional
           absolute values below the cutoff are considered 0. Defaults to 0. eV/Ang**2.
        hermitian : bool, optional
           if true (default), the returned dynamical matrix will be hermitian
        dtype: np.dtype (np.float64)
           default data-type of the matrix
        order: list of str, optional
            the order of which to try and read the dynamical matrix
            By default this is ``['got'/'gout', 'FC']``. Note that ``FC`` corresponds to
            the `fcSileGULP` file (``FORCE_CONSTANTS_2ND``).
        """
        geom = self.read_geometry(**kwargs)

        order = parse_order(kwargs.pop("order", None), ["got", "FC"])
        for f in order:
            v = getattr(self, "_r_dynamical_matrix_{}".format(f.lower()))(
                geom, **kwargs
            )
            if v is None:
                continue

            # Convert the dynamical matrix such that a diagonalization returns eV ^ 2
            scale = constant.hbar / units("Ang", "m") / units("eV amu", "J kg") ** 0.5
            v.data *= scale**2
            v = DynamicalMatrix.fromsp(geom, v)
            if kwargs.get("hermitian", True):
                v = (v + v.transpose()) * 0.5
            return v

        return None

    @sile_fh_open()
    def _r_dynamical_matrix_got(self, geometry, **kwargs):
        """In case the dynamical matrix is read from the file"""
        # Easier for creation of the sparsity pattern
        from scipy.sparse import lil_matrix

        # Default cutoff eV / Ang ** 2
        cutoff = kwargs.get("cutoff", 0.0)
        dtype = kwargs.get("dtype", np.float64)

        nxyz = geometry.no
        dyn = lil_matrix((nxyz, nxyz), dtype=dtype)

        f, _ = self.step_to(self._keys["dyn"])
        if not f:
            info(
                f"{self.__class__.__name__}.read_dynamical_matrix tries to lookup the Dynamical matrix "
                "using key '{self._keys['dyn']}'. "
                "Use .set_dynamical_matrix_key(...) to search for different name."
                "This could not be found found in file: {self.file}"
            )
            return None

        # skip 1 line
        self.readline()

        # default range
        dat = np.empty([nxyz], dtype=dtype)
        i, j = 0, 0
        nxyzm1 = nxyz - 1
        while i < nxyz:
            l = self.readline().strip()
            if len(l) == 0:
                break

            # convert to float list
            ls = [float(x) for x in l.split()]

            k = min(12, nxyz - j)

            # GULP only prints columns corresponding
            # to a full row. Hence the remaining
            # data must be nxyz - j - 1
            dat[j : j + k] = ls[:k]
            j += k

            if j >= nxyz:
                dyn[i, :] = dat[:]
                # step row
                i += 1
                # reset column
                j = 0

        # clean-up for memory
        del dat

        # Convert to COO matrix format
        dyn = dyn.tocoo()

        # Construct mass ** (-.5), so we can check cutoff correctly (in unit eV/Ang**2)
        mass_sqrt = geometry.atoms.mass.repeat(3) ** 0.5
        dyn.data[:] *= mass_sqrt[dyn.row] * mass_sqrt[dyn.col]
        dyn.data[np.fabs(dyn.data) < cutoff] = 0.0
        dyn.data[:] *= 1 / (mass_sqrt[dyn.row] * mass_sqrt[dyn.col])
        dyn.eliminate_zeros()

        return dyn

    _r_dynamical_matrix_gout = _r_dynamical_matrix_got

    def _r_dynamical_matrix_fc(self, geometry, **kwargs):
        # The output of the force constant in the file does not contain the mass-scaling
        # nor the unit conversion
        f = self.dir_file("FORCE_CONSTANTS_2ND")
        if not f.is_file():
            return None

        fc = fcSileGULP(f, "r").read_hessian(**kwargs)

        if fc.shape[0] // 3 != geometry.na:
            warn(
                f"{self.__class__.__name__}.read_dynamical_matrix(FC) inconsistent force constant file, na_file={fc.shape[0]//3}, na_geom={geometry.na}"
            )
            return None
        elif fc.shape[0] != geometry.no:
            warn(
                f"{self.__class__.__name__}.read_dynamical_matrix(FC) inconsistent geometry, no_file={fc.shape[0]}, no_geom={geometry.no}"
            )
            return None

        # Construct orbital mass ** (-.5)
        rmass = 1 / geometry.atoms.mass.repeat(3) ** 0.5

        # Scale to get dynamical matrix
        fc.data[:] *= rmass[fc.row] * rmass[fc.col]

        return fc


# Old-style GULP output
add_sile("gout", gotSileGULP, gzip=True)
add_sile("got", gotSileGULP, gzip=True)
add_sile("out", gotSileGULP, gzip=True)
