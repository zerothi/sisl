# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
from scipy.sparse import SparseEfficiencyWarning, lil_matrix, triu

from sisl import Atom, Geometry, Lattice
from sisl import _array as _a
from sisl._core.sparse import ispmatrix, ispmatrixd
from sisl._help import wrap_filterwarnings
from sisl._internal import set_module
from sisl.messages import warn
from sisl.physics import Hamiltonian

# Import sile objects
from .sile import *

__all__ = ["hamiltonianSile"]


@set_module("sisl.io")
class hamiltonianSile(Sile):
    """Hamiltonian file object"""

    @sile_fh_open()
    def read_geometry(self) -> Geometry:
        """Reading a geometry in regular Hamiltonian format"""

        cell = np.zeros([3, 3], np.float64)
        Z = []
        xyz = []

        nsc = _a.zerosi([3])

        def Z2no(i, no):
            try:
                # pure atomic number
                return int(i), no
            except Exception:
                # both atomic number and no
                j = i.replace("[", " ").replace("]", " ").split()
                return int(j[0]), int(j[1])

        # The format of the geometry file is
        keys = ("atoms", "cell", "lattice", "supercell", "nsc")
        for _ in range(len(keys)):
            _, L = self.step_to(keys, case=False)
            L = L.strip()
            l = L.lower()
            if "supercell" in l or "nsc" in l:
                # We have everything in one line
                l = l.split()[1:]
                for i in range(3):
                    nsc[i] = int(l[i])
            elif "cell" in l or "lattice" in l:
                if "begin" in l:
                    for i in range(3):
                        l = self.readline().split()
                        cell[i, 0] = float(l[0])
                        cell[i, 1] = float(l[1])
                        cell[i, 2] = float(l[2])
                    self.readline()  # step past the block
                else:
                    # We have the diagonal in one line
                    l = l.split()[1:]
                    if len(l) in (3, 6):
                        cell = np.zeros([len(l)], np.float64)
                    else:
                        raise NotImplementedError(
                            "# of lattice components different "
                            "from 3 or 6 values is not implemented"
                        )
                    for i in range(cell.size):
                        cell[i] = float(l[i])
            elif "atoms" in l:
                l = self.readline()
                while not l.startswith("end"):
                    ls = l.split()
                    try:
                        no = int(ls[4])
                    except Exception:
                        no = 1
                    z, no = Z2no(ls[0], no)
                    Z.append({"Z": z, "orbital": [-1.0 for _ in range(no)]})
                    xyz.append([float(f) for f in ls[1:4]])
                    l = self.readline()
                xyz = _a.arrayd(xyz)
                xyz.shape = (-1, 3)
                self.readline()  # step past the block

        # Create geometry with associated lattice and atoms
        geom = Geometry(xyz, atoms=Atom[Z], lattice=Lattice(cell, nsc=nsc))

        return geom

    @sile_fh_open(True)
    def read_hamiltonian(
        self, hermitian: bool = True, dtype=np.float64, **kwargs
    ) -> Hamiltonian:
        """Reads a Hamiltonian (including the geometry)

        Reads the Hamiltonian model
        """
        # Read the geometry in this file
        geom = self.read_geometry()

        # TODO parsing geom.no is wrong for non-collinear spin

        # With the geometry in place we can read in the entire matrix
        # Create a new sparse matrix
        H = lil_matrix((geom.no, geom.no_s), dtype=dtype)
        S = lil_matrix((geom.no, geom.no_s), dtype=dtype)

        def i2o(geom, i):
            try:
                # pure orbital
                return int(i)
            except Exception:
                # ia[o]
                # atom ia and the orbital o
                j = i.replace("[", " ").replace("]", " ").split()
                return geom.a2o(int(j[0])) + int(j[1])

        # Start reading in the supercell
        while True:
            found, l = self.step_to("matrix", allow_reread=False)
            if not found:
                break

            # Get supercell specification it the block
            #   begin matrix <supercell>
            ls = l.split()
            try:
                isc = np.array([int(ls[i]) for i in range(2, 5)], np.int32)
            except Exception:
                isc = np.array([0, 0, 0], np.int32)

            off1 = geom.sc_index(isc) * geom.no
            off2 = geom.sc_index(-isc) * geom.no
            l = self.readline()
            while not l.startswith("end"):
                ls = l.split()
                jo = i2o(geom, ls[0])
                io = i2o(geom, ls[1])
                h = float(ls[2])
                try:
                    s = float(ls[3])
                except IndexError:
                    s = 0.0
                H[jo, io + off1] = h
                S[jo, io + off1] = s
                if hermitian:
                    H[io, jo + off2] = h
                    S[io, jo + off2] = s
                l = self.readline()

        if np.abs(S).sum() == geom.no:
            S = None

        return Hamiltonian.fromsp(geom, H, S)

    @sile_fh_open()
    def write_geometry(self, geometry: Geometry, fmt: str = ".8f", **kwargs) -> None:
        """Writes the geometry to the output file

        Parameters
        ----------
        geometry :
              The geometry we wish to write
        """

        # The format of the geometry file is
        # for now, pretty stringent
        # Get cell_fmt
        cell_fmt = fmt
        if "cell_fmt" in kwargs:
            cell_fmt = kwargs["cell_fmt"]
        xyz_fmt = fmt

        self._write("begin cell\n")
        # Write the cell
        fmt_str = "  {{0:{0}}} {{1:{0}}} {{2:{0}}}\n".format(cell_fmt)
        for i in range(3):
            self._write(fmt_str.format(*geometry.cell[i, :]))
        self._write("end cell\n")

        # Write number of super cells in each direction
        self._write("\nsupercell {:d} {:d} {:d}\n".format(*geometry.nsc))

        # Write all atomic positions along with the specie type
        self._write("\nbegin atoms\n")
        fmt1_str = "  {{0:d}} {{1:{0}}} {{2:{0}}} {{3:{0}}}\n".format(xyz_fmt)
        fmt2_str = "  {{0:d}}[{{1:d}}] {{2:{0}}} {{3:{0}}} {{4:{0}}}\n".format(xyz_fmt)

        for ia in geometry:
            Z = geometry.atoms[ia].Z
            no = geometry.atoms[ia].no
            if no == 1:
                self._write(fmt1_str.format(Z, *geometry.xyz[ia, :]))
            else:
                self._write(fmt2_str.format(Z, no, *geometry.xyz[ia, :]))

        self._write("end atoms\n")

    @wrap_filterwarnings("ignore", category=SparseEfficiencyWarning)
    @sile_fh_open()
    def write_hamiltonian(
        self, H: Hamiltonian, hermitian: bool = True, **kwargs
    ) -> None:
        """Writes the Hamiltonian model to the file

        Writes a Hamiltonian model to the intrinsic Hamiltonian file format.
        The file can be constructed by the implicit force of Hermiticity,
        or without.

        Utilizing the Hermiticity we reduce the file-size by approximately
        50%.

        Parameters
        ----------
        H :
        hermitian :
            whether the stored data is halved using the Hermitian property of the
            Hamiltonian.

        Notes
        -----
        This file format can only be used to write unpolarized spin configurations.
        """
        # We use the upper-triangular form of the Hamiltonian
        # and the overlap matrix for hermitian problems

        geom = H.geometry
        is_orthogonal = H.orthogonal

        if not H.spin.is_unpolarized:
            raise NotImplementedError(
                f"{self!r}.write_hamiltonian can only write "
                "un-polarized Hamiltonians."
            )

        # First write the geometry
        self.write_geometry(geom, **kwargs)

        # We default to the advanced layout if we have more than one
        # orbital on any one atom
        advanced = kwargs.get(
            "advanced", np.any(np.array([a.no for a in geom.atoms.atom], np.int32) > 1)
        )

        fmt = kwargs.get("fmt", "g")
        if advanced:
            fmt1_str = " {{0:d}}[{{1:d}}] {{2:d}}[{{3:d}}] {{4:{0}}}\n".format(fmt)
            fmt2_str = (
                " {{0:d}}[{{1:d}}] {{2:d}}[{{3:d}}] {{4:{0}}} {{5:{0}}}\n".format(fmt)
            )
        else:
            fmt1_str = f" {{0:d}} {{1:d}} {{2:{fmt}}}\n"
            fmt2_str = " {{0:d}} {{1:d}} {{2:{0}}} {{3:{0}}}\n".format(fmt)

        # We currently force the model to be finalized
        # before we can write it
        # This should be easily circumvented
        # TODO more spin configurations
        N = len(H)
        h = H.tocsr(0)
        if not is_orthogonal:
            S = H.tocsr(H.S_idx)

        # If the model is Hermitian we can
        # do with writing out half the entries
        if hermitian:
            herm_acc = kwargs.get("herm_acc", 1e-6)

            h_ht = H - H.transpose(conjugate=True, spin=True)
            amax = np.abs(h_ht._csr._D).max()
            if amax > herm_acc:
                warn(
                    f"{self!r}.write_hamiltonian could not assert the matrix to be hermitian "
                    "within the accuracy required ({amax})."
                )
                hermitian = False
            del h_ht

        # numpy arrays are not pickable, so we convert to a tuple of ints
        # to ensure we can use hash ops on the set.
        def conv2int_tuple(sc_off):
            return tuple(map(int, sc_off))

        # This small set determines which supercell connections
        # we write. In case we are writing the Hermitian values
        # We'll only write half + 1 of the supercells!
        write_isc = set(map(conv2int_tuple, geom.lattice.sc_off))

        if hermitian:

            # Remove half of the supercell connections
            # We simply retain a list of connections we wish to write
            write_isc = set()

            # binary priority
            priority = np.array([4, 2, 1])

            def choice(isc):
                nonlocal priority
                return sum(priority[isc > 0])

            for i, isc in enumerate(geom.lattice.sc_off):
                # Select the isc which has the most positive numbers
                priority_i = choice(isc)
                priority_j = choice(-isc)

                # We have ^\dagger element, remove it
                jsc = conv2int_tuple(-isc)
                isc = conv2int_tuple(isc)

                # a set won't double add anything
                # so no need to check
                if priority_i > priority_j:
                    write_isc.add(isc)
                else:
                    write_isc.add(jsc)

        # Start writing of the model
        # We loop on all super-cells
        for i, isc in enumerate(geom.lattice.sc_off):

            # check if we should write this sc_off
            if conv2int_tuple(isc) not in write_isc:
                continue

            # check if the connections belong in the primary unit-cell
            is_primary = np.all(isc == 0)

            # Check that we have any contributions in this
            # sub-section, and immediately remove zeros
            Hsub = h[:, i * N : (i + 1) * N]
            Hsub.eliminate_zeros()
            if not is_orthogonal:
                Ssub = S[:, i * N : (i + 1) * N]
                Ssub.eliminate_zeros()

            if hermitian and is_primary:
                # only write upper right of primary unit-cell connections
                Hsub = triu(Hsub, format="csr")
                if not is_orthogonal:
                    Ssub = triu(Ssub, format="csr")

            # Ensure that when h[i,i] == 0, we still write something
            # Since S != 0.
            if is_primary:
                Hsub.setdiag(Hsub.diagonal())

            if Hsub.getnnz() == 0:
                # Skip writing this isc
                continue

            # We have a contribution, write out the information
            self._write("\nbegin matrix {:d} {:d} {:d}\n".format(*isc))
            if advanced:
                for io, jo, hh in ispmatrixd(Hsub):
                    o = np.array([io, jo], np.int32)
                    a = geom.o2a(o)
                    o = o - geom.a2o(a)

                    s = 0.0
                    if not is_orthogonal:
                        s = Ssub[io, jo]
                    elif is_primary:
                        if io == jo:
                            s = 1.0
                    if s == 0.0:
                        self._write(fmt1_str.format(a[0], o[0], a[1], o[1], hh))
                    else:
                        self._write(fmt2_str.format(a[0], o[0], a[1], o[1], hh, s))
            else:
                for io, jo, hh in ispmatrixd(Hsub):
                    s = 0.0
                    if not is_orthogonal:
                        s = Ssub[io, jo]
                    elif is_primary:
                        if io == jo:
                            s = 1.0
                    if s == 0.0:
                        self._write(fmt1_str.format(io, jo, hh))
                    else:
                        self._write(fmt2_str.format(io, jo, hh, s))

            self._write("end matrix {:d} {:d} {:d}\n".format(*isc))


add_sile("ham", hamiltonianSile, case=False, gzip=True)
