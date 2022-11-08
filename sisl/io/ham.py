# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import triu
from scipy.sparse import SparseEfficiencyWarning

# Import sile objects
from .sile import *

from sisl._internal import set_module
from sisl.messages import warn
from sisl import Geometry, Atom, SuperCell
from sisl.sparse import ispmatrix, ispmatrixd
from sisl.physics import Hamiltonian
from sisl._help import wrap_filterwarnings
from sisl import _array as _a


__all__ = ['hamiltonianSile']


@set_module("sisl.io")
class hamiltonianSile(Sile):
    """ Hamiltonian file object """

    @sile_fh_open()
    def read_geometry(self):
        """ Reading a geometry in regular Hamiltonian format """

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
                j = i.replace('[', ' ').replace(']', ' ').split()
                return int(j[0]), int(j[1])

        # The format of the geometry file is
        keys = ['atoms', 'cell', 'supercell', 'nsc']
        for _ in range(len(keys)):
            _, l = self.step_to(keys, case=False)
            l = l.strip()
            if 'supercell' in l.lower() or 'nsc' in l.lower():
                # We have everything in one line
                l = l.split()[1:]
                for i in range(3):
                    nsc[i] = int(l[i])
            elif 'cell' in l.lower():
                if 'begin' in l.lower():
                    for i in range(3):
                        l = self.readline().split()
                        cell[i, 0] = float(l[0])
                        cell[i, 1] = float(l[1])
                        cell[i, 2] = float(l[2])
                    self.readline()  # step past the block
                else:
                    # We have everything in one line
                    l = l.split()[1:]
                    for i in range(3):
                        cell[i, i] = float(l[i])
                    # TODO incorporate rotations
            elif 'atoms' in l.lower():
                l = self.readline()
                while not l.startswith('end'):
                    ls = l.split()
                    try:
                        no = int(ls[4])
                    except Exception:
                        no = 1
                    z, no = Z2no(ls[0], no)
                    Z.append({'Z': z, 'orbital': [-1. for _ in range(no)]})
                    xyz.append([float(f) for f in ls[1:4]])
                    l = self.readline()
                xyz = _a.arrayd(xyz)
                xyz.shape = (-1, 3)
                self.readline()  # step past the block

        # Create geometry with associated supercell and atoms
        geom = Geometry(xyz, atoms=Atom[Z], sc=SuperCell(cell, nsc))

        return geom

    @sile_fh_open()
    def read_hamiltonian(self, hermitian=True, dtype=np.float64, **kwargs):
        """ Reads a Hamiltonian (including the geometry)

        Reads the Hamiltonian model
        """
        # Read the geometry in this file
        geom = self.read_geometry()

        # Rewind to ensure we can read the entire matrix structure
        self.fh.seek(0)

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
                j = i.replace('[', ' ').replace(']', ' ').split()
                return geom.a2o(int(j[0])) + int(j[1])

        # Start reading in the supercell
        while True:
            found, l = self.step_to('matrix', allow_reread=False)
            if not found:
                break

            # Get supercell
            ls = l.split()
            try:
                isc = np.array([int(ls[i]) for i in range(2, 5)], np.int32)
            except Exception:
                isc = np.array([0, 0, 0], np.int32)

            off1 = geom.sc_index(isc) * geom.no
            off2 = geom.sc_index(-isc) * geom.no
            l = self.readline()
            while not l.startswith('end'):
                ls = l.split()
                jo = i2o(geom, ls[0])
                io = i2o(geom, ls[1])
                h = float(ls[2])
                try:
                    s = float(ls[3])
                except Exception:
                    s = 0.
                H[jo, io + off1] = h
                S[jo, io + off1] = s
                if hermitian:
                    H[io, jo + off2] = h
                    S[io, jo + off2] = s
                l = self.readline()

        return Hamiltonian.fromsp(geom, H, S)

    @sile_fh_open()
    def write_geometry(self, geometry, fmt='.8f', **kwargs):
        """
        Writes the geometry to the output file

        Parameters
        ----------
        geometry : Geometry
              The geometry we wish to write
        """

        # The format of the geometry file is
        # for now, pretty stringent
        # Get cell_fmt
        cell_fmt = fmt
        if 'cell_fmt' in kwargs:
            cell_fmt = kwargs['cell_fmt']
        xyz_fmt = fmt

        self._write('begin cell\n')
        # Write the cell
        fmt_str = '  {{0:{0}}} {{1:{0}}} {{2:{0}}}\n'.format(cell_fmt)
        for i in range(3):
            self._write(fmt_str.format(*geometry.cell[i, :]))
        self._write('end cell\n')

        # Write number of super cells in each direction
        self._write('\nsupercell {:d} {:d} {:d}\n'.format(*geometry.nsc))

        # Write all atomic positions along with the specie type
        self._write('\nbegin atoms\n')
        fmt1_str = '  {{0:d}} {{1:{0}}} {{2:{0}}} {{3:{0}}}\n'.format(xyz_fmt)
        fmt2_str = '  {{0:d}}[{{1:d}}] {{2:{0}}} {{3:{0}}} {{4:{0}}}\n'.format(
            xyz_fmt)

        for ia in geometry:
            Z = geometry.atoms[ia].Z
            no = geometry.atoms[ia].no
            if no == 1:
                self._write(fmt1_str.format(Z, *geometry.xyz[ia, :]))
            else:
                self._write(fmt2_str.format(Z, no, *geometry.xyz[ia, :]))

        self._write('end atoms\n')

    @wrap_filterwarnings("ignore", category=SparseEfficiencyWarning)
    @sile_fh_open()
    def write_hamiltonian(self, H, hermitian=True, **kwargs):
        """ Writes the Hamiltonian model to the file

        Writes a Hamiltonian model to the intrinsic Hamiltonian file format.
        The file can be constructed by the implict force of Hermiticity,
        or without.

        Utilizing the Hermiticity we reduce the file-size by approximately
        50%.

        Parameters
        ----------
        H : `Hamiltonian` model
        hermitian : boolean=True
            whether the stored data is halved using the Hermitian property
        """
        # We use the upper-triangular form of the Hamiltonian
        # and the overlap matrix for hermitian problems

        geom = H.geometry

        # First write the geometry
        self.write_geometry(geom, **kwargs)

        # We default to the advanced layuot if we have more than one
        # orbital on any one atom
        advanced = kwargs.get('advanced', np.any(
            np.array([a.no for a in geom.atoms.atom], np.int32) > 1))

        fmt = kwargs.get('fmt', 'g')
        if advanced:
            fmt1_str = ' {{0:d}}[{{1:d}}] {{2:d}}[{{3:d}}] {{4:{0}}}\n'.format(
                fmt)
            fmt2_str = ' {{0:d}}[{{1:d}}] {{2:d}}[{{3:d}}] {{4:{0}}} {{5:{0}}}\n'.format(
                fmt)
        else:
            fmt1_str = f' {{0:d}} {{1:d}} {{2:{fmt}}}\n'
            fmt2_str = ' {{0:d}} {{1:d}} {{2:{0}}} {{3:{0}}}\n'.format(fmt)

        # We currently force the model to be finalized
        # before we can write it
        # This should be easily circumvented
        h = H.tocsr(0)
        if not H.orthogonal:
            S = H.tocsr(H.S_idx)

        # If the model is Hermitian we can
        # do with writing out half the entries
        if hermitian:
            herm_acc = kwargs.get('herm_acc', 1e-6)
            # We check whether it is Hermitian (not S)
            for i, isc in enumerate(geom.sc.sc_off):
                oi = i * geom.no
                oj = geom.sc_index(-isc) * geom.no
                # get the difference between the ^\dagger elements
                diff = h[:, oi:oi + geom.no] - \
                    h[:, oj:oj + geom.no].transpose()
                diff.eliminate_zeros()
                if np.any(np.abs(diff.data) > herm_acc):
                    amax = np.amax(np.abs(diff.data))
                    warn(f"The model could not be asserted to be Hermitian "
                         "within the accuracy required ({amax}).")
                    hermitian = False
                del diff

        if hermitian:
            # Remove all double stuff
            for i, isc in enumerate(geom.sc.sc_off):
                if np.any(isc < 0):
                    # We have ^\dagger element, remove it
                    o = i * geom.no
                    # Ensure that we remove all nullified quantities
                    # (setting elements to zero will add them internally
                    #  :(, hence this actually constructs the full matrix
                    # Therefore we do it on a row basis, to limit memory
                    # requirements
                    for j in range(geom.no):
                        h[j, o:o + geom.no] = 0.
                        h.eliminate_zeros()
                        if not H.orthogonal:
                            S[j, o:o + geom.no] = 0.
                            S.eliminate_zeros()
            o = geom.sc_index(np.zeros([3], np.int32))
            # Get upper-triangular matrix of the unit-cell H and S
            ut = triu(h[:, o:o + geom.no], k=0).tocsr()
            for j in range(geom.no):
                h[j, o:o + geom.no] = 0.
                h[j, o:o + geom.no] = ut[j, :]
                h.eliminate_zeros()
            if not H.orthogonal:
                ut = triu(S[:, o:o + geom.no], k=0).tocsr()
                for j in range(geom.no):
                    S[j, o:o + geom.no] = 0.
                    S[j, o:o + geom.no] = ut[j, :]
                    S.eliminate_zeros()

                # Ensure that S and H have the same sparsity pattern
                for jo, io in ispmatrix(S):
                    h[jo, io] = h[jo, io]

            del ut

        # Start writing of the model
        # We loop on all super-cells
        for i, isc in enumerate(geom.sc.sc_off):
            # Check that we have any contributions in this
            # sub-section
            Hsub = h[:, i * geom.no:(i + 1) * geom.no]
            if not H.orthogonal:
                Ssub = S[:, i * geom.no:(i + 1) * geom.no]
            if Hsub.getnnz() == 0:
                continue
            # We have a contribution, write out the information
            self._write('\nbegin matrix {:d} {:d} {:d}\n'.format(*isc))
            if advanced:
                for jo, io, hh in ispmatrixd(Hsub):
                    o = np.array([jo, io], np.int32)
                    a = geom.o2a(o)
                    o = o - geom.a2o(a)
                    if not H.orthogonal:
                        s = Ssub[jo, io]
                    elif jo == io:
                        s = 1.
                    else:
                        s = 0.
                    if s == 0.:
                        self._write(fmt1_str.format(a[0], o[0], a[1], o[1], hh))
                    else:
                        self._write(
                            fmt2_str.format(
                                a[0], o[0], a[1], o[1], hh, s))
            else:
                for jo, io, hh in ispmatrixd(Hsub):
                    if not H.orthogonal:
                        s = Ssub[jo, io]
                    elif jo == io:
                        s = 1.
                    else:
                        s = 0.
                    if s == 0.:
                        self._write(fmt1_str.format(jo, io, hh))
                    else:
                        self._write(fmt2_str.format(jo, io, hh, s))
            self._write('end matrix {:d} {:d} {:d}\n'.format(*isc))


add_sile('ham', hamiltonianSile, case=False, gzip=True)
