"""
Sile object for reading/writing TB in/output
"""
from __future__ import print_function, division

# Import sile objects
from sisl.io.sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl.tb import TightBinding

import numpy as np

__all__ = ['TBSile']


class TBSile(Sile):
    """ Tight-binding file object """

    def read_geom(self):
        """ Reading a geometry in regular TB format """
        if not hasattr(self, 'fh'):
            # The file-handle has not been opened
            with self:
                return self.read_geom()

        cell = np.zeros([3, 3], np.float64)
        Z = []
        xyz = []

        nsc = np.zeros([3], np.int32)

        def Z2no(i, no):
            try:
                # pure atomic number
                return int(i), no
            except:
                # both atomic number and no
                j = i.replace('[', ' ').replace(']', ' ').split()
                return int(j[0]), int(j[1])

        # The format of the geometry file is
        keys = ['atoms', 'cell', 'supercell', 'nsc']
        for _ in range(len(keys)):
            f, l = self.step_to(keys, case=False)
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
                    except:
                        no = 1
                    z, no = Z2no(ls[0], no)
                    Z.append({'Z': z, 'orbs': no})
                    xyz.append([float(f) for f in ls[1:4]])
                    l = self.readline()
                xyz = np.array(xyz, np.float64)
                xyz.shape = (-1, 3)
                self.readline()  # step past the block

        # Return the geometry
        # Create list of atoms
        geom = Geometry(xyz, atoms=Atom[Z], sc=SuperCell(cell, nsc))

        return geom

    def read_tb(self, hermitian=True, dtype=np.float64, **kwargs):
        """ Reads a tight-binding model (including the geometry)

        Reads the tight-binding model
        """
        if not hasattr(self, 'fh'):
            # The file-handle has not been opened
            with self:
                return self.read_tb(hermitian=hermitian, dtype=dtype, **kwargs)

        # Read the geometry in this file
        geom = self.read_geom()

        # Rewind to ensure we can read the entire matrix structure
        self.fh.seek(0)

        # With the geometry in place we can read in the entire matrix
        # Create a new sparse matrix
        from scipy.sparse import lil_matrix
        H = lil_matrix((geom.no, geom.no_s), dtype=dtype)
        S = lil_matrix((geom.no, geom.no_s), dtype=dtype)

        def i2o(geom, i):
            try:
                # pure orbital
                return int(i)
            except:
                # ia[o]
                # atom ia and the orbital o
                j = i.replace('[', ' ').replace(']', ' ').split()
                return geom.a2o(int(j[0])) + int(j[1])

        # Start reading in the supercell
        while True:
            found, l = self.step_to('matrix')
            if not found:
                break

            # Get supercell
            ls = l.split()
            try:
                isc = np.array([int(ls[i]) for i in range(2, 5)], np.int32)
            except:
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
                except:
                    s = 0.
                H[jo, io + off1] = h
                S[jo, io + off1] = s
                if hermitian:
                    S[io, jo + off2] = s
                    H[io, jo + off2] = h
                l = self.readline()

        return TightBinding.sp2tb(geom, H, S)

    def write_geom(self, geom, **kwargs):
        """
        Writes the geometry to the output file

        Parameters
        ----------
        geom: Geometry
              The geometry we wish to write
        """
        if not hasattr(self, 'fh'):
            # The file-handle has not been opened
            with self as fh:
                fh.write_geom(geom, **kwargs)
            return

        # The format of the geometry file is
        # for now, pretty stringent
        # Get cell_fmt
        cell_fmt = '.5f'
        if 'fmt' in kwargs:
            cell_fmt = kwargs['fmt']
        if 'cell_fmt' in kwargs:
            cell_fmt = kwargs['cell_fmt']
        xyz_fmt = '.4e'
        if 'fmt' in kwargs:
            xyz_fmt = kwargs['fmt']
        if 'xyz_fmt' in kwargs:
            xyz_fmt = kwargs['xyz_fmt']

        self._write('begin cell\n')
        # Write the cell
        fmt_str = '  {{0:{0}}} {{1:{0}}} {{2:{0}}}\n'.format(cell_fmt)
        for i in range(3):
            self._write(fmt_str.format(*geom.cell[i, :]))
        self._write('end cell\n')

        # Write number of super cells in each direction
        self._write('\nsupercell {0:d} {1:d} {2:d}\n'.format(*geom.nsc))

        # Write all atomic positions along with the specie type
        self._write('\nbegin atoms\n')
        fmt1_str = '  {{0:d}} {{1:{0}}} {{2:{0}}} {{3:{0}}}\n'.format(xyz_fmt)
        fmt2_str = '  {{0:d}}[{{1:d}}] {{2:{0}}} {{3:{0}}} {{4:{0}}}\n'.format(
            xyz_fmt)

        for ia in geom:
            Z = geom.atoms[ia].Z
            no = geom.atoms[ia].orbs
            if no == 1:
                self._write(fmt1_str.format(Z, *geom.xyz[ia, :]))
            else:
                self._write(fmt2_str.format(Z, no, *geom.xyz[ia, :]))

        self._write('end atoms\n')

    def write_tb(self, tb, hermitian=True, **kwargs):
        """ Writes the tight-binding model to the file

        Writes a tight-binding model to the tight-binding file.
        The file can be constructed by the implict force of Hermiticity,
        or without.

        Utilizing the Hermiticity we reduce the file-size by approximately
        50%.

        Parameters
        ----------
        tb : `TightBinding` model
        hermitian : boolean=True
            whether the stored data is halved using the Hermitian property

        """
        tb.finalize()

        if not hasattr(self, 'fh'):
            # The file-handle has not been opened
            with self:
                return self.write_tb(tb, hermitian=hermitian, **kwargs)

        # We use the upper-triangular form of the Hamiltonian
        # and the overlap matrix
        if hermitian:
            from scipy.sparse import triu

        geom = tb.geom

        # First write the geometry
        self.write_geom(geom)

        # We default to the advanced layuot if we have more than one
        # orbital on any one atom
        advanced = kwargs.get('advanced', np.any(
            np.array([a.orbs for a in tb.atoms], np.int32) > 1))

        fmt = kwargs.get('fmt', 'g')
        if advanced:
            fmt1_str = ' {{0:d}}[{{1:d}}] {{2:d}}[{{3:d}}] {{4:{0}}}\n'.format(
                fmt)
            fmt2_str = ' {{0:d}}[{{1:d}}] {{2:d}}[{{3:d}}] {{4:{0}}} {{5:{0}}}\n'.format(
                fmt)
        else:
            fmt1_str = ' {{0:d}} {{1:d}} {{2:{0}}}\n'.format(fmt)
            fmt2_str = ' {{0:d}} {{1:d}} {{2:{0}}} {{3:{0}}}\n'.format(fmt)

        # We currently force the model to be finalized
        # before we can write it
        # This should be easily circumvented
        H, S = tb.tocsr()
        # If the model is Hermitian we can
        # do with writing out half the entries
        if hermitian:
            herm_acc = kwargs.get('herm_acc', 1e-6)
            # We check whether it is Hermitian
            for i, isc in enumerate(geom.sc.sc_off):
                oi = i * geom.no
                oj = geom.sc_index(-isc) * geom.no
                # get the difference between the ^\dagger elements
                diff = H[:, oi:oi + geom.no] - \
                    H[:, oj:oj + geom.no].transpose()
                diff.eliminate_zeros()
                diff = diff.tocoo()
                if np.any(np.abs(diff.data) > herm_acc):
                    amax = np.amax(np.abs(diff.data))
                    warnings.warn(
                        'The model could not be asserted to be Hermitian within the accuracy required ({0}).'.format(amax),
                        UserWarning)
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
                        H[j, o:o + geom.no] = 0.
                        H.eliminate_zeros()
                        S[j, o:o + geom.no] = 0.
                        S.eliminate_zeros()
            o = geom.sc_index(np.zeros([3], np.int32))
            # Get upper-triangular matrix of the unit-cell H and S
            ut = triu(H[:, o:o + geom.no], k=0).tocsr()
            for j in range(geom.no):
                H[j, o:o + geom.no] = 0.
                H[j, o:o + geom.no] = ut[j, :]
                H.eliminate_zeros()
            ut = triu(S[:, o:o + geom.no], k=0).tocsr()
            for j in range(geom.no):
                S[j, o:o + geom.no] = 0.
                S[j, o:o + geom.no] = ut[j, :]
                S.eliminate_zeros()
            del ut

            # Ensure that S and H have the same sparsity pattern
            S = S.tocoo()
            for jo, io, s in zip(S.row, S.col, S.data):
                H[jo, io] = H[jo, io]
            S = S.tocsr()

        # Start writing of the model
        # We loop on all super-cells
        for i, isc in enumerate(geom.sc.sc_off):
            # Check that we have any contributions in this
            # sub-section
            Hsub = H[:, i * geom.no:(i + 1) * geom.no].tocoo()
            Ssub = S[:, i * geom.no:(i + 1) * geom.no]
            if Hsub.getnnz() == 0:
                continue
            # We have a contribution, write out the information
            self._write('\nbegin matrix {0:d} {1:d} {2:d}\n'.format(*isc))
            if advanced:
                for jo, io, h in zip(Hsub.row, Hsub.col, Hsub.data):
                    s = Ssub[jo, io]
                    o = np.array([jo, io], np.int32)
                    a = geom.o2a(o)
                    o = o - geom.a2o(a)
                    if s == 0.:
                        self._write(fmt1_str.format(a[0], o[0], a[1], o[1], h))
                    else:
                        self._write(
                            fmt2_str.format(
                                a[0], o[0], a[1], o[1], h, s))
            else:
                for jo, io, h in zip(Hsub.row, Hsub.col, Hsub.data):
                    s = Ssub[jo, io]
                    if s == 0.:
                        self._write(fmt1_str.format(jo, io, h))
                    else:
                        self._write(fmt2_str.format(jo, io, h, s))
            self._write('end matrix {0:d} {1:d} {2:d}\n'.format(*isc))


if __name__ == "__main__":
    # Create geometry
    alat = 3.57
    dist = alat * 3. ** .5 / 4
    C = Atom(Z=6, R=dist * 1.01, orbs=1)
    geom = Geometry(np.array([[0, 0, 0], [1, 1, 1]], np.float64) * alat / 4,
                    atoms=C, sc=SuperCell(np.array([[0, 1, 1],
                                                    [1, 0, 1],
                                                    [1, 1, 0]], np.float64) * alat / 2))
    # Write stuff
    print('Writing TBSile')
    geom.write(TBSile('diamond.tb', 'w'))
    print('Reading TBSile')
    geomr = TBSile('diamond.tb', 'r').read_geom()
    print(geomr)
    print(geomr.cell)
    print(geomr.xyz)

    # Create the tight-binding model
    geom.set_supercell([3, 3, 3])
    tb = TightBinding(geom, nc=1)
    for ia in tb.geom:
        idx = tb.close(ia, dR=(0.1, dist * 1.01))
        tb[ia, idx[0]] = (0., 1.)
        tb[ia, idx[1]] = (-.2, 0.)
    print('Before', tb.tocsr()[0])
    TBSile('diamond.tb', 'w').write_tb(tb)
    tbr = TBSile('diamond.tb', 'r').read_tb()
    print('After', tbr.tocsr()[0])

    print('Before', tb.tocsr()[0])
    TBSile('diamondh.tb', 'w').write_tb(tb, hermitian=False)
    tbr = TBSile('diamondh.tb', 'r').read_tb(hermitian=False)
    print('After', tbr.tocsr()[0])
