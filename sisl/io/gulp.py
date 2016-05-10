"""
Sile object for reading/writing GULP in/output
"""
from __future__ import print_function

# Import sile objects
from sisl.io.sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl.tb import PhononTightBinding

import numpy as np

__all__ = ['GULPSile']


class GULPSile(Sile):
    """ GULP file object """

    def _setup(self):
        """ Setup `GULPSile` after initialization """

        self._keys = {
            'sc': 'Final Cartesian lattice vectors',
            'geom': 'Final fractional coordinates',
            'dyn': 'Real Dynamical matrix',
        }

    def set_key(self, segment, key):
        """ Sets the segment lookup key """
        if key is not None:
            self._keys[segment] = key

    def set_sc_key(self, key):
        """ Overwrites internal key lookup value for the cell vectors """
        self.set_key('sc', key)

    def read_sc(self, key=None):
        """ Reads a `SuperCell` and creates the GULP cell """
        self.set_sc_key(key)
        if not hasattr(self, 'fh'):
            # The file-handle has not been opened
            with self:
                return self.read_sc(key)

        f, _ = self.step_to(self._keys['sc'])
        if not f:
            raise ValueError(
                ('GULPSile tries to lookup the SuperCell vectors '
                 'using key "' + self._keys['sc'] + '". \n'
                 'Use ".set_sc_key(...)" to search for different name.\n'
                 'This could not be found found in file: "' + self.file + '".'))

        # skip 1 line
        self.readline()
        cell = np.empty([3, 3], np.float64)
        for i in range(3):
            l = self.readline().split()
            cell[i, 0] = float(l[0])
            cell[i, 1] = float(l[1])
            cell[i, 2] = float(l[2])

        return SuperCell(cell)

    def set_geom_key(self, key):
        """ Overwrites internal key lookup value for the geometry vectors """
        self.set_key('geom', key)

    def read_geom(self, key=None):
        """ Reads a geometry and creates the GULP dynamical geometry """
        self.set_geom_key(key)

        if not hasattr(self, 'fh'):
            # The file-handle has not been opened
            with self:
                return self.read_geom(key)

        # create default supercell
        sc = SuperCell([1, 1, 1])

        for sc_geom in range(2):
            # Step to either the geometry or
            f, ki, _ = self.step_either([self._keys['sc'], self._keys['geom']])
            if not f and ki == 0:
                raise ValueError(
                    ('GULPSile tries to lookup the SuperCell vectors '
                     'using key "' + self._keys['sc'] + '". \n'
                     'Use ".set_sc_key(...)" to search for different name.\n'
                     'This could not be found found in file: "' + self.file + '".'))
            elif f and ki == 0:
                # supercell
                self.readline()
                cell = np.zeros([3, 3], np.float64)
                for i in range(3):
                    l = self.readline().split()
                    cell[i, 0] = float(l[0])
                    cell[i, 1] = float(l[1])
                    cell[i, 2] = float(l[2])
                sc = SuperCell(cell)

            elif not f and ki == 1:
                raise ValueError(
                    ('GULPSile tries to lookup the Geometry coordinates '
                     'using key "' + self._keys['geom'] + '". \n'
                     'Use ".set_geom_key(...)" to search for different name.\n'
                     'This could not be found found in file: "' + self.file + '".'))
            elif f and ki == 1:

                # We skip 5 lines
                for i in range(5):
                    self.readline()

                Z = []
                xyz = []
                while True:
                    l = self.readline()
                    if l[0] == '-':
                        break

                    ls = l.split()
                    Z.append({'Z': ls[1], 'orbs': 3})
                    xyz.append(map(float, ls[3:6]))

                # Convert to array and correct size
                xyz = np.array(xyz, np.float64)
                xyz.shape = (-1, 3)

                if len(Z) == 0 or len(xyz) == 0:
                    raise ValueError(
                        'Could not read in cell information and/or coordinates')

            elif not f:
                # could not find either cell or geometry
                raise ValueError(
                    ('GULPSile tries to lookup the SuperCell or Geometry.\n'
                     'None succeeded, ensure file has correct format.\n'
                     'This could not be found found in file: "' + self.file + '".'))

        # as the cell may be read in after the geometry we have
        # to wait until here to convert from fractional
        if 'fractional' in self._keys['geom'].lower():
            # Correct for fractional coordinates
            xyz[:, 0] *= np.sum(sc.cell[:, 0])
            xyz[:, 1] *= np.sum(sc.cell[:, 1])
            xyz[:, 2] *= np.sum(sc.cell[:, 2])

        # Return the geometry
        return Geometry(xyz, atoms=Atom[Z], sc=sc)

    def set_dyn_key(self, key):
        """ Overwrites internal key lookup value for the dynamical matrix vectors """
        self.set_key('dyn', key)

    set_tb_key = set_dyn_key

    def read_tb(self, **kwargs):
        """ Returns a GULP tight-binding model for the output of GULP """
        if not hasattr(self, 'fh'):
            # The file-handle has not been opened
            with self:
                return self.read_tb(**kwargs)

        dtype = kwargs.get('dtype', np.float64)

        geom = self.read_geom(**kwargs)

        # Easier for creation of the sparsity pattern
        from scipy.sparse import lil_matrix, diags

        dyn = lil_matrix((geom.no, geom.no), dtype=dtype)

        f, _ = self.step_to(self._keys['dyn'])
        if not f:
            raise ValueError(
                ('GULPSile tries to lookup the Dynamical matrix '
                 'using key "' + self._keys['dyn'] + '". '
                 'Use .set_dyn_key(...) to search for different name.'
                 'This could not be found found in file: "' + self.file + '".'))

        # skip 1 line
        self.readline()

        # default range
        no = geom.no
        dat = np.empty([no], dtype=dtype)
        i = 0
        j = 0
        while True:
            l = self.readline().strip()
            if len(l) == 0:
                break

            # convert to float list
            ls = map(float, l.split())

            if j + 12 <= no:
                # Here the full line can fit for the same row
                dat[j:j + 12] = ls[:12]
                j += 12
                if j >= no:
                    dyn[i, :] = dat[:]
                    # step row
                    i += 1
                    # reset column
                    j = 0
                    if i >= no:
                        break
            else:
                # add the values (12 values == 3*4)
                # for atoms on each line
                for k in range(4):
                    dat[j:j + 3] = ls[k * 3:(k + 1) * 3]

                    j += 3
                    if j >= no:
                        dyn[i, :] = dat[:]
                        i += 1
                        j = 0
                        if i >= no:
                            break

        # clean-up for memory
        del dat

        # Convert to COO format
        dyn = dyn.tocoo()

        # Convert the GULP data to standard units
        dyn.data[:] *= (521.469 * 1.23981e-4) ** 2

        # Create "fake" overlap matrix
        ones = np.ones(dyn.shape[0], dtype=dtype)
        S = diags(ones, 0, shape=dyn.shape)
        S = S.tocsr()
        del ones

        return PhononTightBinding.sp2tb(geom, dyn, S)


if __name__ == "__main__":
    pass
