"""
Sile object for reading TBtrans AV binary files
"""
from __future__ import print_function, division

import warnings
import numpy as np
import itertools
from numbers import Integral
# The sparse matrix for the orbital/bond currents
from scipy.sparse import csr_matrix, lil_matrix

# Import sile objects
from sisl.utils import *
from ..sile import *
from .sile import SileCDFSIESTA
from .tbtrans import tbtncSileSiesta, phtncSileSiesta


# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl._help import _str
from sisl._help import _range as range
from sisl.units.siesta import unit_convert

__all__ = ['tbtavncSileSiesta', 'phtavncSileSiesta']

Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')
Ry2K = unit_convert('Ry', 'K')
eV2Ry = unit_convert('eV', 'Ry')


class tbtavncSileSiesta(tbtncSileSiesta):
    """ TBtrans average file object 

    This `SileCDF` implements the writing of the TBtrans output `*.TBT.AV.nc` sile which contains
    the k-averaged quantities related to the NEGF code TBtrans.

    Although the TBtrans code is in fortran and the resulting NetCDF file variables
    are in fortran indexing (1-based), everything is returned as Python indexing (0-based)
    when scripting.

    This is vital when using this `Sile`.

    Note that when using the command-line utility ``sdata`` the indexing is fortran based 
    because the data handlers are meant for _easy_ use.
    """
    _trans_type = 'TBT'
    _k_avg = True

    @property
    def nkpt(self):
        """ Always return 1, this is to signal other routines """
        return 1

    def write_tbtav(self, *args, **kwargs):
        """ Wrapper for writing the k-averaged TBT.AV.nc file. 

        This write _requires_ the TBT.nc `Sile` object passed as the first argument,
        or as the keyword `from=tbt` argument.

        Parameters
        ----------
        from : ``tbtncSileSiesta``
          the TBT.nc file object that has the k-sampled quantities.
        """

        if 'from' in kwargs:
            tbt = kwargs['from']
        elif len(args) > 0:
            tbt = args[0]
        else:
            raise ValueError("tbtncSileSiesta has not been passed to write the averaged file")

        if not isinstance(tbt, tbtncSileSiesta):
            raise ValueError('first argument of tbtavncSileSiesta.write *must* be a tbtncSileSiesta object')

        # Notify if the object is not in write mode.
        sile_raise_write(self)

        head = self

        def copy_attr(f, t):
            t.setncatts({att: f.getncattr(att) for att in f.ncattrs()})

        # Retrieve k-weights
        nkpt = len(tbt.dimensions['nkpt'])
        wkpt = np.asarray(tbt.variables['wkpt'][:], np.float64)

        # First copy and re-create all entries in the output file
        for dvg in tbt:
            # Iterate all:
            #  root,
            #  dimensions,
            #  variables
            #  sub-groups, (start over again)

            # Root group
            if tbt.isDataset(dvg):
                # Copy attributes and continue
                copy_attr(dvg, self)
                continue

            # Ensure the group exists
            if tbt.isGroup(dvg):
                grp = self.createGroup(dvg.path)
                copy_attr(dvg, grp)
                continue

            # Ensure the group exists... (the above case handles groups)
            grp = self.createGroup(dvg.group().path)

            if tbt.isDimension(dvg):

                # In case the dimension is the k-point one
                # we remove that dimension
                if 'nkpt' == dvg.name:
                    continue

                # Simply re-create the dimension
                if dvg.isunlimited():
                    grp.createDimension(dvg.name, None)
                else:
                    grp.createDimension(dvg.name, len(dvg))

                continue

            # It *must* be a variable now

            # Quickly skip the k-point variable and the weights
            if dvg.name in ['kpt', 'wkpt']:
                continue

            # Down-scale the k-point dimension
            if 'nkpt' in dvg.dimensions:
                # Remove that dimension
                dims = list(dvg.dimensions)
                # Create slice
                idx = dims.index('nkpt')
                dims.pop(idx)
                dims = tuple(dims)
                has_kpt = True

            else:
                dims = dvg.dimensions[:]
                has_kpt = False

            v = grp.createVariable(dvg.name, dvg.dtype,
                                   dimensions=dims,
                                   **dvg.filters())

            # Copy attributes
            copy_attr(dvg, v)

            # Copy values
            if has_kpt:
                # Instead of constantly reading-writing to disk
                # (if buffer is too small)
                # we create a temporary array to hold the averaged
                # quantities.
                # This should only be faster for very large variables
                if idx == 0:
                    dat = np.asarray(dvg[0][:] * wkpt[0])
                    for k in range(1, nkpt):
                        dat += dvg[k][:] * wkpt[k]
                    v[:] = dat[:]
                else:
                    for slc in iter_shape(dvg.shape[:idx]):
                        dat = np.asarray(dvg[slc][0][:] * wkpt[0])
                        for k in range(1, nkpt):
                            dat += dvg[slc][k][:] * wkpt[k]
                        v[slc][:] = dat[:]
                del dat
            else:
                v[:] = dvg[:]

        # Update the source attribute to signal the originating file
        self.setncattr('source', 'k-average of: ' + tbt._file)
        self.sync()


add_sile('TBT.AV.nc', tbtavncSileSiesta)
# Add spin-dependent files
add_sile('TBT_DN.AV.nc', tbtavncSileSiesta)
add_sile('TBT_UP.AV.nc', tbtavncSileSiesta)


class phtavncSileSiesta(tbtavncSileSiesta):
    """ PHtrans file object """
    _trans_type = 'PHT'
    pass

add_sile('PHT.AV.nc', phtavncSileSiesta)
