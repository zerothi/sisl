"""
Sile object for reading TBtrans binary files
"""
from __future__ import print_function

import warnings
import numpy as np

# The sparse matrix for the orbital/bond currents
from scipy.sparse import csr_matrix, lil_matrix

# Check against integers
from numbers import Integral

# Import sile objects
from .sile import SileCDFSIESTA
from ..sile import *
from sisl.utils import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl.units.siesta import unit_convert

__all__ = ['tbtncSileSiesta', 'phtncSileSiesta']

__all__ += ['dHncSileSiesta']

Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')


class tbtncSileSiesta(SileCDFSIESTA):
    """ TBtrans file object """
    _trans_type = 'TBT'

    def _get_var(self, name, tree=None):
        """ Local method to get the NetCDF variable """
        if tree is None:
            return self.variables[name]

        g = self
        if isinstance(tree, list):
            for t in tree:
                g = g.groups[t]
        else:
            g = g.groups[tree]

        return g.variables[name]

    @Sile_fh_open
    def _data(self, name, tree=None):
        """ Local method for obtaining the data from the SileCDF.

        This method checks how the file is access, i.e. whether
        data is stored in the object or it should be read consequtively.
        """
        if self._access > 0:
            if name in self.__data:
                return self.__data[name]
        with self:
            return self._get_var(name, tree=tree)[:]

    @Sile_fh_open
    def _data_avg(self, name, tree=None, avg=False):
        """ Local method for obtaining the data from the SileCDF.

        This method checks how the file is access, i.e. whether
        data is stored in the object or it should be read consequtively.
        """
        if self._access > 0:
            if name in self.__data:
                return self.__data[name]

        v = self._get_var(name, tree=tree)
        wkpt = self.wkpt

        # Perform normalization
        orig_shape = v.shape
        if isinstance(avg, bool):
            if avg:
                nk = len(wkpt)
                data = v[0, ...] * wkpt[0]
                for i in range(1, nk):
                    data += v[i, :] * wkpt[i]
                data.shape = orig_shape[1:]
            else:
                data = v[:]

        elif isinstance(avg, Integral):
            data = v[avg, ...] * wkpt[avg]
            data.shape = orig_shape[1:]

        else:
            # We assume avg is some kind of itterable
            data = v[avg[0], ...] * wkpt[avg[0]]
            for i in range(1, len(avg)):
                data += v[avg[i], ...] * wkpt[avg[i]]
            data.shape = orig_shape[1:]

        # Return data
        return data

    
    @Sile_fh_open
    def _data_E(self, name, tree=None, avg=False, E=None):
        """ Local method for obtaining the data from the SileCDF using an E index.

        """
        if E is None:
            return self._data_avg(name, tree, avg)

        # Ensure that it is an index
        iE = self.E2idx(E)

        if self._access > 0:
            raise RuntimeError(
                "data_E is not allowed for access-contained items.")

        v = self._get_var(name, tree=tree)
        wkpt = self.wkpt

        # Perform normalization
        orig_shape = v.shape

        if isinstance(avg, bool):
            if avg:
                nk = len(wkpt)
                data = np.array(v[0, iE, ...]) * wkpt[0]
                for i in range(1, nk):
                    data += v[i, iE, ...] * wkpt[i]
                data.shape = orig_shape[1:]
            else:
                data = np.array(v[:, iE, ...])

        elif isinstance(avg, Integral):
            data = np.array(v[avg, iE, ...]) * wkpt[avg]
            data.shape = orig_shape[1:]

        else:
            # We assume avg is some kind of itterable
            data = v[avg[0], iE, ...] * wkpt[avg[0]]
            for i in range(1, len(avg)):
                data += v[avg[i], iE, ...] * wkpt[avg[i]]
            data.shape = orig_shape[1:]

        # Return data
        return data

    
    def _setup(self):
        """ Setup the special object for data containing """
        self.__data = dict()

        if self._access > 0:

            # Fake double calls
            access = self._access
            self._access = 0

            # There are certain elements which should
            # be minimal on memory but allow for
            # fast access by the object.
            for d in ['cell', 'xa', 'lasto',
                      'a_dev', 'pivot',
                      'kpt', 'wkpt', 'E']:
                self.__data[d] = self._data(d)

            # Create the geometry in the data file
            self.__data['_geom'] = self.read_geom()

            # Reset the access pattern
            self._access = access


    @Sile_fh_open
    def read_sc(self):
        """ Returns `SuperCell` object from a .TBT.nc file """
        cell = np.array(np.copy(self.cell), dtype=np.float64)
        cell.shape = (3, 3)

        return SuperCell(cell)

    
    @Sile_fh_open
    def read_geom(self):
        """ Returns Geometry object from a .TBT.nc file """
        sc = self.read_sc()

        xyz = np.array(np.copy(self.xa), dtype=np.float64)
        xyz.shape = (-1, 3)

        # Create list with correct number of orbitals
        lasto = np.array(np.copy(self.lasto), dtype=np.int32)
        nos = np.append([lasto[0]], np.diff(lasto))
        nos = np.array(nos, np.int32)

        # Default to Hydrogen atom with nos[ia] orbitals
        # This may be counterintuitive but there is no storage of the
        # actual species
        atms = [Atom(Z='H', orbs=o) for o in nos]

        # Create and return geometry object
        geom = Geometry(xyz, atms, sc=sc)

        return geom

    def write_geom(self):
        """ This does not work """
        raise ValueError(self.__class__.__name__ + " can not write a geometry")

    # This class also contains all the important quantities elements of the
    # file.

    @property
    def geom(self):
        """ Returns the associated geometry from the TBT file """
        return self.read_geom()

    @property
    def cell(self):
        """ Unit cell in file """
        return self._data('cell') * Bohr2Ang

    @property
    def na(self):
        """ Returns number of atoms in the cell """
        return int(len(self.dimensions['na_u']))
    na_u = na

    @property
    def no(self):
        """ Returns number of orbitals in the cell """
        return int(len(self.dimensions['no_u']))
    no_u = no

    @property
    def xa(self):
        """ Atomic coordinates in file """
        return self._data('xa') * Bohr2Ang
    xyz = xa

    # Device atoms and other quantities
    @property
    def na_d(self):
        """ Number of atoms in the device region """
        return len(self.dimensions['na_d'])
    na_dev = na_d

    @property
    def a_d(self):
        """ Atomic indices (1-based) of device atoms """
        return self._data('a_dev')
    a_dev = a_d

    @property
    def pivot(self):
        """ Pivot table of device orbitals to obtain input sorting """
        return self._data('pivot')
    pvt = pivot

    @property
    def lasto(self):
        """ Last orbital of corresponding atom """
        return self._data('lasto')

    @property
    def no_d(self):
        """ Number of orbitals in the device region """
        return int(len(self.dimensions['no_d']))

    @property
    def kpt(self):
        """ Sampled k-points in file """
        return self._data('kpt')

    @property
    def wkpt(self):
        """ Weights of k-points in file """
        return self._data('wkpt')

    @property
    def nkpt(self):
        """ Number of k-points in file """
        return len(self.dimensions['nkpt'])

    @property
    def E(self):
        """ Sampled energy-points in file """
        return self._data('E') * Ry2eV

    def Eindex(self, E):
        """ Return the closest energy index corresponding to the energy `E`"""
        if isinstance(E, Integral):
            return E
        return np.abs(self._data('E') - E / Ry2eV).argmin()

    @property
    def ne(self):
        """ Number of energy-points in file """
        return len(self.dimensions['ne'])
    nE = ne

    @property
    @Sile_fh_open
    def elecs(self):
        """ List of electrodes """
        elecs = self.groups.keys()

        # in cases of not calculating all
        # electrode transmissions we must ensure that
        # we add the last one
        var = self.groups[elecs[0]].variables.keys()
        for tvar in var:
            if tvar.endswith('.T'):
                tvar = tvar.split('.')[0]
                if tvar not in elecs:
                    elecs.append(tvar)
        return elecs
    electrodes = elecs
    Electrodes = elecs
    Elecs = elecs

    def chemical_potential(self, elec):
        """ Return the chemical potential associated with the electrode `elec` """
        return self._data('mu', elec)
    mu = chemical_potential

    def electronic_temperature(self, elec):
        """ Return temperature of the electrode electronic distribution """
        return self._data('kT', elec)
    kT = electronic_temperature

    def transmission(self, elec_from, elec_to, avg=True):
        """ Return the transmission from `from` to `to`.

        The transmission between two electrodes may be retrieved
        from the `Sile`.

        Parameters
        ----------
        elec_from: str
           the originating electrode
        elec_to: str
           the absorbing electrode (different from `elec_from`)
        avg: bool (True)
           whether the returned transmission is k-averaged
        """
        if elec_from == elec_to:
            raise ValueError(
                "Supplied elec_from and elec_to must not be the same.")

        return self._data_avg(elec_to + '.T', elec_from, avg=avg)
    T = transmission

    def transmission_eig(self, elec_from, elec_to, avg=True):
        """ Return the transmission eigenvalues from `from` to `to`.

        The transmission eigenvalues between two electrodes may be retrieved
        from the `Sile`.

        Parameters
        ----------
        elec_from: str
           the originating electrode
        elec_to: str
           the absorbing electrode (different from `elec_from`)
        avg: bool (True)
           whether the returned eigenvalues are k-averaged
        """
        if elec_from == elec_to:
            raise ValueError(
                "Supplied elec_from and elec_to must not be the same.")

        return self._data_avg(elec_to + '.T.Eig', elec_from, avg=avg)
    Teig = transmission_eig

    def transmission_bulk(self, elec, avg=True):
        """ Return the bulk transmission in the `elec` electrode

        Parameters
        ----------
        elec: str
           the bulk electrode
        avg: bool (True)
           whether the returned transmission is k-averaged
        """
        return self._data_avg('T', elec, avg=avg)
    Tbulk = transmission_bulk

    def DOS(self, avg=True):
        """ Return the Green function DOS.

        Parameters
        ----------
        avg: bool (True)
           whether the returned DOS is k-averaged
        """
        return self._data_avg('DOS', avg=avg)
    DOS_Gf = DOS

    def ADOS(self, elec, avg=True):
        """ Return the DOS of the spectral function from `elec`.

        Parameters
        ----------
        elec: str
           electrode originating spectral function
        avg: bool (True)
           whether the returned DOS is k-averaged
        """
        return self._data_avg('ADOS', elec, avg=avg)
    DOS_A = ADOS

    def DOS_bulk(self, elec, avg=True):
        """ Return the bulk DOS of `elec`.

        Parameters
        ----------
        elec: str
           electrode where the bulk DOS is returned
        avg: bool (True)
           whether the returned DOS is k-averaged
        """
        return self._data_avg('DOS', elec, avg=avg)
    BulkDOS = DOS_bulk


    def orbital_current(self, elec, E=None, avg=True):
        """ Return the orbital current originating from `elec`.

        This will return a sparse matrix (``scipy.sparse.csr_matrix``).
        The sparse matrix may be interacted with like a normal
        matrix although it enables extremely big matrices.

        Parameters
        ----------
        elec: str
           the electrode of originating electrons
        E: int (None)
           the energy index of the orbital current
           If `None` two objects will be returned, 1) the ``csr_matrix`` of the orbital currents , 2) all the currents (J), you may do:
            >>> J, mat = orbital_current(elec)
            >>> mat.data[:] = J[E,:]
           otherwise it will only return:
            >>> mat.data[:] = J[E,:]
           which is (far) less memory consuming.
        avg: bool (True)
           whether the orbital current returned is k-averaged
        """

        # Get column indices
        col = np.array(self.variables['list_col'][:], np.int32) - 1
        # Create row-pointer
        tmp = np.cumsum(self.variables['n_col'][:])
        size = len(tmp)
        mat_size = (size, size)
        ptr = np.empty([size + 1], np.int32)
        ptr[0] = 0
        ptr[1:] = tmp[:]
        del tmp

        if E is None:
            # Return both the data and the corresponding
            # sparse matrix
            J = self._data_avg('J', elec, avg=avg)
            if len(J.shape) == 2:
                mat = csr_matrix( (J[0, :], col, ptr), shape=mat_size)
            else:
                mat = csr_matrix( (J[0, 0, :], col, ptr), shape=mat_size)
            return mat, J

        else:
            J = self._data_E('J', elec, avg, E)

        return csr_matrix( (J, col, ptr), shape=mat_size)


    def bond_current(self, Jij, sum="+"):
        """ Return the bond-current between atoms (sum of orbital currents)

        Parameters
        ----------
        Jij: ``scipy.sparse.csr_matrix``
           the orbital currents as retrieved from `orbital_current`
        sum: str ("+")
           this value may be "+"/"-"/"all"
           If "+" is supplied only the positive orbital currents are used,
           for "-", only the negatev orbital currents are used,
           else return both.
        """

        # We convert to atomic bond-currents
        J = lil_matrix( (self.na_u, self.na_u), dtype=mat.dtype)

        # Create the iterator across the sparse pattern
        tmp = Jij.tocoo()
        it = np.nditer([self.o2a(tmp.row), self.o2a(tmp.col), tmp.data],
                       flags=['buffered'], op_flags=['readonly'])

        # Perform reduction
        if "+" in sum:
            for ja, ia, d in it:
                if d > 0:
                    J[ja, ia] += d

        elif "-" in sum:
            for ja, ia, d in it:
                if d < 0:
                    J[ja, ia] -= d

        else:
            for ja, ia, d in it:
                J[ja, ia] += d

        # Delete iterator
        del it

        # Rescale to correct magnitude
        J.data[:] *= .5

        # Now we have the bond-currents
        # convert and sort
        mat = J.tocsr()
        mat.sort_indices()

        return mat


    def atom_current(self, Jij, activity=True):
        r""" Return the atom-current of atoms

        Parameters
        ----------
        Jij: ``scipy.sparse.csr_matrix``
           the orbital currents as retrieved from `orbital_current`
        activity: bool (True)
           whether the activity current is returned.
           This is defined using these two equations:

           .. math::
              J_I^{|a|} &=\frac{1}{2} \sum_J \big| \sum_{\nu\in I}\sum_{\mu\in J} J_{\nu\mu} \big|
              J_I^{|o|} &=\frac{1}{2} \sum_J \sum_{\nu\in I}\sum_{\mu\in J} \big| J_{\nu\mu} \big|

           If `activity = False` it returns

           .. math::
              J_I^{|a|}

           and if `activity = True` it returns

           .. math::
              J_I^{\mathcal A} = \sqrt{ J_I^{|a|} J_I^{|o|} }

        """
        
        # Convert to csr format (just ensure it)
        tmp = Jij.tocsr()

        # List of atomic currents
        Ja = np.zeros([self.na_u], np.float64)
        Jo = np.zeros([self.na_u], np.float64)

        # We already know which atoms are the device atoms...
        atom = self.a_dev

        # Create local lasto
        lasto = np.append([0],self.geom.lasto)

        # Faster function calls
        nabs = np.abs
        nsum = np.sum

        # Calculate individual bond-currents between atoms
        for ia in atom:
            for ja in atom:
                
                # we also include ia == ja (that should be zero anyway)
                t = tmp[lasto[ia-1]:lasto[ia],lasto[ja-1]:lasto[ja]].data
                
                # Calculate both the orbital and atomic normalized current
                Jo[ia-1] += nsum(nabs(t))
                Ja[ia-1] += nabs(nsum(t))

        del t

        # If it is the activity current, we return the geometric mean...
        if activity:
            Ja = np.sqrt( Ja * Jo )

        # Scale correctly
        Ja *= 0.5
            
        return Ja


    @dec_default_AP("Extract data from a TBT.nc file")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """

        # We limit the import to occur here
        import argparse as arg

        d = {
            "_tbt": self,
            "_data_header" : [],
            "_data" : [],
            "_Arng" : None,
            "_Ascale" : 1. / len(self.pivot), 
            "_Erng" : None,
            "_krng" : None,
        }
        namespace = default_namespace(**d)

        # Ensure the namespace is populated
        ensure_namespace(p, namespace)


        def dec_ensure_E(func):
            """ This decorater ensures that E is the first element in the _data container """
            def assign_E(self, *args, **kwargs):
                for arg in args:
                    if not isinstance(arg, TBTNamespace):
                        continue
                    ns = arg
                    break
                if len(ns._data) == 0:
                    # We immediately extract the energies
                    if ns._Erng is None:
                        ns._data.append(ns._tbt.E[:])
                    else:
                        ns._data.append(ns._tbt.E[ns._Erng])
                    ns._data_header.append('Energy[eV]')
                return func(self, *args, **kwargs)
            return assign_E

        # Energy grabs
        class ERange(arg.Action):
            def __call__(self, parser, ns, value, option_string=None):
                Emap = strmap(float, value, recursive=False, sep=':')
                # Convert to actual indices
                E = []
                for begin, end in Emap:
                    E.append(range(ns._tbt.Eindex(begin), ns._tbt.Eindex(end)+1))
                ns._Erng = np.array(E, np.int32).flatten()
        p.add_argument('--energy', '-E',
                       action=ERange,
                       help='Denote the sub-section of energies that are extracted: "-1:0,1:2" [eV]')

        # k-range
        class kRange(arg.Action):
            def __call__(self, parser, ns, value, option_string=None):
                ns._krng = lstranges(strmap(int, value, recursive=False, sep='-'))
        p.add_argument('--kpoint', '-k',
                       action=kRange,
                       help='Denote the sub-section of k-indices that are extracted.')

        # Try and add the atomic specification
        class AtomRange(arg.Action):
            def __call__(self, parser, ns, value, option_string=None):
                # Immediately convert to proper indices
                geom = ns._tbt.read_geom()
                ranges = lstranges(strmap(int, value, sep='-'))
                # we have only a subset of the orbitals
                orbs = []
                no = 0
                asarray = np.asarray
                for atoms in ranges:
                    if isinstance(atoms, list):
                        # Get atoms
                        ia = asarray(atoms[0], np.int32)
                        ob = geom.a2o(ia - 1, True)
                        no += len(ob)
                        ob = ob[asarray(atoms[1], np.int32) - 1]
                    else:
                        ia = asarray(atoms[0], np.int32)
                        ob = geom.a2o(ia - 1, True)
                        no += len(ob)
                    orbs.append(ob)
                # Add one to make the c-index equivalent to the f-index
                orbs = np.array(orbs, np.int32).flatten() + 1
                pivot = np.where(np.in1d(ns._tbt.pivot, orbs))[0]

                if len(orbs) != len(pivot):
                    print('Device atoms:')
                    tmp = ns._tbt.a_dev[:]
                    tmp.sort()
                    print(tmp[:])
                    raise ValueError('Atomic/Orbital requests are not fully included in the device region.')
                ns._Arng = pivot
                # Correct the scale to the correct number of orbitals
                ns._Ascale = 1. / no

        p.add_argument('--atom', '-a',
                       action=AtomRange,
                       help='Limit orbital resolved quantities to a sub-set of atoms/orbitals: "1-2[3,4]" will yield the 1st and 2nd atom and their 3rd and fourth orbital. Multiple comma-separated specifications are allowed.')


        class DataT(arg.Action):
            @dec_collect_actions
            @dec_ensure_E
            def __call__(self, parser, ns, values, option_string=None):
                e1 = values[0]
                if e1 not in ns._tbt.elecs:
                    raise ValueError('Electrode: "'+e1+'" cannot be found in the specified file.')
                e2 = values[1]
                if e2 not in ns._tbt.elecs:
                    raise ValueError('Electrode: "'+e2+'" cannot be found in the specified file.')

                # Grab the information
                if ns._krng is None:
                    data = ns._tbt.transmission(e1, e2, avg=True)
                else:
                    data = ns._tbt.transmission(e1, e2, avg=ns._krng)
                ns._data.append(data[ns._Erng,...])
                ns._data_header.append('T:{}-{}[G]'.format(e1, e2))
        p.add_argument('--transmission', '-T',nargs=2, metavar=('ELEC1','ELEC2'),
                       action=DataT,
                       help='Store the transmission between two electrodes.')

        class DataDOS(arg.Action):
            @dec_collect_actions
            @dec_ensure_E
            def __call__(self, parser, ns, value, option_string=None):
                if not value is None:
                    # we are storing the spectral DOS
                    e = value
                    if e not in ns._tbt.elecs:
                        raise ValueError('Electrode: "'+e1+'" cannot be found in the specified file.')
                    # Grab the information
                    if ns._krng is None:
                        data = ns._tbt.ADOS(e, avg=True)
                    else:
                        data = ns._tbt.ADOS(e, avg=ns._krng)
                    ns._data_header.append('ADOS:{}[1/eV]'.format(e))
                else:
                    if ns._krng is None:
                        data = ns._tbt.DOS(avg=True)
                    else:
                        data = ns._tbt.DOS(avg=ns._krng)
                    ns._data_header.append('DOS[1/eV]')
                # Grab out the atomic ranges
                if not ns._Arng is None:
                    orig_shape = data.shape
                    data = data[...,ns._Arng]
                # Select the energies, even if _Erng is None, this will work!
                data = np.sum(data[ns._Erng,...], axis=-1).flatten()
                ns._data.append(data * ns._Ascale)
        p.add_argument('--dos', '-D', nargs='?', metavar='ELEC',
                       action=DataDOS, default=None,
                       help="""Store the DOS. If no electrode is specified, it is Green function, else it is the spectral function.""")
        p.add_argument('--ados', '-AD', nargs=1, metavar='ELEC',
                       action=DataDOS, default=None,
                       help="""Store the spectral DOS, same as --dos.""")

        class DataTEig(arg.Action):
            @dec_collect_actions
            @dec_ensure_E
            def __call__(self, parser, ns, values, option_string=None):
                e1 = values[0]
                if e1 not in ns._tbt.elecs:
                    raise ValueError('Electrode: "'+e1+'" cannot be found in the specified file.')
                e2 = values[1]
                if e2 not in ns._tbt.elecs:
                    raise ValueError('Electrode: "'+e2+'" cannot be found in the specified file.')

                # Grab the information
                if ns._krng is None:
                    data = ns._tbt.transmission_eig(e1, e2, avg=True)
                else:
                    data = ns._tbt.transmission_eig(e1, e2, avg=ns._krng)
                # The shape is: k, E, neig
                neig = data.shape[-1]
                for eig in range(neig):
                    ns._data.append(data[ns._Erng,eig])
                    ns._data_header.append('Teig({}):{}-{}[G]'.format(eig+1, e1, e2))
        p.add_argument('--transmission-eig', '-Teig',nargs=2, metavar=('ELEC1','ELEC2'),
                       action=DataTEig,
                       help='Store the transmission eigenvalues between two electrodes.')

        class Out(arg.Action):
            @dec_run_actions
            def __call__(self, parser, ns, value, option_string=None):
                from sisl.io import TableSile
                out = value[0]
                TableSile(out).write(np.array(ns._data), header=ns._data_header)

                # Clean all data
                ns._data_header = []
                ns._data = []
                # These are expert options
                ns._Arng = None
                ns._Ascale = 1. / len(ns._tbt.pivot)
                ns._Erng = None
                ns._krng = None
        p.add_argument('--out','-o', nargs=1, action=Out,
                       help='Store the currently collected information (at its current invocation) to the out file.')

        return p, namespace


add_sile('TBT.nc', tbtncSileSiesta)

class phtncSileSiesta(tbtncSileSiesta):
    """ PHtrans file object """
    _trans_type = 'PHT'
    pass

add_sile('PHT.nc', phtncSileSiesta)


# The deltaH nc file
class dHncSileSiesta(SileCDFSIESTA):
    """ TBtrans delta-H file object """

    @Sile_fh_open
    def write_geom(self, geom):
        """
        Creates the NetCDF file and writes the geometry information
        """
        sile_raise_write(self)

        # Create initial dimensions
        self._crt_dim(self, 'one', 1)
        self._crt_dim(self, 'n_s', np.prod(geom.nsc))
        self._crt_dim(self, 'xyz', 3)
        self._crt_dim(self, 'no_s', np.prod(geom.nsc) * geom.no)
        self._crt_dim(self, 'no_u', geom.no)
        self._crt_dim(self, 'na_u', geom.na)

        # Create initial geometry
        v = self._crt_var(self, 'nsc', 'i4', ('xyz',))
        v.info = 'Number of supercells in each unit-cell direction'
        v = self._crt_var(self, 'lasto', 'i4', ('na_u',))
        v.info = 'Last orbital of equivalent atom'
        v = self._crt_var(self, 'xa', 'f8', ('na_u', 'xyz'))
        v.info = 'Atomic coordinates'
        v.unit = 'Bohr'
        v = self._crt_var(self, 'cell', 'f8', ('xyz', 'xyz'))
        v.info = 'Unit cell'
        v.unit = 'Bohr'

        # Create designation of the creation
        self.method = 'sisl'

        # Save stuff
        self.variables['nsc'][:] = geom.nsc
        self.variables['xa'][:] = geom.xyz / Bohr2Ang
        self.variables['cell'][:] = geom.cell / Bohr2Ang

        bs = self._crt_grp(self, 'BASIS')
        b = self._crt_var(bs, 'basis', 'i4', ('na_u',))
        b.info = "Basis of each atom by ID"

        orbs = np.empty([geom.na], np.int32)

        for ia, a, isp in geom.iter_species():
            b[ia] = isp + 1
            orbs[ia] = a.orbs
            if a.tag in bs.groups:
                # Assert the file sizes
                if bs.groups[a.tag].Number_of_orbitals != a.orbs:
                    raise ValueError(
                        'File ' +
                        self.file +
                        ' has erroneous data in regards of ' +
                        'of the already stored dimensions.')
            else:
                ba = bs.createGroup(a.tag)
                ba.ID = np.int32(isp + 1)
                ba.Atomic_number = np.int32(a.Z)
                ba.Mass = a.mass
                ba.Label = a.tag
                ba.Element = a.symbol
                ba.Number_of_orbitals = np.int32(a.orbs)

        # Store the lasto variable as the remaining thing to do
        self.variables['lasto'][:] = np.cumsum(orbs)


    def _add_lvl(self, ilvl):
        """
        Simply adds and returns a group if it does not
        exist it will be created
        """
        slvl = 'LEVEL-'+str(ilvl)
        if slvl in self.groups:
            lvl = self._crt_grp(self, slvl)
        else:
            lvl = self._crt_grp(self, slvl)
            if ilvl in [2, 4]:
                self._crt_dim(lvl, 'nkpt', None)
                v = self._crt_var(lvl, 'kpt', 'f8', ('nkpt','xyz'),
                                  attr = {'info' :'k-points for dH values',
                                          'unit' : 'b**-1'})
            if ilvl in [3, 4]: 
                self._crt_dim(lvl, 'ne', None)
                v = self._crt_var(lvl, 'E', 'f8', ('ne',),
                                  attr = {'info' :'Energy points for dH values',
                                          'unit' : 'Ry'})
            
        return lvl


    @Sile_fh_open
    def write_es(self, ham, **kwargs):
        """ Writes Hamiltonian model to file

        Parameters
        ----------
        ham : `Hamiltonian` model
           the model to be saved in the NC file
        spin : int, 0
           the spin-index of the Hamiltonian object that is stored.
        """
        # Ensure finalizations
        ham.finalize()

        # Ensure that the geometry is written
        self.write_geom(ham.geom)

        self._crt_dim(self, 'spin', ham._spin)


        # Determine the type of dH we are storing...
        k = kwargs.get('k', None)
        E = kwargs.get('E', None)
        
        if (k is None) and (E is None):
            ilvl = 1
        elif (k is not None) and (E is None):
            ilvl = 2
        elif (k is None) and (E is not None):
            ilvl = 3
        elif (k is not None) and (E is not None):
            ilvl = 4
        else:
            print(k,E)
            raise ValueError("This is wrongly implemented!!!")

        lvl = self._add_lvl(ilvl)

        # Append the sparsity pattern
        # Create basis group
        if 'n_col' in lvl.variables:
            if len(lvl.dimensions['nnzs']) != ham.nnz:
                raise ValueError("The sparsity pattern stored in dH *MUST* be equivalent for "
                                 "all dH entries [nnz].")
            if np.any(lvl.variables['n_col'][:] != ham._data.ncol[:]):
                raise ValueError("The sparsity pattern stored in dH *MUST* be equivalent for "
                                 "all dH entries [n_col].")
            if np.any(lvl.variables['list_col'][:] != ham._data.col[:]+1):
                raise ValueError("The sparsity pattern stored in dH *MUST* be equivalent for "
                                 "all dH entries [list_col].")
        else:
            self._crt_dim(lvl, 'nnzs', ham._data.col.shape[0])
            v = self._crt_var(lvl, 'n_col', 'i4', ('no_u',))
            v.info = "Number of non-zero elements per row"
            v[:] = ham._data.ncol[:]
            v = self._crt_var(lvl, 'list_col', 'i4', ('nnzs',),
                              chunksizes=(len(ham._data.col),), **self._cmp_args)
            v.info = "Supercell column indices in the sparse format"
            v[:] = ham._data.col[:] + 1  # correct for fortran indices
            v = self._crt_var(lvl, 'isc_off', 'i4', ('n_s', 'xyz'))
            v.info = "Index of supercell coordinates"
            v[:] = ham.geom.sc.sc_off[:, :]


        warn_E = True
        if ilvl in [3,4]:
            Es = np.array(lvl.variables['E'][:]) / Ry2eV

            iE = 0
            if len(Es) > 0:
                iE = np.argmin(np.abs(Es - E))
                if abs(Es[iE] - E) > 0.0001:
                    # accuracy of 0.1 meV

                    # create a new entry
                    iE = len(Es)
                    lvl.variables['E'][iE] = E * Ry2eV
                    warn_E = False
            else:
                warn_E = False

        warn_k = True
        if ilvl in [2,4]:
            kpt = np.array(lvl.variables['kpt'][:])

            ik = 0
            if len(kpt) > 0:
                ik = np.argmin(np.sum(np.abs(kpt - k[None,:]), axis=1))
                if np.allclose(kpt[ik,:], k, atol=0.0001):
                    # accuracy of 0.0001 

                    # create a new entry
                    ik = len(kpt)
                    lvl.variables['kpt'][ik, :] = k
                    warn_k = False
            else:
                warn_k = False


        if ilvl == 4 and warn_k and warn_E and False:
            # As soon as we have put the second k-point and the first energy
            # point, this warning will proceed...
            # I.e. even though the variable has not been set, it will WARN
            # Hence we out-comment this for now...
            warnings.warn('Overwriting k-point {0} and energy point {1} correction.'.format(ik,iE), UserWarning) 
        elif ilvl == 3 and warn_E:
            warnings.warn('Overwriting energy point {0} correction.'.format(iE), UserWarning) 
        elif ilvl == 2 and warn_k:
            warnings.warn('Overwriting k-point {0} correction.'.format(ik), UserWarning)

        if ilvl == 1:
            dim = ('spin', 'nnzs')
            sl = [slice(None)] * 2
            csize = [1] * 2
        elif ilvl == 2:
            dim = ('nkpt', 'spin', 'nnzs')
            sl = [slice(None)] * 3
            sl[0] = ik
            csize = [1] * 3
        elif ilvl == 3:
            dim = ('ne', 'spin', 'nnzs')
            sl = [slice(None)] * 3
            sl[0] = iE
            csize = [1] * 3
        elif ilvl == 4:
            dim = ('nkpt', 'ne', 'spin', 'nnzs')
            sl = [slice(None)] * 4
            sl[0] = ik
            sl[1] = iE
            csize = [1] * 4

        # Number of non-zero elements
        csize[-1] = ham.nnz

        if ham._data._D.dtype.kind == 'c':
            v1 = self._crt_var(lvl, 'RedH', 'f8', dim,
                               chunksizes=csize, 
                               attr = {'info' : "Real part of dH",
                                       'unit' : "Ry"}, **self._cmp_args)
            for i in range(ham.spin):
                sl[-2] = i
                v1[sl] = ham._data._D[:, i].real / Ry2eV ** ham._E_order

            v2 = self._crt_var(lvl, 'ImdH', 'f8', dim,
                               chunksizes=csize, 
                               attr = {'info' : "Imaginary part of dH",
                                       'unit' : "Ry"}, **self._cmp_args)
            for i in range(ham.spin):
                sl[-2] = i
                v2[sl] = ham._data._D[:, i].imag / Ry2eV ** ham._E_order

        else:
            v = self._crt_var(lvl, 'dH', 'f8', dim,
                              chunksizes=csize, 
                              attr = {'info' : "dH",
                                      'unit' : "Ry"},  **self._cmp_args)
            for i in range(ham.spin):
                sl[-2] = i
                v[sl] = ham._data._D[:, i] / Ry2eV ** ham._E_order


add_sile('dH.nc', dHncSileSiesta)

