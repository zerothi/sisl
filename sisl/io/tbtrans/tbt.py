from __future__ import print_function, division

from numbers import Integral
try:
    from StringIO import StringIO
except Exception:
    from io import StringIO

import numpy as np
from numpy import in1d, sqrt
import itertools

# The sparse matrix for the orbital/bond currents
from scipy.sparse import csr_matrix
from scipy.sparse import isspmatrix_csr
from scipy.sparse import SparseEfficiencyWarning

# Import sile objects
from ..sile import add_sile, sile_raise_write
from ._cdf import _devncSileTBtrans
from sisl.utils import *
import sisl._array as _a

from sisl import Geometry, Atoms
from sisl import units, constant
from sisl.messages import warn, info, SislError
from sisl._help import _range as range
from sisl._help import wrap_filterwarnings
from sisl.unit.siesta import unit_convert
from sisl.physics.distribution import fermi_dirac
from sisl.physics.densitymatrix import DensityMatrix


__all__ = ['tbtncSileTBtrans', 'tbtavncSileTBtrans']

Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')
Ry2K = unit_convert('Ry', 'K')
eV2Ry = unit_convert('eV', 'Ry')


class tbtncSileTBtrans(_devncSileTBtrans):
    r""" TBtrans output file object

    Implementation of the TBtrans output ``*.TBT.nc`` files which contains
    calculated quantities related to the NEGF code TBtrans.

    Although the TBtrans code is in fortran and the resulting NetCDF file variables
    are in fortran indexing (1-based), everything is returned as Python indexing (0-based)
    when using Python scripts.

    In the following equations we will use this notation:

    * :math:`\alpha` and :math:`\beta` are atomic indices
    * :math:`\nu` and :math:`\mu` are orbital indices

    A word on DOS normalization:

    All the device region DOS functions may request a normalization depending
    on a variety of functions. You are highly encouraged to read the documentation for
    the `norm` function and to consider the benefit of using the ``norm='atom'``
    normalization to more easily compare various partitions of DOS.

    Notes
    -----
    The API for this class are largely equivalent to the arguments of the `sdata` command-line
    tool, with the execption that the command-line tool uses Fortran indexing numbers (1-based).
    """
    _trans_type = 'TBT'
    _E2eV = Ry2eV

    _k_avg = False

    def write_tbtav(self, *args, **kwargs):
        """ Convert this to a TBT.AV.nc file, i.e. all k dependent quantites are averaged out.

        This command will overwrite any previous file with the ending TBT.AV.nc and thus
        will not take notice of any older files.

        Parameters
        ----------
        file : str
            output filename
        """
        f = self._file.replace('.nc', '.AV.nc')
        if len(args) > 0:
            f = args[0]
        f = kwargs.get('file', f)
        tbtavncSileTBtrans(f, mode='w', access=0).write_tbtav(self)

    def _value_avg(self, name, tree=None, kavg=False):
        """ Local method for obtaining the data from the SileCDF.

        This method checks how the file is access, i.e. whether
        data is stored in the object or it should be read consequtively.
        """
        if self._access > 0:
            if name in self._data:
                return self._data[name]

        try:
            v = self._variable(name, tree=tree)
        except KeyError as err:
            group = None
            if isinstance(tree, list):
                group = '.'.join(tree)
            elif not tree is None:
                group = tree
            if not group is None:
                raise KeyError(self.__class__.__name__ + ' could not retrieve key "{}.{}" due to missing flags in the input file.'.format(group, name))
            raise KeyError(self.__class__.__name__ + ' could not retrieve key "{}" due to missing flags in the input file.'.format(name))

        if self._k_avg:
            return v[:]

        # Perform normalization
        orig_shape = v.shape
        if isinstance(kavg, bool):
            if kavg:
                wkpt = self.wk
                nk = len(wkpt)
                data = v[0, ...] * wkpt[0]
                for i in range(1, nk):
                    data += v[i, :] * wkpt[i]
                data.shape = orig_shape[1:]
            else:
                data = v[:]

        elif isinstance(kavg, Integral):
            data = v[kavg, ...]
            data.shape = orig_shape[1:]

        else:
            raise ValueError(self.__class__.__name__ + ' requires kavg argument to be either bool or an integer corresponding to the k-point index.')

        # Return data
        return data

    def _value_E(self, name, tree=None, kavg=False, E=None):
        """ Local method for obtaining the data from the SileCDF using an E index.

        """
        if E is None:
            return self._value_avg(name, tree, kavg)

        # Ensure that it is an index
        iE = self.Eindex(E)

        try:
            v = self._variable(name, tree=tree)
        except KeyError:
            group = None
            if isinstance(tree, list):
                group = '.'.join(tree)
            elif not tree is None:
                group = tree
            if not group is None:
                raise KeyError(self.__class__.__name__ + ' could not retrieve key "{}.{}" due to missing flags in the input file.'.format(group, name))
            raise KeyError(self.__class__.__name__ + ' could not retrieve key "{}" due to missing flags in the input file.'.format(name))
        if self._k_avg:
            return v[iE, ...]

        wkpt = self.wk

        # Perform normalization
        orig_shape = v.shape

        if isinstance(kavg, bool):
            if kavg:
                nk = len(wkpt)
                data = np.array(v[0, iE, ...]) * wkpt[0]
                for i in range(1, nk):
                    data += v[i, iE, ...] * wkpt[i]
                data.shape = orig_shape[2:]
            else:
                data = np.array(v[:, iE, ...])

        elif isinstance(kavg, Integral):
            data = np.array(v[kavg, iE, ...])
            data.shape = orig_shape[2:]

        else:
            raise ValueError(self.__class__.__name__ + ' requires kavg argument to be either bool or an integer corresponding to the k-point index.')

        # Return data
        return data

    def transmission(self, elec_from=0, elec_to=1, kavg=True):
        r""" Transmission from `elec_from` to `elec_to`.

        The transmission between two electrodes may be retrieved
        from the `Sile`.

        The transmission is calculated as:

        .. math::

            T(E) = \mathrm{Tr}[\mathbf{G}\boldsymbol\Gamma_{\mathrm{from}}\mathbf{G}^\dagger\boldsymbol\Gamma_{\mathrm{to}}]

        where all quantities are energy dependent.

        Parameters
        ----------
        elec_from: str, int, optional
           the originating electrode
        elec_to: str, int, optional
           the absorbing electrode (different from `elec_from`)
        kavg: bool, int, optional
           whether the returned transmission is k-averaged, or an explicit (unweighed) k-point
           is returned

        See Also
        --------
        transmission_eig : the transmission decomposed in eigenchannels
        transmission_bulk : the total transmission in a periodic lead
        reflection : total reflection back into the electrode
        """
        elec_from = self._elec(elec_from)
        elec_to = self._elec(elec_to)
        if elec_from == elec_to:
            raise ValueError(self.__class__.__name__ + ".transmission elec_from and elec_to must not be the same.")

        return self._value_avg(elec_to + '.T', elec_from, kavg=kavg)

    def reflection(self, elec=0, kavg=True, from_single=False):
        r""" Reflection into electrode `elec`

        The reflection into electrode `elec` is calculated as:

        .. math::

             R(E) = T_{\mathrm{bulk}}(E) - \sum_{\mathrm{to}} T_{\mathrm{elec}\to\mathrm{to}}(E)

        Another way of calculating the reflection is via:

        .. math::

             R(E) = T_{\mathrm{bulk}}(E) - \big\{i \mathrm{Tr}[(\mathbf G-\mathbf G^\dagger)\boldsymbol\Gamma_{\mathrm{elec}}]
                   - \mathrm{Tr}[\mathbf G\boldsymbol\Gamma_{\mathrm{elec}}\mathbf G^\dagger\boldsymbol\Gamma_{\mathrm{elec}}]\big\}

        Both are identical, however, numerically they may be different. Particularly when the bulk transmission
        is very large compared to the transmission to the other electrodes one should prefer the first equation.

        Parameters
        ----------
        elec: str, int, optional
           the backscattered electrode
        kavg: bool, int, optional
           whether the returned reflection is k-averaged, or an explicit (unweighed) k-point
           is returned
        from_single: bool, optional
           whether the reflection is calculated using the Green function and a
           single scattering matrix Eq. (2) above (true), otherwise Eq. (1) will be used (false).

        See Also
        --------
        transmission : the total transmission
        transmission_eig : the transmission decomposed in eigenchannels
        transmission_bulk : the total transmission in a periodic lead
        """
        elec = self._elec(elec)
        BT = self.transmission_bulk(elec, kavg=kavg)

        # Find full transmission out of electrode
        if from_single:
            T = self._value_avg(elec + '.T', elec, kavg=kavg) - self._value_avg(elec + '.C', elec, kavg=kavg)
        else:
            T = 0.
            for to in self.elecs:
                to = self._elec(to)
                if elec == to:
                    continue
                T = T + self.transmission(elec, to, kavg=kavg)

        return BT - T

    def transmission_eig(self, elec_from=0, elec_to=1, kavg=True):
        """ Transmission eigenvalues from `elec_from` to `elec_to`.

        Parameters
        ----------
        elec_from: str, int, optional
           the originating electrode
        elec_to: str, int, optional
           the absorbing electrode (different from `elec_from`)
        kavg: bool, int, optional
           whether the returned transmission eigenvalues are k-averaged, or an explicit (unweighed) k-point
           is returned

        See Also
        --------
        transmission : the total transmission
        transmission_bulk : the total transmission in a periodic lead
        """
        elec_from = self._elec(elec_from)
        elec_to = self._elec(elec_to)
        if elec_from == elec_to:
            raise ValueError(self.__class__.__name__ + ".transmission_eig elec_from and elec_to must not be the same.")

        return self._value_avg(elec_to + '.T.Eig', elec_from, kavg=kavg)

    def transmission_bulk(self, elec=0, kavg=True):
        """ Bulk transmission for the `elec` electrode

        The bulk transmission is equivalent to creating a 2 terminal device with
        electrode `elec` tiled 3 times.

        Parameters
        ----------
        elec: str, int, optional
           the bulk electrode
        kavg: bool, int, optional
           whether the returned transmission are k-averaged, or an explicit (unweighed) k-point
           is returned

        See Also
        --------
        transmission : the total transmission
        transmission_eig : the transmission decomposed in eigenchannels
        reflection : total reflection back into the electrode
        """
        return self._value_avg('T', self._elec(elec), kavg=kavg)

    def norm(self, atom=None, orbital=None, norm='none'):
        r""" Normalization factor depending on the input

        The normalization can be performed in one of the below methods.
        In the following :math:`N` refers to the normalization constant
        that is to be used (i.e. the divisor):

        ``'none'``
           :math:`N=1`
        ``'all'``
           :math:`N` equals the number of orbitals in the total device region.
        ``'atom'``
           :math:`N` equals the total number of orbitals in the selected
           atoms. If `orbital` is an argument a conversion of `orbital` to the equivalent
           unique atoms is performed, and subsequently the total number of orbitals on the
           atoms is used. This makes it possible to compare the fraction of orbital DOS easier.
        ``'orbital'``
           :math:`N` is the sum of selected orbitals, if `atom` is specified, this
           is equivalent to the 'atom' option.

        Parameters
        ----------
        atom : array_like of int or bool, optional
           only return for a given set of atoms (default to all).
           *NOT* allowed with `orbital` keyword
        orbital : array_like of int or bool, optional
           only return for a given set of orbitals (default to all)
           *NOT* allowed with `atom` keyword
        norm : {'none', 'atom', 'orbital', 'all'}
           how the normalization of the summed DOS is performed (see `norm` routine)
        """
        # Cast to lower
        norm = norm.lower()
        if norm == 'none':
            NORM = 1.
        elif norm in ['all', 'atom', 'orbital']:
            NORM = float(self.no_d)
        else:
            raise ValueError(self.__class__.__name__ + '.norm error on norm keyword in when requesting normalization!')

        # If the user simply requests a specific norm
        if atom is None and orbital is None:
            return NORM

        # Now figure out what to do
        if atom is None:
            # Get pivoting indices to average over
            if norm == 'orbital':
                NORM = float(len(self.o2p(orbital)))
            elif norm == 'atom':
                geom = self.geom
                a = np.unique(geom.o2a(orbital))
                # Now sum the orbitals per atom
                NORM = float(_a.sumi(geom.firsto[a+1] - geom.firsto[a]))
            return NORM

        if not orbital is None:
            raise ValueError(self.__class__.__name__ + '.norm both atom and orbital cannot be specified!')

        # atom is specified, this will result in the same normalization
        # regardless of norm == [orbital, atom] since it is all orbitals
        # on the given atoms.
        if norm in ['orbital', 'atom']:
            NORM = float(len(self.a2p(atom)))

        return NORM

    def _DOS(self, DOS, atom, orbital, sum, norm):
        """ Averages/sums the DOS

        Parameters
        ----------
        DOS : numpy.ndarray
           data to process
        atom : array_like of int, optional
           only return for a given set of atoms (default to all).
           *NOT* allowed with `orbital` keyword
        orbital : array_like of int, optional
           only return for a given set of orbitals (default to all)
           *NOT* allowed with `atom` keyword
        sum : bool, optional
           whether the returned quantities are summed or returned *as is*, i.e. resolved per atom/orbital.
        norm : {'none', 'atom', 'orbital', 'all'}
           how the normalization of the summed DOS is performed (see `norm` routine)

        Returns
        -------
        numpy.ndarray
            in order of the geometry orbitals (i.e. pivoted back to the device region).
            If `atom` or `orbital` is specified they are returned in that order.
        """
        if not atom is None and not orbital is None:
            raise ValueError(('Both atom and orbital keyword in DOS request '
                              'cannot be specified, only one at a time.'))
        # Cast to lower
        norm = norm.lower()
        if norm == 'none':
            NORM = 1.
        elif norm in ['all', 'atom', 'orbital']:
            NORM = float(self.no_d)
        else:
            raise ValueError('Error on norm keyword in DOS request')

        if atom is None and orbital is None:
            # We simply return *everything*
            if sum:
                return DOS.sum(-1) / NORM
            # We return the sorted DOS
            p = np.argsort(self.pivot())
            return DOS[..., p] / NORM

        # Now figure out what to do
        if atom is None:
            # orbital *must* be specified

            # Get pivoting indices to average over
            p = self.o2p(orbital)
            if norm == 'orbital':
                NORM = float(len(p))
            elif norm == 'atom':
                geom = self.geom
                a = np.unique(geom.o2a(orbital))
                # Now sum the orbitals per atom
                NORM = float(_a.sumi(geom.firsto[a+1] - geom.firsto[a]))

            if sum:
                return DOS[..., p].sum(-1) / NORM
            # Else, we have to return the full subset
            return DOS[..., p] / NORM

        # atom is specified
        # Return the pivoting orbitals for the atom
        p = self.a2p(atom)
        if norm in ['orbital', 'atom']:
            NORM = float(len(p))

        if sum or isinstance(atom, Integral):
            # Regardless of SUM, when requesting a single atom
            # we return it
            return DOS[..., p].sum(-1) / NORM

        # We default the case where 1-orbital systems are in use
        # Then it becomes *very* easy
        if len(p) == len(atom):
            return DOS[..., p] / NORM

        # This is the multi-orbital case...

        # We will return per-atom
        shp = list(DOS.shape[:-1])
        nDOS = np.empty(shp + [len(atom)], DOS.dtype)

        # Quicker than re-creating the geometry on every instance
        geom = self.geom

        # Sum for new return stuff
        for i, a in enumerate(atom):
            pvt = self.a2p(a)
            if len(pvt) == 0:
                nDOS[..., i] = 0.
            else:
                nDOS[..., i] = DOS[..., pvt].sum(-1) / NORM

        return nDOS

    def DOS(self, E=None, kavg=True, atom=None, orbital=None, sum=True, norm='none'):
        r""" Green function density of states (DOS) (1/eV).

        Extract the DOS on a selected subset of atoms/orbitals in the device region

        .. math::

           \mathrm{DOS}(E) = -\frac{1}{\pi N} \sum_{\nu\in \mathrm{atom}/\mathrm{orbital}} \Im \mathbf{G}_{\nu\nu}(E)

        The normalization constant (:math:`N`) is defined in the routine `norm` and depends on the
        arguments.

        Parameters
        ----------
        E : float or int, optional
           optionally only return the DOS of atoms at a given energy point
        kavg: bool, int, optional
           whether the returned DOS is k-averaged, or an explicit (unweighed) k-point
           is returned
        atom : array_like of int or bool, optional
           only return for a given set of atoms (default to all).
           *NOT* allowed with `orbital` keyword
        orbital : array_like of int or bool, optional
           only return for a given set of orbitals (default to all)
           *NOT* allowed with `atom` keyword
        sum : bool, optional
           whether the returned quantities are summed or returned *as is*, i.e. resolved per atom/orbital.
        norm : {'none', 'atom', 'orbital', 'all'}
           how the normalization of the summed DOS is performed (see `norm` routine)

        See Also
        --------
        ADOS : the spectral density of states from an electrode
        BDOS : the bulk density of states in an electrode
        """
        return self._DOS(self._value_E('DOS', kavg=kavg, E=E), atom, orbital, sum, norm) * eV2Ry

    def ADOS(self, elec=0, E=None, kavg=True, atom=None, orbital=None, sum=True, norm='none'):
        r""" Spectral density of states (DOS) (1/eV).

        Extract the spectral DOS from electrode `elec` on a selected subset of atoms/orbitals in the device region

        .. math::
           \mathrm{ADOS}_\mathfrak{el}(E) = \frac{1}{2\pi N} \sum_{\nu\in \mathrm{atom}/\mathrm{orbital}} [\mathbf{G}(E)\Gamma_\mathfrak{el}\mathbf{G}^\dagger]_{\nu\nu}(E)

        The normalization constant (:math:`N`) is defined in the routine `norm` and depends on the
        arguments.

        Parameters
        ----------
        elec: str, int, optional
           electrode originating spectral function
        E : float or int, optional
           optionally only return the DOS of atoms at a given energy point
        kavg: bool, int, optional
           whether the returned DOS is k-averaged, or an explicit (unweighed) k-point
           is returned
        atom : array_like of int or bool, optional
           only return for a given set of atoms (default to all).
           *NOT* allowed with `orbital` keyword
        orbital : array_like of int or bool, optional
           only return for a given set of orbitals (default to all)
           *NOT* allowed with `atom` keyword
        sum : bool, optional
           whether the returned quantities are summed or returned *as is*, i.e. resolved per atom/orbital.
        norm : {'none', 'atom', 'orbital', 'all'}
           how the normalization of the summed DOS is performed (see `norm` routine).

        See Also
        --------
        DOS : the total density of states (including bound states)
        BDOS : the bulk density of states in an electrode
        """
        elec = self._elec(elec)
        return self._DOS(self._value_E('ADOS', elec, kavg=kavg, E=E), atom, orbital, sum, norm) * eV2Ry

    def BDOS(self, elec=0, E=None, kavg=True, sum=True, norm='none'):
        r""" Bulk density of states (DOS) (1/eV).

        Extract the bulk DOS from electrode `elec` on a selected subset of atoms/orbitals in the device region

        .. math::

           \mathrm{BDOS}_\mathfrak{el}(E) = -\frac{1}{\pi} \Im\mathbf{G}(E)

        Parameters
        ----------
        elec: str, int, optional
           electrode where the bulk DOS is returned
        E : float or int, optional
           optionally only return the DOS of atoms at a given energy point
        kavg: bool, int, optional
           whether the returned DOS is k-averaged, or an explicit (unweighed) k-point
           is returned
        sum : bool, optional
           whether the returned quantities are summed or returned *as is*, i.e. resolved per atom/orbital.
        norm : {'none', 'atom', 'orbital', 'all'}
           whether the returned quantities are summed or normed by total number of orbitals.
           Currently one cannot extract DOS per atom/orbital.

        See Also
        --------
        DOS : the total density of states (including bound states)
        ADOS : the spectral density of states from an electrode
        """
        # The bulk DOS is already normalized per non-expanded cell
        # Hence the non-normalized quantity needs to be multiplied by
        #  product(bloch)
        elec = self._elec(elec)
        if norm in ['atom', 'orbital', 'all']:
            # This is normalized per non-expanded unit-cell, so no need to do Bloch
            N = 1. / len(self._dimension('no_u', elec))
        else:
            N = 1.
        if sum:
            return self._value_E('DOS', elec, kavg=kavg, E=E).sum(-1) * eV2Ry * N
        else:
            return self._value_E('DOS', elec, kavg=kavg, E=E) * eV2Ry * N

    def _E_T_sorted(self, elec_from, elec_to, kavg=True):
        """ Internal routine for returning energies and transmission in a sorted array """
        E = self.E
        idx_sort = np.argsort(E)
        # Get transmission
        T = self.transmission(elec_from, elec_to, kavg)
        return E[idx_sort], T[idx_sort]

    def current(self, elec_from=0, elec_to=1, kavg=True):
        r""" Current from `from` to `to` using the k-weights and energy spacings in the file.

        Calculates the current as:

        .. math::
           I(\mu_t - \mu_f) = \frac{e}{h}\int\!\mathrm{d}E\, T(E) [n_F(\mu_t, k_B T_t) - n_F(\mu_f, k_B T_f)]

        The chemical potential and the temperature are taken from this object.

        Parameters
        ----------
        elec_from: str, int, optional
           the originating electrode
        elec_to: str, int, optional
           the absorbing electrode (different from `elec_from`)
        kavg: bool, int, optional
           whether the returned current is k-averaged, or an explicit (unweighed) k-point
           is returned

        See Also
        --------
        current_parameter : to explicitly set the electronic temperature and chemical potentials
        chemical_potential : routine that defines the chemical potential of the queried electrodes
        kT : routine that defines the electronic temperature of the queried electrodes
        """
        mu_f = self.chemical_potential(elec_from)
        kt_f = self.kT(elec_from)
        mu_t = self.chemical_potential(elec_to)
        kt_t = self.kT(elec_to)
        return self.current_parameter(elec_from, mu_f, kt_f,
                                      elec_to, mu_t, kt_t, kavg)

    def current_parameter(self, elec_from, mu_from, kt_from,
                          elec_to, mu_to, kt_to, kavg=True):
        r""" Current from `from` to `to` using the k-weights and energy spacings in the file.

        Calculates the current as:

        .. math::
           I(\mu_t - \mu_f) = \frac{e}{h}\int\!\mathrm{d}E\, T(E) [n_F(\mu_t, k_B T_t) - n_F(\mu_f, k_B T_f)]

        The chemical potential and the temperature are passed as arguments to
        this routine.

        Parameters
        ----------
        elec_from: str, int
           the originating electrode
        mu_from: float
           the chemical potential of the electrode (in eV)
        kt_from: float
           the electronic temperature of the electrode (in eV)
        elec_to: str, int
           the absorbing electrode (different from `elec_from`)
        mu_to: float
           the chemical potential of the electrode (in eV)
        kt_to: float
           the electronic temperature of the electrode (in eV)
        kavg: bool, int, optional
           whether the returned current is k-averaged, or an explicit (unweighed) k-point
           is returned

        See Also
        --------
        current : which calculates the current with the chemical potentials and temperatures set in the TBtrans calculation
        """
        elec_from = self._elec(elec_from)
        elec_to = self._elec(elec_to)
        # Get energies
        E, T = self._E_T_sorted(elec_from, elec_to, kavg)

        # We expect the tbtrans calcluation was created with the simple
        #   mid-rule!
        # The mid-rule is equivalent to adding a dE = (E[1] - E[0]) / 2
        # to both ends.
        dE = E[1] - E[0]

        # Check that the lower bound is sufficient
        print_warning = mu_from - kt_from * 3 < E[0] - dE / 2 or \
                        mu_to - kt_to * 3  < E[0] - dE / 2
        print_warning = mu_from + kt_from * 3 > E[-1] + dE / 2 or \
                        mu_to + kt_to * 3  > E[-1] + dE / 2 or \
                        print_warning
        if print_warning:
            # We should pretty-print a table of data
            m = max(len(elec_from), len(elec_to), 15)
            s = ("{:"+str(m)+"s} {:9.3f} : {:9.3f} eV\n").format('Energy range', E[0] - dE / 2, E[-1] + dE / 2)
            s += ("{:"+str(m)+"s} {:9.3f} : {:9.3f} eV\n").format(elec_from, mu_from - kt_from * 3, mu_from + kt_from * 3)
            s += ("{:"+str(m)+"s} {:9.3f} : {:9.3f} eV\n").format(elec_to, mu_to - kt_to * 3, mu_to + kt_to * 3)
            min_e = min(mu_from - kt_from * 3, mu_to - kt_to * 3)
            max_e = max(mu_from + kt_from * 3, mu_to + kt_to * 3)
            s += ("{:"+str(m)+"s} {:9.3f} : {:9.3f} eV\n").format('dFermi function', min_e, max_e)

            warn(self.__class__.__name__ + ".current_parameter cannot "
                 "accurately calculate the current due to the calculated energy range. "
                 "Increase the calculated energy-range.\n" + s)

        I = (T * dE * (fermi_dirac(E, kt_from, mu_from) - fermi_dirac(E, kt_to, mu_to))).sum()
        return I * constant.q / constant.h('eV s')

    def _check_Teig(self, func_name, TE, eps=0.001):
        """ Internal method to check whether all transmission eigenvalues are present """
        if np.any(np.logical_and.reduce(TE > eps, axis=-1)):
            info('{}.{} does possibly not have all relevant transmission eigenvalues in the '
                 'calculation. For some energy values all transmission eigenvalues are above {}!'.format(self.__class__.__name__,
                                                                                                         func_name, eps))

    def shot_noise(self, elec_from=0, elec_to=1, classical=False, kavg=True):
        r""" Shot-noise term `from` to `to` using the k-weights

        Calculates the shot-noise term according to `classical` (also known as the Poisson value).
        If `classical` is True the shot-noise calculated is:

        .. math::
           S_P(E, V) = \frac{2e^2}{h}|V|\sum_k\sum_n T_{k,n}(E) w_k = \frac{2e^3}{h}|V|T(E)

        while for `classical` False (default) the Fermi-Dirac statistics is taken into account:

        .. math::
           S(E, V) = \frac{2e^2}{h}|V|\sum_k\sum_n T_{k,n}(E) [1 - T_{k,n}(E)] w_k

        Raises
        ------
        SislInfo
            If *all* of the calculated :math:`T_{k,n}(E)` values in the file are above 0.001.

        Parameters
        ----------
        elec_from: str, int, optional
           the originating electrode
        elec_to: str, int, optional
           the absorbing electrode (different from `elec_from`)
        classical: bool, optional
           which shot-noise to calculate, default to non-classical
        kavg: bool, int, optional
           whether the returned shot-noise is k-averaged, or an explicit (unweighed) k-point
           is returned

        See Also
        --------
        fano : the ratio between the quantum mechanial and the classical shot noise.
        noise_power : temperature dependent noise power
        """
        mu_f = self.chemical_potential(elec_from)
        mu_t = self.chemical_potential(elec_to)
        # The applied bias between the two electrodes
        eV = abs(mu_f - mu_t)
        # Pre-factor
        # 2 e ^ 3 V / h
        # Note that h in eV units will cancel the units in the applied bias
        noise_const = 2 * constant.q ** 2 * (eV / constant.h('eV s'))
        if classical:
            # Calculate the Poisson shot-noise (equal to 2eI in the low T and zero kT limit)
            return noise_const * self.transmission(elec_from, elec_to, kavg=kavg)

        # Non-classical
        if isinstance(kavg, bool):
            if not kavg:
                # The user wants it k-resolved
                T = self.transmission_eig(elec_from, elec_to, kavg=False)
                self._check_Teig('shot_noise', T)
                return noise_const * (T * (1 - T)).sum(-1)

            # We need to manually weigh the k-points
            wkpt = self.wkpt

            T = self.transmission_eig(elec_from, elec_to, kavg=0)
            self._check_Teig('shot_noise', T)
            sn = (T * (1 - T)).sum(-1) * wkpt[0]
            for ik in range(1, self.nkpt):
                T = self.transmission_eig(elec_from, elec_to, kavg=ik)
                self._check_Teig('shot_noise', T)
                sn += (T * (1 - T)).sum(-1) * wkpt[ik]

        else:
            T = self.transmission_eig(elec_from, elec_to, kavg=kavg)
            self._check_Teig('shot_noise', T)
            sn = (T * (1 - T)).sum(-1)

        return noise_const * sn

    def noise_power(self, elec_from=0, elec_to=1, kavg=True):
        r""" Noise power `from` to `to` using the k-weights and energy spacings in the file (temperature dependent)

        Calculates the noise power as

        .. math::
           S(V) = \frac{2e^2}{h}\sum_k\sum_n \int\mathrm d E
                  \big\{T_{k,n}(E)[f_L(1-f_L)+f_R(1-f_R)] +
                        T_{k,n}(E)[1 - T_{k,n}(E)](f_L - f_R)^2\big\} w_k

        Where :math:`f_i` are the Fermi-Dirac distributions for the electrodes.

        Raises
        ------
        SislInfo
            If *all* of the calculated :math:`T_{k,n}(E)` values in the file are above 0.001.

        Parameters
        ----------
        elec_from: str, int, optional
           the originating electrode
        elec_to: str, int, optional
           the absorbing electrode (different from `elec_from`)
        kavg: bool, int, optional
           whether the returned noise-power is k-averaged, or an explicit (unweighed) k-point
           is returned

        See Also
        --------
        fano : the ratio between the quantum mechanial and the classical shot noise.
        shot_noise : shot-noise term (zero temperature limit)
        """
        kT_f = self.kT(elec_from)
        kT_t = self.kT(elec_to)
        mu_f = self.chemical_potential(elec_from)
        mu_t = self.chemical_potential(elec_to)
        fd_f = fermi_dirac(self.E, kT_f, mu_f)
        fd_t = fermi_dirac(self.E, kT_t, mu_t)

        # Get the energy spacing (probably we should add a routine)
        dE = self.E[1] - self.E[0]

        # Pre-calculate the factors
        eq_fac = dE * (fd_f * (1 - fd_f) + fd_t * (1 - fd_t))
        neq_fac = dE * (fd_f - fd_t) ** 2
        del fd_f, fd_t

        # Pre-factor
        # 2 e ^ 2 / h
        # Note that h in eV units will cancel the units in the dE integration
        noise_const = 2 * constant.q ** 2 / constant.h('eV s')

        # Determine the k-average
        if isinstance(kavg, bool):
            if not kavg:
                # The user wants it k-resolved
                T = self.transmission_eig(elec_from, elec_to, kavg=False)
                self._check_Teig('noise_power', T)
                return noise_const * ((T.sum(-1) * eq_fac).sum(-1) + ((T * (1 - T)).sum(-1) * neq_fac).sum(-1))

            # We need to manually weigh the k-points
            wkpt = self.wkpt

            T = self.transmission_eig(elec_from, elec_to, kavg=0)
            self._check_Teig('noise_power', T)
            # Separate the calculation into two terms (see Ya.M. Blanter, M. Buttiker, Physics Reports 336 2000)
            np = ((T.sum(-1) * eq_fac).sum(-1) + ((T * (1 - T)).sum(-1) * neq_fac).sum(-1)) * wkpt[0]
            for ik in range(1, self.nkpt):
                T = self.transmission_eig(elec_from, elec_to, kavg=ik)
                self._check_Teig('noise_power', T)
                np += ((T.sum(-1) * eq_fac).sum(-1) + ((T * (1 - T)).sum(-1) * neq_fac).sum(-1)) * wkpt[ik]

        else:
            T = self.transmission_eig(elec_from, elec_to, kavg=kavg)
            self._check_Teig('noise_power', T)
            np = (T.sum(-1) * eq_fac).sum(-1) + ((T * (1 - T)).sum(-1) * neq_fac).sum(-1)

        # Do final conversion
        return noise_const * np

    def fano(self, elec_from=0, elec_to=1, kavg=True, zero_T=1e-6):
        r""" The Fano-factor for the calculation (requires calculated transmission eigenvalues)

        Calculate the Fano factor defined as (or through the shot-noise):

        .. math::
           F(E) &= \frac{\sum_{k,n} T_{k,n}(E)[1 - T_{k,n}(E)] w_k}{\sum_{k,n} T_{k,n}(E) w_k}
           \\
                &= S(E, V) / S_P(E, V)

        Notes
        -----
        The default `zero_T` may change in the future.
        This calculation will *only* work for non-polarized calculations since the divisor needs
        to be the spin-sum.
        The current implementation uses the full transmission as the divisor.

        Examples
        --------

        For a spin-polarized calculation one should calculate the Fano factor as:

        >>> up = get_sile('siesta.TBT_UP.nc')
        >>> down = get_sile('siesta.TBT_DN.nc')
        >>> fano = up.fano() * up.transmission() + down.fano() * down.transmission()
        >>> fano /= up.transmission() + down.transmission()

        Parameters
        ----------
        elec_from: str, int, optional
           the originating electrode
        elec_to: str, int, optional
           the absorbing electrode (different from `elec_from`)
        kavg: bool, int, optional
           whether the returned Fano factor is k-averaged, or an explicit (unweighed) k-point
           is returned. In any case the divisor will always be the k-averaged transmission.
        zero_T : float, optional
           any transmission eigen value lower than this value will be treated as exactly 0.

        See Also
        --------
        shot_noise : shot-noise term (zero temperature limit)
        noise_power : temperature dependent noise power
        """
        def dividend(T):
            T[T <= zero_T] = 0.
            return (T * (1 - T)).sum(-1)

        if isinstance(kavg, bool):
            if not kavg:
                # The user wants it k-resolved
                T = self.transmission_eig(elec_from, elec_to, kavg=False)
                self._check_Teig('fano', T)
                fano = dividend(T)
                T = self.transmission(elec_from, elec_to)
                fano /= T[None, :]
                fano[:, T <= 0.] = 0.
                return fano

            # We need to manually weigh the k-points
            wkpt = self.wkpt

            T = self.transmission_eig(elec_from, elec_to, kavg=0)
            self._check_Teig('fano', T)
            fano = dividend(T) * wkpt[0]
            for ik in range(1, self.nkpt):
                T = self.transmission_eig(elec_from, elec_to, kavg=ik)
                self._check_Teig('fano', T)
                fano += dividend(T) * wkpt[ik]

        else:
            T = self.transmission_eig(elec_from, elec_to, kavg=kavg)
            self._check_Teig('fano', T)
            fano = dividend(T)

        # Divide by k-averaged transmission
        T = self.transmission(elec_from, elec_to)
        fano /= T
        fano[T <= 0.] = 0.
        return fano

    def _sparse_data(self, data, elec, E, kavg=True, isc=None):
        """ Internal routine for retrieving sparse data (orbital current, COOP) """
        # Get the geometry for obtaining the sparsity pattern.
        if elec is not None:
            elec = self._elec(elec)

        geom = self.geom

        # These are the row-pointers...
        rptr = np.insert(_a.cumsumi(self._value('n_col')), 0, 0)

        # Get column indices
        col = self._value('list_col') - 1

        # Default matrix size
        mat_size = [geom.no, geom.no_s]

        # Figure out the super-cell indices that are requested
        # First we figure out the indices, then
        # we build the array of allowed columns
        if isc is None:
            isc = [None, None, None]

        if isc[0] is None and isc[1] is None and isc[2] is None:
            all_col = None

        else:
            # The user has requested specific supercells
            # Here we create a list of supercell interactions.

            nsc = np.copy(geom.nsc)
            # Shorten to the unit-cell if there are no more
            for i in [0, 1, 2]:
                if nsc[i] == 1:
                    isc[i] = 0
                if not isc[i] is None:
                    nsc[i] = 1

            # Small function for creating the supercells allowed
            def ret_range(val, req):
                i = val // 2
                if req is None:
                    return range(-i, i+1)
                return [req]
            x = ret_range(nsc[0], isc[0])
            y = ret_range(nsc[1], isc[1])
            z = ret_range(nsc[2], isc[2])

            # Make a shrinking logical array for selecting a subset of the
            # orbital currents...
            all_col = _a.emptyi(len(x) * len(y) * len(z))
            for i, (ix, iy, iz) in enumerate(itertools.product(x, y, z)):
                all_col[i] = geom.sc_index([ix, iy, iz])

            # If the user requests a single supercell index, we will
            # return a square matrix
            if len(all_col) == 1:
                mat_size[1] = mat_size[0]

            # Transfer all_col to the range
            all_col = array_arange(all_col * geom.no,
                                   n=_a.fulli(len(all_col), geom.no))

            # Create a logical array for sub-indexing
            all_col = in1d(col, _a.arrayi(all_col))
            col = col[all_col]

            # recreate row-pointer
            cnz = np.count_nonzero
            def func(ptr1, ptr2):
                return cnz(all_col[ptr1:ptr2])
            tmp = _a.arrayi(list(map(func, rptr[:geom.no], rptr[1:])))
            rptr = np.insert(_a.cumsumi(tmp), 0, 0)
            del tmp

        if all_col is None:
            D = self._value_E(data, elec, kavg, E)
        else:
            D = self._value_E(data, elec, kavg, E)[..., all_col]

        return csr_matrix((D, col, rptr), shape=mat_size)

    def _sparse_data_orb_to_atom(self, Dij, uc=False):
        """ Reduce orbital sparse data to atomic sparse data

        Parameters
        ----------
        Dij : scipy.sparse.csr_matrix
           the input data
        uc : bool, optional
           whether the returned data are only in the unit-cell.
           If ``True`` this will return a sparse matrix of ``shape = (self.na, self.na)``,
           else, it will return a sparse matrix of ``shape = (self.na, self.na * self.n_s)``.
           One may figure out the connections via `~sisl.geometry.Geometry.sc_index`.
        """
        geom = self.geom
        na = geom.na
        o2a = geom.o2a

        if not uc:
            uc = Dij.shape[0] == Dij.shape[1]

        # We convert to atomic bond-currents
        if uc:
            Dab = csr_matrix((na, na), dtype=Dij.dtype)

            def map_col(c):
                return o2a(c) % na

        else:
            Dab = csr_matrix((na, na * geom.n_s), dtype=Dij.dtype)

            map_col = o2a

        # Lets do array notation for speeding up the computations
        if not isspmatrix_csr(Dij):
            Dij = Dij.tocsr()

        # Check for the simple case of 1-orbital systems
        if geom.na == geom.no:
            # In this case it is extremely easy!
            # Just copy to the new data

            # Transfer all columns to the new columns
            Dab.indptr[:] = Dij.indptr.copy()
            if uc:
                Dab.indices = (Dij.indices % na).astype(np.int32, copy=False)
            else:
                Dab.indices = Dij.indices.copy()

        else:
            # The multi-orbital case

            # Loop all atoms to make the new pointer array
            # I.e. a consecutive array of pointers starting from
            #   firsto[.] .. lasto[.]
            iptr = Dij.indptr
            # Get first orbital
            fo = geom.firsto
            # Automatically create the new index pointer
            # from first and last orbital
            indptr = np.insert(_a.cumsumi(iptr[fo[1:]] - iptr[fo[:-1]]), 0, 0)

            # Now we have a new indptr, and the column indices have also
            # been processed.
            Dab.indptr[:] = indptr[:]
            # Transfer all columns to the new columns
            Dab.indices = map_col(Dij.indices).astype(np.int32, copy=False)

        # Copy data
        Dab.data = np.copy(Dij.data)
        # Note that we do not sum duplicates as that depends on the next routine
        # I.e. sometimes we want to remove negative values, etc.
        return Dab

    def orbital_current(self, elec, E, kavg=True, isc=None, only='all'):
        r""" Orbital current originating from `elec` as a sparse matrix

        This will return a sparse matrix, see ``scipy.sparse.csr_matrix`` for details.
        Each matrix element of the sparse matrix corresponds to the orbital indices of the
        underlying geometry.

        When requesting orbital-currents it is vital to consider how the data needs to be analysed
        before extracting the data. For instance, if only local currents are interesting one should
        use ``only='+'``. While if one is interested in the transmission between subset of orbitals,
        ``only='all'`` is the correct method.

        For inexperienced users it is adviced to try out all three values of ``only`` to ensure
        the correct physics is obtained.

        This becomes even more important when the orbital currents are calculated with magnetic
        fields. With :math:`\mathbf B` fields local current loops may form and current does
        not necessarily flow along the transport direction.

        Parameters
        ----------
        elec: str, int
           the electrode of originating electrons
        E: float or int
           the energy or the energy index of the orbital current. If an integer
           is passed it is the index, otherwise the index corresponding to
           ``Eindex(E)`` is used.
        kavg: bool, int, optional
           whether the returned orbital current is k-averaged, or an explicit (unweighed) k-point
           is returned
        isc: array_like, optional
           the returned bond currents from the unit-cell (``[None, None, None]``) to
           the given supercell, the default is all orbital currents for the supercell.
           To only get unit cell orbital currents, pass ``[0, 0, 0]``.
        only : {'all', '+', '-'}
           which orbital currents to return, all, positive or negative values only.
           Default to ``'all'`` because it can then be used in the subsequent default
           arguments for `bond_current_from_orbital` and `atom_current_from_orbital`.

        Examples
        --------
        >>> Jij = tbt.orbital_current(0, -1.0) # orbital current @ E = -1 eV originating from electrode ``0``
        >>> Jij[10, 11] # orbital current from the 11th to the 12th orbital

        See Also
        --------
        bond_current_from_orbital : transfer the orbital current to bond current
        bond_current : the bond current (orbital current summed over orbitals)
        atom_current_from_orbital : transfer the orbital current to atomic current
        atom_current : the atomic current for each atom (scalar representation of bond-currents)
        vector_current : an atomic field current for each atom (Cartesian representation of bond-currents)
        """
        J = self._sparse_data('J', elec, E, kavg, isc)

        if only == '+':
            J.data[J.data < 0] = 0
        elif only == '-':
            J.data[J.data > 0] = 0
        elif only != 'all':
            raise ValueError(self.__class__.__name__ + '.orbital_current "only" keyword has '
                             'wrong value ["all", "+", "-"] allowed.')

        return J

    def bond_current_from_orbital(self, Jij, only='+', uc=False):
        r""" Bond-current between atoms (sum of orbital currents) from an external orbital current

        Conversion routine from orbital currents into bond currents.

        The bond currents are a sum over all orbital currents:

        .. math::
           J_{\alpha\beta} = \sum_{\nu\in\alpha}\sum_{\mu\in\beta} J_{\nu\mu}

        where if

        * ``only='+'``:
          only :math:`J_{\nu\mu} > 0` are summed onto the corresponding atom,
        * ``only='-'``:
          only :math:`J_{\nu\mu} < 0` are summed onto the corresponding atom,
        * ``only='all'``:
          all :math:`J_{\nu\mu}` are summed onto the corresponding atom.

        Parameters
        ----------
        Jij : scipy.sparse.csr_matrix
           the orbital currents as retrieved from `orbital_current`
        only : {'+', '-', 'all'}
           If "+" is supplied only the positive orbital currents are used,
           for "-", only the negative orbital currents are used,
           else return both.
        uc : bool, optional
           whether the returned bond-currents are only in the unit-cell.
           If ``True`` this will return a sparse matrix of ``shape = (self.na, self.na)``,
           else, it will return a sparse matrix of ``shape = (self.na, self.na * self.n_s)``.
           One may figure out the connections via `~sisl.geometry.Geometry.sc_index`.

        Examples
        --------
        >>> Jij = tbt.orbital_current(0, -1.0) # orbital current @ E = -1 eV originating from electrode ``0``
        >>> Jab = tbt.bond_current_from_orbital(Jij)
        >>> Jab[2,3] # bond current between atom 3 and 4

        See Also
        --------
        orbital_current : the orbital current between individual orbitals
        bond_current : the bond current (orbital current summed over orbitals)
        atom_current : the atomic current for each atom (scalar representation of bond-currents)
        vector_current : an atomic field current for each atom (Cartesian representation of bond-currents)
        """
        Jab = self._sparse_data_orb_to_atom(Jij, uc)

        if only == '+':
            Jab.data[Jab.data < 0] = 0
        elif only == '-':
            Jab.data[Jab.data > 0] = 0
        elif only != 'all':
            raise ValueError(self.__class__.__name__ + '.bond_current_from_orbital "only" keyword has '
                             'wrong value ["+", "-", "all"] allowed.')

        # Do in-place operations by removing all the things not required
        Jab.sum_duplicates()

        return Jab

    def bond_current(self, elec, E, kavg=True, isc=None, only='+', uc=False):
        """ Bond-current between atoms (sum of orbital currents)

        Short hand function for calling `orbital_current` and `bond_current_from_orbital`.

        Parameters
        ----------
        elec : str, int
           the electrode of originating electrons
        E : float or int
           A `float` for energy in eV, `int` for explicit energy index
           Unlike `orbital_current` this may not be `None` as the down-scaling of the
           orbital currents may not be equivalent for all energy points.
        kavg : bool, int, optional
           whether the returned bond current is k-averaged, or an explicit (unweighed) k-point
           is returned
        isc : array_like, optional
           the returned bond currents from the unit-cell (``[None, None, None]``) (default) to
           the given supercell. If ``[None, None, None]`` is passed all
           bond currents are returned.
        only : {'+', '-', 'all'}
           If "+" is supplied only the positive orbital currents are used,
           for "-", only the negative orbital currents are used,
           else return the sum of both. Please see discussion in `orbital_current`.
        uc : bool, optional
           whether the returned bond-currents are only in the unit-cell.
           If `True` this will return a sparse matrix of ``shape = (self.na, self.na)``,
           else, it will return a sparse matrix of ``shape = (self.na, self.na * self.n_s)``.
           One may figure out the connections via `~sisl.geometry.Geometry.sc_index`.

        Examples
        --------
        >>> Jij = tbt.orbital_current(0, -1.0, only='+') # orbital current @ E = -1 eV originating from electrode ``0``
        >>> Jab1 = tbt.bond_current_from_orbital(Jij)
        >>> Jab2 = tbt.bond_current(0, -1.0)
        >>> Jab1 == Jab2
        True

        See Also
        --------
        orbital_current : the orbital current between individual orbitals
        bond_current_from_orbital : transfer the orbital current to bond current
        atom_current : the atomic current for each atom (scalar representation of bond-currents)
        vector_current : an atomic field current for each atom (Cartesian representation of bond-currents)
        """
        Jij = self.orbital_current(elec, E, kavg, isc, only=only)

        return self.bond_current_from_orbital(Jij, uc=uc, only=only)

    def atom_current_from_orbital(self, Jij, activity=True):
        r""" Atomic current of atoms by passing the orbital current

        The atomic current is a single number specifying a figure of the *magnitude*
        current flowing through each atom. It is thus *not* a quantity that can be related to
        the physical current flowing in/out of atoms but is merely a number that provides an
        idea of *how much* current this atom is redistributing.

        The atomic current may have two meanings based on these two equations

        .. math::
            J_\alpha^{|a|} &=\frac{1}{2} \sum_\beta \Big| \sum_{\nu\in \alpha}\sum_{\mu\in \beta} J_{\nu\mu} \Big|
            \\
            J_\alpha^{|o|} &=\frac{1}{2} \sum_\beta \sum_{\nu\in \alpha}\sum_{\mu\in \beta} \big| J_{\nu\mu} \big|

        If the *activity* current is requested (``activity=True``)
        :math:`J_\alpha^{\mathcal A} = \sqrt{ J_\alpha^{|a|} J_\alpha^{|o|} }` is returned.

        If ``activity=False`` :math:`J_\alpha^{|a|}` is returned.

        For geometries with all atoms only having 1-orbital, they are equivalent.

        Generally the activity current is a more rigorous figure of merit for the current
        flowing through an atom. More so than than the summed absolute atomic current due to
        the following reasoning. The activity current is a geometric mean of the absolute bond current
        and the absolute orbital current. This means that if there is an atom with a large orbital current
        it will have a larger activity current.

        Parameters
        ----------
        Jij: scipy.sparse.csr_matrix
           the orbital currents as retrieved from `orbital_current`
        activity: bool, optional
           ``True`` to return the activity current, see explanation above

        Examples
        --------
        >>> Jij = tbt.orbital_current(0, -1.03) # orbital current @ E = -1 eV originating from electrode ``0``
        >>> Ja = tbt.atom_current_from_orbital(Jij)
        """
        # Create the bond-currents with all summations
        Jab = self.bond_current_from_orbital(Jij, only='all')
        # We take the absolute and sum it over all connecting atoms
        Ja = np.asarray(abs(Jab).sum(1)).ravel()

        if activity:
            # Calculate the absolute summation of all orbital
            # currents and transfer it to a bond-current
            Jab = self.bond_current_from_orbital(abs(Jij), only='all')

            # Sum to make it per atom, it is already the absolute
            Jo = np.asarray(Jab.sum(1)).ravel()

            # Return the geometric mean of the atomic current X orbital
            # current.
            Ja = sqrt(Ja * Jo)

        # Scale correctly
        Ja *= 0.5

        return Ja

    def atom_current(self, elec, E, kavg=True, activity=True):
        """ Atomic current of atoms

        Short hand function for calling `orbital_current` and `atom_current_from_orbital`.

        Parameters
        ----------
        elec: str, int
           the electrode of originating electrons
        E: float or int
           the energy or energy index of the atom current.
        kavg: bool, int, optional
           whether the returned atomic current is k-averaged, or an explicit (unweighed) k-point
           is returned
        activity: bool, optional
           whether the activity current is returned, see `atom_current_from_orbital` for details.

        See Also
        --------
        orbital_current : the orbital current between individual orbitals
        bond_current_from_orbital : transfer the orbital current to bond current
        bond_current : the bond current (orbital current summed over orbitals)
        vector_current : an atomic field current for each atom (Cartesian representation of bond-currents)
        """
        Jorb = self.orbital_current(elec, E, kavg)

        return self.atom_current_from_orbital(Jorb, activity=activity)

    @wrap_filterwarnings("ignore", category=SparseEfficiencyWarning)
    def vector_current_from_bond(self, Jab):
        r""" Vector for each atom being the sum of bond-current times the normalized bond between the atoms

        The vector current is defined as:

        .. math::
              \mathbf J_\alpha = \sum_\beta \frac{r_\beta - r_\alpha}{|r_\beta - r_\alpha|} \cdot J_{\alpha\beta}

        Where :math:`J_{\alpha\beta}` is the bond current between atom :math:`\alpha` and :math:`\beta` and
        :math:`r_\alpha` are the atomic coordinates.

        Parameters
        ----------
        Jab: scipy.sparse.csr_matrix
           the bond currents as retrieved from `bond_current`

        Returns
        -------
        numpy.ndarray
            array of vectors per atom in the Geometry (only non-zero for device atoms)

        See Also
        --------
        orbital_current : the orbital current between individual orbitals
        bond_current_from_orbital : transfer the orbital current to bond current
        bond_current : the bond current (orbital current summed over orbitals)
        atom_current : the atomic current for each atom (scalar representation of bond-currents)
        """
        geom = self.geom

        na = geom.na
        # vector currents
        Ja = _a.zerosd([na, 3])

        # Loop atoms in the device region
        # These are the only atoms which may have bond-currents,
        # So no need to loop over any other atoms
        getrow = Jab.getrow
        Rij = geom.Rij

        for ia in self.a_dev:
            # Get csr matrix
            Jia = getrow(ia)

            # Set diagonal to zero
            Jia[0, ia] = 0.
            # Remove the diagonal (prohibits the calculation of the
            # norm of the zero vector, hence required)
            Jia.eliminate_zeros()

            # Now calculate the vector elements
            # Remark that the vector goes from ia -> ja
            rv = Rij(ia, Jia.indices)
            rv = rv / sqrt((rv ** 2).sum(1))[:, None]
            Ja[ia, :] = (Jia.data[:, None] * rv).sum(0)

        return Ja

    def vector_current(self, elec, E, kavg=True, only='+'):
        """ Vector for each atom describing the *mean* path for the current travelling through the atom

        See `vector_current_from_bond` for details.

        Parameters
        ----------
        elec: str or int
           the electrode of originating electrons
        E: float or int
           the energy or energy index of the vector current.
           Unlike `orbital_current` this may not be `None` as the down-scaling of the
           orbital currents may not be equivalent for all energy points.
        kavg: bool, int, optional
           whether the returned vector current is k-averaged, or an explicit (unweighed) k-point
           is returned
        only : {'+', '-', 'all'}
           By default only sum *outgoing* vector currents (``'+'``).
           The *incoming* vector currents may be retrieved by ``'-'``, while the
           average incoming and outgoing direction can be obtained with ``'all'``.
           In the last case the vector currents are divided by 2 to ensure the length
           of the vector is compatibile with the other options given a pristine system.

        Returns
        -------
        numpy.ndarray
            array of vectors per atom in the Geometry (only non-zero for device atoms)

        See Also
        --------
        orbital_current : the orbital current between individual orbitals
        bond_current_from_orbital : transfer the orbital current to bond current
        bond_current : the bond current (orbital current summed over orbitals)
        atom_current : the atomic current for each atom (scalar representation of bond-currents)
        """
        # Imperative that we use the entire supercell structure to
        # retain vectors crossing the boundaries
        Jab = self.bond_current(elec, E, kavg, only=only)

        if only == 'all':
            # When we divide by two one can *always* compare the bulk
            # vector currents using either of the sum-rules.
            # I.e. it will be much easier to distinguish differences
            # between "incoming" and "outgoing".
            return self.vector_current_from_bond(Jab) / 2

        return self.vector_current_from_bond(Jab)

    def density_matrix(self, E, kavg=True, isc=None, geometry=None):
        r""" Density matrix from the Green function at energy `E` (1/eV)

        The density matrix can be used to calculate the LDOS in real-space.

        The :math:`\mathrm{LDOS}(E, \mathbf r)` may be calculated using the `~sisl.physics.DensityMatrix.density`
        routine. Basically the LDOS in real-space may be calculated as

        .. math::
            \rho_{\mathbf G}(E, \mathbf r) = -\frac{1}{\pi}\sum_{\nu\mu}\phi_\nu(\mathbf r)\phi_\mu(\mathbf r) \Im[\mathbf G_{\nu\mu}(E)]

        where :math:`\phi` are the orbitals. Note that the broadening used in the TBtrans calculations
        ensures the broadening of the density, i.e. it should not be necessary to perform energy
        averages over the density matrices.

        Parameters
        ----------
        E : float or int
           the energy or the energy index of density matrix. If an integer
           is passed it is the index, otherwise the index corresponding to
           ``Eindex(E)`` is used.
        kavg: bool, int, optional
           whether the returned density matrix is k-averaged, or an explicit (unweighed) k-point
           is returned
        isc: array_like, optional
           the returned density matrix from unit-cell (``[None, None, None]``) to
           the given supercell, the default is all density matrix elements for the supercell.
           To only get unit cell orbital currents, pass ``[0, 0, 0]``.
        geometry: Geometry, optional
           geometry that will be associated with the density matrix. By default the
           geometry contained in this file will be used. However, then the
           atomic species are probably incorrect, nor will the orbitals contain
           the basis-set information required to generate the required density
           in real-space.

        See Also
        --------
        Adensity_matrix : spectral function density matrix

        Returns
        -------
        DensityMatrix
            object containing the Geometry and the density matrix elements
        """
        return self.Adensity_matrix(None, E, kavg, isc, geometry=geometry)

    def Adensity_matrix(self, elec, E, kavg=True, isc=None, geometry=None):
        r""" Spectral function density matrix at energy `E` (1/eV)

        The density matrix can be used to calculate the LDOS in real-space.

        The :math:`\mathrm{LDOS}(E, \mathbf r)` may be calculated using the `~sisl.physics.DensityMatrix.density`
        routine. Basically the LDOS in real-space may be calculated as

        .. math::
            \rho_{\mathbf A_{\mathfrak{el}}}(E, \mathbf r) = \frac{1}{2\pi}\sum_{\nu\mu}\phi_\nu(\mathbf r)\phi_\mu(\mathbf r) \Re[\mathbf A_{\mathfrak{el}, \nu\mu}(E)]

        where :math:`\phi` are the orbitals. Note that the broadening used in the TBtrans calculations
        ensures the broadening of the density, i.e. it should not be necessary to perform energy
        averages over the density matrices.

        Parameters
        ----------
        elec: str or int
           the electrode of originating electrons
        E : float or int
           the energy or the energy index of density matrix. If an integer
           is passed it is the index, otherwise the index corresponding to
           ``Eindex(E)`` is used.
        kavg: bool, int, optional
           whether the returned density matrix is k-averaged, or an explicit (unweighed) k-point
           is returned
        isc: array_like, optional
           the returned density matrix from unit-cell (``[None, None, None]``) to
           the given supercell, the default is all density matrix elements for the supercell.
           To only get unit cell orbital currents, pass ``[0, 0, 0]``.
        geometry: Geometry, optional
           geometry that will be associated with the density matrix. By default the
           geometry contained in this file will be used. However, then the
           atomic species are probably incorrect, nor will the orbitals contain
           the basis-set information required to generate the required density
           in real-space.

        See Also
        --------
        density_matrix : Green function density matrix

        Returns
        -------
        DensityMatrix
            object containing the Geometry and the density matrix elements
        """
        dm = self._sparse_data('DM', elec, E, kavg, isc) * eV2Ry
        # Now create the density matrix object
        geom = self.read_geometry()
        if geometry is None:
            DM = DensityMatrix.fromsp(geom, dm)
        else:
            if geom.no != geometry.no:
                raise ValueError(self.__class__.__name__ + '.Adensity_matrix requires input geometry to contain the correct number of orbitals. Please correct input!')
            DM = DensityMatrix.fromsp(geometry, dm)
        return DM

    def orbital_COOP(self, E, kavg=True, isc=None):
        r""" Orbital COOP analysis of the Green function

        This will return a sparse matrix, see ``scipy.sparse.csr_matrix`` for details.
        Each matrix element of the sparse matrix corresponds to the COOP of the
        underlying geometry.

        The COOP analysis can be written as:

        .. math::
            \mathrm{COOP}^{\mathbf G}_{\nu\mu} = \frac{-1}{2\pi}
              \Im\big[(\mathbf G - \mathbf G^\dagger)_{\nu\mu} \mathbf S_{\mu\nu} \big]

        The sum of the COOP DOS is equal to the DOS:

        .. math::
            \mathrm{DOS}_{\nu} = \sum_\mu \mathrm{COOP}^{\mathbf G}_{\nu\mu}

        One can calculate the (diagonal) balanced COOP analysis, see JPCM 15 (2003),
        7751-7761 for details. The DBCOOP is given by:

        .. math::
            D &= \sum_\nu \mathrm{COOP}^{\mathbf G}_{\nu\nu}
            \\
            \mathrm{DBCOOP}^{\mathbf G}_{\nu\mu} &= \mathrm{COOP}^{\mathbf G}_{\nu\mu} / D

        The BCOOP can be looked up in the reference above.

        Parameters
        ----------
        E: float or int
           the energy or the energy index of COOP. If an integer
           is passed it is the index, otherwise the index corresponding to
           ``Eindex(E)`` is used.
        kavg: bool, int, optional
           whether the returned COOP is k-averaged, or an explicit (unweighed) k-point
           is returned
        isc: array_like, optional
           the returned COOP from unit-cell (``[None, None, None]``) to
           the given supercell, the default is all COOP for the supercell.
           To only get unit cell orbital currents, pass ``[0, 0, 0]``.

        Examples
        --------
        >>> COOP = tbt.orbital_COOP(-1.0) # COOP @ E = -1 eV 
        >>> COOP[10, 11] # COOP value between the 11th and 12th orbital
        >>> COOP.sum(1).A[tbt.o_dev, 0] == tbt.DOS(sum=False)[tbt.Eindex(-1.0)]
        >>> D = COOP.diagonal().sum()
        >>> DBCOOP = COOP / D

        See Also
        --------
        atom_COOP_from_orbital : transfer an orbital COOP to atomic COOP
        atom_COOP : atomic COOP analysis of the Green function
        orbital_ACOOP : orbital resolved COOP analysis of the spectral function
        atom_ACOOP : atomic COOP analysis of the spectral function
        orbital_COHP : orbital resolved COHP analysis of the Green function
        atom_COHP_from_orbital : atomic COHP analysis from an orbital COHP
        atom_COHP : atomic COHP analysis of the Green function
        orbital_ACOHP : orbital resolved COHP analysis of the spectral function
        atom_ACOHP : atomic COHP analysis of the spectral function
        """
        return self.orbital_ACOOP(None, E, kavg, isc)

    def orbital_ACOOP(self, elec, E, kavg=True, isc=None):
        r""" Orbital COOP analysis of the spectral function

        This will return a sparse matrix, see `~scipy.sparse.csr_matrix` for details.
        Each matrix element of the sparse matrix corresponds to the COOP of the
        underlying geometry.

        The COOP analysis can be written as:

        .. math::
            \mathrm{COOP}^{\mathbf A}_{\nu\mu} = \frac{1}{2\pi} \Re\big[\mathbf A_{\nu\mu} \mathbf S_{\mu\nu} \big]

        The sum of the COOP DOS is equal to the DOS:

        .. math::
            \mathrm{ADOS}_{\nu} = \sum_\mu \mathrm{COOP}^{\mathbf A}_{\nu\mu}

        One can calculate the (diagonal) balanced COOP analysis, see JPCM 15 (2003),
        7751-7761 for details. The DBCOOP is given by:

        .. math::
            D &= \sum_\nu \mathrm{COOP}^{\mathbf A}_{\nu\nu}
            \\
            \mathrm{DBCOOP}^{\mathbf A}_{\nu\mu} &= \mathrm{COOP}^{\mathbf A}_{\nu\mu} / D

        The BCOOP can be looked up in the reference above.

        Parameters
        ----------
        elec: str or int
           the electrode of the spectral function
        E: float or int
           the energy or the energy index of COOP. If an integer
           is passed it is the index, otherwise the index corresponding to
           ``Eindex(E)`` is used.
        kavg: bool, int, optional
           whether the returned COOP is k-averaged, or an explicit (unweighed) k-point
           is returned
        isc: array_like, optional
           the returned COOP from unit-cell (``[None, None, None]``) to
           the given supercell, the default is all COOP for the supercell.
           To only get unit cell orbital currents, pass ``[0, 0, 0]``.

        Examples
        --------
        >>> ACOOP = tbt.orbital_ACOOP(0, -1.0) # COOP @ E = -1 eV from ``0`` spectral function
        >>> ACOOP[10, 11] # COOP value between the 11th and 12th orbital
        >>> ACOOP.sum(1).A[tbt.o_dev, 0] == tbt.ADOS(0, sum=False)[tbt.Eindex(-1.0)]
        >>> D = ACOOP.diagonal().sum()
        >>> ADBCOOP = ACOOP / D

        See Also
        --------
        orbital_COOP : orbital resolved COOP analysis of the Green function
        atom_COOP_from_orbital : transfer an orbital COOP to atomic COOP
        atom_COOP : atomic COOP analysis of the Green function
        atom_ACOOP : atomic COOP analysis of the spectral function
        orbital_COHP : orbital resolved COHP analysis of the Green function
        atom_COHP_from_orbital : atomic COHP analysis from an orbital COHP
        atom_COHP : atomic COHP analysis of the Green function
        orbital_ACOHP : orbital resolved COHP analysis of the spectral function
        atom_ACOHP : atomic COHP analysis of the spectral function
        """
        COOP = self._sparse_data('COOP', elec, E, kavg, isc) * eV2Ry
        return COOP

    def atom_COOP_from_orbital(self, COOP, uc=False):
        r""" Calculate the atomic COOP curve from the orbital COOP

        The atomic COOP are a sum over all orbital COOP:

        .. math::
            \mathrm{COOP}_{\alpha\beta} = \sum_{\nu\in\alpha}\sum_{\mu\in\beta} \mathrm{COOP}_{\nu\mu}

        Parameters
        ----------
        COOP : scipy.sparse.csr_matrix
           the orbital COOP as retrieved from `orbital_COOP` or `orbital_ACOOP`
        uc : bool, optional
           whether the returned COOP are only in the unit-cell.
           If ``True`` this will return a sparse matrix of ``shape = (self.na, self.na)``,
           else, it will return a sparse matrix of ``shape = (self.na, self.na * self.n_s)``.
           One may figure out the connections via `~sisl.geometry.Geometry.sc_index`.

        See Also
        --------
        orbital_COOP : orbital resolved COOP analysis of the Green function
        orbital_ACOOP : orbital resolved COOP analysis of the spectral function
        atom_COOP : atomic COOP analysis of the Green function
        """
        COOP = self._sparse_data_orb_to_atom(COOP, uc)
        COOP.sum_duplicates()
        return COOP

    def atom_COOP(self, E, kavg=True, isc=None, uc=False):
        r""" Atomic COOP curve of the Green function

        Parameters
        ----------
        E: float or int
           the energy or the energy index of COOP. If an integer
           is passed it is the index, otherwise the index corresponding to
           ``Eindex(E)`` is used.
        kavg: bool, int, optional
           whether the returned COOP is k-averaged, or an explicit (unweighed) k-point
           is returned
        isc: array_like, optional
           the returned COOP from unit-cell (``[None, None, None]``) to
           the given supercell, the default is all COOP for the supercell.
           To only get unit cell orbital currents, pass ``[0, 0, 0]``.
        uc : bool, optional
           whether the returned COOP are only in the unit-cell.
           If ``True`` this will return a sparse matrix of ``shape = (self.na, self.na)``,
           else, it will return a sparse matrix of ``shape = (self.na, self.na * self.n_s)``.
           One may figure out the connections via `~sisl.geometry.Geometry.sc_index`.

        See Also
        --------
        orbital_COOP : orbital resolved COOP analysis of the Green function
        atom_COOP_from_orbital : transfer an orbital COOP to atomic COOP
        atom_ACOOP : atomic COOP analysis of the spectral function
        atom_COHP : atomic COHP analysis of the Green function
        """
        return self.atom_ACOOP(None, E, kavg, isc, uc)

    def atom_ACOOP(self, elec, E, kavg=True, isc=None, uc=False):
        r""" Atomic COOP curve of the spectral function

        Parameters
        ----------
        elec: str or int
           the electrode of the spectral function
        E: float or int
           the energy or the energy index of COOP. If an integer
           is passed it is the index, otherwise the index corresponding to
           ``Eindex(E)`` is used.
        kavg: bool, int, optional
           whether the returned COOP is k-averaged, or an explicit (unweighed) k-point
           is returned
        isc: array_like, optional
           the returned COOP from unit-cell (``[None, None, None]``) to
           the given supercell, the default is all COOP for the supercell.
           To only get unit cell orbital currents, pass ``[0, 0, 0]``.
        uc : bool, optional
           whether the returned COOP are only in the unit-cell.
           If ``True`` this will return a sparse matrix of ``shape = (self.na, self.na)``,
           else, it will return a sparse matrix of ``shape = (self.na, self.na * self.n_s)``.
           One may figure out the connections via `~sisl.geometry.Geometry.sc_index`.

        See Also
        --------
        orbital_COOP : orbital resolved COOP analysis of the Green function
        atom_COOP_from_orbital : transfer an orbital COOP to atomic COOP
        atom_COOP : atomic COOP analysis of the Green function
        atom_ACOHP : atomic COHP analysis of the spectral function
        """
        COOP = self.orbital_ACOOP(elec, E, kavg, isc)
        return self.atom_COOP_from_orbital(COOP, uc)

    def orbital_COHP(self, E, kavg=True, isc=None):
        r""" Orbital resolved COHP analysis of the Green function

        This will return a sparse matrix, see ``scipy.sparse.csr_matrix`` for details.
        Each matrix element of the sparse matrix corresponds to the COHP of the
        underlying geometry.

        The COHP analysis can be written as:

        .. math::
            \mathrm{COHP}^{\mathbf G}_{\nu\mu} = \frac{-1}{2\pi}
              \Im\big[(\mathbf G - \mathbf G^\dagger)_{\nu\mu} \mathbf H_{\mu\nu} \big]

        Parameters
        ----------
        E: float or int
           the energy or the energy index of COHP. If an integer
           is passed it is the index, otherwise the index corresponding to
           ``Eindex(E)`` is used.
        kavg: bool, int, optional
           whether the returned COHP is k-averaged, or an explicit (unweighed) k-point
           is returned
        isc: array_like, optional
           the returned COHP from unit-cell (``[None, None, None]``) to
           the given supercell, the default is all COHP for the supercell.
           To only get unit cell orbital currents, pass ``[0, 0, 0]``.

        Examples
        --------
        >>> COHP = tbt.orbital_COHP(-1.0) # COHP @ E = -1 eV 
        >>> COHP[10, 11] # COHP value between the 11th and 12th orbital

        See Also
        --------
        atom_COHP_from_orbital : atomic COHP analysis from an orbital COHP
        atom_COHP : atomic COHP analysis of the Green function
        orbital_ACOHP : orbital resolved COHP analysis of the spectral function
        atom_ACOHP : atomic COHP analysis of the spectral function
        orbital_COOP : orbital resolved COOP analysis of the Green function
        atom_COOP_from_orbital : transfer an orbital COOP to atomic COOP
        atom_COOP : atomic COOP analysis of the Green function
        orbital_ACOOP : orbital resolved COOP analysis of the spectral function
        atom_ACOOP : atomic COOP analysis of the spectral function
        """
        return self.orbital_ACOHP(None, E, kavg, isc)

    def orbital_ACOHP(self, elec, E, kavg=True, isc=None):
        r""" Orbital resolved COHP analysis of the spectral function

        This will return a sparse matrix, see ``scipy.sparse.csr_matrix`` for details.
        Each matrix element of the sparse matrix corresponds to the COHP of the
        underlying geometry.

        The COHP analysis can be written as:

        .. math::
            \mathrm{COHP}^{\mathbf A}_{\nu\mu} = \frac{1}{2\pi} \Re\big[\mathbf A_{\nu\mu}
                \mathbf H_{\nu\mu} \big]

        Parameters
        ----------
        elec: str or int
           the electrode of the spectral function
        E: float or int
           the energy or the energy index of COHP. If an integer
           is passed it is the index, otherwise the index corresponding to
           ``Eindex(E)`` is used.
        kavg: bool, int, optional
           whether the returned COHP is k-averaged, or an explicit (unweighed) k-point
           is returned
        isc: array_like, optional
           the returned COHP from unit-cell (``[None, None, None]``) to
           the given supercell, the default is all COHP for the supercell.
           To only get unit cell orbital currents, pass ``[0, 0, 0]``.

        See Also
        --------
        orbital_COHP : orbital resolved COHP analysis of the Green function
        atom_COHP_from_orbital : atomic COHP analysis from an orbital COHP
        atom_COHP : atomic COHP analysis of the Green function
        atom_ACOHP : atomic COHP analysis of the spectral function
        orbital_COOP : orbital resolved COOP analysis of the Green function
        atom_COOP_from_orbital : transfer an orbital COOP to atomic COOP
        atom_COOP : atomic COOP analysis of the Green function
        orbital_ACOOP : orbital resolved COOP analysis of the spectral function
        atom_ACOOP : atomic COOP analysis of the spectral function
        """
        COHP = self._sparse_data('COHP', elec, E, kavg, isc)
        return COHP

    def atom_COHP_from_orbital(self, COHP, uc=False):
        r""" Calculate the atomic COHP curve from the orbital COHP

        The atomic COHP are a sum over all orbital COHP:

        .. math::
            \mathrm{COHP}_{\alpha\beta} = \sum_{\nu\in\alpha}\sum_{\mu\in\beta} \mathrm{COHP}_{\nu\mu}

        Parameters
        ----------
        COHP : scipy.sparse.csr_matrix
           the orbital COHP as retrieved from `orbital_COHP` or `orbital_ACOHP`
        uc : bool, optional
           whether the returned COHP are only in the unit-cell.
           If ``True`` this will return a sparse matrix of ``shape = (self.na, self.na)``,
           else, it will return a sparse matrix of ``shape = (self.na, self.na * self.n_s)``.
           One may figure out the connections via `~sisl.geometry.Geometry.sc_index`.

        See Also
        --------
        orbital_COHP : orbital resolved COHP analysis of the Green function
        orbital_ACOHP : orbital resolved COHP analysis of the spectral function
        atom_COHP : atomic COHP analysis of the Green function
        """
        return self.atom_COOP_from_orbital(COHP, uc)

    def atom_COHP(self, E, kavg=True, isc=None, uc=False):
        r""" Atomic COHP curve of the Green function

        Parameters
        ----------
        E: float or int
           the energy or the energy index of COHP. If an integer
           is passed it is the index, otherwise the index corresponding to
           ``Eindex(E)`` is used.
        kavg: bool, int, optional
           whether the returned COHP is k-averaged, or an explicit (unweighed) k-point
           is returned
        isc: array_like, optional
           the returned COHP from unit-cell (``[None, None, None]``) to
           the given supercell, the default is all COHP for the supercell.
           To only get unit cell orbital currents, pass ``[0, 0, 0]``.
        uc : bool, optional
           whether the returned COHP are only in the unit-cell.
           If ``True`` this will return a sparse matrix of ``shape = (self.na, self.na)``,
           else, it will return a sparse matrix of ``shape = (self.na, self.na * self.n_s)``.
           One may figure out the connections via `~sisl.geometry.Geometry.sc_index`.

        See Also
        --------
        orbital_COHP : orbital resolved COHP analysis of the Green function
        atom_COHP_from_orbital : transfer an orbital COHP to atomic COHP
        atom_ACOHP : atomic COHP analysis of the spectral function
        atom_COOP : atomic COOP analysis of the Green function
        """
        return self.atom_ACOHP(None, E, kavg, isc, uc)

    def atom_ACOHP(self, elec, E, kavg=True, isc=None, uc=False):
        r""" Atomic COHP curve of the spectral function

        Parameters
        ----------
        elec: str or int
           the electrode of the spectral function
        E: float or int
           the energy or the energy index of COHP. If an integer
           is passed it is the index, otherwise the index corresponding to
           ``Eindex(E)`` is used.
        kavg: bool, int, optional
           whether the returned COHP is k-averaged, or an explicit (unweighed) k-point
           is returned
        isc: array_like, optional
           the returned COHP from unit-cell (``[None, None, None]``) to
           the given supercell, the default is all COHP for the supercell.
           To only get unit cell orbital currents, pass ``[0, 0, 0]``.
        uc : bool, optional
           whether the returned COHP are only in the unit-cell.
           If ``True`` this will return a sparse matrix of ``shape = (self.na, self.na)``,
           else, it will return a sparse matrix of ``shape = (self.na, self.na * self.n_s)``.
           One may figure out the connections via `~sisl.geometry.Geometry.sc_index`.

        See Also
        --------
        orbital_COHP : orbital resolved COHP analysis of the Green function
        atom_COHP_from_orbital : transfer an orbital COHP to atomic COHP
        atom_COHP : atomic COHP analysis of the Green function
        atom_ACOOP : atomic COOP analysis of the spectral function
        """
        COHP = self.orbital_ACOHP(elec, E, kavg, isc)
        return self.atom_COHP_from_orbital(COHP, uc)

    def read_data(self, *args, **kwargs):
        """ Read specific type of data.

        This is a generic routine for reading different parts of the data-file.

        Parameters
        ----------
        geom: bool, optional
           return the geometry
        atom_current: bool, optional
           return the atomic current flowing through an atom (the *activity* current)
        vector_current: bool, optional
           return the orbital currents as vectors
        """
        val = []
        for kw in kwargs:

            if kw in ['geom', 'geometry']:
                if kwargs[kw]:
                    val.append(self.read_geometry())

            elif kw == 'atom_current':
                if kwargs[kw]:
                    # TODO we need some way of handling arguments.
                    val.append(self.atom_current(*args))

            elif kw == 'vector_current':
                if kwargs[kw]:
                    # TODO we need some way of handling arguments.
                    val.append(self.vector_current(*args))

        if len(val) == 0:
            val = None
        elif len(val) == 1:
            val = val[0]
        return val

    def info(self, elec=None):
        """ Information about the calculated quantities available for extracting in this file

        Parameters
        ----------
        elec : str or int
           the electrode to request information from
        """
        if not elec is None:
            elec = self._elec(elec)

        # Create a StringIO object to retain the information
        out = StringIO()
        # Create wrapper function
        def prnt(*args, **kwargs):
            option = kwargs.pop('option', None)
            if option is None:
                print(*args, file=out)
            else:
                print('{:60s}[{}]'.format(' '.join(args), ', '.join(option)), file=out)

        def truefalse(bol, string, fdf=None):
            if bol:
                prnt("  + " + string + ": true")
            else:
                prnt("  - " + string + ": false", option=fdf)

        # Retrieve the device atoms
        prnt("Device information:")
        if self._k_avg:
            prnt("  - all data is k-averaged")
        else:
            # Print out some more information related to the
            # k-point sampling.
            # However, we still do not know whether TRS is
            # applied.
            kpt = self.k
            nA = len(np.unique(kpt[:, 0]))
            nB = len(np.unique(kpt[:, 1]))
            nC = len(np.unique(kpt[:, 2]))
            prnt(("  - number of kpoints: {} <- "
                   "[ A = {} , B = {} , C = {} ] (time-reversal unknown)").format(self.nk, nA, nB, nC))
        prnt("  - energy range:")
        E = self.E
        Em, EM = np.amin(E), np.amax(E)
        dE = np.diff(E)
        dEm, dEM = np.amin(dE) * 1000, np.amax(dE) * 1000 # convert to meV
        if (dEM - dEm) < 1e-3: # 0.001 meV
            prnt("     {:.5f} -- {:.5f} eV  [{:.3f} meV]".format(Em, EM, dEm))
        else:
            prnt("     {:.5f} -- {:.5f} eV  [{:.3f} -- {:.3f} meV]".format(Em, EM, dEm, dEM))
        prnt("  - imaginary part (eta): {:.4f} meV".format(self.eta() * 1e3))
        prnt("  - atoms with DOS (fortran indices):")
        prnt("     " + list2str(self.a_dev + 1))
        prnt("  - number of BTD blocks: {}".format(self.n_btd()))
        truefalse('DOS' in self.variables, "DOS Green function", ['TBT.DOS.Gf'])
        truefalse('DM' in self.variables, "Density matrix Green function", ['TBT.DM.Gf'])
        truefalse('COOP' in self.variables, "COOP Green function", ['TBT.COOP.Gf'])
        truefalse('COHP' in self.variables, "COHP Green function", ['TBT.COHP.Gf'])
        if elec is None:
            elecs = self.elecs
        else:
            elecs = [elec]

        # Print out information for each electrode
        for elec in elecs:
            if not elec in self.groups:
                prnt("  * no information available")
                continue

            try:
                bloch = self.bloch(elec)
            except:
                bloch = [1] * 3
            try:
                n_btd = self.n_btd(elec)
            except:
                n_btd = 'unknown'
            prnt()
            prnt("Electrode: {}".format(elec))
            prnt("  - number of BTD blocks: {}".format(n_btd))
            prnt("  - Bloch: [{}, {}, {}]".format(*bloch))
            gelec = self.groups[elec]
            if 'TBT' in self._trans_type:
                prnt("  - chemical potential: {:.4f} eV".format(self.chemical_potential(elec)))
                prnt("  - electron temperature: {:.2f} K".format(self.electron_temperature(elec)))
            else:
                prnt("  - phonon temperature: {:.4f} K".format(self.phonon_temperature(elec)))
            prnt("  - imaginary part (eta): {:.4f} meV".format(self.eta(elec) * 1e3))
            truefalse('DOS' in gelec.variables, "DOS bulk", ['TBT.DOS.Elecs'])
            truefalse('ADOS' in gelec.variables, "DOS spectral", ['TBT.DOS.A'])
            truefalse('J' in gelec.variables, "orbital-current", ['TBT.Current.Orb'])
            truefalse('DM' in gelec.variables, "Density matrix spectral", ['TBT.DM.A'])
            truefalse('COOP' in gelec.variables, "COOP spectral", ['TBT.COOP.A'])
            truefalse('COHP' in gelec.variables, "COHP spectral", ['TBT.COHP.A'])
            truefalse('T' in gelec.variables, "transmission bulk", ['TBT.T.Bulk'])
            truefalse(elec + '.T' in gelec.variables, "transmission out", ['TBT.T.Out'])
            truefalse(elec + '.C' in gelec.variables, "transmission out correction", ['TBT.T.Out'])
            truefalse(elec + '.C.Eig' in gelec.variables, "transmission out correction (eigen)", ['TBT.T.Out', 'TBT.T.Eig'])
            for elec2 in self.elecs:
                # Skip it self, checked above in .T and .C
                if elec2 == elec:
                    continue
                truefalse(elec2 + '.T' in gelec.variables, "transmission -> " + elec2)
                truefalse(elec2 + '.T.Eig' in gelec.variables, "transmission (eigen) -> " + elec2, ['TBT.T.Eig'])

        s = out.getvalue()
        out.close()
        return s

    @default_ArgumentParser(description="Extract data from a TBT.nc file")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """

        # We limit the import to occur here
        import argparse

        namespace = default_namespace(_tbt=self,
                                      _geometry=self.geom,
                                      _data=[], _data_description=[], _data_header=[],
                                      _norm='none',
                                      _Ovalue='', _Orng=None, _Erng=None,
                                      _krng=True)

        def ensure_E(func):
            """ This decorater ensures that E is the first element in the _data container """

            def assign_E(self, *args, **kwargs):
                ns = args[1]
                if len(ns._data) == 0:
                    # We immediately extract the energies
                    ns._data.append(ns._tbt.E[ns._Erng].flatten())
                    ns._data_header.append('Energy[eV]')
                return func(self, *args, **kwargs)
            return assign_E

        # Correct the geometry species information
        class GeometryAction(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):

                old_g = ns._geometry.copy()

                # Now read the file to read the geometry from
                g = Geometry.read(value)

                # Make sure g has the same # of orbitals
                atoms = [None] * len(old_g)
                for a, idx in g.atom:
                    for i in idx:
                        atoms[i] = a.copy(orbital=old_g.atoms[i].R)
                g._atoms = Atoms(atoms)

                ns._geometry = g
        p.add_argument('--geometry', '-G',
                       action=GeometryAction,
                       help=('Update the geometry of the output file, this enables one to set the species correctly,'
                             ' note this only affects output-files where species are important'))

        class ERange(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                E = ns._tbt.E
                Emap = strmap(float, value, E.min(), E.max())
                # Convert to actual indices
                E = []
                for begin, end in Emap:
                    if begin is None and end is None:
                        ns._Erng = None
                        return
                    elif begin is None:
                        E.append(range(ns._tbt.Eindex(end)+1))
                    elif end is None:
                        E.append(range(ns._tbt.Eindex(begin), len(ns._tbt.E)))
                    else:
                        E.append(range(ns._tbt.Eindex(begin), ns._tbt.Eindex(end)+1))
                # Issuing unique also sorts the entries
                ns._Erng = np.unique(_a.arrayi(E).flatten())
        p.add_argument('--energy', '-E', action=ERange,
                       help="""Denote the sub-section of energies that are extracted: "-1:0,1:2" [eV]

                       This flag takes effect on all energy-resolved quantities and is reset whenever --plot or --out is called""")

        # k-range
        class kRange(argparse.Action):

            @collect_action
            def __call__(self, parser, ns, value, option_string=None):
                try:
                    ns._krng = int(value)
                except:
                    # Parse it as an array
                    if ',' in value:
                        k = map(float, value.split(','))
                    else:
                        k = map(float, value.split())
                    k = list(k)
                    if len(k) != 3:
                        raise ValueError('Argument --kpoint *must* be an integer or 3 values to find the corresponding k-index')
                    ns._krng = ns._tbt.kindex(k)
                # Add a description on which k-point this is
                k = ns._tbt.k[ns._krng]
                ns._data_description.append('Data is extracted at k-point: [{} {} {}]'.format(k[0], k[1], k[2]))

        if not self._k_avg:
            p.add_argument('--kpoint', '-k', action=kRange,
                           help="""Denote a specific k-index or comma/white-space separated k-point that is extracted, default to k-averaged quantity.
                           For specific k-points the k weight will not be used.

                           This flag takes effect on all k-resolved quantities and is reset whenever --plot or --out is called""")

        # The normalization method
        class NormAction(argparse.Action):

            @collect_action
            def __call__(self, parser, ns, value, option_string=None):
                ns._norm = value
        p.add_argument('--norm', '-N', action=NormAction, default='atom',
                       choices=['none', 'atom', 'orbital', 'all'],
                       help="""Specify the normalization method; "none") no normalization, "atom") total orbitals in selected atoms,
                       "orbital") selected orbitals or "all") total orbitals in the device region.

                       This flag only takes effect on --dos and --ados and is reset whenever --plot or --out is called""")

        # Try and add the atomic specification
        class AtomRange(argparse.Action):

            @collect_action
            def __call__(self, parser, ns, value, option_string=None):
                value = value.replace(' ', '')

                # Immediately convert to proper indices
                geom = ns._geometry
                a_dev = ns._tbt.a_dev[:] + 1

                # Sadly many shell interpreters does not
                # allow simple [] because they are expansion tokens
                # in the shell.
                # We bypass this by allowing *, [, {
                # * will "only" fail if files are named accordingly, else
                # it will be passed as-is.
                #       {    [    *
                sep = ['c', 'b', '*']
                failed = True
                while failed and len(sep) > 0:
                    try:
                        ranges = lstranges(strmap(int, value, a_dev.min(), a_dev.max(), sep.pop()))
                        failed = False
                    except:
                        pass
                if failed:
                    print(value)
                    raise ValueError("Could not parse the atomic/orbital ranges")

                # we have only a subset of the orbitals
                orbs = []
                no = 0
                asarrayi = _a.asarrayi
                for atoms in ranges:
                    if isinstance(atoms, list):
                        # this will be
                        #  atoms[0] == atom
                        #  atoms[1] == list of orbitals on the atom
                        if atoms[0] not in a_dev:
                            continue

                        # Get atoms and orbitals
                        ob = geom.a2o(atoms[0] - 1, True)
                        # We normalize for the total number of orbitals
                        # on the requested atoms.
                        # In this way the user can compare directly the DOS
                        # for same atoms with different sets of orbitals and the
                        # total will add up.
                        no += len(ob)
                        ob = ob[asarrayi(atoms[1]) - 1]
                    else:
                        if atoms not in a_dev:
                            continue
                        ob = geom.a2o(atoms - 1, True)
                        no += len(ob)
                    orbs.append(ob)

                if len(orbs) == 0:
                    print('Device atoms:')
                    print('  ', list2str(a_dev))
                    print('Input atoms:')
                    print('  ', value)
                    raise ValueError('Atomic/Orbital requests are not fully included in the device region.')

                # Add one to make the c-index equivalent to the f-index
                orbs = np.concatenate(orbs).flatten()

                # Check that the requested orbitals are all in the device region
                if len(orbs) != len(ns._tbt.o2p(orbs)):
                    # This should in principle never be called because of the
                    # checks above.
                    print('Device atoms:')
                    print('  ', list2str(a_dev))
                    print('Input atoms:')
                    print('  ', value)
                    raise ValueError('Atomic/Orbital requests are not fully included in the device region.')

                ns._Ovalue = value
                ns._Orng = orbs

        p.add_argument('--atom', '-a', type=str, action=AtomRange,
                       help="""Limit orbital resolved quantities to a sub-set of atoms/orbitals: "1-2[3,4]" will yield the 1st and 2nd atom and their 3rd and fourth orbital. Multiple comma-separated specifications are allowed. Note that some shells does not allow [] as text-input (due to expansion), {, [ or * are allowed orbital delimiters.

                       This flag takes effect on all atom/orbital resolved quantities (except BDOS, transmission_bulk) and is reset whenever --plot or --out is called""")

        class DataT(argparse.Action):

            @collect_action
            @ensure_E
            def __call__(self, parser, ns, values, option_string=None):
                e1 = ns._tbt._elec(values[0])
                if e1 not in ns._tbt.elecs:
                    raise ValueError('Electrode: "'+e1+'" cannot be found in the specified file.')
                e2 = ns._tbt._elec(values[1])
                if e2 not in ns._tbt.elecs:
                    if e2.strip() == '.':
                        for e2 in ns._tbt.elecs:
                            if e2 != e1:
                                try: # catches if T isn't calculated
                                    self(parser, ns, [e1, e2], option_string)
                                except:
                                    pass
                        return
                    raise ValueError('Electrode: "'+e2+'" cannot be found in the specified file.')

                # Grab the information
                data = ns._tbt.transmission(e1, e2, kavg=ns._krng)[ns._Erng]
                data.shape = (-1,)
                ns._data.append(data)
                ns._data_header.append('T[G0]:{}-{}'.format(e1, e2))
                ns._data_description.append('Column {} is transmission from {} to {}'.format(len(ns._data), e1, e2))
        p.add_argument('-T', '--transmission', nargs=2, metavar=('ELEC1', 'ELEC2'),
                       action=DataT,
                       help='Store transmission between two electrodes.')

        class DataBT(argparse.Action):

            @collect_action
            @ensure_E
            def __call__(self, parser, ns, value, option_string=None):
                e = ns._tbt._elec(value[0])
                if e not in ns._tbt.elecs:
                    if e.strip() == '.':
                        for e in ns._tbt.elecs:
                            try: # catches if B isn't calculated
                                self(parser, ns, [e], option_string)
                            except:
                                pass
                        return
                    raise ValueError('Electrode: "'+e+'" cannot be found in the specified file.')

                # Grab the information
                data = ns._tbt.transmission_bulk(e, kavg=ns._krng)[ns._Erng]
                data.shape = (-1,)
                ns._data.append(data)
                ns._data_header.append('BT[G0]:{}'.format(e))
                ns._data_description.append('Column {} is bulk-transmission'.format(len(ns._data)))
        p.add_argument('-BT', '--transmission-bulk', nargs=1, metavar='ELEC',
                       action=DataBT,
                       help='Store bulk transmission of an electrode.')

        class DataDOS(argparse.Action):

            @collect_action
            @ensure_E
            def __call__(self, parser, ns, value, option_string=None):
                if not value is None:
                    # we are storing the spectral DOS
                    e = ns._tbt._elec(value)
                    if e not in ns._tbt.elecs:
                        raise ValueError('Electrode: "'+e+'" cannot be found in the specified file.')
                    data = ns._tbt.ADOS(e, kavg=ns._krng, orbital=ns._Orng, norm=ns._norm)
                    ns._data_header.append('ADOS[1/eV]:{}'.format(e))
                else:
                    data = ns._tbt.DOS(kavg=ns._krng, orbital=ns._Orng, norm=ns._norm)
                    ns._data_header.append('DOS[1/eV]')
                NORM = int(ns._tbt.norm(orbital=ns._Orng, norm=ns._norm))

                # The flatten is because when ns._Erng is None, then a new
                # dimension (of size 1) is created
                ns._data.append(data[ns._Erng].flatten())
                if ns._Orng is None:
                    ns._data_description.append('Column {} is sum of all device atoms+orbitals with normalization 1/{}'.format(len(ns._data), NORM))
                else:
                    ns._data_description.append('Column {} is atoms[orbs] {} with normalization 1/{}'.format(len(ns._data), ns._Ovalue, NORM))

        p.add_argument('--dos', '-D', nargs='?', metavar='ELEC',
                       action=DataDOS, default=None,
                       help="""Store DOS. If no electrode is specified, it is Green function, else it is the spectral function.""")
        p.add_argument('--ados', '-AD', metavar='ELEC',
                       action=DataDOS, default=None,
                       help="""Store spectral DOS, same as --dos but requires an electrode-argument.""")

        class DataDOSBulk(argparse.Action):

            @collect_action
            @ensure_E
            def __call__(self, parser, ns, value, option_string=None):

                # we are storing the Bulk DOS
                e = ns._tbt._elec(value[0])
                if e not in ns._tbt.elecs:
                    raise ValueError('Electrode: "'+e+'" cannot be found in the specified file.')
                # Grab the information
                data = ns._tbt.BDOS(e, kavg=ns._krng, sum=False)
                ns._data_header.append('BDOS[1/eV]:{}'.format(e))
                # Select the energies, even if _Erng is None, this will work!
                no = data.shape[-1]
                data = np.mean(data[ns._Erng, ...], axis=-1).flatten()
                ns._data.append(data)
                ns._data_description.append('Column {} is sum of all electrode[{}] atoms+orbitals with normalization 1/{}'.format(len(ns._data), e, no))
        p.add_argument('--bulk-dos', '-BD', nargs=1, metavar='ELEC',
                       action=DataDOSBulk, default=None,
                       help="""Store bulk DOS of an electrode.""")

        class DataTEig(argparse.Action):

            @collect_action
            @ensure_E
            def __call__(self, parser, ns, values, option_string=None):
                e1 = ns._tbt._elec(values[0])
                if e1 not in ns._tbt.elecs:
                    raise ValueError('Electrode: "'+e1+'" cannot be found in the specified file.')
                e2 = ns._tbt._elec(values[1])
                if e2 not in ns._tbt.elecs:
                    if e2.strip() == '.':
                        for e2 in ns._tbt.elecs:
                            if e1 != e2:
                                try: # catches if T-eig isn't calculated
                                    self(parser, ns, [e1, e2], option_string)
                                except:
                                    pass
                        return
                    raise ValueError('Electrode: "'+e2+'" cannot be found in the specified file.')

                # Grab the information
                data = ns._tbt.transmission_eig(e1, e2, kavg=ns._krng)
                # The shape is: E, neig
                neig = data.shape[-1]
                for eig in range(neig):
                    ns._data.append(data[ns._Erng, ..., eig].flatten())
                    ns._data_header.append('Teig({})[G0]:{}-{}'.format(eig+1, e1, e2))
                    ns._data_description.append('Column {} is transmission eigenvalues from electrode {} to {}'.format(len(ns._data), e1, e2))
        p.add_argument('--transmission-eig', '-Teig', nargs=2, metavar=('ELEC1', 'ELEC2'),
                       action=DataTEig,
                       help='Store transmission eigenvalues between two electrodes.')

        class Info(argparse.Action):
            """ Action to print information contained in the TBT.nc file, helpful before performing actions """

            def __call__(self, parser, ns, value, option_string=None):
                # First short-hand the file
                print(ns._tbt.info(value))

        p.add_argument('--info', '-i', action=Info, nargs='?', metavar='ELEC',
                       help='Print out what information is contained in the TBT.nc file, optionally only for one of the electrodes.')

        class Out(argparse.Action):

            @run_actions
            def __call__(self, parser, ns, value, option_string=None):

                out = value[0]

                try:
                    # We figure out if the user wants to write
                    # to a geometry
                    obj = get_sile(out, mode='w')
                    if hasattr(obj, 'write_geometry'):
                        with obj as fh:
                            fh.write_geometry(ns._geometry)
                        return
                    raise NotImplementedError
                except:
                    pass

                if len(ns._data) == 0:
                    # do nothing if data has not been collected
                    print("No data has been collected in the arguments, nothing will be written, have you forgotten arguments?")
                    return

                from sisl.io import tableSile
                tableSile(out, mode='w').write(*ns._data,
                                               comment=ns._data_description,
                                               header=ns._data_header)
                # Clean all data
                ns._data_description = []
                ns._data_header = []
                ns._data = []
                # These are expert options
                ns._norm = 'none'
                ns._Ovalue = ''
                ns._Orng = None
                ns._Erng = None
                ns._krng = True
        p.add_argument('--out', '-o', nargs=1, action=Out,
                       help='Store currently collected information (at its current invocation) to the out file.')

        class AVOut(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                if value is None:
                    ns._tbt.write_tbtav()
                else:
                    ns._tbt.write_tbtav(value)
        p.add_argument('--tbt-av', action=AVOut, nargs='?', default=None,
                       help='Create "{0}" with the k-averaged quantities of this file.'.format(self.file.replace('TBT.nc', 'TBT.AV.nc')))

        class Plot(argparse.Action):

            @run_actions
            def __call__(self, parser, ns, value, option_string=None):

                if len(ns._data) == 0:
                    # do nothing if data has not been collected
                    print("No data has been collected in the arguments, nothing will be plotted, have you forgotten arguments?")
                    return

                from matplotlib import pyplot as plt
                plt.figure()

                def _get_header(header):
                    val_info = header.split(':')
                    if len(val_info) == 1:
                        # We smiply have the data
                        return val_info[0].split('[')[0]
                    # We have a value *and* the electrode
                    return '{}:{}'.format(val_info[0].split('[')[0], val_info[1])

                is_DOS = True
                is_T = True
                for i in range(1, len(ns._data)):
                    plt.plot(ns._data[0], ns._data[i], label=_get_header(ns._data_header[i]))
                    is_DOS &= 'DOS' in ns._data_header[i]
                    is_T &= 'T' in ns._data_header[i]

                if is_DOS:
                    plt.ylabel('DOS [1/eV]')
                elif is_T:
                    plt.ylabel('Transmission [G0]')
                else:
                    plt.ylabel('mixed units')
                plt.xlabel('E - E_F [eV]')

                plt.legend(loc=8, ncol=3, bbox_to_anchor=(0.5, 1.0))
                if value is None:
                    plt.show()
                else:
                    plt.savefig(value)

                # Clean all data
                ns._data_description = []
                ns._data_header = []
                ns._data = []
                # These are expert options
                ns._norm = 'none'
                ns._Ovalue = ''
                ns._Orng = None
                ns._Erng = None
                ns._krng = True
        p.add_argument('--plot', '-p', action=Plot, nargs='?', metavar='FILE',
                       help='Plot the currently collected information (at its current invocation).')

        return p, namespace


# The average files
# These are essentially equivalent to the TBT.nc files
# with the exception that the k-points have been averaged out.
class tbtavncSileTBtrans(tbtncSileTBtrans):
    """ TBtrans average file object

    This `Sile` implements the writing of the TBtrans output ``*.TBT.AV.nc`` sile which contains
    the k-averaged quantities related to the NEGF code TBtrans.

    See `tbtncSileTBtrans` for details as this object is essentially a copy of it.
    """
    _trans_type = 'TBT'
    _k_avg = True
    _E2eV = Ry2eV

    @property
    def nkpt(self):
        """ Always return 1, this is to signal other routines """
        return 1

    @property
    def wkpt(self):
        """ Always return [1.], this is to signal other routines """
        return _a.onesd(1)

    def write_tbtav(self, *args, **kwargs):
        """ Wrapper for writing the k-averaged TBT.AV.nc file.

        This write *requires* the TBT.nc `Sile` object passed as the first argument,
        or as the keyword ``from=tbt`` argument.

        Parameters
        ----------
        from : tbtncSileTBtrans
          the TBT.nc file object that has the k-sampled quantities.
        """

        if 'from' in kwargs:
            tbt = kwargs['from']
        elif len(args) > 0:
            tbt = args[0]
        else:
            raise SislError("tbtncSileTBtrans has not been passed to write the averaged file")

        if not isinstance(tbt, tbtncSileTBtrans):
            raise ValueError('first argument of tbtavncSileTBtrans.write *must* be a tbtncSileTBtrans object')

        # Notify if the object is not in write mode.
        sile_raise_write(self)

        def copy_attr(f, t):
            t.setncatts({att: f.getncattr(att) for att in f.ncattrs()})

        # Retrieve k-weights
        nkpt = len(tbt.dimensions['nkpt'])
        wkpt = _a.asarrayd(tbt.variables['wkpt'][:])

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

    # Denote default writing routine
    _write_default = write_tbtav


for _name in ['shot_noise', 'noise_power', 'fano']:
    setattr(tbtavncSileTBtrans, _name, None)


add_sile('TBT.nc', tbtncSileTBtrans)
# Add spin-dependent files
add_sile('TBT_DN.nc', tbtncSileTBtrans)
add_sile('TBT_UP.nc', tbtncSileTBtrans)
add_sile('TBT.AV.nc', tbtavncSileTBtrans)
# Add spin-dependent files
add_sile('TBT_DN.AV.nc', tbtavncSileTBtrans)
add_sile('TBT_UP.AV.nc', tbtavncSileTBtrans)
