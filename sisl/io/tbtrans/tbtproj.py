from __future__ import print_function, division

try:
    from StringIO import StringIO
except Exception:
    from io import StringIO
import numpy as np

from sisl._help import _str
from sisl.utils import *
from sisl.unit.siesta import unit_convert

from ..sile import add_sile
from .tbt import tbtncSileTBtrans


__all__ = ['tbtprojncSileTBtrans']

Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')
Ry2K = unit_convert('Ry', 'K')
eV2Ry = unit_convert('eV', 'Ry')


class tbtprojncSileTBtrans(tbtncSileTBtrans):
    """ TBtrans projection file object """
    _trans_type = 'TBT.Proj'

    @classmethod
    def _mol_proj_elec(self, elec_mol_proj):
        """ Parse the electrode-molecule-projection str/tuple into the molecule-projected-electrode

        Parameters
        ----------
        elec_mol_proj : str or tuple
           electrode-molecule-projection
        """
        if isinstance(elec_mol_proj, _str):
            elec_mol_proj = elec_mol_proj.split('.')
        if len(elec_mol_proj) == 1:
            return elec_mol_proj
        return [elec_mol_proj[i] for i in [1, 2, 0]]

    @property
    def elecs(self):
        """ List of electrodes """
        elecs = []

        # in cases of not calculating all
        # electrode transmissions we must ensure that
        # we add the last one
        for group in self.groups.keys():
            if group in elecs:
                continue
            if 'mu' in self.groups[group].variables.keys():
                elecs.append(group)
        return elecs

    @property
    def molecules(self):
        """ List of regions where state projections may happen """
        mols = []
        for mol in self.groups.keys():
            if len(self.groups[mol].groups) > 0:
                # this is a group with groups!
                mols.append(mol)
        return mols

    def projections(self, molecule):
        """ List of projections on `molecule`

        Parameters
        ----------
        molecule : str
            name of molecule to retrieve projections on
        """
        mol = self.groups[molecule]
        return list(mol.groups.keys())

    def ADOS(self, elec_mol_proj, E=None, kavg=True, atom=None, orbital=None, sum=True, norm='none'):
        r""" Projected spectral density of states (DOS) (1/eV)

        Extract the projected spectral DOS from electrode `elec` on a selected subset of atoms/orbitals in the device region

        .. math::
           \mathrm{ADOS}_\mathfrak{el}(E) = \frac{1}{2\pi N} \sum_{\nu\in \mathrm{atom}/\mathrm{orbital}} [\mathbf{G}(E)|i\rangle\langle i|\Gamma_\mathfrak{el}|i\rangle\langle i|\mathbf{G}^\dagger]_{\nu\nu}(E)

        where :math:`|i\rangle` may be a sum of states.
        The normalization constant (:math:`N`) is defined in the routine `norm` and depends on the
        arguments.

        Parameters
        ----------
        elec_mol_proj: str or tuple
           originating projected spectral function (<electrode>.<molecule>.<projection>)
        E : float or int, optional
           optionally only return the DOS of atoms at a given energy point
        kavg: bool, int or array_like, optional
           whether the returned DOS is k-averaged, an explicit k-point
           or a selection of k-points
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
        """
        mol_proj_elec = self._mol_proj_elec(elec_mol_proj)
        return self._DOS(self._value_E('ADOS', mol_proj_elec, kavg=kavg, E=E), atom, orbital, sum, norm) * eV2Ry

    def transmission(self, elec_mol_proj_from, elec_mol_proj_to, kavg=True):
        """ Transmission from `mol_proj_elec_from` to `mol_proj_elec_to`

        Parameters
        ----------
        elec_mol_proj_from: str or tuple
           the originating scattering projection (<electrode>.<molecule>.<projection>)
        elec_mol_proj_to: str or tuple
           the absorbing scattering projection (<electrode>.<molecule>.<projection>)
        kavg: bool, int or array_like, optional
           whether the returned transmission is k-averaged, an explicit k-point
           or a selection of k-points

        See Also
        --------
        transmission_eig : projected transmission decomposed in eigenchannels
        """
        mol_proj_elec = self._mol_proj_elec(elec_mol_proj_from)
        if not isinstance(elec_mol_proj_to, _str):
            elec_mol_proj_to = '.'.join(elec_mol_proj_to)
        return self._value_avg(elec_mol_proj_to + '.T', mol_proj_elec, kavg=kavg)

    def transmission_eig(self, elec_mol_proj_from, elec_mol_proj_to, kavg=True):
        """ Transmission eigenvalues from `elec_mol_proj_from` to `elec_mol_proj_to`

        Parameters
        ----------
        elec_mol_proj_from: str or tuple
           the originating scattering projection (<electrode>.<molecule>.<projection>)
        elec_mol_proj_to: str or tuple
           the absorbing scattering projection (<electrode>.<molecule>.<projection>)
        kavg: bool, int or array_like, optional
           whether the returned transmission is k-averaged, an explicit k-point
           or a selection of k-points

        See Also
        --------
        transmission : projected transmission
        """
        mol_proj_elec = self._mol_proj_elec(elec_mol_proj_from)
        if not isinstance(elec_mol_proj_to, _str):
            elec_mol_proj_to = '.'.join(elec_mol_proj_to)
        return self._value_avg(elec_mol_proj_to + '.T.Eig', mol_proj_elec, kavg=kavg)

    def Adensity_matrix(self, elec_mol_proj, E, kavg=True, isc=None, geometry=None):
        r""" Projected spectral function density matrix at energy `E` (1/eV)

        The projected density matrix can be used to calculate the LDOS in real-space.

        The :math:`\mathrm{LDOS}(E, \mathbf r)` may be calculated using the `~sisl.physics.DensityMatrix.density`
        routine. Basically the LDOS in real-space may be calculated as

        .. math::
            \rho_{\mathbf A_{\mathfrak{el}}}(E, \mathbf r) = \frac{1}{2\pi}\sum_{\nu\mu}\phi_\nu(\mathbf r)\phi_\mu(\mathbf r) \Re[\mathbf A_{\mathfrak{el}, \nu\mu}(E)]

        where :math:`\phi` are the orbitals. Note that the broadening used in the TBtrans calculations
        ensures the broadening of the density, i.e. it should not be necessary to perform energy
        averages over the density matrices.

        Parameters
        ----------
        elec_mol_proj: str or tuple
           the projected electrode of originating electrons
        E : float or int
           the energy or the energy index of density matrix. If an integer
           is passed it is the index, otherwise the index corresponding to
           ``Eindex(E)`` is used.
        kavg: bool, int or array_like, optional
           whether the returned density matrix is k-averaged, an explicit k-point
           or a selection of k-points
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

        Returns
        -------
        DensityMatrix: the object containing the Geometry and the density matrix elements
        """
        mol_proj_elec = self._mol_proj_elec(elec_mol_proj)
        dm = self._sparse_data('DM', mol_proj_elec, E, kavg, isc) * eV2Ry
        # Now create the density matrix object
        geom = self.read_geometry()
        if geometry is None:
            DM = DensityMatrix.fromsp(geom, dm)
        else:
            if geom.no != geometry.no:
                raise ValueError(self.__class__.__name__ + '.Adensity_matrix requires input geometry to contain the correct number of orbitals. Please correct input!')
            DM = DensityMatrix.fromsp(geometry, dm)
        return DM

    def orbital_ACOOP(self, elec_mol_proj, E, kavg=True, isc=None):
        r""" Orbital COOP analysis of the projected spectral function

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
        elec_mol_proj: str or tuple
           the electrode of the spectral function
        E: float or int
           the energy or the energy index of COOP. If an integer
           is passed it is the index, otherwise the index corresponding to
           ``Eindex(E)`` is used.
        kavg: bool, int or array_like, optional
           whether the returned COOP is k-averaged, an explicit k-point
           or a selection of k-points
        isc: array_like, optional
           the returned COOP from unit-cell (``[None, None, None]``) to
           the given supercell, the default is all COOP for the supercell.
           To only get unit cell orbital currents, pass ``[0, 0, 0]``.

        Examples
        --------
        >>> ACOOP = tbt.orbital_ACOOP('Left.C60.HOMO', -1.0) # COOP @ E = -1 eV from ``Left.C60.HOMO`` spectral function
        >>> ACOOP[10, 11] # COOP value between the 11th and 12th orbital
        >>> ACOOP.sum(1).A[tbt.o_dev, 0] == tbt.ADOS(0, sum=False)[tbt.Eindex(-1.0)]
        >>> D = ACOOP.diagonal().sum()
        >>> ADBCOOP = ACOOP / D

        See Also
        --------
        atom_COOP_from_orbital : transfer an orbital COOP to atomic COOP
        atom_ACOOP : atomic COOP analysis of the projected spectral function
        atom_COHP_from_orbital : atomic COHP analysis from an orbital COHP
        orbital_ACOHP : orbital resolved COHP analysis of the projected  spectral function
        atom_ACOHP : atomic COHP analysis of the projected spectral function
        """
        mol_proj_elec = self._mol_proj_elec(elec_mol_proj)
        COOP = self._sparse_data('COOP', mol_proj_elec, E, kavg, isc) * eV2Ry
        return COOP

    def orbital_ACOHP(self, elec_mol_proj, E, kavg=True, isc=None):
        r""" Orbital COHP analysis of the projected spectral function

        This will return a sparse matrix, see ``scipy.sparse.csr_matrix`` for details.
        Each matrix element of the sparse matrix corresponds to the COHP of the
        underlying geometry.

        The COHP analysis can be written as:

        .. math::
            \mathrm{COHP}^{\mathbf A}_{\nu\mu} = \frac{1}{2\pi} \Re\big[\mathbf A_{\nu\mu}
                \mathbf H_{\nu\mu} \big]

        Parameters
        ----------
        elec_mol_proj: str or tuple
           the electrode of the projected spectral function
        E: float or int
           the energy or the energy index of COHP. If an integer
           is passed it is the index, otherwise the index corresponding to
           ``Eindex(E)`` is used.
        kavg: bool, int or array_like, optional
           whether the returned COHP is k-averaged, an explicit k-point
           or a selection of k-points
        isc: array_like, optional
           the returned COHP from unit-cell (``[None, None, None]``) to
           the given supercell, the default is all COHP for the supercell.
           To only get unit cell orbital currents, pass ``[0, 0, 0]``.

        See Also
        --------
        atom_COHP_from_orbital : atomic COHP analysis from an orbital COHP
        atom_ACOHP : atomic COHP analysis of the projected spectral function
        atom_COOP_from_orbital : transfer an orbital COOP to atomic COOP
        orbital_ACOOP : orbital resolved COOP analysis of the projected spectral function
        atom_ACOOP : atomic COOP analysis of the projected spectral function
        """
        mol_proj_elec = self._mol_proj_elec(elec_mol_proj)
        COHP = self._sparse_data('COHP', mol_proj_elec, E, kavg, isc)
        return COHP

    @default_ArgumentParser(description="Extract data from a TBT.Proj.nc file")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        p, namespace = super(tbtprojncSileTBtrans, self).ArgumentParser(p, *args, **kwargs)

        # We limit the import to occur here
        import argparse

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

        class InfoMols(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                print(' '.join(ns._tbt.molecules))
        p.add_argument('--molecules', '-M', nargs=0,
                       action=InfoMols,
                       help="""Show molecules in the projection file""")

        class InfoProjs(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                print(' '.join(ns._tbt.projections(value[0])))
        p.add_argument('--projections', '-P', nargs=1, metavar='MOL',
                       action=InfoProjs,
                       help="""Show projections on molecule.""")

        class DataDOS(argparse.Action):

            @collect_action
            @ensure_E
            def __call__(self, parser, ns, value, option_string=None):
                data = ns._tbt.ADOS(value, kavg=ns._krng, orbital=ns._Orng, norm=ns._norm)
                ns._data_header.append('ADOS[1/eV]:{}'.format(value))
                NORM = int(ns._tbt.norm(orbital=ns._Orng, norm=ns._norm))

                # The flatten is because when ns._Erng is None, then a new
                # dimension (of size 1) is created
                ns._data.append(data[ns._Erng].flatten())
                if ns._Orng is None:
                    ns._data_description.append('Column {} is sum of all device atoms+orbitals with normalization 1/{}'.format(len(ns._data), NORM))
                else:
                    ns._data_description.append('Column {} is atoms[orbs] {} with normalization 1/{}'.format(len(ns._data), ns._Ovalue, NORM))
        p.add_argument('--ados', '-AD', metavar='E.M.P',
                       action=DataDOS, default=None,
                       help="""Store projected spectral DOS""")

        class DataT(argparse.Action):

            @collect_action
            @ensure_E
            def __call__(self, parser, ns, values, option_string=None):
                elec_mol_proj1 = values[0]
                elec_mol_proj2 = values[1]

                # Grab the information
                data = ns._tbt.transmission(elec_mol_proj1, elec_mol_proj2, kavg=ns._krng)[ns._Erng]
                data.shape = (-1,)
                ns._data.append(data)
                ns._data_header.append('T[G0]:{}-{}'.format(elec_mol_proj1, elec_mol_proj2))
                ns._data_description.append('Column {} is transmission from {} to {}'.format(len(ns._data), elec_mol_proj1, elec_mol_proj2))
        p.add_argument('-T', '--transmission', nargs=2, metavar=('E.M.P1', 'E.M.P2'),
                       action=DataT,
                       help='Store transmission between two projections.')

        class DataTEig(argparse.Action):

            @collect_action
            @ensure_E
            def __call__(self, parser, ns, values, option_string=None):
                elec_mol_proj1 = values[0]
                elec_mol_proj2 = values[1]

                # Grab the information
                data = ns._tbt.transmission_eig(elec_mol_proj1, elec_mol_proj2, kavg=ns._krng)[ns._Erng]
                neig = data.shape[-1]
                for eig in range(neig):
                    ns._data.append(data[ns._Erng, ..., eig].flatten())
                    ns._data_header.append('Teig({})[G0]:{}-{}'.format(eig+1, elec_mol_proj1, elec_mol_proj2))
                    ns._data_description.append('Column {} is transmission eigenvalues from electrode {} to {}'.format(len(ns._data), elec_mol_proj1, elec_mol_proj2))
        p.add_argument('-Teig', '--transmission-eig', nargs=2, metavar=('E.M.P1', 'E.M.P2'),
                       action=DataTEig,
                       help='Store transmission eigenvalues between two projections.')

        return p, namespace

    def info(self, molecule=None):
        """ Information about the calculated quantities available for extracting in this file

        Parameters
        ----------
        molecule : str or int
           the molecule to request information from
        """
        # Create a StringIO object to retain the information
        out = StringIO()
        # Create wrapper function
        def prnt(*args, **kwargs):
            option = kwargs.pop('option', None)
            if option is None:
                print(*args, file=out)
            else:
                print('{:70s}[{}]'.format(' '.join(args), ', '.join(option)), file=out)

        def truefalse(bol, string, fdf=None, suf=2):
            if bol:
                true(string, fdf, suf)
            else:
                prnt("{}- {}: false".format(' ' * suf, string), option=fdf)

        def true(string, fdf=None, suf=2):
            prnt("{}+ {}: true".format(' ' * suf, string), option=fdf)

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
        if molecule is None:
            mols = self.molecules
        else:
            mols = [molecule]

        def _get_all(opt, vars):
            out = []
            indices = []
            for i, var in enumerate(vars):
                if var.endswith(opt):
                    out.append(var[:-len(opt)])
                    indices.append(i)
            indices.sort(reverse=True)
            for i in indices:
                vars.pop(i)
            return out

        def _print_to(ns, var):
            elec_mol_proj = var.split('.')
            if len(elec_mol_proj) == 1:
                prnt(" " * ns + "-> {elec}".format(elec=elec_mol_proj[0]))
            elif len(elec_mol_proj) == 3:
                elec2, mol2, proj2 = elec_mol_proj
                prnt(" " * ns + "-> {elec}|{mol}|{proj}".format(elec=elec2, mol=mol2, proj=proj2))

        def _print_to_full(s, vars):
            if len(vars) == 0:
                return
            ns = len(s)
            prnt(s)
            for var in vars:
                _print_to(ns, var)

        # Print out information for each electrode
        for mol in mols:
            opt = {'mol1': mol}
            gmol = self.groups[mol]
            prnt()
            prnt("Molecule: {}".format(mol))
            prnt("  - molecule atoms (fortran indices):")
            prnt("     " + list2str(gmol.variables['atom'][:]))

            projs = self.projections(mol)
            prnt("  - number of projections: {}".format(len(projs)))
            for proj in projs:
                opt['proj1'] = proj
                gproj = gmol.groups[proj]
                prnt("    > Projection: {mol1}|{proj1}".format(**opt))
                prnt("      - number of states: {}".format(len(gproj.dimensions['nlvl'])))
                # Figure out the electrode projections
                elecs = gproj.groups.keys()
                for elec in elecs:
                    opt['elec1'] = elec
                    gelec = gproj.groups[elec]
                    vars = list(gelec.variables.keys()) # ensure a copy
                    prnt("      > Electrode: {elec1}|{mol1}|{proj1}".format(**opt))

                    # Loop and figure out what is in it.
                    if 'ADOS' in vars:
                        vars.pop(vars.index('ADOS'))
                        true("DOS spectral", ['TBT.Projs.DOS.A'], suf=8)
                    if 'J' in vars:
                        vars.pop(vars.index('J'))
                        true("orbital-current", ['TBT.Projs.Current.Orb'], suf=8)
                    if 'DM' in vars:
                        vars.pop(vars.index('DM'))
                        true("Density matrix spectral", ['TBT.Projs.DM.A'], suf=8)
                    if 'COOP' in vars:
                        vars.pop(vars.index('COOP'))
                        true("COOP spectral", ['TBT.Projs.COOP.A'], suf=8)
                    if 'COHP' in vars:
                        vars.pop(vars.index('COHP'))
                        true("COHP spectral", ['TBT.Projs.COHP.A'], suf=8)

                    # Retrieve all vars with transmissions
                    vars_T = _get_all('.T', vars)
                    vars_Teig = _get_all('.T.Eig', vars)
                    vars_C = _get_all('.C', vars)
                    vars_Ceig = _get_all('.C.Eig', vars)

                    _print_to_full("        + transmission:", vars_T)
                    _print_to_full("        + transmission (eigen):", vars_Teig)
                    _print_to_full("        + transmission out corr.:", vars_C)
                    _print_to_full("        + transmission out corr. (eigen):", vars_Ceig)

        # Finally there may be only RHS projections in which case the remaining groups are for
        # *pristine* electrodes
        for elec in self.elecs:
            gelec = self.groups[elec]
            vars = list(gelec.variables.keys()) # ensure a copy

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

            # Retrieve all vars with transmissions
            vars_T = _get_all('.T', vars)
            vars_Teig = _get_all('.T.Eig', vars)
            vars_C = _get_all('.C', vars)
            vars_Ceig = _get_all('.C.Eig', vars)

            _print_to_full("  + transmission:", vars_T)
            _print_to_full("  + transmission (eigen):", vars_Teig)
            _print_to_full("  + transmission out corr.:", vars_C)
            _print_to_full("  + transmission out corr. (eigen):", vars_Ceig)

        s = out.getvalue()
        out.close()
        return s

    def eigenstate(self, molecule, k=None, all=True):
        r""" Return the eigenstate on the projected `molecule`

        The eigenstate object will contain the geometry as the parent object.
        The eigenstate will be in the Lowdin basis:

        .. math::
            |\psi'_i\rangle = \mathbf S^{1/2} |\psi_i\rangle

        Parameters
        ----------
        molecule : str
           name of the molecule to retrieve the eigenstate from
        k : optional
           k-index for retrieving a specific k-point (default to all)
        all : bool, optional
           whether all states should be returned

        Returns
        -------
        EigenstateElectron
        """
        if 'PHT' in self._trans_type:
            from sisl.physics import EigenmodePhonon as cls
        else:
            from sisl.physics import EigenstateElectron as cls

        mol = self.groups[molecule]
        if all and ('states' in mol.variables or 'Restates' in mol.variables):
            suf = 'states'
        else:
            all = False
            suf = 'state'

        is_gamma = suf in mol.variables
        if is_gamma:
            state = mol.variables[suf][:]
        else:
            state = mol.variables['Re' + suf][:] + 1j * mol.variables['Im' + suf][:]
        eig = mol.variables['eig'][:]

        if eig.ndim > 1:
            raise NotImplementedError(self.__class__.__name__ + ".eigenstate currently does not implement "
                                      "the k-point version.")

        geom = self.read_geometry()
        if all:
            return cls(state, eig, parent=geom)
        lvl = mol.variables['lvl'][:]
        lvl = np.where(lvl > 0, lvl - 1, lvl) + mol.HOMO_index
        return cls(state, eig[lvl], parent=geom)


for _name in ['current', 'current_parameter',
              'shot_noise', 'noise_power', 'fano',
              'density_matrix',
              'orbital_COOP', 'atom_COOP',
              'orbital_COHP', 'atom_COHP']:
    setattr(tbtprojncSileTBtrans, _name, None)


add_sile('TBT.Proj.nc', tbtprojncSileTBtrans)
# Add spin-dependent files
add_sile('TBT_DN.Proj.nc', tbtprojncSileTBtrans)
add_sile('TBT_UP.Proj.nc', tbtprojncSileTBtrans)
