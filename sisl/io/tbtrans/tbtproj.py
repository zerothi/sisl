from __future__ import print_function, division

try:
    from StringIO import StringIO
except Exception:
    from io import StringIO
import numpy as np

from sisl.utils import list2str
from sisl.unit.siesta import unit_convert

from ..sile import add_sile
from .tbt import tbtncSileTBtrans


__all__ = ['tbtprojncSileTBtrans', 'phtprojncSileTBtrans']

Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')
Ry2K = unit_convert('Ry', 'K')
eV2Ry = unit_convert('eV', 'Ry')


class tbtprojncSileTBtrans(tbtncSileTBtrans):
    """ TBtrans projection file object """
    _trans_type = 'TBT.Proj'

    def _elec(self, mol_proj_elec):
        """ In projections we re-use the _* methods from tbtncSileTBtrans by forcing _elec to return its argument """
        return mol_proj_elec

    def eta(self):
        r""" Device region :math:`\eta` value """
        return self._value('eta')[0] * Ry2eV

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
                print('{:60s}[{}]'.format(' '.join(args), ', '.join(option)), file=out)

        def truefalse(bol, string, fdf=None, suf=2):
            if bol:
                true(string, fdf, suf)
            else:
                prnt("{}- {}: false".format(' ' * suf, string), option=fdf)

        def true(string, fdf=None, suf=2):
            prnt("{}+ {}: true".format(' ' * suf, string))

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
                prnt("    > Projection: {}".format(proj))
                prnt("      - number of states: {}".format(len(gproj.dimensions['nlvl'])))
                # Figure out the electrode projections
                elecs = gproj.groups.keys()
                for elec in elecs:
                    opt['elec1'] = elec
                    gelec = gproj.groups[elec]
                    vars = list(gelec.variables.keys()) # ensure a copy

                    # Loop and figure out what is in it.
                    if 'ADOS' in vars:
                        vars.pop(vars.index('ADOS'))
                        true("DOS spectral {elec1}|{proj1}".format(**opt), ['TBT.Projs.DOS.A'], suf=6)
                    if 'J' in vars:
                        vars.pop(vars.index('J'))
                        true("orbital-current {elec1}|{proj1}".format(**opt), ['TBT.Projs.Current.Orb'], suf=6)
                    if 'DM' in vars:
                        vars.pop(vars.index('DM'))
                        true("Density matrix spectral {elec1}|{proj1}".format(**opt), ['TBT.Projs.DM.A'], suf=6)
                    if 'COOP' in vars:
                        vars.pop(vars.index('COOP'))
                        true("COOP spectral {elec1}|{proj1}".format(**opt), ['TBT.Projs.COOP.A'], suf=6)
                    if 'COHP' in vars:
                        vars.pop(vars.index('COHP'))
                        true("COHP spectral {elec1}|{proj1}".format(**opt), ['TBT.Projs.COHP.A'], suf=6)

                    # Retrieve all vars with transmissions
                    vars_T = _get_all('.T', vars)
                    vars_Teig = _get_all('.T.Eig', vars)
                    vars_C = _get_all('.C', vars)
                    vars_Ceig = _get_all('.C.Eig', vars)

                    _print_to_full("      + transmission: {elec1}|{mol1}.{proj1}".format(**opt), vars_T)
                    _print_to_full("      + transmission (eigen): {elec1}|{mol1}.{proj1}".format(**opt), vars_Teig)
                    _print_to_full("      + transmission out corr.: {elec1}|{mol1}.{proj1}".format(**opt), vars_C)
                    _print_to_full("      + transmission out corr. (eigen): {elec1}|{mol1}.{proj1}".format(**opt), vars_Ceig)

        # Finally there may be only RHS projections in which case the remaining groups are for
        # *pristine* electrodes
        elecs = [elec for elec in self.groups.keys() if len(self.groups[elec].groups) == 0]
        for elec in elecs:
            gelec = self.groups[elec]
            vars = list(gelec.variables.keys()) # ensure a copy
            prnt("")
            prnt("Electrode: {}".format(elec))

            # Retrieve all vars with transmissions
            vars_T = _get_all('.T', vars)
            vars_Teig = _get_all('.T.Eig', vars)
            vars_C = _get_all('.C', vars)
            vars_Ceig = _get_all('.C.Eig', vars)

            _print_to_full("  + transmission: {elec}".format(elec=elec), vars_T)
            _print_to_full("  + transmission (eigen): {elec}".format(elec=elec), vars_Teig)
            _print_to_full("  + transmission out corr.: {elec}".format(elec=elec), vars_C)
            _print_to_full("  + transmission out corr. (eigen): {elec}".format(elec=elec), vars_Ceig)

        s = out.getvalue()
        out.close()
        return s


#    def eigenstate(self, molecule, k=None, all=True):
#        r""" Return the eigenstate on the projected `molecule`
#
#        The eigenstate object will contain the geometry as the parent object.
#        The eigenstate will be in the Lowdin basis:
#        .. math::
#            |\psi'_i\rangle = \mathbf S^{1/2} |\psi_i\rangle
#
#        Parameters
#        ----------
#        molecule : str
#           name of the molecule to retrieve the eigenstate from
#        k : optional
#           k-index for retrieving a specific k-point (default to all)
#        all : bool, optional
#           whether all states should be returned
#
#        Returns
#        -------
#        EigenstateElectron
#        """
#        if 'PHT' in self._trans_type:
#            from sisl.physics import EigenmodePhonon as cls
#        else:
#            from sisl.physics import EigenstateElectronn as cls
#
#        mol = self.groups[molecule]
#        suf = 'state'
#        if all:
#            if ('states' not in mol.variable) and ('Restates' not in mol.variable):
#                suf = 'states'
#
#        is_gamma = suf in mol.variable
#        if is_gamma:
#            state = mol.variable['Re' + suf][:] + 1j * mol.variable['Im' + suf][:]
#        else:
#            state = mol.variable[suf][:]
#        eig = mol.variable['eig'][:]


class phtprojncSileTBtrans(tbtprojncSileTBtrans):
    """ PHtrans projection file object """
    _trans_type = 'PHT.Proj'


# Clean up methods
for _name in ['elecs', 'chemical_potential', 'mu',
              'electron_temperature', 'kT',
              'current', 'current_parameter',
              'shot_noise', 'fano', 'density_matrix',
              'orbital_COOP', 'atom_COOP',
              'orbital_COHP', 'atom_COHP']:
    setattr(tbtprojncSileTBtrans, _name, None)


add_sile('TBT.Proj.nc', tbtprojncSileTBtrans)
# Add spin-dependent files
add_sile('TBT_DN.Proj.nc', tbtprojncSileTBtrans)
add_sile('TBT_UP.Proj.nc', tbtprojncSileTBtrans)

add_sile('PHT.Proj.nc', phtprojncSileTBtrans)
