from __future__ import print_function

import numpy as np

from sisl._help import xml_parse
from ..sile import add_sile
from .sile import SileSiesta
from sisl.messages import warn
from sisl._array import arrayd, emptyd
from sisl.atom import PeriodicTable, Atom, Atoms
from sisl.geometry import Geometry
from sisl.orbital import AtomicOrbital
from sisl.unit.siesta import unit_convert


__all__ = ['pdosSileSiesta']

Bohr2Ang = unit_convert('Bohr', 'Ang')


class pdosSileSiesta(SileSiesta):
    """ Projected DOS file with orbital information

    Data file containing the PDOS as calculated by Siesta.
    """

    def read_data(self, as_dataarray=False):
        r""" Returns data associated with the PDOS file

        For spin-polarized calculations the returned values are up/down, orbitals, energy.
        For non-collinear calculations the returned values are sum/x/y/z, orbitals, energy.

        Parameters
        ----------
        as_dataarray: bool, optional
           If True the returned PDOS is a `xarray.DataArray` with energy, spin
           and orbital information as coordinates in the data.
           The geometry, unit and Fermi level are stored as attributes in the
           DataArray.

        Returns
        -------
        geom : Geometry instance with positions, atoms and orbitals.
        E : the energies at which the PDOS has been evaluated at (if Fermi-level present in file energies are shifted to :math:`E - E_F = 0`).
        PDOS : an array of DOS with dimensions ``(nspin, geom.no, len(E))`` (with different spin-components) or ``(geom.no, len(E))`` (spin-symmetric).
        DataArray : if `as_dataarray` is True, only this data array is returned, in this case all data can be post-processed using the `xarray` selection routines.
        """
        # Get the element-tree
        root = xml_parse(self.file).getroot()

        # Get number of orbitals
        nspin = int(root.find('nspin').text)
        # Try and find the fermi-level
        Ef = root.find('fermi_energy')
        E = arrayd(list(map(float, root.find('energy_values').text.split())))
        if Ef is None:
            warn(str(self) + '.read_data could not locate the Fermi-level in the XML tree, using E_F = 0. eV')
        else:
            Ef = float(Ef.text)
            E -= Ef
        ne = len(E)

        # All coordinate, atoms and species data
        xyz = []
        atoms = []
        atom_species = []
        def ensure_size(ia):
            while len(atom_species) <= ia:
                atom_species.append(None)
                xyz.append(None)
        def ensure_size_orb(ia, i):
            while len(atoms) <= ia:
                atoms.append([])
            while len(atoms[ia]) <= i:
                atoms[ia].append(None)

        if nspin == 4:
            def process(D):
                tmp = np.empty(D.shape[0], D.dtype)
                tmp[:] = D[:, 3]
                D[:, 3] = D[:, 0] - D[:, 1]
                D[:, 0] = D[:, 0] + D[:, 1]
                D[:, 1] = D[:, 2]
                D[:, 2] = tmp[:]
                return D
        else:
            def process(D):
                return D

        if as_dataarray:
            import xarray as xr
            if nspin == 1:
                spin = ['sum']
            elif nspin == 2:
                spin = ['up', 'down']
            elif nspin == 4:
                spin = ['sum', 'x', 'y' 'z']

            # Dimensions of the PDOS data-array
            dims = ['E', 'spin', 'n', 'l', 'm', 'zeta', 'polarization']

            shape = (ne, nspin, 1, 1, 1, 1, 1)
            def to(o, DOS):
                # Coordinates for this dataarray
                coords = [E, spin,
                          [o.n], [o.l], [o.m], [o.Z], [o.P]]

                return xr.DataArray(data=process(DOS).reshape(shape),
                                    dims=dims, coords=coords, name='PDOS')

        else:
            def to(o, DOS):
                return process(DOS)

        D = []
        for orb in root.findall('orbital'):

            # Short-hand function to retrieve integers for the attributes
            def oi(name):
                return int(orb.get(name))

            # Get indices
            ia = oi('atom_index') - 1
            i = oi('index') - 1

            species = orb.get('species')

            # Create the atomic orbital
            try:
                Z = oi('Z')
            except:
                try:
                    Z = PeriodicTable().Z(species)
                except:
                    # Unknown
                    Z = -1

            try:
                P = orb.get('P') == 'true'
            except:
                P = False

            ensure_size(ia)
            xyz[ia] = list(map(float, orb.get('position').split()))
            atom_species[ia] = Z

            # Construct the atomic orbital
            O = AtomicOrbital(n=oi('n'), l=oi('l'), m=oi('m'), Z=oi('z'), P=P)

            # We know that the index is far too high. However,
            # this ensures a consecutive orbital
            ensure_size_orb(ia, i)
            atoms[ia][i] = O

            # it is formed like : spin-1, spin-2 (however already in eV)
            DOS = arrayd(list(map(float, orb.find('data').text.split()))).reshape(-1, nspin)

            if as_dataarray:
                if len(D) == 0:
                    D = to(O, DOS)
                else:
                    D = D.combine_first(to(O, DOS))
            else:
                D.append(process(DOS))

        # Now we need to parse the data
        # First reduce the atom
        atoms = [[o for o in a if o] for a in atoms]
        atoms = Atoms([Atom(Z, os) for Z, os in zip(atom_species, atoms)])
        geom = Geometry(arrayd(xyz) * Bohr2Ang, atoms)

        if as_dataarray:
            # Add attributes
            D.attrs['geometry'] = geom
            D.attrs['units'] = '1/eV'
            if Ef is None:
                D.attrs['Ef'] = 'Unknown'
            else:
                D.attrs['Ef'] = Ef

            return D

        D = np.moveaxis(np.stack(D, axis=0), 2, 0)
        if nspin == 1:
            return geom, E, D[0]
        return geom, E, D


# PDOS files are:
# They contain the same file (same xml-data)
# However, pdos.xml is preferred because it has higher precision.
#  siesta.PDOS
add_sile('PDOS', pdosSileSiesta, case=False, gzip=True)
#  pdos.xml/siesta.PDOS.xml
add_sile('PDOS.xml', pdosSileSiesta, case=False, gzip=True)
