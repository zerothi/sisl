from .data_source import DataSource
from sisl.utils.mathematics import fnorm

import numpy as np

class BondData(DataSource):
    
    def get(self, geometry, bonds):
        raise NotImplementedError("")

    pass

def _get_bond_lengths(geometry, bonds):
    # Get an array with the coordinates defining the start and end of each bond.
    # The array will be of shape (nbonds, 2, 3)
    coords = geometry[bonds]
    # Take the diff between the end and start -> shape (nbonds, 1 , 3)
    # And then the norm of each vector -> shape (nbonds, 1, 1)
    # Finally, we just ravel it to an array of shape (nbonds, )
    return fnorm(np.diff(coords, axis=1), axis=-1).ravel()

class BondLength(BondData):
    
    ndim = 1
    
    def get(self, geometry, bonds):
        return _get_bond_lengths(geometry, bonds)

class BondStrain(BondData):
    
    ndim = 1

    def __init__(self, ref_geometry):
        self._ref_geometry = ref_geometry
        super().__init__()
    
    def get(self, geometry, bonds):
        assert self._ref_geometry.na == geometry.na, (f"Geometry provided (na={geometry.na}) does not have the"
            f" same number of atoms as the reference geometry (na={self._ref_geometry.na})")

        ref_bond_lengths = _get_bond_lengths(self._ref_geometry, bonds)
        bond_lengths = _get_bond_lengths(geometry, bonds)

        return (bond_lengths - ref_bond_lengths) / ref_bond_lengths

class BondFromAtoms(BondData):
    def __init__(self, atom_data):
        self.atom_data = atom_data

    def get(self, geometry, bonds):

        atom_data = self.atom_data
        if isinstance(atom_data, DataSource):
            atom_data = atom_data.get(geometry)

        return atom_data[bonds[:, 0]]
