import numpy as np

from sisl.atom import AtomGhost, PeriodicTable

from .data_source import DataSource


class AtomData(DataSource):
    def function(self, geometry, atoms=None):
        raise NotImplementedError("")

@AtomData.from_func
def AtomCoords(geometry, atoms=None):
    return geometry.xyz[atoms]

@AtomData.from_func
def AtomX(geometry, atoms=None):
    return geometry.xyz[atoms, 0]

@AtomData.from_func
def AtomY(geometry, atoms=None):
    return geometry.xyz[atoms, 1]

@AtomData.from_func
def AtomZ(geometry, atoms=None):
    return geometry.xyz[atoms, 2]

@AtomData.from_func
def AtomFCoords(geometry, atoms=None):
    return geometry.sub(atoms).fxyz

@AtomData.from_func
def AtomFx(geometry, atoms=None):
    return geometry.sub(atoms).fxyz[:, 0]

@AtomData.from_func
def AtomFy(geometry, atoms=None):
    return geometry.sub(atoms).fxyz[:, 1]

@AtomData.from_func
def AtomFz(geometry, atoms=None):
    return geometry.sub(atoms).fxyz[:, 2]

@AtomData.from_func
def AtomR(geometry, atoms=None):
    return geometry.sub(atoms).maxR(all=True)

@AtomData.from_func
def AtomZ(geometry, atoms=None):
    return geometry.sub(atoms).atoms.Z

@AtomData.from_func
def AtomNOrbitals(geometry, atoms=None):
    return geometry.sub(atoms).orbitals

class AtomDefaultColors(AtomData):

    _atoms_colors = {
        "H": "#cccccc",
        "O": "red",
        "Cl": "green",
        "N": "blue",
        "C": "grey",
        "S": "yellow",
        "P": "orange",
        "Au": "gold",
        "else": "pink"
    }

    def function(self, geometry, atoms=None):
        return np.array([
            self._atoms_colors.get(atom.symbol, self._atoms_colors["else"])
            for atom in geometry.sub(atoms).atoms
        ])

@AtomData.from_func
def AtomIsGhost(geometry, atoms=None, fill_true=True, fill_false=False):
    return np.array([
        fill_true if isinstance(atom, AtomGhost) else fill_false
        for atom in geometry.sub(atoms).atoms
    ])

@AtomData.from_func
def AtomPeriodicTable(geometry, atoms=None, what=None, pt=PeriodicTable):
    if not isinstance(pt, PeriodicTable):
        pt = pt()
    function = getattr(pt, what)
    return function(geometry.sub(atoms).atoms.Z)