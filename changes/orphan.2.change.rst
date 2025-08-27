added `oxyz` to extract coordinates of orbitals

This allows easier access to orbital coordinates
rather than having to convert indices to atoms
and then to supercell coordinates.

Basically equivalent to:
```python
atoms = geom.o2a(orbitals)
geom.axyz(atoms) == geom.oxyz(orbitals)
```
