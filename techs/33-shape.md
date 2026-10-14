# 33 · Shapes

Owns: `src/sisl/shape/*`

Paths are relative to `src/sisl/shape/`.

## Contract

`Shape` (`base.py`) has a `center` and a `to` dispatcher.

- `within_index(xyz, rtol=1e-8)` returns unique `int` indices of the points in the shape. Each shape implements it.
- `within(xyz)` is a boolean mask from `within_index`. One point gives one `bool`.
- `xyz in shape` is `True` when all points are in the shape.
- `PureShape` adds `volume` and `expand`. Use it for shapes with an exact volume.

### `to` conversions

- `to.Ellipsoid` falls back to `to.Sphere().to.Ellipsoid()`.
- `to.Cuboid` falls back to `to.Ellipsoid().to.Cuboid()`.
- Thus a new shape must implement at least `to.Sphere`. The sphere must contain all of the shape.
- `Grid.index(shape)` uses the smaller of `to.Cuboid` and `to.Ellipsoid`. A tight bound makes it faster.
- `toSphere`, `toEllipsoid` and `toCuboid` are deprecated.

## Shapes

| Shape | File | Notes |
|---|---|---|
| `Cuboid(v, center or origin)` | `prism4.py` | `v` is a scalar, 3 values or 3×3. `origin = center - sum(v) / 2`. |
| `Cube` | `prism4.py` | Subclass of `Cuboid`. |
| `Ellipsoid(v, center)` | `ellipsoid.py` | Makes non-orthogonal vectors orthogonal, with a warning. |
| `Sphere` | `ellipsoid.py` | Subclass of `Ellipsoid`. |
| `EllipticalCylinder` | `_cylinder.py` | |
| `NullShape` | `base.py` | Contains nothing. `volume = 0`. |

`within_index` maps the points to the frame of the shape. Then it tests the fractional coordinates.
The tests use Cython helpers from `sisl/_indices.pyx`.

## Tolerance

`rtol` is relative to the fractional frame of the shape.
Thus the absolute tolerance increases with the size of the shape.

## Composite shapes

| Operator | Class | Index operation |
|---|---|---|
| `&` | `AndShape` | `intersect1d` |
| `\|`, `+` | `OrShape` | `union1d` |
| `^` | `XOrShape` | `setxor1d` |
| `-` | `SubShape` | `setdiff1d` |

- The composite copies both shapes.
- The index operations use `assume_unique=True`. Thus `within_index` must return unique indices.
- `center` is the mean of the two centers. Setting it moves both shapes.
- `volume` is `-1.0` (not defined).
- `to.Sphere` gives a sphere that contains the composite.
  `AndShape` uses the intersection of two spheres. It gives `NullShape().to.Sphere()` when they do not overlap.

## Use in Geometry

- `geometry[shape]` selects atoms in the unit cell only.
- `within_sc(shapes, isc)` treats the last shape as the largest. It uses it to screen the atoms first.
  Each shape removes the atoms that earlier shapes found. The result is shells.
- `within` loops over all `sc_off` with `within_sc`.
- `close` and `close_sc` use spheres directly. They do not use `Shape` objects.

## Add a shape

1. Subclass `PureShape` (or `Shape` when the volume is not known). Call `super().__init__(center)`.
2. Implement `within_index`, `copy`, `scale`, `expand` and `volume`.
3. Register `to.Sphere`. Register `to.Cuboid` and `to.Ellipsoid` for tighter bounds (see `prism4.py`).
4. Export the class in `shape/__init__.py`.

## Known defects

- `base.py` defines `ToEllipsoidDispatcher` for composites but registers the base `ToEllipsoidDispatch`.
- The doc-string of `Ellipsoid.within_index` calls `rtol` absolute. It is relative.
