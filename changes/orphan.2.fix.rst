`BrillouinZone.copy` keeps the full state of all Brillouin zone classes

A copy of a `BandStructure` now keeps its discontinuities (``None`` in the path),
so `BandStructure.lineartick` and `BandStructure.lineark` are the same as for the original.
A copy of a `GaussKronrodGrid` no longer raises an error.
