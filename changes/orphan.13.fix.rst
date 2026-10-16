Fixed explicit `BandStructure` divisions over a discontinuity

A `~sisl.physics.BandStructure` with a list of ``divisions`` and a discontinuity
(``None`` in ``points``) created k-points between the two points of the
discontinuity. The division of that segment is now set to 1, as for an integer
``divisions``.
