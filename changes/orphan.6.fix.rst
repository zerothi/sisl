Fixed time-reversal symmetry of a displaced `MonkhorstPack`

A `~sisl.physics.MonkhorstPack` with ``trs=True`` and a displacement different from
0 and 1/2 gave wrong Brillouin zone averages. It now uses time-reversal symmetry only
when the k-points are symmetric for :math:`\mathbf k\to-\mathbf k`.
