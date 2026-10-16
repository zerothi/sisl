Fixed the number of points of `BrillouinZone.param_circle` for a discretization length

`~sisl.physics.BrillouinZone.param_circle` with a float ``N_or_dk`` now uses the
circumference to get the number of points. Before, it used the area of the circle,
which gave too few points for radii below :math:`2` 1/Ang.
