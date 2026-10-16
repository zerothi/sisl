Fixed the circle plane of `BrillouinZone.param_circle`

`~sisl.physics.BrillouinZone.param_circle` now creates the circle perpendicular to
``normal``. Before, the circle was in a wrong plane for normals not along the
third Cartesian direction.
