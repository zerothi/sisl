Fixed the weights of a `BrillouinZone` with a single k-point as a flat list

``BrillouinZone(parent, [kx, ky, kz])`` now has one weight. Before, it had 3 weights
for the single k-point.
