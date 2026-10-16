`MonkhorstPack` raises a `ValueError` for directions without k-points

``MonkhorstPack(parent, [10, 0, 1])`` now raises a `ValueError`. Before, it returned a
grid without k-points (and a division by zero warning), unless all directions were 0.
