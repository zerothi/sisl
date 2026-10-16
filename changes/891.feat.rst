Adaptive Brillouin zone integration with `~sisl.physics.GaussKronrodGrid`

`~sisl.physics.AdaptiveBrillouinZone` is the base class of Brillouin zones that
select their k-points from the integrand. Integrate with
``bz.integrate(func)``, where ``func(k)`` returns a scalar or an array.
`~sisl.physics.GaussKronrodGrid` refines boxes of a 15-point Gauss-Kronrod rule
where the integrand changes fast, e.g. Green functions with a small broadening.
``max_k`` limits the total number of k-points, or (as a sequence) the k-point
density along each direction. Time-reversal symmetry (``trs``) halves the
integrated domain, as for `~sisl.physics.MonkhorstPack`.
