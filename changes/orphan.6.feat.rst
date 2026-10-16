Adaptive Brillouin zone integration with `~sisl.physics.GaussKronrodGrid`

`~sisl.physics.AdaptiveBrillouinZone` is the base class of Brillouin zones that
select their k-points from the integrand. Integrate with
``bz.integrate(func)``, where ``func(k)`` returns a scalar or an array.
`~sisl.physics.GaussKronrodGrid` refines boxes of a 15-point Gauss-Kronrod rule
where the integrand changes fast, e.g. Green functions with a small broadening.
