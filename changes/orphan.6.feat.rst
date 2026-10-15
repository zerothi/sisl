Adaptive Brillouin zone integration with `~sisl.physics.GaussKronrod` and `~sisl.physics.Farey`

`~sisl.physics.AdaptiveBrillouinZone` is the base class of Brillouin zones that
select their k-points from the integrand. Integrate with
``bz.integrate(func)``, where ``func(k)`` returns a scalar or an array.
`~sisl.physics.GaussKronrod` refines boxes of a 15-point Gauss-Kronrod rule
where the integrand changes fast, e.g. Green functions with a small broadening.
`~sisl.physics.Farey` uses the nested Farey grids of Chen *et al.*,
Phys. Rev. B **106**, 155102 (2022).
