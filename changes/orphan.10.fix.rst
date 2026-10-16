`GaussKronrodGrid.integrate` warns for an integrand that is not finite

`~sisl.physics.GaussKronrodGrid.integrate` now warns and stops when the integrand
returns NaN or inf. Before, it returned the NaN as a converged value.
