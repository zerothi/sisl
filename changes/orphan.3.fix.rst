`projection` arguments for all functions has been streamlined

The `projection` argument has gotten a major overhaul.
Now the projections are generalized and streamlined across
the code base using a common `comply_projection` method
that decides on what it should convert to.

All old values are still allowed, but newer ones will be preferred:

Here are all the allowed (new) projection options:

- `matrix` matrix product, `ij` components
- `trace` return sum of the `ii` components
- `diagonal` return the `ii` components
- `hadamard` elementwise `ij` components (not equivalent to `matrix`!)
- `hadamard:atoms` elementwise `ij` components, but summed for each atom
