Fixed `Grid.append`, `AndersonMixer`, `iter_shape`, `size_to_num` and `Orbital.equal`

`~sisl.Grid.append` now copies the data of both grids; before it returned zeros.
`~sisl.mixing.AndersonMixer` now keeps one history step and does Anderson mixing;
before it silently did linear mixing. Its results therefore change.
`~sisl.utils.misc.iter_shape` now yields all indices as tuples, also for 1D shapes.
`~sisl.utils.misc.size_to_num` now converts numbers (in MB) to the requested unit.
`Orbital.equal` and `AtomicOrbital.equal` now compare a string against the orbital name.
`tbtavncSileTBtrans.write_tbtav` now k-averages variables where ``nkpt`` is not the
first dimension; before it raised a `NameError`.
