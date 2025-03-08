Now ``geometry.to(ase.Atoms)`` will work

The dispatch method expected it to be ``geometry.to[ase.Atoms]()``
which is counter-intuitive as all the other dispatchers does
not require this. The `__getitem__` method will still work
as that will get you the method by witch the dispatch
will happen.
