All `E` arguments for tbtSile cannot be indices

Prior versions of sisl allowed one to use indices
instead of `E` arguments. However, this led to
confusion when dealing with ``E=0.0``. E.g.
``E=0`` and ``E=0.0`` could behave differently.

Now, everything is handled via energies.
For looping those, its better to do:

.. code::

   tbt = tbt...Sile(...)
   iE = tbt.Eindex(0.84)
   E = tbt.E[iE] # will get you the closest energy point to 0.84

   # or for looping:
   for E in tbt.E:
       ... do something
