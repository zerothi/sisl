.. _physics.brillouinzone:

Brillouin zone
==============

.. currentmodule:: sisl.physics

The Brillouin zone objects are all special classes enabling easy manipulation
of an underlying physical quantity.

Quite often a physical quantity will be required to be averaged, or calculated individually
over a number of k-points. In this regard the Brillouin zone objects can help.

The BrillouinZone object allows direct looping of contained k-points while invoking
particular methods from the contained object.
This is best shown with an example:

>>> import sisl as si
>>> H = si.Hamiltonian(...)
>>> bz = si.BrillouinZone(H)
>>> bz.apply.array.eigh()

This will calculate eigenvalues for all k-points associated with the `BrillouinZone` and
return everything as an array. The `~sisl.physics.BrillouinZone.dispatch` property of
the `BrillouinZone` object has several use cases (here ``array`` is shown).

This may be extremely convenient when calculating band-structures:

>>> H = si.Hamiltonian(...)
>>> bs = si.BandStructure(H, [[0, 0, 0], [0.5, 0, 0]], 100)
>>> bs_eig = bs.apply.array.eigh()
>>> plt.plot(bs.lineark(), bs_eig)

and then you have all eigenvalues for all the k-points along the path.


Multiple quantities
-------------------

Sometimes one may want to post-process the data for each k-point.
As an example lets post-process the DOS on a per k-point basis while
calculating the average:

>>> H = si.Hamiltonian(...)
>>> mp = si.MonkhorstPack(H, [10, 10, 10])
>>> E = np.linspace(-2, 2, 100)
>>> def wrap_DOS(eigenstate):
...    # Calculate the DOS for the eigenstates
...    DOS = eigenstate.DOS(E)
...    # Calculate the velocity for the eigenstates
...    v = eigenstate.velocity()
...    V = (v ** 2).sum(1)
...    return DOS.reshape(-1, 1) * v ** 2 / V.reshape(-1, 1)
>>> DOS = mp.apply.average.eigenstate(wrap=wrap_DOS, eta=True)

This will, calculate the Monkhorst pack k-averaged DOS split into 3 Cartesian
directions based on the eigenstates velocity direction. This method of manipulating
the result can be extremely powerful to calculate many quantities while running an
efficient `BrillouinZone` average. The `eta` flag will print, to stdout, a progress-bar.
The usage of the ``wrap`` method are also passed optional arguments, ``parent`` which is
``H`` in the above example. ``k`` and ``weight`` are the current k-point and weight of the
corresponding k-point. An example could be to manipulate the DOS depending on the k-point and
weight:

>>> H = si.Hamiltonian(...)
>>> mp = si.MonkhorstPack(H, [10, 10, 10])
>>> E = np.linspace(-2, 2, 100)
>>> def wrap_DOS(eigenstate, k, weight):
...    # Calculate the DOS for the eigenstates and weight by k_x and weight
...    return eigenstate.DOS(E) * k[0] * weight
>>> DOS = mp.apply.sum.eigenstate(wrap=wrap_DOS, eta=True)

When using wrap to calculate more than one quantity per eigenstate it may be advantageous
to use `~sisl.oplist` to handle cases of `BrillouinZone.apply.average` and `BrillouinZone.apply.sum`.

>>> H = si.Hamiltonian(...)
>>> mp = si.MonkhorstPack(H, [10, 10, 10])
>>> E = np.linspace(-2, 2, 100)
>>> def wrap_multiple(eigenstate):
...    # Calculate DOS/PDOS for eigenstates
...    DOS = eigenstate.DOS(E)
...    PDOS = eigenstate.PDOS(E)
...    # Calculate velocity for the eigenstates
...    v = eigenstate.velocity()
...    return si.oplist([DOS, PDOS, v])
>>> DOS, PDOS, v = mp.apply.average.eigenstate(wrap=wrap_multiple, eta=True)

Which does mathematical operations (averaging/summing) using `~sisl.oplist`.


In some cases quantities are needed for all :math:`k` points and in such cases
it may not always be that the returned quantities are commensurate.
Lets re-use the previous ``wrap_multiple`` function and try and return the
full quantity:

>>> DOS_PDOS_v = mp.apply.eigenstate(wrap=wrap_multiple, eta=True)

This will raise an error since ``wrap_multiple`` returns an `oplist` (same as a `list`)
and thus is unable to convert this into an equivalent `numpy.ndarray`. Additionally
this can not be merged together in a single `numpy.ndarray` since the shapes of the returned
quantities are not commensurate. One cannot concatenate the 3 different quantities.

To accomblish this one may use an ``zip`` flag where the two lines are equivalent:

>>> DOS, PDOS, v = mp.apply.array.renew(zip=True).eigenstate(wrap=wrap_multiple, eta=True)
>>> DOS, PDOS, v = mp.apply(zip=True).array.eigenstate(wrap=wrap_multiple, eta=True)

and the data is unpacked as wanted.


Parallel calculations
---------------------

The ``apply`` method looping k-points may be explicitly parallelized.
To run parallel do:

>>> H = si.Hamiltonian(...)
>>> mp = si.MonkhorstPack(H, [10, 10, 10])
>>> with mp.apply.renew(pool=True) as par:
...     par.array.eigh()

This requires you also have the package ``pathos`` available.
The above will run in parallel using a default number of processors
in priority:

1. Environment variable ``SISL_NUM_PROCS``
2. Return value of ``os.cpu_count()``.

Note that this may interfere with BLAS implementation which defaults
to use all CPU's for threading. The total processors/threads that will
be created is ``SISL_NUM_PROCS * OMP_NUM_THREADS``. Try and ensure this is below
the actual core-count of your machine (or the number of requested cores in a
HPC environment).


Alternatively one can control the number of processors locally by doing:

>>> H = si.Hamiltonian(...)
>>> mp = si.MonkhorstPack(H, [10, 10, 10])
>>> with mp.apply.renew(pool=2) as par:
...     par.eigh()

which will request 2 processors (regardless of core-count).
As a last resort you can pass your own ``Pool`` of workers that
will be used for the parallel processing.

>>> from multiprocessing import Pool
>>> pool = Pool(4)
>>> H = si.Hamiltonian(...)
>>> mp = si.MonkhorstPack(H, [10, 10, 10])
>>> with mp.apply.renew(pool=pool) as par:
...     par.array.eigh()

The ``Pool`` should implement some standard methods that are
existing in the ``pathos`` enviroment such as ``Pool.restart`` and ``Pool.terminate``
and ``imap`` and ``uimap`` methods. See the ``pathos`` documentation for detalis.


.. autosummary::
   :toctree: generated/

   BrillouinZone
   MonkhorstPack
   BandStructure
