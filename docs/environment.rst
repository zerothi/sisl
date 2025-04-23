.. _environment:

Environment variables
=====================

sisl understands some environment variables that may be used to tweak, or change
the default behavior of sisl.

Here we list the different options:


``SISL_NUM_PROCS = 1``
   Default the number of processors used in parallel calculations.
   Currently only certain Brillouin zone operations has this enabled.

   Please test this for your machine before relying on it giving a lot
   of performance. Especially in conjunction with the ``OMP_NUM_THREADS``
   flag for OpenMP in linear algebra libraries.
   Benchmark and see if it actually improves (certain combinations will
   severely hurt performance, while others will greatly improve performance).
   This value together with ``OMP_NUM_THREADS`` and ``SISL_PAR_CHUNKSIZE``
   should be tuned for your case study.

   It is *very* easy to encounter deadlocks when using parallel processing in
   `sisl`. The main problem is the interplay between the BLAS library and the
   threading in the multiprocessing library. Always try first by maximizing
   ``SISL_NUM_PROCS`` and set ``OMP_NUM_THREADS=1`` to test whether it works
   on your machine, then further tuning can be done by ensuring that
   ``SISL_NUM_PROCS * OMP_NUM_THREADS <= CORES``.

   Generally one can get the maximum number of cores by:

   .. code-block:: python

      import os
      nprocs = len(os.sched_getaffinity(0))
      # If your CPU has hyper-threads, then you have to divide by 2:
      nprocs = nprocs // 2

``SISL_PAR_CHUNKSIZE = 0.2``
   Default size of chunk for each processor when running parallel things.

   The chunksize can be an `int`, in which case it is directly determining
   the chunksize.
   If it is a `float`, it is understood as the fraction of chunks each
   processor will get, so a chunksize of ``0.5`` means that each processor
   gets 2 chunks (the actual chunksize will be adjusted depending on the number
   of iterations). For ``0.1``, it results in 10 chunks per processor.

   When doing parallel jobs the chunksize can greatly improve efficiency.
   For example a chunksize of 1 and 2 processors for a brillouinzone object
   means an execution like this:

   .. code-block:: shell

      <initialization>
      proc-0: k[0]
      proc-1: k[1]
      <collection of results>
      proc-0: k[2]
      proc-1: k[3]
      <collection of results>
      ...

   With a chunksize of 2 it will look like:

   .. code-block:: shell

      <initialization>
      proc-0: k[0]
      proc-0: k[1]
      proc-1: k[2]
      proc-1: k[3]
      <collection of results>
      ...

   this can greatly improve performance, however, a too high chunksize can
   hurt performance, as well as a too low.
   A rule-of-thumb would be to select a chunksize such that
   the number of iterations is divisible by the chunksize.
   For `BrillouinZone` objects, the number of iterations is equivalent to the number
   of k-points.

``SISL_SHOW_PROGRESS = false``
   Certain sisl routines has a builtin progress bar. This variable can default
   whether or not those will be shown. It can be nice for *slow* brillouinzone calculations
   to see if progress is actually being made.

``SISL_IO_DEFAULT = ''``
   The default IO methods `sisl.get_sile` will select files with this file-endings.
   For instance there are many ``stdout`` file types (for each DFT code).
   Setting this to ``Siesta`` would force all files to first search for classes ending
   in ``Siesta`` (see `sisl.io` for class names).

``SISL_TMP = '.sisl_tmp'``
   certain internal methods of sisl will use a temporary folder for storing data.
   The default is a new folder in the currently executed directory.

``SISL_FILES_TESTS``
   Full path to a folder containing tests files. Primarily used for developers.

   This can be used like this:

   .. code-block:: shell

      git clone -b stripped --single-branch https://github.com/zerothi/sisl-files.git
      SISL_FILES_TESTS=$(pwd)/sisl-files pytest --pyargs sisl

``SISL_CONFIGDIR = ~/.config/sisl``
   where certain configuration files should be stored.

   Currently not in use.

``SISL_LOG_FILE``
   if provided `sisl` will log to the provided file.

``SISL_LOG_LEVEL = "info"``
   the log-level used if writing to a log-file.
   The value will be taken from the `logging` module,
   so it should be a variable in that module.

``SISL_CODATA = 2010 | 2014 | 2018 | 2022``
   default internal values used for units and constants.


Code specific environment variables
-----------------------------------

Siesta
^^^^^^

``SISL_UNIT_SIESTA = codata2018 | legacy``
   determine the default units for Siesta files.

   Since Siesta 5.0, the default units are updated to follow
   the CODATA 2018 values. Since Siesta 6.0 it will be 2022.
   The change from 4 to 5 meant that quite a bit of
   results changed. This will force the internal variables
   to be consistent with these changes. Adapt to the same version used
   by your Siesta version.
