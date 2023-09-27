.. _environment:

Environment variables
=====================

sisl understands some environment variables that may be used to tweak, or change
the default behaviour of sisl.

Here we list the different options:


``SISL_NUM_PROCS = 1``
   Default the number of processors used in parallel calculations.
   Currently only certain Brillouin zone operations has this enabled.

   Please test this for your machine before relying on it giving a lot
   of performance. Especially in conjunction with the ``OMP_NUM_THREADS``
   flag for OpenMP in linear algebra libraries.
   Benchmark and see if it actually improves (certain combinations will
   severly hurt performance).

``SISL_VIZ_AUTOLOAD == false``
   whether or not to autoload the visualization module.
   The visualization module imports many dependent modules.
   If you run small scripts that does not use the `sisl.viz` module, then
   it is recommended to keep this to be false.

``SISL_SHOW_PROGRESS = false``
   Certain sisl routines has a builtin progress bar. This variable can default
   whether or not those will be shown. It can be nice for *slow* brillouinzone calculations
   to see if progress is actually being made.

``SISL_IO_DEFAULT = ''``
   The default IO methods `sisl.get_sile` will select files with this file-endings.
   For instance there are many ``stdout`` file types (for each DFT code).
   Setting this to ``Siesta`` would force all files to first search for Siesta file
   endings (see `sisl.io` for class names).

``SISL_TMP = '.sisl_tmp'``
   certain internal methods of sisl will use a temporary folder for storing data.
   The default is a new folder in the currently executed directory.

``SISL_FILES_TESTS``
   Full path to a folder containing tests files. Primarily used for developers.

``SISL_CONFIGDIR = $HOME/.config/sisl``
   where certain configuration files should be stored.

   Currently not in use.


Code specific environment variables
-----------------------------------

Siesta
^^^^^^

``SISL_UNIT_SIESTA = codata2018 | legacy``
   determine the default units for Siesta files.

   Since Siesta 5.0, the default units are updated to follow
   the CODATA 2018 values. This means that quite a bit of
   results changed. This will force the internal variables
   to be consistent with this.

