r"""
================================
TBtrans (:mod:`sisl.io.tbtrans`)
================================

.. module:: sisl.io.tbtrans
   :noindex:

File objects for interaction with the `TBtrans`_ code.

The TBtrans code is a tight-binding transport code implementing
the widely used non-equilibrium Green function method.

It is primarily implemented for the support of TranSiesta (DFT+NEGF)
as a backend for calculating transport for self-consistent DFT software.

Here we show a variety of supplement files that allows the extracting, manipulation
and creation of files supported in TBtrans.

The basic file is the `tbtncSileTBtrans` which is a file to extract information
from a TBtrans output file (typically named: ``siesta.TBT.nc``).
The following will interact with the TBtrans file:

>>> tbt = sisl.get_sile('siesta.TBT.nc') # doctest: +SKIP
>>> tbt.E # doctest: +SKIP
>>> tbt.a_d # doctest: +SKIP

Importantly one may retrieve quantities such as DOS, transmissions,
transmission eigenvalues etc.

>>> tbt.transmission() # from electrode 0 -> 1 (default) # doctest: +SKIP
>>> tbt.transmission(0, 1) # from electrode 0 -> 1 # doctest: +SKIP
>>> tbt.transmission(0, 2) # from electrode 0 -> 2 # doctest: +SKIP
>>> tbt.ADOS(0, E=1.) # k-average, total spectral DOS from 0th electrode # doctest: +SKIP
>>> tbt.density_matrix(E=1.) # k-average, density matrix from Green function at 1. eV # doctest: +SKIP


The above extraction of data is the most frequent use of this module.

Data extraction files
^^^^^^^^^^^^^^^^^^^^^

- `tbtncSileTBtrans` (electron TBtrans output)
- `tbtavncSileTBtrans` (electron k-averaged TBtrans output)
- `tbtsencSileTBtrans` (electron TBtrans self-energy output)
- `tbtprojncSileTBtrans` (projected TBtrans output)


Support files to complement TBtrans
-----------------------------------
- `deltancSileTBtrans` adding :math:`\delta H` or :math:`\delta\Sigma` elements
  to a TBtrans calculation
- `tbtgfSileTBtrans` manual creation of the self-energies for TBtrans.


PHtrans
-------

PHtrans is the same program as TBtrans, it however uses the dynamical matrix to calculate
phonon transport.

- `phtncSileTBtrans` (phonon PHtrans output)
- `phtsencSileTBtrans` (phonon PHtrans self-energy output)
- `phtavncSileTBtrans` (phonon k-averaged PHtrans output)
- `phtprojncSileTBtrans` (projected PHtrans output)

"""
from .sile import *

from .binaries import *
from .delta import *
from .se import *
from .tbt import *
from .tbtproj import *

__all__ = [s for s in dir() if not s.startswith('_')]
