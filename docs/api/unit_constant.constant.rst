
Constants
=========

.. currentmodule:: sisl.constant


A pre-set set of physical constants. The SI units are following the *new* convention
that takes effect on 20 May 2019.

The currently stored constants are (all are given in SI units):

.. autosummary::
   :toctree: generated/

   PhysicalConstant
   q
   c
   h
   hbar
   m_e
   m_p
   G0
   G

All constants may be used like an ordinary float (which converts it to a float):

>>> c
299792458.0 m/s
>>> c * 2
599584916

while one can just as easily convert the units (which ensures thay stay like another `PhysicalConstant`):

>>> c('Ang/ps')
2997924.58 Ang/ps
