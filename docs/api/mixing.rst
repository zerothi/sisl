.. _mixing:

Mixing self-consistent quantities
=================================

.. currentmodule:: sisl.mixing

Mixing various quantities in self-consistent manners are quite frequent.
This module enables a variety of methods based on the Pulay (DIIS) mixing
methods and may be used for externally driven SC cycles.

Container classes
-----------------

Mixing makes use of so called *metrics* and several steps of quantities
stored in *history*.

The basic classes that are used internally are

.. autosummary::
   :toctree: generated/

   Metric
   History


Mixing algorithms
-----------------

.. autosummary::
   :toctree: generated/

   LinearMixer
   DIISMixer
   PulayMixer
   AdaptiveDIISMixer
   AdaptivePulayMixer
   