"""
Mixing objects
==============

Implementation of various mixing methods.

Mixing methods are typical objects used for speeding up convergence
for self-consistent methods.

sisl implements a variety of the standard methods used in the electronic
structure community and its easy implementations allows easy testing for
new methods.
"""

from .base import *
from .linear import *
from .diis import *
