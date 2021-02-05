"""
Linear algebra
==============

Although `numpy` and `scipy` provides a large set of
linear algebra routines, sisl re-implements some of them with
a reduced memory and/or computational effort. This is because
`numpy.linalg` and `scipy.linalg` routines are defaulting
to a large variety of checks to assert the input matrices.

sisl implements its own variants which has interfaces much
like `numpy` and `scipy`.

   inv
   solve
   eig
   eigh
   svd
   eigs
   eigsh

"""
from .base import *

__all__ = [s for s in dir() if not s.startswith('_')]
