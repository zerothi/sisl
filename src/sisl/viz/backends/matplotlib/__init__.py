# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
r"""Matplotlib
==========

Implementations of the sisl-provided matplotlib backends.

"""
import matplotlib

from .backend import MatplotlibBackend, MatplotlibMultiplePlotBackend, MatplotlibSubPlotsBackend
from ._plots import *
