#!/usr/bin/env python
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import pstats

# Script for analysing profile scripts created by the
# cProfile module.
import sys

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    raise ValueError("Must supply a profile file-name")

stat = pstats.Stats(fname)

# We sort against total-time
stat.sort_stats("tottime")
# Only print the first 20% of the routines.
stat.print_stats("sisl", 0.2)
