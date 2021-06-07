#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
A setup.py script to use setuptools, which gives egg goodness.
"""

from setuptools import setup
exec(compile(open('setup.py').read(), 'setup.py', 'exec'))
