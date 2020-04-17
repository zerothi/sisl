#!/usr/bin/env python3
"""
A setup.py script to use setuptools, which gives egg goodness.
"""

from setuptools import setup
exec(compile(open('setup.py').read(), 'setup.py', 'exec'))
