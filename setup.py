#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Manipulating of DFT output, geometries and creating tight-binding parameter sets for NEGF transport"""
import setuptools
# we will use scikit-build's setup procedure
from skbuild import setup

# This requires some name-mangling provided by 'package_dir' option
# Using namespace packages allows others to provide exactly the same package
# without causing namespace problems.
packages = setuptools.find_namespace_packages(where="src")

metadata = dict(
    # Ensure the packages are being found in the correct locations
    package_dir={"": "src"},
    packages=packages,
)

if __name__ == "__main__":
    setup(**metadata)
