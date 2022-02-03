#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Manipulating of DFT output, geometries and creating tight-binding parameter sets for NEGF transport
"""

# We have used a paradigm following pandas and Cython web-page documentation.

if __doc__ is None:
    __doc__ = """Manipulating of DFT output, geometries and creating tight-binding parameter sets for NEGF transport"""

DOCLINES = __doc__.split("\n")

import sys
import multiprocessing
import os

# pkg_resources is part of setuptools
import pkg_resources

# We should *always* import setuptools prior to Cython/distutils
import setuptools


def _ospath(path):
    """ Changes '/' separators to OS separators """
    return os.path.join(*path.split('/'))

# Macros for use when compiling stuff
macros = []

try:
    import Cython
    # if available we can cythonize stuff

    _CYTHON_VERSION = Cython.__version__
    from Cython.Build import cythonize

    # We currently do not have any restrictions on Cython (I think?)
    # If so, simply put it here, and we will not use it
    _CYTHON_INSTALLED = True
except ImportError:
    _CYTHON_VERSION = None
    _CYTHON_INSTALLED = False
    cythonize = lambda x, *args, **kwargs: x  # dummy func

if _CYTHON_INSTALLED:
    # The import of Extension must be after the import of Cython, otherwise
    # we do not get the appropriately patched class.
    # See https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
    # Now we can import cython distutils
    from Cython.Distutils.old_build_ext import old_build_ext as cython_build_ext

    cython = True
else:
    cython = False


# Allow users to remove cython step (forcefully)
# This may break compilation, but at least users should be aware
if "--no-cythonize" in sys.argv:
    sys.argv.remove("--no-cythonize")
    cython = False


# Check if users requests coverage of Cython sources
if "--with-cython-coverage" in sys.argv:
    linetrace = True
    sys.argv.remove("--with-cython-coverage")
else:
    linetrace = False

# Define Cython directives
# We shouldn't rely on sources having the headers filled
# with directives.
# Cleaner to have them here, and possibly on a per file
# basis (if needed).
# That could easily be added at ext_cython place
directives = {"linetrace": False, "language_level": 3}
if linetrace:
    # https://pypkg.com/pypi/pytest-cython/f/tests/example-project/setup.py
    directives["linetrace"] = True
    directives["emit_code_comments"] = True
    macros.extend([("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")])

# Check if users requests checking fortran passing copies
if "--f2py-report-copy" in sys.argv:
    macros.append(("F2PY_REPORT_ON_ARRAY_COPY", "1"))
    sys.argv.remove("--f2py-report-copy")

# We will *only* use setuptools
# Although setuptools is not shipped with the standard library, I think
# this is ok since it should get installed pretty easily.
from setuptools import Command, Extension
from setuptools import find_packages, find_namespace_packages

# Patch to allow fortran sources in setup
# build_ext requires numpy setup
# Also for extending build schemes we require build_ext from numpy.distutils
from distutils.command.sdist import sdist
from numpy.distutils.command.build_ext import build_ext as numpy_build_ext
from numpy.distutils.core import Extension as FortranExtension
from numpy.distutils.core import setup
from numpy import __version__ as np_version
print(f"sisl-build: numpy.__version__ = {np_version}")
if not cython:
    cython_build_ext = numpy_build_ext


# Custom command classes
cmdclass = {}


# Now create the build extensions
class CythonCommand(cython_build_ext):
    """
    Custom distutils command subclassed from Cython.Distutils.build_ext
    to compile pyx->c, and stop there. All this does is override the
    C-compile method build_extension() with a no-op.
    """

    def build_extension(self, ext):
        pass

if cython:
    # we have cython and generate c codes directly
    suffix = ".pyx"
    cmdclass["cython"] = CythonCommand
else:
    suffix = ".c"


# Retrieve the compiler information
from numpy.distutils.system_info import get_info
# use flags defined in numpy
all_info = get_info('ALL')

# Define compilation flags
extra_compile_args = ""
extra_link_args = extra_compile_args

# in numpy>=1.16.0, silence build warnings about deprecated API usage
macros.append(("NPY_NO_DEPRECATED_API", "0"))
# Do not expose multiple platform Cython code.
# We do not need it
#  https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#integrating-multiple-modules
macros.append(("CYTHON_NO_PYINIT_EXPORT", "1"))


class EnsureSource_sdist(sdist):
    """Ensure Cython has runned on all pyx files (i.e. we need c sources)."""

    def initialize_options(self):
        super().initialize_options()

    def run(self):
        if "cython" in cmdclass:
            self.run_command("cython")
        else:
            for ext, ext_d in ext_cython.items():
                pyx = ext_d.get("pyxfile", f"{ext}.pyx")
                source = f"{pyx[:-4]}.c"
                msg = (f".c-source file '{source}' not found.\n"
                       "Run 'setup.py cython' to convert {pyx} to {source} before sdist."
                )
                assert os.path.isfile(source), msg
        super().run()

cmdclass["sdist"] = EnsureSource_sdist


# Cython extensions can't be merged as a single module
# It requires a single source file.
# In this scheme all modules are named the same as their pyx files.
# If the module name should change, simply manually put the pyxfile.
ext_cython = {
    "sisl._indices": {},
    "sisl._math_small": {},
    "sisl._sparse": {
        "depends": [_ospath("sisl/_indices.pxd")]
    },
    "sisl._supercell": {},
    "sisl.physics._bloch": {},
    "sisl.physics._phase": {},
    "sisl.physics._matrix_utils": {},
    "sisl.physics._matrix_k": {},
    "sisl.physics._matrix_dk": {},
    "sisl.physics._matrix_ddk": {},
    "sisl.physics._matrix_phase3": {},
    "sisl.physics._matrix_phase3_nc": {},
    "sisl.physics._matrix_phase3_so": {},
    "sisl.physics._matrix_phase": {},
    "sisl.physics._matrix_phase_nc_diag": {},
    "sisl.physics._matrix_phase_nc": {},
    "sisl.physics._matrix_phase_so": {},
    "sisl.physics._matrix_sc_phase": {
        "depends": [_ospath("sisl/_sparse.pxd")]
    },
    "sisl.physics._matrix_sc_phase_nc_diag": {
        "depends": [_ospath("sisl/_sparse.pxd"),
                    _ospath("sisl/physics/_matrix_utils.pxd")
        ]
    },
    "sisl.physics._matrix_sc_phase_nc": {
        "depends": [_ospath("sisl/_sparse.pxd"),
                    _ospath("sisl/physics/_matrix_utils.pxd")
        ]
    },
    "sisl.physics._matrix_sc_phase_so": {
        "depends": [_ospath("sisl/_sparse.pxd"),
                    _ospath("sisl/physics/_matrix_utils.pxd")
        ]
    },
}

# List of extensions for setup(...)
extensions = []
for name, data in ext_cython.items():
    # Create pyx-file name
    # Default to module name + .pyx
    pyxfile = data.get("pyxfile", f"{name}.pyx").replace(".", os.path.sep)
    extensions.append(
        Extension(name,
                  sources=[f"{pyxfile[:-4]}{suffix}"] + data.get("sources", []),
                  depends=data.get("depends", []),
                  include_dirs=data.get("include", []),
                  language=data.get("language", "c"),
                  define_macros=macros + data.get("macros", []),
                  extra_compile_args=extra_compile_args,
                  extra_link_args=extra_link_args)
    )


# Specific Fortran extensions
ext_fortran = {
    "sisl.io.siesta._siesta": {
        "sources": [_ospath(f"sisl/io/siesta/_src/{f}") for f in
                    ["io_m.f90",
                     "siesta_sc_off.f90",
                     "hsx_read.f90", "hsx_write.f90",
                     "dm_read.f90", "dm_write.f90",
                     "tshs_read.f90", "tshs_write.f90",
                     "grid_read.f90", "grid_write.f90",
                     "gf_read.f90", "gf_write.f90",
                     "tsde_read.f90", "tsde_write.f90",
                     "hs_read.f90",
                     "wfsx_read.f90"]
        ],
    },
}

for name, data in ext_fortran.items():
    ext = FortranExtension(name,
        sources=data.get("sources"),
        depends=data.get("depends", []),
        include_dirs=data.get("include", None),
        define_macros=macros + data.get("macros", []),
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    extensions.append(ext)


class EnsureBuildExt(numpy_build_ext):
    """
    Override build-ext to check whether compilable sources are present
    This merely pretty-prints a better error message.

    Note we require build_ext to inherit from numpy.distutils since
    we need fortran sources.
    """

    def check_cython_extensions(self, extensions):
        for ext in extensions:
            for src in ext.sources:
                if not os.path.exists(src):
                    print(f"{ext.name}: -> {ext.sources}")
                    raise Exception(
                        f"""Cython-generated file '{src}' not found.
                        Cython is required to compile sisl from a development branch.
                        Please install Cython or download a release package of sisl.
                        """)

    def build_extensions(self):
        self.check_cython_extensions(self.extensions)
        numpy_build_ext.build_extensions(self)

# Override build_ext command (typically called by setuptools)
cmdclass["build_ext"] = EnsureBuildExt


# Run cythonizer
def cythonizer(extensions, *args, **kwargs):
    """
    Skip cythonizer (regardless) when running

    * clean
    * sdist

    Otherwise if `cython` is True, we will cythonize sources.
    """
    if "clean" in sys.argv or "sdist" in sys.argv:
        # https://github.com/cython/cython/issues/1495
        return extensions

    elif not cython:
        raise RuntimeError("Cannot cythonize without Cython installed.")

    import argparse

    # Allow parallel flags to be used while cythonizing
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", type=int, dest="parallel")
    parser.add_argument("--parallel", type=int, dest="parallel")
    parsed, _ = parser.parse_known_args()

    if parsed.parallel:
        kwargs["nthreads"] = max(0, parsed.parallel)

    # Extract Cython extensions
    # And also other extensions to store them
    other_extensions = []
    cython_extensions = []
    for ext in extensions:
        if ext.name in ext_cython:
            cython_extensions.append(ext)
        else:
            other_extensions.append(ext)

    return other_extensions + cythonize(cython_extensions, *args, quiet=False, **kwargs)


# This will locate all sisl* packages
packages = find_packages()

# This requires some name-mangling provided by 'package_dir' option
# Using namespace packages allows others to provide exactly the same package
# without causing namespace problems.
packages += map(lambda x: f"sisl_toolbox.{x}", find_namespace_packages(where="toolbox"))


# Please update MANIFEST.in file for stuff to be shipped in the distribution.
# Otherwise we should use package_data to ensure it gets installed.
package_data = {p: ["*.pxd"] for p in packages}
package_data["sisl_toolbox.siesta.minimizer"] = ["*.yaml"]


metadata = dict(
    # Correct the cmdclass
    cmdclass=cmdclass,

    # Ensure the packages are being found in the correct locations
    package_dir={"sisl_toolbox": "toolbox"},
    # This forces MANIFEST.in usage
    include_package_data=True,
    package_data=package_data,
    packages=packages,
    ext_modules=cythonizer(extensions, compiler_directives=directives),
)

cwd = os.path.abspath(os.path.dirname(__file__))
if not os.path.exists(_ospath(cwd + "/PKG-INFO")):
    # Generate Cython sources, unless building from source release
    # generate_cython()
    pass


if __name__ == "__main__":

    # Freeze to support parallel compilation when using spawn instead of fork
    multiprocessing.freeze_support()
    setup(**metadata)
