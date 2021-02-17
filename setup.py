#!/usr/bin/env python3

"""
sisl: Generic library for manipulating DFT output, geometries and tight-binding parameter sets
"""

# We have used a paradigm following pandas and Cython web-page documentation.

if __doc__ is None:
    __doc__ = """sisl: Generic library for manipulating DFT output, geometries and tight-binding parameter sets"""

DOCLINES = __doc__.split("\n")

import sys
import subprocess
import multiprocessing
import os
import os.path as osp
import argparse
from functools import reduce

# pkg_resources are part of setuptools
import pkg_resources

# We should *always* import setuptools prior to Cython/distutils
import setuptools


def _ospath(path):
    """ Changes '/' separators to OS separators """
    return osp.join(*path.split('/'))

# Define a list of minimum versions
min_version ={
    "python": "3.6",
    "numpy": "1.13",
    "pyparsing": "1.5.7",
    "xarray": "0.10.0",
}

viz = {
    "plotly": [
        'dill >= 0.3.2', # see https://github.com/pfebrer/sisl/issues/11
        'pathos',
        'plotly',
        'pandas',
        "xarray >= " + min_version["xarray"],
        'scikit-image'
    ],
    "blender": [
    ], # for when blender enters
    "ase": [
        "ase" # for Jonas's implementation
    ],
}

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
from setuptools import find_packages

# Patch to allow fortran sources in setup
# build_ext requires numpy setup
# Also for extending build schemes we require build_ext from numpy.distutils
from distutils.command.sdist import sdist
from numpy.distutils.command.build_ext import build_ext as numpy_build_ext
from numpy.distutils.core import Extension as FortranExtension
from numpy.distutils.core import setup
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

# All our extensions depend on numpy/core/include
numpy_incl = pkg_resources.resource_filename("numpy", _ospath("core/include"))

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
                  include_dirs=[numpy_incl] + data.get("include", []),
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


class InfoCommand(Command):
    """
    Custom distutils command to create the sisl/info.py file.
    It will additionally print out standard information abou
    the version.
    """
    description = "create info.py file"
    user_options = []
    boolean_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print('info RUNNING')

cmdclass["info.py"] = InfoCommand


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

    # Retrieve numpy include directories for headesr
    numpy_incl = pkg_resources.resource_filename("numpy", _ospath("core/include"))

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


MAJOR = 0
MINOR = 11
MICRO = 0
ISRELEASED = True
VERSION = f"{MAJOR}.{MINOR}.{MICRO}"
GIT_REVISION = "ad878e687045793c53d0c628c5345832beb60695"
REVISION_YEAR = 2021


DISTNAME = "sisl"
LICENSE = "LGPLv3"
AUTHOR = "sisl developers"
URL = "https://github.com/zerothi/sisl"
DOWNLOAD_URL = "https://github.com/zerothi/sisl/releases"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/zerothi/sisl/issues",
    "Documentation": "https://zerothi.github.io/sisl",
    "Source Code": "https://github.com/zerothi/sisl",
}
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Cython",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Utilities",
]


# The install_requires should also be the
# requirements for the actual running of sisl
setuptools_kwargs = {
    "python_requires": ">= " + min_version["python"],
    "install_requires": [
        "setuptools",
        "numpy >= " + min_version["numpy"],
        "scipy",
        "netCDF4",
        "pyparsing >= " + min_version["pyparsing"],
    ],
    "setup_requires": [
        "numpy >= " + min_version["numpy"],
    ],
    "extras_require": {
        # We currently use xarray for additional data-analysis
        # And tqdm for progressbars
        "analysis": [
            "xarray >= " + min_version["xarray"],
            "tqdm",
        ],
        "viz": reduce(lambda a, b: a + b, viz.values()),
        "visualization": reduce(lambda a, b: a + b, viz.values()),
        "viz-plotly": viz["plotly"],
        "viz-blender": viz["blender"],
        "viz-ase": viz["ase"],
    },
    "zip_safe": False,
}


def readme():
    if not osp.exists("README.md"):
        return ""
    return open("README.md", "r").read()

metadata = dict(
    name=DISTNAME,
    author=AUTHOR,
    maintainer=AUTHOR,
    description="Python interface for tight-binding model creation and analysis of DFT output. Input mechanism for large scale transport calculations using NEGF TBtrans (TranSiesta)",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/zerothi/sisl",
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    # Ensure the packages are being found in the correct locations
    package_dir={"sisl_toolbox": "toolbox"},
    package_data={},
    packages=
    # We need to add sisl.* since that recursively adds modules
    find_packages(include=["sisl", "sisl.*"])
    +
    # Add toolboxes
    # This requires some name-mangling since we can't place them
    # in the correct place unless we use 'package_dir' and this trick.
    # 1. Here we list files as they should appear in packages for end-users
    # 2. In 'package_dir' we defer the package name to the local file path
    list(map(lambda x: f"sisl_toolbox.{x}", find_packages("toolbox"))),
    ext_modules=cythonizer(extensions, compiler_directives=directives),
    entry_points={
        "console_scripts":
        ["sgeom = sisl.geometry:sgeom",
         "sgrid = sisl.grid:sgrid",
         "sdata = sisl.utils._sisl_cmd:sisl_cmd",
         "sisl = sisl.utils._sisl_cmd:sisl_cmd",
         # Add toolbox CLI
         "stoolbox = sisl_toolbox.cli:stoolbox_cli",
         "ts_poisson = sisl_toolbox.transiesta.poisson.poisson_explicit:poisson_explicit_cli",
         ]
        #"splotly = sisl.viz.plotly.splot:splot",
    },
    classifiers=CLASSIFIERS,
    platforms="any",
    project_urls=PROJECT_URLS,
    cmdclass=cmdclass,
    **setuptools_kwargs
)

cwd = osp.abspath(osp.dirname(__file__))
if not osp.exists(_ospath(cwd + "/PKG-INFO")):
    # Generate Cython sources, unless building from source release
    # generate_cython()
    pass


def git_version():
    global GIT_REVISION, ISRELEASED

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env).communicate()[0]
        return out.strip().decode("ascii")

    current_path = osp.dirname(osp.realpath(__file__))

    try:
        # Get top-level directory
        git_dir = _minimal_ext_cmd(["git", "rev-parse", "--show-toplevel"])
        # Assert that the git-directory is consistent with this setup.py script
        if git_dir != current_path:
            raise ValueError("Not executing the top-setup.py script")

        # Get latest revision tag
        rev = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        # Get latest tag
        tag = _minimal_ext_cmd(["git", "describe", "--abbrev=0"])
        # Get number of commits since tag
        count = _minimal_ext_cmd(["git", "rev-list", tag + "..", "--count"])
        if len(count) == 0:
            count = "1"
        # Ensure we have the correct ISRELEASED tag
        ISRELEASED = int(count) == 0
        # Get year
        year = int(_minimal_ext_cmd(["git", "show", "-s", "--format=%ci"]).split("-")[0])
        print("sisl-install: using git revision")
    except Exception as e:
        print("sisl-install: using internal shipped revisions")
        # Retain the revision name
        rev = GIT_REVISION
        # Assume it is on tag
        count = "0"
        year = REVISION_YEAR

    return rev, int(count), year


def write_version(filename=_ospath("sisl/info.py")):
    version_str = '''# This file is automatically generated from sisl setup.py
released = {released}

# Git information (specific commit, etc.)
git_revision = "{git}"
git_revision_short = git_revision[:7]
git_count = {count}

# Version information
major   = {version[0]}
minor   = {version[1]}
micro   = {version[2]}
version = ".".join(map(str,[major, minor, micro]))
release = version

if git_count > 2 and not released:
    # Add git-revision to the version string
    version += "+" + str(git_count)

# BibTeX information if people wish to cite
bibtex = f"""@misc{{{{zerothi_sisl,
    author = {{{{Papior, Nick}}}},
    title  = {{{{sisl: v{{version}}}}}},
    year   = {{{{{rev_year}}}}},
    doi    = {{{{10.5281/zenodo.597181}}}},
    url    = {{{{https://doi.org/10.5281/zenodo.597181}}}},
}}}}"""

def cite():
    return bibtex
'''
    # If we are in git we try and fetch the
    # git version as well
    GIT_REV, GIT_COUNT, REV_YEAR = git_version()
    with open(filename, "w") as fh:
        fh.write(version_str.format(version=[MAJOR, MINOR, MICRO],
                                    released=ISRELEASED,
                                    count=GIT_COUNT,
                                    rev_year=REV_YEAR, git=GIT_REV))


if __name__ == "__main__":

    # First figure out if we should define the
    # version file
    if "only-version" in sys.argv:
        # Figure out if we should write a specific file
        print("Only creating the version file")
        write_version()
        sys.exit(0)

    try:
        # Create version file
        # if allowed
        write_version()
    except Exception as e:
        print("Could not write sisl/info.py:")
        print(str(e))

    if ISRELEASED:
        metadata["version"] = VERSION
    else:
        metadata["version"] = VERSION + "-dev"

    # Freeze to support parallel compilation when using spawn instead of fork
    multiprocessing.freeze_support()
    setup(**metadata)
