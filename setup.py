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
import os
import os.path as osp

# pkg_resources are part of setuptools
import pkg_resources


# Define a list of minimum versions
min_version ={
    "numpy": "1.13",
}


try:
    import Cython
    # if available we can cythonize stuff

    _CYTHON_VERSION = Cython.__version__
    from Cython.Build import cythonize

    # We currently do not have any restrictions on Cython (I think?)
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
    from Cython.Distutils.old_build_ext import old_build_ext as build_ext

    cython = True
    from Cython import Tempita as tempita
else:
    # overload the old one
    from distutils.command.build_ext import build_ext

    cython = False


# We will *only* use setuptools
# Although setuptools is not shipped with the standard library, I think
# this is ok since it should get installed pretty easily.
from distutils.command.build import build # just to have it before Cython
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist
from setuptools import Command, Extension
from setuptools import find_packages
from setuptools import setup
# Patch to allow fortran sources
from numpy.distutils.core import setup
from numpy.distutils.core import Extension as FortranExtension


# Custom command classes
cmdclass = {}


# Now create the build extensions
class CythonCommand(build_ext):
    """
    Custom distutils command subclassed from Cython.Distutils.build_ext
    to compile pyx->c, and stop there. All this does is override the
    C-compile method build_extension() with a no-op.
    """
    # note this is used from Pandas (a clever hack)
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
macros = []

# in numpy>=1.16.0, silence build warnings about deprecated API usage
macros.append(("NPY_NO_DEPRECATED_API", "0"))
# Do not expose multiple platform Cython code.
# We do not need it
#  https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#integrating-multiple-modules
macros.append(("CYTHON_NO_PYINIT_EXPORT", "1"))


_pyxfiles = [
    "sisl/_indices.pyx",
    "sisl/_math_small.pyx",
    "sisl/physics/_bloch.pyx",
    "sisl/physics/_matrix_ddk.pyx",
    "sisl/physics/_matrix_dk.pyx",
    "sisl/physics/_matrix_k.pyx",
    "sisl/physics/_matrix_phase3.pyx",
    "sisl/physics/_matrix_phase_nc_diag.pyx",
    "sisl/physics/_matrix_phase_nc.pyx",
    "sisl/physics/_matrix_phase.pyx",
    "sisl/physics/_matrix_phase_so.pyx",
    "sisl/physics/_phase.pyx",
    "sisl/physics/_phase.pyx",
    "sisl/_sparse.pyx",
    "sisl/_supercell.pyx",
]

# Prepopulate the ext_data to create extensions
# Later, when we complicate things more, we
# may need the dictionary to add include statements etc.
# I.e. ext_data[...] = {pyxfile: ...,
#                       include: ...,
#                       depends: ...,
#                       sources: ...}
# All our extensions depend on numpy/core/include
numpy_incl = pkg_resources.resource_filename("numpy", "core/include")

ext_data = {}
for pyx in _pyxfiles:
    # remove ".pyx"
    pyx_src = pyx[:-4]
    pyx_mod = pyx_src.replace("/", ".")
    ext_data[pyx_mod] = {
        "pyxfile": pyx_src,
        "include": [numpy_incl]
    }
    ext_data[pyx_mod] = {"pyxfile": pyx_src}



# List of extensions for setup(...)
extensions = []
for name, data in ext_data.items():
    sources = [data["pyxfile"] + ".c"] + data.get("sources", [])

    # Get options for extensions
    include = data.get("include", None)

    obj = Extension(name,
                    sources=sources,
                    depends=data.get("depends", []),
                    include_dirs=include,
                    language=data.get("language", "c"),
                    define_macros=data.get("macros", macros),
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
    )

    extensions.append(obj)


# Specific Fortran extensions
ext_fortran = {
    "sisl.io.siesta._siesta": {
        "sources": [f"sisl/io/siesta/_src/{f}" for f in
                    ("io_m.f90",
                     "siesta_sc_off.f90",
                     "hsx_read.f90", "hsx_write.f90",
                     "dm_read.f90", "dm_write.f90",
                     "tshs_read.f90", "tshs_write.f90",
                     "grid_read.f90", "grid_write.f90",
                     "gf_read.f90", "gf_write.f90",
                     "tsde_read.f90", "tsde_write.f90",
                     "hs_read.f90",
                     "wfsx_read.f90")
        ],
    },
}

for name, data in ext_fortran.items():
    sources = data.get("sources")

    # Get options for extensions
    include = data.get("include", None)

    obj = FortranExtension(name,
        sources=sources,
        depends=data.get("depends", []),
        include_dirs=include,
        define_macros=data.get("macros", macros),
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    extensions.append(obj)


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
    


MAJOR = 0
MINOR = 9
MICRO = 8
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
GIT_REVISION = "13a327bd8e27d689f119bafdf38519bab7f6e0f6"
REVISION_YEAR = 2020



DISTNAME = "sisl"
LICENSE = 'LGPLv3',
AUTHOR = "Nick Papior"
URL = "https://github.com/zerothi/sisl",
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
    "python_requires": ">= 3.6",
    "install_requires": [
        "setuptools",
        "numpy >= " + min_version["numpy"],
        "scipy",
        "netCDF4",
        "pyparsing >= 1.5.7",
    ],
    "setup_requires": [
        "numpy >= " + min_version["numpy"],
    ],
    "extras_require": {
        # We currently use xarray for additional data-analysis
        # And tqdm for progressbars
        "analysis": [
            "xarray >= 0.10.0",
            "tqdm",
        ],
    },
    "zip_safe": False,
}

# Create list of all sub-directories with
#   __init__.py files...
packages = ['sisl']
for subdir, dirs, files in os.walk('sisl'):
    if '__init__.py' in files:
        packages.append(subdir.replace(os.sep, '.'))
        if 'tests' in dirs:
            packages.append(subdir.replace(os.sep, '.') + '.tests')


def readme():
    if not osp.exists('README.md'):
        return ""
    return open('README.md', 'r').read()

metadata = dict(
    name=DISTNAME,
    maintainer=AUTHOR,
    description="Python interface for tight-binding model creation and analysis of DFT output. Input mechanism for large scale transport calculations using NEGF TBtrans (TranSiesta)",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="http://github.com/zerothi/sisl",
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    packages=find_packages(include=["sisl", "sisl.*"]),
    ext_modules=extensions,
    entry_points={
        'console_scripts':
        ['sgeom = sisl.geometry:sgeom',
         'sgrid = sisl.grid:sgrid',
         'sdata = sisl.utils.sdata:sdata',
         'sisl = sisl.utils.sdata:sdata']
    },
    classifiers=CLASSIFIERS,
    platforms="any",
    project_urls = PROJECT_URLS,
    **setuptools_kwargs
)

cwd = osp.abspath(osp.dirname(__file__))
if not osp.exists(osp.join(cwd, 'PKG-INFO')):
    # Generate Cython sources, unless building from source release
    # generate_cython()
    pass

def git_version():
    global GIT_REVISION

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env).communicate()[0]
        return out.strip().decode('ascii')

    current_path = osp.dirname(osp.realpath(__file__))

    try:
        # Get top-level directory
        git_dir = _minimal_ext_cmd(['git', 'rev-parse', '--show-toplevel'])
        # Assert that the git-directory is consistent with this setup.py script
        if git_dir != current_path:
            raise ValueError('Not executing the top-setup.py script')

        # Get latest revision tag
        rev = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        # Get latest tag
        tag = _minimal_ext_cmd(['git', 'describe', '--abbrev=0'])
        # Get number of commits since tag
        count = _minimal_ext_cmd(['git', 'rev-list', tag + '..', '--count'])
        if len(count) == 0:
            count = '1'
        # Get year
        year = int(_minimal_ext_cmd(['git', 'show', '-s', '--format=%ci']).split('-')[0])
        print('sisl-install: using git revision')
    except Exception as e:
        print('sisl-install: using internal shipped revisions')
        # Retain the revision name
        rev = GIT_REVISION
        # Assume it is on tag
        count = '0'
        year = REVISION_YEAR

    return rev, int(count), year


def write_version(filename='sisl/info.py'):
    version_str = """# This file is automatically generated from sisl setup.py
released = {released}

# Git information (specific commit, etc.)
git_revision = '{git}'
git_revision_short = git_revision[:7]
git_count = {count}

# Version information
major   = {version[0]}
minor   = {version[1]}
micro   = {version[2]}
version = '.'.join(map(str,[major, minor, micro]))
release = version

if git_count > 2 and not released:
    # Add git-revision to the version string
    version += '+' + str(git_count)

# BibTeX information if people wish to cite
bibtex = '''@misc{{{{zerothi_sisl,
    author = {{{{Papior, Nick}}}},
    title  = {{{{sisl: v{{0}}}}}},
    year   = {{{{{rev_year}}}}},
    doi    = {{{{10.5281/zenodo.597181}}}},
    url    = {{{{https://doi.org/10.5281/zenodo.597181}}}},
}}}}'''.format(version)

def cite():
    return bibtex
"""
    # If we are in git we try and fetch the
    # git version as well
    GIT_REV, GIT_COUNT, REV_YEAR = git_version()
    with open(filename, 'w') as fh:
        fh.write(version_str.format(version=[MAJOR, MINOR, MICRO],
                                    released=ISRELEASED,
                                    count=GIT_COUNT,
                                    rev_year=REV_YEAR, git=GIT_REV))


if __name__ == '__main__':

    # First figure out if we should define the
    # version file
    try:
        only_idx = sys.argv.index('only-version')
    except:
        only_idx = 0
    if only_idx > 0:
        # Figure out if we should write a specific file
        print("Only creating the version file")
        if len(sys.argv) > only_idx + 1:
            vF = sys.argv[only_idx+1]
            write_version(vF)
        else:
            write_version()
        sys.exit(0)

    try:
        # Create version file
        # if allowed
        write_version()
    except Exception as e:
        print('Could not write sisl/info.py:')
        print(str(e))

    if ISRELEASED:
        metadata['version'] = VERSION
    else:
        metadata['version'] = VERSION + '-dev'

    # Main setup of python modules
    setup(**metadata)
