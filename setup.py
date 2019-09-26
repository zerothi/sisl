#!/usr/bin/env python
"""
Library to create/handle geometries and tight-binding parameters in Python. Made with DFT in mind.
"""

from __future__ import print_function

if __doc__ is None:
    __doc__ = """sisl: Generic library for manipulating DFT output, geometries and tight-binding parameter sets"""

DOCLINES = __doc__.split("\n")

import sys
import subprocess
import os
import os.path as osp

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Topic :: Software Development
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Physics
Topic :: Utilities
"""

MAJOR = 0
MINOR = 9
MICRO = 7
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
GIT_REVISION = "01a5bf18d6f322bb4dff5fa31936881c3446b811"
REVISION_YEAR = 2019


# The MANIFEST should be updated (which it only is
# if it does not exist...)
# So we try and delete it...
if osp.exists('MANIFEST'):
    os.remove('MANIFEST')


# The install_requires should also be the
# requirements for the actual running of sisl
install_requires = [
    'six',
    'setuptools',
    'numpy>=1.13',
    'scipy>=0.18',
    'netCDF4',
    'pyparsing>=1.5.7',
]

# Create list of all sub-directories with
#   __init__.py files...
packages = ['sisl']
for subdir, dirs, files in os.walk('sisl'):
    if '__init__.py' in files:
        packages.append(subdir.replace(os.sep, '.'))
        if 'tests' in 'dirs':
            packages.append(subdir.replace(os.sep, '.') + '.tests')


def readme():
    if not osp.exists('README.md'):
        return ""
    return open('README.md', 'r').read()

metadata = dict(
    name='sisl',
    maintainer="Nick Papior",
    maintainer_email="nickpapior@gmail.com",
    description="Python interface for tight-binding model creation and analysis of DFT output. Input mechanism for large scale transport calculations using NEGF TBtrans (TranSiesta)",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="http://github.com/zerothi/sisl",
    download_url="http://github.com/zerothi/sisl/releases",
    license='LGPLv3',
    packages=packages,
    entry_points={
        'console_scripts':
        ['sgeom = sisl.geometry:sgeom',
         'sgrid = sisl.grid:sgrid',
         'sdata = sisl.utils.sdata:sdata',
         'sisl = sisl.utils.sdata:sdata']
    },
    classifiers=[_f.strip() for _f in CLASSIFIERS.split('\n') if _f],
    platforms=['Unix', 'Linux', 'Mac OS-X', 'Windows'],
    install_requires=install_requires,
    tests_require=['pytest'],
    zip_safe=False,
    extras_require={
        # We currently use xarray for additional data-analysis
        # And tqdm for progressbars
        'analysis': ['xarray>=0.10.0', 'tqdm'],
    },
)

# If pytest is installed, add it to setup_requires
try:
    import pytest
    metadata['setup_requires'] = ['pytest-runner']
except:
    pass

cwd = osp.abspath(osp.dirname(__file__))
if not osp.exists(osp.join(cwd, 'PKG-INFO')):
    # Generate Cython sources, unless building from source release
    # generate_cython()
    pass


# Generate configuration
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('sisl')
    config.get_version('sisl/info.py')

    return config


# With credits from NUMPY developers we use this
# routine to get the git-tag


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

    # Be sure to import this before numpy setup
    from setuptools import setup

    try:
        # Now we import numpy distutils for installation.
        # Note that this should work, also when
        from numpy.distutils.core import setup
        metadata['configuration'] = configuration
    except:
        if ISRELEASED:
            metadata['version'] = VERSION
        else:
            metadata['version'] = VERSION + '-dev'

    if 'sdist' in sys.argv:
        from distutils.command.sdist import sdist
        metadata['cmdclass'] = {'sdist': sdist}

    # Main setup of python modules
    setup(**metadata)
