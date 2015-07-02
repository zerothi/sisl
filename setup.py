#!/usr/bin/env python
"""
sids: Library to create/mingle/handle geometries and tight-binding sets in python.
"""

from __future__ import print_function

DOCLINES = __doc__.split("\n")

import sys, subprocess
import os, os.path as osp

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: LPGL
Programming Language :: C
Programming Language :: Python :: 2
Programming Language :: Python :: 3
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

MAJOR               = 0
MINOR               = 1
MICRO               = 2
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

def generate_cython():
    cwd = osp.abspath(osp.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                          osp.join(cwd, 'tools', 'cythonize.py'),
                          'sids'],
                         cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")

metadata = dict(
    name = 'sids',
    maintainer = "Nick Papior Andersen",
    maintainer_email = "nickpapior@gmail.com",
    description = DOCLINES[0],
    long_description = "\n".join(DOCLINES[2:]),
    download_url = "https://github.com/zerothi/sids/releases",
    license = 'LGPLv3',
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    test_suite='nose.collector',
    )

from numpy.distutils.core import setup

cwd = osp.abspath(osp.dirname(__file__))
if not osp.exists(osp.join(cwd, 'PKG-INFO')):
    # Generate Cython sources, unless building from source release
    #generate_cython()
    pass

# Generate configuration
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    
    config.add_subpackage('sids')
    
    return config

metadata['version'] = VERSION
if not ISRELEASED:
    metadata['version'] = VERSION + '-dev'
    metadata['configuration'] = configuration
    
if __name__ == '__main__':

    # Main setup of python modules
    setup(**metadata)
