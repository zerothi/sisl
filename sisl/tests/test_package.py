from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np

import sisl

class TestVersion(object):

    def test_version(self):
        print(sisl.__version__)
        print(sisl.__major__)
        print(sisl.__minor__)
        print(sisl.__micro__)
        print(sisl.info.__version__)
        print(sisl.info.__major__)
        print(sisl.info.__minor__)
        print(sisl.info.__micro__)
        print(sisl.info.release)
        print(sisl.info.git_revision)
        print(sisl.info.git_revision_short)

        
