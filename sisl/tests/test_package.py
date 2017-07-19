from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np

import sisl


@attr('version')
class TestVersion(object):

    def test_version(self):
        sisl.__version__
        sisl.__major__
        sisl.__minor__
        sisl.__micro__
        sisl.info.version
        sisl.info.major
        sisl.info.minor
        sisl.info.micro
        sisl.info.release
        sisl.info.git_revision
        sisl.info.git_revision_short
