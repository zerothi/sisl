from __future__ import print_function, division

import pytest

from sisl.utils.cmd import *

pytestmark = pytest.mark.utils


def test_default_namespace1():
    d = {'a': 1}
    dd = default_namespace(**d)
    assert dd.a == d['a']


def test_collect_input1():
    argv = ['test.xyz', '--stneohus stnaoeu', '-a', 'aote']
    argv_out, in_file = collect_input(argv)

    assert in_file == 'test.xyz'
    assert len(argv) == len(argv_out) + 1


def test_collect_arguments1():
    ap, ns, argv = collect_arguments([])
    assert len(argv) == 0


def test_collect_arguments2():
    ap, ns, argv = collect_arguments([], input=True)
    assert len(argv) == 0
