import pytest

import argparse
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


def test_collect_arguments3():
    with pytest.raises(ValueError):
        collect_arguments(['this.file.never.exists'], input=True)


def test_decorators1():

    # Create a default argument parser
    @default_ArgumentParser('SPBS', description='MY DEFAULT STUFF')
    def myArgParser(self, p=None, *args, **kwargs):
        return p

    p = myArgParser(None)
    assert "SPBS" in p.format_help()
    assert "MY DEFAULT STUFF" in p.format_help()

    p = argparse.ArgumentParser(description='SECOND DEFAULT')
    p = myArgParser(None, p)
    assert "SPBS" not in p.format_help()
    assert "MY DEFAULT STUFF" not in p.format_help()
    assert "SECOND DEFAULT" in p.format_help()


def test_decorators2():

    p = argparse.ArgumentParser()
    ns = default_namespace(my_default='test')

    class Act1(argparse.Action):
        @run_collect_action
        def __call__(self, parser, ns, value, option_string=None):
            setattr(ns, 'act1', value)

    class Act2(argparse.Action):
        @run_collect_action
        def __call__(self, parser, ns, value, option_string=None):
            assert ns.act1 is not None
            setattr(ns, 'act2', value)

    class Act3(argparse.Action):
        @collect_action
        def __call__(self, parser, ns, value, option_string=None):
            setattr(ns, 'act3', value)

    class Act4(argparse.Action):
        def __call__(self, parser, ns, value, option_string=None):
            with pytest.raises(AttributeError):
                assert ns.act3 is None
            setattr(ns, 'act4', value)

    class Act5(argparse.Action):
        @run_actions
        def __call__(self, parser, ns, value, option_string=None):
            pass

    p.add_argument('--a1', action=Act1)
    p.add_argument('--a2', action=Act2)
    p.add_argument('--a3', action=Act3)
    p.add_argument('--a4', action=Act4)
    p.add_argument('--a5', action=Act5, nargs=0)

    # Run arguments
    argv = '--a1 v1 --a2 v2 --a3 v3 --a4 v4'.split()
    args = p.parse_args(argv, namespace=ns)

    assert args.my_default == 'test'

    assert args.act1 == 'v1'
    assert args.act2 == 'v2'
    with pytest.raises(AttributeError):
        assert args.act3 is None
    assert args.act4 == 'v4'

    args = p.parse_args(argv + ['--a5'], namespace=ns)

    assert args.my_default == 'test'

    assert args.act1 == 'v1'
    assert args.act2 == 'v2'
    assert args.act3 == 'v3'
    assert args.act4 == 'v4'
