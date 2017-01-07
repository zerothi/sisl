"""
Generic utility to create commonly used ArgumentParser
options etc.
"""

from sisl.utils.ranges import strmap, strseq


def argv_negative_fix(argv):
    """ Fixes the `argv` list by adding a space for input that may be float's """
    rgv = []
    for a in argv:
        try:
            strseq(complex, a)
            strmap(complex, a)
        except:
            rgv.append(a)
        else:
            rgv.append(' ' + a)
    return rgv


def default_namespace(*args, **kwargs):
    class CustomNamespace(object):
        pass
    namespace = CustomNamespace()
    namespace._actions_run = False
    namespace._actions = []
    for key in kwargs:
        setattr(namespace, key, kwargs[key])
    return namespace


def ensure_namespace(p, ns):
    """
    Ensure a namespace passed in case it is not.

    This is currently a hack for:
       https://bugs.python.org/issue27859
    """
    old_parse_known_args = p.parse_known_args

    def parse_known_args(self, argv, namespace=None):
        if namespace is None:
            return old_parse_known_args(argv, ns)
        return old_parse_known_args(argv, namespace)
    p.parse_known_args = parse_known_args


def collect_input(argv):
    """ Function for returning the input file

    This simply creates a shortcut input file and returns
    it.
    """
    import argparse

    # Grap input-file
    p = argparse.ArgumentParser('Parser for input file', add_help=False)
    # Now add the input and output file
    p.add_argument('input_file', nargs='?', default=None)
    # Retrieve the input file
    # (return the remaining options)
    args, argv = p.parse_known_args(argv)

    return argv, args.input_file


def collect_arguments(argv, input=False,
                      argumentparser=None,
                      namespace=None):
    """
    Function for returning the actual arguments depending on the input options.

    This function will create a fake `ArgumentParser` which then
    will pass through the input figuring out which options
    that should be given to the final `ArgumentParser`.

    Parameters
    ----------
    argv : ``list`` of ``str``
       the argument list that comprise the arguments

    input : ``bool``, ``False``
       whether or not the arguments should also gather
       from the input file.
    argumentparser : ``argparse.ArgumentParser``
       the argument parser that should add the options that we find from
       the output and input files.
    namespace : ``Namespace``
       the namespace for the argument parser.
    """

    # First we figure out the input file, and the output file
    import argparse
    import sisl
    import sys, os, os.path as osp

    # Create the default namespace in case there is none
    if namespace is None:
        namespace = default_namespace()

    if input:
        # Grap input-file
        p = argparse.ArgumentParser('Parser for input file', add_help=False)
        # Now add the input and output file
        p.add_argument('input_file', nargs='?', default=None)
        # Retrieve the input file
        # (return the remaining options)
        args, argv = p.parse_known_args(argv)
        input_file = args.input_file
    else:
        input_file = None

    # Grap output-file
    p = argparse.ArgumentParser('Parser for output file', add_help=False)
    p.add_argument('--out', '-o', nargs=1, default=None)

    # Parse the passed args to sort out the input file and
    # the output file
    args, _ = p.parse_known_args(argv)

    if input_file is not None:
        try:
            obj = sisl.get_sile(input_file)
            argumentparser, namespace = obj.ArgumentParser(argumentparser, namespace=namespace,
                                                           **obj._ArgumentParser_args_single())
            # Be sure to add the input file
            setattr(namespace, '_input_file', input_file)
        except Exception as e:
            print(e)
            raise ValueError("File: '"+input_file+"' cannot be found. Please supply a readable file!")

    if args.out is not None:
        try:
            obj = sisl.get_sile(args.out[0], mode='r')
            obj.ArgumentParser_out(argumentparser, namespace=namespace)
        except Exception as e:
            pass

    return argumentparser, namespace, argv


def dec_default_AP(*A_args, **A_kwargs):
    """
    Decorator for routines which takes a parser as argument
    and ensures that it is _not_ `None`.
    """
    def default_AP(func):
        # This requires that the first argument
        # for the function is the parser with default=None
        from argparse import ArgumentParser

        def new_func(self, parser=None, *args, **kwargs):
            if parser is None:
                # Create the new parser and insert in the
                # argument list
                parser = ArgumentParser(*A_args, **A_kwargs)
            return func(self, parser, *args, **kwargs)
        return new_func
    return default_AP


def dec_collect_action(func):
    """
    Decorator for collecting actions until the namespace `_actions_run` is `True`.

    Note that the `Namespace` object is the 2nd argument.
    """

    def collect(self, *args, **kwargs):
        if args[1]._actions_run:
            return func(self, *args, **kwargs)
        # Else we append the actions to be performed
        args[1]._actions.append((self, args, kwargs))
        return None
    return collect


def dec_collect_and_run_action(func):
    """
    Decorator for collecting actions and running.

    Note that the `Namespace` object is the 2nd argument.
    """

    def collect(self, *args, **kwargs):
        if args[1]._actions_run:
            return func(self, *args, **kwargs)
        args[1]._actions.append((self, args, kwargs))
        return func(self, *args, **kwargs)
    return collect


def dec_run_actions(func):
    """
    Decorator for running collected actions.

    Note that the `Namespace` object is the 2nd argument.
    """

    def run(self, *args, **kwargs):
        args[1]._actions_run = True
        # Run all so-far collected actions
        for A, Aargs, Akwargs in args[1]._actions:
            A(*Aargs, **Akwargs)
        args[1]._actions_run = False
        args[1]._actions = []
        return func(self, *args, **kwargs)
    return run
