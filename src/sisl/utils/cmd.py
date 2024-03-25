# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import argparse

from sisl.utils.ranges import strmap, strseq

__all__ = ["argv_negative_fix", "default_namespace"]
__all__ += ["collect_input", "collect_arguments"]
__all__ += ["add_sisl_version_cite_arg"]
__all__ += ["default_ArgumentParser"]
__all__ += ["collect_action", "run_collect_action"]
__all__ += ["run_actions"]
__all__ += ["add_action"]


def argv_negative_fix(argv):
    """Fixes `argv` list by adding a space for input that may be float's

    This function tries to prevent ``'-<>'`` being captured by `argparse`.

    Parameters
    ----------
    argv : list of str
       the arguments passed to an argument-parser
    """
    rgv = []
    for a in argv:
        try:
            strseq(complex, a)
            strmap(complex, a)
        except Exception:
            rgv.append(a)
        else:
            rgv.append(" " + a)
    return rgv


def default_namespace(**kwargs):
    """Ensure the namespace can be used to collect and run the actions

    Parameters
    ----------
    **kwargs : dict
       the dictionary keys added to the namespace object.
    """

    class CustomNamespace:
        pass

    namespace = CustomNamespace()
    namespace._actions_run = False
    namespace._actions = []
    for key in kwargs:
        setattr(namespace, key, kwargs[key])
    return namespace


def add_action(namespace, action, args, kwargs):
    """Add an action to the list of actions to be runned

    Parameters
    ----------
    namespace : obj
       the `argparse` namespace to append the action too
    action : obj
       the `argparse.Action` which is appended to the list of actions
    args : list
       arguments that will be passed to `action` once asked to runs
    kwargs : dict
       keyword arguments passed to `action` once asked to runs
    """
    namespace._actions.append((action, args, kwargs))


def collect_input(argv):
    """Function for returning the input file

    This simply creates a shortcut input file and returns
    it.

    Parameters
    ----------
    argv : list of str
       arguments passed to an `argparse.ArgumentParser`
    """
    # Grap input-file
    p = argparse.ArgumentParser("Parser for input file", add_help=False)
    # Now add the input and output file
    p.add_argument("input_file", nargs="?", default=None)
    # Retrieve the input file
    # (return the remaining options)
    args, argv = p.parse_known_args(argv)

    return argv, args.input_file


def add_sisl_version_cite_arg(parser):
    """Add a sisl version and citation argument to the ArgumentParser for printing (to stdout) the used sisl version

    Parameters
    ----------
    parser : `argparse.ArgumentParser`
       the parser to add the version string too
    """
    from sisl import __bibtex__, __version__

    group = parser.add_argument_group("version information")

    class PrintVersion(argparse.Action):
        def __call__(self, parser, ns, values, option_string=None):
            print(f"sisl: {__version__}")

    group.add_argument(
        "--version",
        nargs=0,
        action=PrintVersion,
        help=f"Show detailed sisl version information (v{__version__})",
    )

    class PrintCite(argparse.Action):
        def __call__(self, parser, ns, values, option_string=None):
            print(f"BibTeX:\n{__bibtex__}")

    group.add_argument(
        "--cite",
        nargs=0,
        action=PrintCite,
        help="Show the citation required when using sisl",
    )


def collect_arguments(argv, input=False, argumentparser=None, namespace=None):
    """Function for returning the actual arguments depending on the input options.

    This function will create a fake `argparse.ArgumentParser` which then
    will pass through the input figuring out which options
    that should be given to the final `argparse.ArgumentParser`.

    Parameters
    ----------
    argv : list of str
       the argument list that comprise the arguments
    input : bool, optional
       whether or not the arguments should also gather
       from the input file.
    argumentparser : argparse.ArgumentParser, optional
       the argument parser that should add the options that we find from
       the output and input files.
    namespace : argparse.Namespace, optional
       the namespace for the argument parser.
    """

    # First we figure out the input file, and the output file
    from sisl.io import get_sile

    # Create the default namespace in case there is none
    if namespace is None:
        namespace = default_namespace()

    if input:
        argv, input_file = collect_input(argv)
    else:
        input_file = None

    # Grap output-file
    p = argparse.ArgumentParser("Parser for output file", add_help=False)
    p.add_argument("--out", "-o", nargs=1, default=None)

    # Parse the passed args to sort out the input file and
    # the output file
    args, _ = p.parse_known_args(argv)

    if input_file is not None:
        try:
            obj = get_sile(input_file)
            argumentparser, namespace = obj.ArgumentParser(
                argumentparser, namespace=namespace, **obj._ArgumentParser_args_single()
            )
            # Be sure to add the input file
            setattr(namespace, "_input_file", input_file)
        except Exception as e:
            print(e)
            raise ValueError(
                f"File: '{input_file}' cannot be found. Please supply a readable file!"
            )

    if args.out is not None:
        try:
            obj = get_sile(args.out[0], mode="r")
            obj.ArgumentParser_out(argumentparser, namespace=namespace)
        except Exception:
            # Allowed pass due to pythonic reading
            pass

    return argumentparser, namespace, argv


def default_ArgumentParser(*A_args, **A_kwargs):
    """
    Decorator for routines which takes a parser as argument
    and ensures that it is _not_ ``None``.
    """

    def default_AP(func):
        # This requires that the first argument
        # for the function is the parser with default=None
        def new_func(self, parser=None, *args, **kwargs):
            if parser is None:
                # Create the new parser and insert in the argument list
                parser = argparse.ArgumentParser(*A_args, **A_kwargs)
            elif "description" in A_kwargs:
                parser.description = A_kwargs["description"]

            return func(self, parser, *args, **kwargs)

        return new_func

    return default_AP


def collect_action(func):
    """
    Decorator for collecting actions until the namespace attrbitute ``_actions_run`` is ``True``.

    Note that the `argparse.Namespace` object is the 2nd argument.
    """

    def collect(self, *args, **kwargs):
        if args[1]._actions_run:
            return func(self, *args, **kwargs)
        # Else we append the actions to be performed
        add_action(args[1], self, args, kwargs)
        return None

    return collect


def run_collect_action(func):
    """
    Decorator for collecting actions and running.

    Note that the `argparse.Namespace` object is the 2nd argument.
    """

    def collect(self, *args, **kwargs):
        if args[1]._actions_run:
            return func(self, *args, **kwargs)
        add_action(args[1], self, args, kwargs)
        return func(self, *args, **kwargs)

    return collect


def run_actions(func):
    """
    Decorator for running collected actions.

    Note that the `argparse.Namespace` object is the 2nd argument.
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
