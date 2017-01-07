#!/usr/bin/env python
"""
Easy conversion of data from different formats to other formats.
"""
from __future__ import print_function, division

import sys
import os
import os.path as osp
import argparse

__all__ = ['sdata']


def argparse_patch(parser):
    """ Patch the argparse module such that one may process the Namespace in subparsers

    This patch have been created by: 
      paul.j3 (http://bugs.python.org/file44363/issue27859test.py)
    and adapted by Nick R. Papior with minor edits.

    Parameters
    ----------
    parser: ArgumentParser
       parser to be patched
    """
    class MySubParsersAction(argparse._SubParsersAction):

        def __call__(self, parser, namespace, values, option_string=None):
            parser_name = values[0]
            arg_strings = values[1:]

            # set the parser name if requested
            if self.dest is not argparse.SUPPRESS:
                setattr(namespace, self.dest, parser_name)

            # select the parser
            try:
                parser = self._name_parser_map[parser_name]
            except KeyError:
                args = {'parser_name': parser_name,
                        'choices': ', '.join(self._name_parser_map)}
                msg = _('unknown parser %(parser_name)r (choices: %(choices)s)') % args
                raise ArgumentError(self, msg)

            # parse all the remaining options into the namespace
            # store any unrecognized options on the object, so that the top
            # level parser can decide what to do with them

            # pass parent namespace (it is now the users responsibility to
            # not have dublicate .default parameters)
            namespace, arg_strings = parser.parse_known_args(arg_strings, namespace)

            ## ORIGINAL
            #subnamespace, arg_strings = parser.parse_known_args(arg_strings, None)
            #for key, value in vars(subnamespace).items():
            #    setattr(namespace, key, value)

            if arg_strings:
                vars(namespace).setdefault(_UNRECOGNIZED_ARGS_ATTR, [])
                getattr(namespace, _UNRECOGNIZED_ARGS_ATTR).extend(arg_strings)
    parser.register('action', 'parsers', MySubParsersAction)


def sdata(argv=None, sile=None):

    from . import cmd

    # The file *MUST* be the first argument
    # (except --help|-h)

    # We cannot create a separate ArgumentParser to retrieve a positional arguments
    # as that will grab the first argument for an option!

    # Start creating the command-line utilities that are the actual ones.
    description = """
This manipulation utility can handle nearly all files in the sisl code in
changing ways. It handles files dependent on type AND content.
    """.format(osp.basename(sys.argv[0]))

    if argv is not None:
        # We keep the arguments
        pass
    elif len(sys.argv) == 1:
        # no arguments
        # fake a help
        argv = ['--help']
    else:
        argv = sys.argv[1:]

    # Ensure that the arguments have pre-pended spaces
    argv = cmd.argv_negative_fix(argv)

    p = argparse.ArgumentParser("Manipulates sisl Sile's for manipulation.",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=description, conflict_handler='resolve')

    # Patch the parser to allow namespace passing in subparsers...
    argparse_patch(p)

    # Now try and figure out the actual arguments
    p, ns, argv = cmd.collect_arguments(argv, input=True,
                                        argumentparser=p)

    # Now the arguments should have been populated
    # and we will sort out if the input options
    # is only a help option.
    if not hasattr(ns, '_input_file'):
        # Then there are no input files...
        # It is difficult to create an adaptable script
        # with no adaptee... ;)
        if not (('--help' in argv) or ('-h' in argv)):
            # Re-create the argument parser with the help description
            argv = ['--help']
        description += """

One *must* supply a file before a help menu can be generated.
The help menu depends on the type of Sile that is specified.
        """
        p = argparse.ArgumentParser("Manipulates sisl Sile's for manipulation.",
                                    formatter_class=argparse.RawDescriptionHelpFormatter,
                                    description=description)

    # We are good to go!!!
    p.parse_args(argv, namespace=ns)

    return 0
