# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A command-line-interface for toolboxes that provide CLI

This is a wrapper with sub-commands the toolboxes that are
accessible.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from sisl._lib._argparse import SislHelpFormatter


class SToolBoxCLI:
    """Run the CLI `stoolbox`"""

    def __init__(self):
        self._cmds = []

    def register(self, setup):
        """Register a setup callback function which creates the subparser

        The ``setup(..)`` command must accept a sub-parser from `argparse` as its
        first argument.

        The only requirements to create a sub-command is to fullfill these requirements:

        1. Create a new parser using ``subp.add_parser``.
        2. Ensure a runner is attached to the subparser through ``.set_defaults(runner=<callable>)``

        A minimal example would be:

        >>> def setup(subp):
        ...    p = subp.add_parser("test-sub")
        ...    def test_sub_method(args):
        ...        print(args)
        ...    p.set_defaults(runner=test_sub_method)
        """
        self._cmds.append(setup)

    def __call__(self, argv=None):

        # Create command-line
        cmd = Path(sys.argv[0])
        p = argparse.ArgumentParser(
            f"{cmd.name}",
            description="Specific toolboxes to aid sisl users",
            formatter_class=SislHelpFormatter,
        )

        info = {
            "title": "Toolboxes",
            "metavar": "TOOL",
        }

        # Check which Python version we have
        version = sys.version_info
        if version.major >= 3 and version.minor >= 7:
            info["required"] = True

        # Create the sub-parser
        subp = p.add_subparsers(**info)

        for cmd in self._cmds:
            cmd(subp, parser_kwargs=dict(formatter_class=p.formatter_class))

        args = p.parse_args(argv)
        args.runner(args)


# First create the class to hold and dynamically create the commands
stoolbox_cli = SToolBoxCLI()


def register_toolbox_cli(func: Callable[[argparse.ArgumentParser], None]):
    """Register a function to the CLI

    Parameters
    ----------
    func :
        a function that gets called with an `argparse.ArgumentParser`
        argument and manipulates it to add the arguments needed for the
        parser.
    """
    global stoolbox_cli
    stoolbox_cli.register(func)
