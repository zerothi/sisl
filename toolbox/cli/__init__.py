""" A command-line-interface for toolboxes that provide CLI

This is a wrapper with sub-commands the toolboxes that are
accessible.
"""


class SToolBoxCLI:
    """ Run the CLI `stoolbox` """

    def __init__(self):
        self._cmds = []

    def register(self, setup):
        """ Register a setup callback function which creates the subparser

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
        import sys
        from pathlib import Path
        import argparse

        # Create command-line
        cmd = Path(sys.argv[0])
        p = argparse.ArgumentParser(f"{cmd.name}",
                                    description="Specific toolboxes to aid sisl users")

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
            cmd(subp)

        args = p.parse_args(argv)
        args.runner(args)


# Populate the commands

# First create the class to hold and dynamically create the commands
stoolbox_cli = SToolBoxCLI()

from sisl_toolbox.transiesta.poisson.fftpoisson_fix import fftpoisson_fix_cli
stoolbox_cli.register(fftpoisson_fix_cli)

from sisl_toolbox.siesta.atom._atom import atom_plot_cli
stoolbox_cli.register(atom_plot_cli)
