# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Import object holding all the CLI
import sys
from pathlib import Path

from sisl_toolbox.cli import register_toolbox_cli

from ._atom import atom_plot

# TODO once 3.10 hits, use `sys.orig_argv[0]` which is non-writeable
_script = Path(sys.argv[0]).name


def atom_plot_cli(subp=None, parser_kwargs={}):
    """Run plotting command for the output of atom"""

    is_sub = not subp is None

    title = "Plotting facility for atom output (run in the atom output directory)"
    if is_sub:
        global _script
        _script = f"{_script} atom-plot"
        p = subp.add_parser("atom-plot", description=title, help=title, **parser_kwargs)
    else:
        import argparse

        p = argparse.ArgumentParser(title, **parser_kwargs)

    p.add_argument(
        "--plot",
        "-P",
        action="append",
        type=str,
        choices=("wavefunction", "charge", "log", "potential"),
        help="""Determine what to plot""",
    )

    p.add_argument("-l", default="spdf", type=str, help="""Which l shells to plot""")

    p.add_argument("--save", "-S", default=None, help="""Save output plots to file.""")

    p.add_argument(
        "--show",
        default=False,
        action="store_true",
        help="""Force showing the plot (only if --save is specified)""",
    )

    p.add_argument(
        "input", type=str, default="INP", help="""Input file name (default INP)"""
    )

    if is_sub:
        p.set_defaults(runner=atom_plot)
    else:
        atom_plot(p.parse_args())


register_toolbox_cli(atom_plot_cli)
