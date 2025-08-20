# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import argparse

from sisl._lib._argparse import get_argparse_parser

# Import object holding all the CLI
from sisl_toolbox.cli import register_toolbox_cli

from .basis_optimization import optimize_basis, write_basis_to_yaml


def basis_cli(subp=None, parser_kwargs={}):
    """Argparse CLI for the basis optimization utilities."""

    # Create main parser
    title = "Basis optimization utilities"
    is_sub = not subp is None
    if is_sub:
        p = subp.add_parser("basis", description=title, help=title, **parser_kwargs)
    else:
        p = argparse.ArgumentParser(title, description=title, **parser_kwargs)

    # If the main parser is executed, just print the help
    p.set_defaults(runner=lambda args: p.print_help())

    # Add the subparsers for the commands
    subp = p.add_subparsers(title="Commands")

    get_argparse_parser(
        write_basis_to_yaml, name="build", subp=subp, parser_kwargs=parser_kwargs
    )
    parser_kwargs["aliases"] = ("optim",)
    get_argparse_parser(
        optimize_basis, name="optimize", subp=subp, parser_kwargs=parser_kwargs
    )


register_toolbox_cli(basis_cli)
