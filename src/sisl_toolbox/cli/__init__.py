# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
""" A command-line-interface for toolboxes that provide CLI

This is a wrapper with sub-commands the toolboxes that are
accessible.
"""

import typer
from ._typer_wrappers import annotate_typer

app = typer.Typer(
    name="Sisl toolbox", 
    help="Specific toolboxes to aid sisl users",
    rich_markup_mode="markdown"
)

from sisl_toolbox.siesta.atom._atom import atom_plot

app.command()(annotate_typer(atom_plot))

stoolbox_cli = app


# Populate the commands

# First create the class to hold and dynamically create the commands
# stoolbox_cli = SToolBoxCLI()

# from sisl_toolbox.transiesta.poisson.fftpoisson_fix import fftpoisson_fix_cli

# stoolbox_cli.register(fftpoisson_fix_cli)

# from sisl_toolbox.siesta.atom._atom import atom_plot_cli

# stoolbox_cli.register(atom_plot_cli)
