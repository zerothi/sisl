# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
""" A command-line-interface for toolboxes that provide CLI

This is a wrapper with sub-commands the toolboxes that are
accessible.
"""

import typer
from ._typer_wrappers import annotate_typer

from sisl_toolbox.siesta.atom._atom import atom_plot
from sisl_toolbox.transiesta.poisson.fftpoisson_fix import fftpoisson_fix

app = typer.Typer(
    name="Sisl toolbox", 
    help="Specific toolboxes to aid sisl users",
    rich_markup_mode="markdown",
    add_completion=False
)

app.command()(annotate_typer(atom_plot))
app.command("ts-fft")(annotate_typer(fftpoisson_fix))

stoolbox_cli = app

