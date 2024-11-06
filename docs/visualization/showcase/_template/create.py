# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import argparse
from pathlib import Path

from sisl.viz.plotutils import get_plot_classes


def create_showcase_nb(cls, force=False):
    """
    Creates a new notebook to showcase a plot class from the showcase template.

    Parameters
    -----------
    cls: str
        the name of the class that you want to
    """
    if cls not in [c.__name__ for c in get_plot_classes()]:
        message = f"We didn't find a plot class with the name '{cls}'"

        if force:
            print(message)
        else:
            raise ValueError(message)

    with open(Path(__file__).parent / "Showcase template.ipynb", "r") as f:
        lines = f.read()

    with open(Path(__file__).parent.parent / f"{cls}.ipynb", "w") as f:
        f.write(lines.replace("<$plotclass$>", cls))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--cls", type=str, required=True)
    parser.add_argument("-f", "--force", action="store_true")

    args = parser.parse_args()
    create_showcase_nb(args.cls, getattr(args, "force", False))
