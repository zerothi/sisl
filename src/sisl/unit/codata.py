# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections import namedtuple
from importlib.resources import as_file, files
from pathlib import Path
from typing import Optional, Union

from sisl._environ import get_environ_variable
from sisl._internal import set_module

__all__ = ["read_codata", "CODATA"]


@set_module("sisl.unit")
def read_codata(year: Union[int, str] = 2022) -> dict:
    """Read a shipped CODATA-XXXX.txt ascii file, as downloaded from NIST

    The CODATA files are downloaded from:

    https://pml.nist.gov/cuu/Constants/

    And any usage should credit the work done by NIST.

    Parameters
    ----------
    year :
        which CODATA year should we read.

    Returns
    -------
    CODATA mapping of values
    """

    # Figure out if the CODATA version is available!
    data_file: Path = files("sisl.unit").joinpath(f"codata_{year}.txt")

    if not data_file.is_file():
        # Locate all files in the sub-directory
        data_dir = data_file.parent
        codata_files = list(map(str, data_dir.glob("codata_*.txt")))
        raise FileNotFoundError(
            f"""\
Requesting codata_{year}.txt file which does not exist!

Only located these CODATA files:
{codata_files}
"""
        )

    Constant = namedtuple("Constant", "name value uncertainty unit")
    data = dict()
    # Store the year it was bound too
    data["year"] = str(year)

    reading_constant = False
    with open(data_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("Quantity"):
                reading_constant = True
                continue

            if not reading_constant:
                continue

            if line.startswith("--------") or not line:
                continue

            # We have something to parse
            line = list(filter(lambda x: x, line.strip().split("  ")))
            name = line[0]

            # if there is a unit, we have it here.
            unit = line[-1].strip()
            line = list(map(lambda x: x.replace(" ", "").replace("...", ""), line[1:]))
            if not line:
                continue

            try:
                float(line[-1])
                unit = None
            except:
                pass

            # Extract data
            value = line[0]
            uncertainty = line[1]

            constant = Constant(name, float(value), uncertainty, unit)
            data[name] = constant

    return data


year = get_environ_variable("SISL_CODATA")
CODATA = read_codata(year)
del year
