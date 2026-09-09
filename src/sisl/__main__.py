# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import sys

from .utils._sisl_cmd import sisl_cmd

if __name__ == "__main__":
    sys.exit(sisl_cmd())
