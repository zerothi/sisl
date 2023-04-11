# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest

import operator as ops
import numpy as np

import sisl.typing as st


pytestmark = pytest.mark.typing

def test_argument():

    def func(a: st.AtomsArgument):
        print(a)

