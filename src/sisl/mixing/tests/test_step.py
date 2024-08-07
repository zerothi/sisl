# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl.mixing import AndersonMixer, LinearMixer, StepMixer

pytestmark = pytest.mark.mixing


def test_step_manual():
    mix1 = LinearMixer()
    mix2 = AndersonMixer()

    # now merge them
    def gen():
        yield mix1
        yield mix1
        yield mix2
        yield mix2

    mixer = StepMixer(gen)
    # Now lets test it
    f = np.random.rand(4) + 1
    df = np.random.rand(4) + 0.2

    # test 3 times
    for _ in range(3):
        # we start with linearmixer
        for _ in range(2):
            assert isinstance(mixer.mixer, LinearMixer)
            mixer(f, df)

        for _ in range(2):
            assert isinstance(mixer.mixer, AndersonMixer)
            mixer(f, df)


def test_step_yield_repeat():
    mix1 = LinearMixer()
    mix2 = AndersonMixer()
    mix1_rep2 = StepMixer.yield_repeat(mix1, 2)
    mix2_rep2 = StepMixer.yield_repeat(mix2, 2)

    # now merge them
    def gen():
        yield from mix1_rep2()
        yield from mix2_rep2()

    mixer = StepMixer(gen)
    # Now lets test it
    f = np.random.rand(4) + 1
    df = np.random.rand(4) + 0.2

    # test 3 times
    for _ in range(3):
        # we start with linearmixer
        for _ in range(2):
            assert isinstance(mixer.mixer, LinearMixer)
            mixer(f, df)

        for _ in range(2):
            assert isinstance(mixer.mixer, AndersonMixer)
            mixer(f, df)


def test_step_yield_repeat_n():
    mix = LinearMixer()
    r = StepMixer.yield_repeat(mix, 1)

    g = r()
    next(g)
    with pytest.raises(StopIteration):
        next(g)

    r = StepMixer.yield_repeat(mix, 2)
    g = r()
    next(g)
    next(g)
    with pytest.raises(StopIteration):
        next(g)


def test_step_yield_chain():
    mix1 = LinearMixer()
    mix2 = AndersonMixer()
    mix1_rep2 = StepMixer.yield_repeat(mix1, 2)
    mix2_rep2 = StepMixer.yield_repeat(mix2, 2)

    # now merge them
    mixer = StepMixer(mix1_rep2, mix2_rep2)

    # Now lets test it
    f = np.random.rand(4) + 1
    df = np.random.rand(4) + 0.2

    # test 3 times
    for _ in range(3):
        # we start with linearmixer
        for _ in range(2):
            assert isinstance(mixer.mixer, LinearMixer)
            mixer(f, df)

        for _ in range(2):
            assert isinstance(mixer.mixer, AndersonMixer)
            mixer(f, df)
