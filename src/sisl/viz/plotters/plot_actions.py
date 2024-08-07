# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""Contains all the individual actions that can be performed on a figure."""

import functools
import inspect
import sys
from typing import Literal, Optional

from ..figure import Figure


def _register_actions(figure_cls: type[Figure]):
    # Take all actions possible from the Figure class
    module = sys.modules[__name__]

    actions = inspect.getmembers(
        figure_cls,
        predicate=lambda x: inspect.isfunction(x) and not x.__name__.startswith("_"),
    )

    for name, function in actions:
        sig = inspect.signature(function)

        @functools.wraps(function)
        def a(*args, __method_name__=function.__name__, **kwargs):
            return dict(method=__method_name__, args=args, kwargs=kwargs)

        a.__signature__ = sig.replace(parameters=list(sig.parameters.values())[1:])
        a.__module__ = module

        setattr(module, name, a)


_register_actions(Figure)


def combined(
    *plotters,
    composite_method: Optional[
        Literal["multiple", "subplots", "multiple_x", "multiple_y", "animation"]
    ] = None,
    provided_list: bool = False,
    **kwargs,
):
    if provided_list:
        plotters = plotters[0]

    return {
        "composite_method": composite_method,
        "plot_actions": plotters,
        "init_kwargs": kwargs,
    }
