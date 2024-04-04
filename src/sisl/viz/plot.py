# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from sisl.messages import deprecate
from sisl.nodes import Workflow


class Plot(Workflow):
    """Base class for all plots"""

    def __getattr__(self, key):
        if key != "nodes":
            # If an ipython key is requested, get the plot and look
            # for the key in the plot. This is simply to enhance
            # interactivity in a python notebook environment.
            # However, this results in a (maybe undesired) behavior:
            # The plot is updated when ipython requests it, without any
            # explicit request to update it. This is how it has worked
            # from the beggining, so it's probably best to keep it like
            # this for now.
            if "ipython" in key:
                output = self.nodes.output.get()
            else:
                output = self.nodes.output._output
            return getattr(output, key)
        else:
            return super().__getattr__(key)

    def merge(self, *others, **kwargs):
        from .plots.merged import merge_plots

        return merge_plots(self, *others, **kwargs)

    def update_settings(self, *args, **kwargs):
        deprecate(
            "f{self.__class__.__name__}.update_settings is deprecated. Please use update_inputs.",
            "0.15",
        )
        return self.update_inputs(*args, **kwargs)

    @classmethod
    def plot_class_key(cls) -> str:
        return cls.__name__.replace("Plot", "").lower()
