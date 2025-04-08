# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from nodify import Workflow

from sisl.messages import deprecate

from .figure import BACKENDS


class Plot(Workflow):
    """Base class for all plots"""

    def __getattr__(self, key):
        if key != "nodes":
            # From the backend input, we find out which class is the figure going to be
            # (even if no figure has been created yet or the latest figure was from a different backend)
            # Then we check if the attribute will be available there. If it will, we update the plot and
            # get the attribute on the updated plot.
            # This is so that things like `plot.show()` work as expected.
            # It has the downside that `.get()` is called even when for example a method of the figure is
            # retreived to get its docs (e.g. in the helper messages of jupyter notebooks)
            selected_backend = self.inputs.get("backend")
            figure_cls = BACKENDS.get(selected_backend)
            if figure_cls is not None and (
                hasattr(figure_cls, key) or figure_cls.fig_has_attr(key)
            ):
                return getattr(self.nodes.output.get(), key)
            else:
                raise AttributeError(
                    f"'{key}' not found in {self.__class__.__name__} with backend '{selected_backend}'"
                )
        else:
            return super().__getattr__(key)

    def merge(self, *others, **kwargs):
        from .plots.merged import merge_plots

        return merge_plots(self, *others, **kwargs)

    def update_settings(self, *args, **kwargs):
        deprecate(
            "f{self.__class__.__name__}.update_settings is deprecated. Please use update_inputs.",
            "0.15",
            "0.17",
        )
        return self.update_inputs(*args, **kwargs)

    @classmethod
    def plot_class_key(cls) -> str:
        return cls.__name__.replace("Plot", "").lower()
