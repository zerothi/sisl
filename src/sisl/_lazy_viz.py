# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""Module that sets a placeholder for the plot method in sisl classes.

When the plot attribute is accessed, sisl.viz is imported and the placeholders are
all removed, presumably to be replaced by the actual plot handlers."""
import sisl

__all__ = ["set_viz_placeholders", "clear_viz_placeholders"]

# Classes that will have a plot attribute placeholder
lazy_viz_classes = [
    sisl.Geometry,
    sisl.Grid,
    sisl.BrillouinZone,
    sisl.SparseCSR,
    sisl.SparseOrbital,
    sisl.SparseAtom,
    sisl.EigenstateElectron,
    sisl.io.Sile,
]


class PlotHandlerPlaceholder:
    """Placeholder to set as "plot" attribute for plotables while sisl.viz is not imported."""

    def __get__(self, instance, owner):
        # Import sisl.viz, which will remove all the placeholders and
        # set the actual plot handlers as the "plot" attribute.
        import sisl.viz  # noqa: F401

        # Return the plot handler
        if instance is None:
            return owner.plot
        else:
            return instance.plot


def set_viz_placeholders():
    """Set the plot attribute to a placeholder for all the classes."""
    placeholder = PlotHandlerPlaceholder()
    # Set the plot attribute to the placeholder for all the classes
    for cls in lazy_viz_classes:
        cls.plot = placeholder


def clear_viz_placeholders():
    for cls in lazy_viz_classes:
        del cls.plot
