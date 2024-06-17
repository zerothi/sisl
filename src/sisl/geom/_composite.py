# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from abc import abstractmethod
from dataclasses import copy, dataclass, fields

from sisl import Geometry
from sisl._internal import set_module
from sisl.messages import SislError, warn

__all__ = ["composite_geometry", "CompositeGeometrySection", "GeometrySection"]


@set_module("sisl.geom")
@dataclass
class CompositeGeometrySection:
    @abstractmethod
    def build_section(self, previous: Geometry) -> Geometry: ...

    @abstractmethod
    def add_section(
        self, geometry: Geometry, geometry_addition: Geometry
    ) -> Geometry: ...

    def _junction_error(self, prev, msg, what):
        """Helper function to raise an error if the junction is not valid.

        It extends the error by specifying details about the sections that
        are being joined.
        """
        msg = f"Error at junction between sections {prev} and {self}. {msg}"
        if what == "raise":
            raise SislError(msg)
        elif what == "warn":
            warn(msg)


@set_module("sisl.geom")
@dataclass
class GeometrySection(CompositeGeometrySection):
    """Basic geometry section which does nothing, it simply returns a geometry

    When attaching two geometries they will be `Geometry.add`, without changes
    """

    geometry: Geometry

    def build_section(self, geometry):
        # it does not matter what the prior geometry was, we just add one
        return self.geometry

    def add_section(self, geometry, geometry_addition):
        return geometry.add(geometry_addition)


def composite_geometry(sections, section_cls, **kwargs):
    """Creates a composite geometry from a list of sections.

    The sections are added one after another in the provided order.

    Parameters
    ----------
    sections: array-like of (_geom_section or Geometry or tuple or dict)
        A list of sections to be added to the ribbon.

        Each section is either a `composite_geometry.section` or something that will
        be parsed to a `composite_geometry.section`.
    section_cls: class, optional
        The class to use for parsing sections.
    **kwargs:
        Keyword arguments used as defaults for the sections when the .
    """

    # Parse sections into Section objects
    def conv(s):
        if isinstance(s, Geometry):
            return GeometrySection(s)
        # If it is some arbitrary type, convert it to a tuple
        if not isinstance(s, (section_cls, tuple, dict)):
            s = (s,)
        # If we arrived here with a tuple, convert it to a dict
        if isinstance(s, tuple):
            s = {field.name: val for field, val in zip(fields(section_cls), s)}
        # At this point it is either a dict or already a section object.
        if isinstance(s, dict):
            return section_cls(**{**kwargs, **s})

        return copy.copy(s)

    # Then loop through all the sections.
    geom = None
    prev = None
    for i, section in enumerate(sections):
        section = conv(section)

        new_addition = section.build_section(prev)

        if i == 0:
            geom = new_addition
        else:
            geom = section.add_section(geom, new_addition)

        prev = section

    return geom


composite_geometry.section = CompositeGeometrySection
