from abc import abstractmethod
from dataclasses import copy, dataclass, fields

from sisl.messages import SislError, warn

__all__ = ["composite_geometry", "CompositeGeometrySection"]


@dataclass
class CompositeGeometrySection:

    @abstractmethod
    def build_section(self, geometry):
        ...

    @abstractmethod
    def add_section(self, geometry, geometry_addition):
        ...

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


def composite_geometry(sections, section_cls, **kwargs):
    """Creates a composite geometry from a list of sections.

    The sections are added one after another in the provided order.

    Parameters
    ----------
    sections: array-like of (_geom_section or tuple or dict)
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
        # If it is some arbitrary type, convert it to a tuple
        if not isinstance(s, (section_cls, tuple, dict)):
            s = (s, )
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
