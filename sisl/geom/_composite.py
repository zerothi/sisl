import abc
import dataclasses
from dataclasses import dataclass, fields

from sisl.messages import warn

__all__ = ['composite_geometry']


@dataclass
class _GeometrySection:

    @abc.abstractmethod
    def add_section(self, geometry, previous, **kwargs):
        """Add `self` to geometry and return a new `Geometry` object, `previous` is the previously added geometry

        Any subclass of `GeometrySection` *must* implement the add function
        which adds two sections together.

        Both `self` and `previous` *must* be an instance of `GeometrySection` for this
        to work.

        Parameters
        ----------
        geometry : Geometry or None
           the geometry to add *this* segment too, if this is None, `previous` shouldn't be referenced
        previous : list of GeometrySection
           the previous sections added to `geometry`, only meaningfull if `geometry` is not None.
           This is a list of sections that describes the order of prior sections added.
        **kwargs :
           special arguments for this sections add mechanism, the arguments will be forced
           to not have names that intersect with the fields of the class.
        """

    def create_error_handler(self, what):
        """Helper function to create a handler for errors

        Parameter
        ---------
        what : str or Exception or Warning
           determines how the function creates the handler
           If equals to raise, it will raise a ValueError, other
           raised types can be forced by explicitly passing them as `what`

        Returns
        -------
        method : callable
            the method will accept 1 single message and any number of arguments as the implementor should
            define what else is written
        """
        if isinstance(what, str):
            # this is to prevent it from hitting issubclass
            # That would require it to be a class (isinstance(what, object))
            pass
        elif issubclass(what, Warning):
            def func(msg, *args, **kwargs):
                import warnings
                warnings.warn(what(msg, *args, **kwargs))
            return func
        elif issubclass(what, Exception):
            def func(msg, *args, **kwargs):
                raise what(msg, *args, **kwargs)
            return func
        else:
            raise ValueError(f"Unknown argument type, must be one of Exception or str, found {type(what)}")

        what = what.lower()
        if what == "warn":
            def func(msg, *args, **kwargs):
                warn(msg, *args, **kwargs)
        elif what in ("raise", "error"):
            def func(msg, *args, **kwargs):
                raise ValueError(msg, *args, **kwargs)
        else:
            raise ValueError(f"Unknown argument value, must be one of 'warn' or 'raise', got {what}")

        return func


def composite_geometry(sections, section_cls=_GeometrySection, **kwargs):
    """Creates a composite geometry from a list of sections.

    The sections are added one after another in the provided order.

    Parameters
    ----------
    sections: array-like of (_geom_section or tuple or dict)
        A list of sections to be added to the ribbon.

        Each section is either a `_geom_section` or something that will
        be parsed to a `_geom_section`.
    section_cls: class, optional
        The class to use for parsing sections.
    **kwargs:
        Keyword arguments used as defaults for the sections when the .
    """
    def strip_kwargs(cls, kwargs, not_in=True):
        """ Strip items in `kwargs` from ``fields(cls)`` """
        field_names = [f.name for f in fields(cls)]

        # Now return a new kwargs based on keys not in field_names
        if not_in:
            return {k: v for k, v in kwargs.items() if k not in field_names}
        return {k: v for k, v in kwargs.items() if k in field_names}

    # Parse sections into Section objects
    def conv(s):
        # If it is some arbitrary type, convert it to a tuple
        if not isinstance(s, (section_cls, tuple, list, dict)):
            s = (s, )

        # If we arrived here with a tuple, convert it to a dict
        if isinstance(s, (tuple, list)):
            s = {field.name: val for field, val in zip(fields(section_cls), s)}

        # At this point it is either a dict or already a section object.
        if isinstance(s, dict):
            # We need to remove any arguments that does not fit this class
            return section_cls(**{**strip_kwargs(section_cls, kwargs, not_in=False), **s})

        # TODO: do we really need a copy here? I removed it
        return s

    # Then loop through all the sections
    # Initialize the returned geometry and the previous one
    geometry = None
    prior_sections = []
    for section in map(conv, sections):

        # add the two sections
        geometry = section.add_section(geometry, prior_sections, **strip_kwargs(section, kwargs))
        prior_sections.append(section)

    return geometry

composite_geometry.Section = _GeometrySection
