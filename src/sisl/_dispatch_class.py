""" Internal class used for subclassing.

This class implements the 

__init_subclass__ method to ensure that classes automatically
create the `to`/`new` methods.

Sometimes these can be required to be inherited as well.

The usage interface should look something like this:

class A(_ToNew,
        to=object used | str | None,
        new=object used | str | None,
)

A.to.register ..
A.new.register ..
"""
from typing import Any, Optional
from collections import namedtuple

from ._dispatcher import ClassDispatcher, TypeDispatcher


class _ToNew:
    """Subclassable for creating the new/to arguments"""

    def __init_subclass__(cls, /,
                          new: Optional[Any] = "default",
                          to: Optional[Any] = "default",
                          when_subclassing: Optional[str] = None,
                          **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Get the allowed actions for subclassing
        prefix = "_tonew"
        allowed_subclassing = ("keep", "new", "copy")

        def_new = ClassDispatcher("new")
        def_to = ClassDispatcher("to")
       
        # check for the default handler of the parent class
        # We find the first base class which has either of the 
        first_base = None
        for b in cls.__bases__:
            if hasattr(b, "to") or hasattr(b, "new"):
                # any one fits
                first_base = b
                break

        if when_subclassing is None and first_base is not None:
            when_subclassing = getattr(first_base, f"{prefix}_when_subclassing")
        else:
            when_subclassing = "new"

        if when_subclassing not in allowed_subclassing:
            raise ValueError(f"when_subclassing should be one of {allowed_subclassing}")

        for attr, arg, def_ in (("to", to, def_to),
                                ("new", new, def_new)):
        
            base = first_base
            for b in cls.__bases__:
                if hasattr(b, attr):
                    base = b
                    break
            
            if base is None and isinstance(arg, str):
                # when there is no base and the class is not 
                # specified. Then regardless a new object will be used.
                arg = def_
        
            if isinstance(arg, str):
                if arg == "default":
                    arg = when_subclassing

                # now we can parse the potential subclassing problem
                if arg == "keep":
                    # signal we do nothing...
                    # this will tap into the higher class structures
                    # registration.
                    arg = None

                elif arg == "new":
                    # always create a new one
                    arg = def_

                elif arg == "copy":
                    # TODO, this might fail if base does not have it...
                    # But then again, it shouldn't be `copy`...
                    # copy it!
                    arg = getattr(base, attr).copy()

            if arg is not None and not isinstance(arg, str):
                setattr(cls, attr, arg)

        setattr(cls, f"{prefix}_when_subclassing", when_subclassing)
