# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Optional, Union

from ._dispatcher import ClassDispatcher

"""Internal class used for subclassing.

This class implements the

__init_subclass__ method to ensure that classes automatically
create the `to`/`new` methods.

Sometimes these can be required to be inherited as well.

The usage interface should look something like this:

class A(_Dispatchs,
        dispatchs=[
            "new", "hello"
        ]
)

A.new.register ..
A.hello.register ..
"""

_log = logging.getLogger(__name__)


class _Dispatchs:
    """Subclassable for creating the dispatch arguments"""

    def __init_subclass__(
        cls,
        /,
        dispatchs: Optional[Union[str, Sequence[Any]]] = None,
        when_subclassing: Optional[str] = None,
        **kwargs,
    ):
        # complete the init_subclass
        super().__init_subclass__(**kwargs)

        # Get the allowed actions for subclassing
        prefix = "_cls_dispatchs"
        allowed_subclassing = ("keep", "new", "copy")

        def find_base(cls, attr):
            # The order of execution, since the implementation search
            # is based on MRO, we should search in that order.
            for base in cls.__mro__:
                if hasattr(base, attr):
                    return base
            return None

        if dispatchs is None:
            # Copy dispatch names when subclassing.
            # I.e. we will search through all the previous ones
            # and copy them.
            dispatchs = []
            for base in cls.__mro__:
                if hasattr(base, f"{prefix}_dispatchs"):
                    dispatchs.extend(getattr(base, f"{prefix}_dispatchs"))

        elif not isinstance(dispatchs, (list, tuple)):
            dispatchs = [dispatchs]

        loop = []
        for attr in dispatchs:
            # argument could be:
            #  dispatchs = [
            #       ("new", "keep"),
            #       "to"
            #  ]
            obj = None
            if isinstance(attr, (list, tuple)):
                attr, obj = attr
            elif hasattr(attr, "_attr_name"):
                attr, obj = getattr(attr, "_attr_name"), attr

            if attr in cls.__dict__:
                raise ValueError(f"The attribute {attr} already exists on {cls!r}")

            base = find_base(cls, attr)
            if base is None:
                # this is likely the first instance of the class
                # So one cannot do anything but specifying stuff
                when_subcls = None
                if obj is None:
                    obj = ClassDispatcher(attr)
            else:
                when_subcls = getattr(base, f"{prefix}_when_subclassing")
                if obj is None:
                    obj = when_subcls

            if isinstance(obj, str):
                if obj == "new":
                    obj = ClassDispatcher(attr)
                elif obj == "keep":
                    obj = None
                elif obj == "copy":
                    obj = getattr(base, attr).copy()
            loop.append((attr, obj, when_subcls))

        if when_subclassing is None:
            # first non-None value
            when_subclassing = "copy"
            for _, _, when_subcls in loop:
                if when_subcls is not None:
                    when_subclassing = when_subcls
        _log.debug(f"{cls!r} when_subclassing = {when_subclassing}")

        if when_subclassing not in allowed_subclassing:
            raise ValueError(
                f"when_subclassing should be one of {allowed_subclassing}, got {when_subclassing}"
            )

        for attr, obj, _ in loop:
            if obj is None:
                _log.debug(f"Doing nothing for {attr} on class {cls!r}")
            else:
                _log.debug(f"Inserting {attr}={obj!r} onto class {cls!r}")
                setattr(cls, attr, obj)

        setattr(cls, f"{prefix}_when_subclassing", when_subclassing)
        setattr(cls, f"{prefix}_dispatchs", [attr for attr, _, _ in loop])
