# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from sisl._dispatcher import AbstractDispatch, ClassDispatcher


class ModelDispatcher(ClassDispatcher):
    """Container for dispatch models"""

    pass


class BaseModel:
    """Base class used for inheritance for creating separate models"""

    ref = ModelDispatcher(
        "ref",
        type_dispatcher=None,
        obj_getattr=lambda obj, key: (_ for _ in ()).throw(
            AttributeError(
                (
                    f"{obj}.to does not implement '{key}' "
                    f"dispatcher, are you using it incorrectly?"
                )
            )
        ),
    )

    def __getattr__(self, attr):
        return getattr(self.ref, attr)


# Each model should inherit from this


class ReferenceDispatch(AbstractDispatch):
    """Base dispatcher that implemnets different models"""

    pass
