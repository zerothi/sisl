# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This is a single dispatch method that works with class methods that have annotations.
# TODO when forward refs work with annontations
# from __future__ import annotations

from functools import singledispatchmethod as real_singledispatchmethod


class singledispatchmethod(real_singledispatchmethod):
    def register(self, cls, method=None):
        if hasattr(cls, "__func__"):
            setattr(cls, "__annotations__", cls.__func__.__annotations__)
        return self.dispatcher.register(cls, func=method)

    def __get__(self, obj, cls=None):
        _method = super().__get__(obj, cls)
        _method.dispatcher = self.dispatcher
        return _method
