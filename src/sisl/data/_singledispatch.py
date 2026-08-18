# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This is a single dispatch method that works with class methods that have annotations.

from functools import singledispatchmethod as real_singledispatchmethod


class singledispatchmethod(real_singledispatchmethod):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.names = ["base"]

    def register(self, name: str, cls, method=None):
        if hasattr(cls, "__func__"):
            setattr(cls, "__annotations__", cls.__func__.__annotations__)

        self.names.append(name)

        ret = self.dispatcher.register(cls, func=method)
        return ret

    def __get__(self, obj, cls=None):
        _method = super().__get__(obj, cls)
        _method.dispatcher = self.dispatcher
        _method.names = self.names
        return _method
