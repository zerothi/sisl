# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from typing import Literal

__all__ = ["GHFormat", "GHLink"]


class GHFormat:

    def __init__(self, type: Literal["issues", "discussions", "pull"]):
        self._type = type

    def is_valid(self, number: str) -> bool:
        try:
            return int(number) > 0
        except ValueError:
            return False

    def format(self, number: str) -> str:
        if not self.is_valid(number):
            return ""  # no formatting, explicitly requesting no output

        prefix = {
            "issues": "GH",
            "pull": "PR",
            "discussions": "D",
        }[self._type]

        return f"{prefix}{number}"

    def __mod__(self, number: str) -> str:
        """To enable *old-style* formatting."""
        return self.format(number)

    def __eq__(self, other) -> bool:
        return self._type == other._type

    def __hash__(self) -> int:
        return hash(self._type)

    def __setstate__(self, d):
        self.__init__(d["type"])

    def __getstate__(self):
        return {"type": self._type}


class GHLink(GHFormat):
    url = "https://github.com/zerothi/sisl"

    def format(self, number: str) -> str:
        if not self.is_valid(number):
            return f"{self.url}/{self._type}"  # no formatting, explicitly requesting no output

        return f"{self.url}/{self._type}/{number}"

    def __hash__(self) -> int:
        return hash((self.url, self._type))
