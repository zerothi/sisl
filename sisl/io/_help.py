# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from re import compile as re_compile


__all__ = ["starts_with_list", "header_to_dict"]


def starts_with_list(l, comments):
    for comment in comments:
        if l.strip().startswith(comment):
            return True
    return False


def header_to_dict(header):
    """ Convert a header line with 'key=val key1=val1' sequences to a single dictionary """
    e = re_compile(r"(\S+)=")

    # 1. Remove *any* entry with 0 length
    # 2. Ensure it is a list
    # 3. Reverse the list order (for popping)
    kv = list(filter(lambda x: len(x.strip()) > 0, e.split(header)))[::-1]

    # Now create the dictionary
    d = {}
    while len(kv) >= 2:
        # We have reversed the list
        key = kv.pop().strip(' =') # remove white-space *and* =
        val = kv.pop().strip() # remove outer whitespace
        d[key] = val

    return d
