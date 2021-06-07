# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
__all__ = ['starts_with_list']


def starts_with_list(l, comments):
    for comment in comments:
        if l.strip().startswith(comment):
            return True
    return False
