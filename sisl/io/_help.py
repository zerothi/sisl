from __future__ import print_function, division

__all__ = ['starts_with_list']


def starts_with_list(l, comments):
    for comment in comments:
        if l.startswith(comment):
            return True
    return False
