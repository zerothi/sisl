""" Helper functions for IO reading
"""
from __future__ import print_function, division


def starts_with_list(l, comments):
    for comment in comments:
        if l.startswith(comment):
            return True
    return False
