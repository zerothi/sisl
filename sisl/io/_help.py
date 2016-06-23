""" Helper functions for IO reading
"""
from __future__ import print_function, division
import sys


# Local function for extending the broadcasted functions
def extendall(tbl, mod):
    tbl.extend(sys.modules[mod].__dict__['__all__'])


def starts_with_list(l, comments):
    for comment in comments:
        if l.startswith(comment):
            return True
    return False
