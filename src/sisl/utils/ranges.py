# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import re
from itertools import groupby

__all__ = ["strmap", "strseq", "lstranges", "erange", "list2str", "fileindex"]


# Function to change a string to a range of integers
def strmap(func, s, start=None, end=None, sep="b"):
    """Parse a string as though it was a slice and map all entries using ``func``.

    Parameters
    ----------
    func : function
       function to parse every match with
    s    : str
       the string that should be parsed
    start  : optional
       the replacement in case the LHS of the delimiter is not present
    end  : optional
       the replacement in case the RHS of the delimiter is not present
    sep  : {"b", "c"}
       separator used, ``"b"`` is square brackets, ``"c"``, curly braces

    Examples
    --------
    >>> strmap(int, "1")
    [1]
    >>> strmap(int, "1-2")
    [(1, 2)]
    >>> strmap(int, "-2", end=4)
    [2]
    >>> strmap(int, "1-")
    [(1, None)]
    >>> strmap(int, "1-", end=4)
    [(1, 4)]
    >>> strmap(int, "1-10[2-3]")
    [((1, 10), [(2, 3)])]
    """
    if sep in ("b", "[", "]"):
        segment = re.compile(r"\[(.+)\]\[(.+)\]|(.+)\[(.+)\]|(.+)")
        sep1, sep2 = "[", "]"
    elif sep in ("c", "{", "}"):
        segment = re.compile(r"\{(.+)\}\{(.+)\}|(.+)\{(.+)\}|(.+)")
        sep1, sep2 = "{", "}"
    else:
        raise ValueError("strmap: unknown separator for the sequence")

    # Create list
    s = s.replace(" ", "")
    if len(s) == 0:
        return [None]
    elif s in ["-", ":"]:
        return [(start, end)]

    commas = s.split(",")

    # Collect all the comma separated quantities that
    # may be selected by [..,..]
    i = 0
    while i < len(commas) - 1:
        if commas[i].count(sep1) == commas[i].count(sep2):
            i = i + 1
        else:
            # there must be more [ than ]
            commas[i] = commas[i] + "," + commas[i + 1]
            del commas[i + 1]

    # Check the last input...
    i = len(commas) - 1
    if commas[i].count(sep1) != commas[i].count(sep2):
        raise ValueError(f"Unbalanced string: not enough {sep1} and {sep2}")

    # Now we have a comma-separated list
    # with collected brackets.
    l = []
    for seg in commas:
        # Split it in groups of reg-exps
        m = segment.findall(seg)[0]

        if len(m[0]) > 0:
            # this is: [..][..]
            rhs = strmap(func, m[1], start, end, sep)
            for el in strmap(func, m[0], start, end, sep):
                l.append((el, rhs))

        elif len(m[2]) > 0:
            # this is: ..[..]
            l.append(
                (strseq(func, m[2], start, end), strmap(func, m[3], start, end, sep))
            )

        elif len(m[4]) > 0:
            l.append(strseq(func, m[4], start, end))

    return l


def strseq(cast, s, start=None, end=None):
    """Accept a string and return the casted tuples of content based on ranges.

    Parameters
    ----------
    cast : function
       parser of the individual elements
    s : str
       string with content

    Examples
    --------
    >>> strseq(int, "3")
    3
    >>> strseq(int, "3-6")
    (3, 6)
    >>> strseq(int, "-2")
    -2
    >>> strseq(int, "-2", end=7)
    5
    >>> strseq(int, "-2:2", end=7)
    (-2, 2)
    >>> strseq(int, "-2-2", end=7)
    (-2, 2)
    >>> strseq(int, "3-")
    (3, None)
    >>> strseq(int, "3:2:7")
    (3, 2, 7)
    >>> strseq(int, "3:2:", end=8)
    (3, 2, 8)
    >>> strseq(int, ":2:", start=2)
    (2, 2, None)
    >>> strseq(float, "3.2:6.3")
    (3.2, 6.3)
    """
    if ":" in s:
        s = [ss.strip() for ss in s.split(":")]
    elif s.startswith("-") and len(s) > 1:
        if s.count("-") > 1:
            s = [ss.strip() for ss in s[1:].split("-")]
            s[0] = f"-{s[0]}"
    elif "-" in s:
        s = [ss.strip() for ss in s.split("-")]
    if isinstance(s, list):
        if len(s[0]) == 0:
            s[0] = start
        if len(s[-1]) == 0:
            s[-1] = end
        return tuple(cast(ss) if ss is not None else None for ss in s)
    ret = cast(s)
    if ret < 0 and end is not None:
        return ret + end
    return ret


def erange(start, step, end=None):
    """Returns the range with both ends includede"""
    if end is None:
        return range(start, step + 1)
    return range(start, end + 1, step)


def lstranges(lst, cast=erange, end=None):
    """Convert a `strmap` list into expanded ranges"""
    l = []
    # If an entry is a tuple, it means it is either
    # a range 0-1 == tuple(0, 1), or
    # a sub-range
    #   0[0-1], 0-1[0-1]
    if isinstance(lst, tuple):
        if len(lst) == 3:
            l.extend(cast(*lst))
        else:
            head = lstranges(lst[0], cast, end)
            bot = lstranges(lst[1], cast, end)
            if isinstance(head, list):
                for el in head:
                    l.append([el, bot])
            elif isinstance(bot, list):
                l.append([head, bot])
            else:
                l.extend(cast(head, bot))

    elif isinstance(lst, list):
        for lt in lst:
            ls = lstranges(lt, cast, end)
            if isinstance(ls, list):
                l.extend(ls)
            else:
                l.append(ls)
    else:
        if lst is None and end is not None:
            return cast(0, end)
        return lst
    return l


def list2str(lst):
    """Convert a list of elements into a string of ranges

    Examples
    --------
    >>> list2str([2, 4, 5, 6])
    "2, 4-6"
    >>> list2str([2, 4, 5, 6, 8, 9])
    "2, 4-6, 8-9"
    """
    lst = lst[:]
    lst.sort()
    # Create positions
    pos = [j - i for i, j in enumerate(lst)]
    t = 0
    rng = ""
    for _, els in groupby(pos):
        ln = len(list(els))
        el = lst[t]
        if t > 0:
            rng += ", "
        t += ln
        if ln == 1:
            rng += str(el)
        # elif ln == 2:
        #    rng += "{}, {}".format(str(el), str(el+ln-1))
        else:
            rng += "{}-{}".format(el, el + ln - 1)
    return rng


# Function to retrieve an optional index from the
# filename
#   file[0] returns:
#     file, 0
#   file returns:
#     file, None
#   file[0-1] returns
#     file, [0,1]
def fileindex(f, cast=int):
    """Parses a filename string into the filename and the indices.

    This range can be formatted like this:
      file[1,2,3-6]
    in which case it will return:
      file, [1,2,3,4,5,6]

    Parameters
    ----------
    f : str
       filename to parse
    cast : function
       the function to cast the bracketed value

    Examples
    --------
    >>> fileindex("Hello[0]")
    ("Hello", 0)
    >>> fileindex("Hello[0-2]")
    ("Hello", [0, 1, 2])
    """

    if "[" not in f:
        return f, None

    # Grab the filename
    f = f.split("[")
    fname = f.pop(0)
    # Re-join and remove the last "]"
    f = "[".join(f)
    if f[-1] == "]":
        f = f[:-1]
    ranges = strmap(cast, f)
    rng = lstranges(ranges)
    if len(rng) == 1:
        return fname, rng[0]
    return fname, rng
