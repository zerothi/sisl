# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import ast
import functools
import importlib
import inspect
import operator as op
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from math import pi
from numbers import Integral
from typing import Any, Union

__all__ = ["merge_instances", "str_spec", "direction", "listify", "angle"]
__all__ += ["iter_shape", "math_eval", "allow_kwargs"]
__all__ += ["import_attr", "lazy_import"]
__all__ += ["PropertyDict", "NotNonePropertyDict"]
__all__ += ["size_to_num", "size_to_elements"]


# supported operators
_operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
    ast.USub: op.neg,
}


def math_eval(expr: str) -> Any:
    """Evaluate a mathematical expression using a safe evaluation method

    Parameters
    ----------
    expr :
       the string to be evaluated using math

    Examples
    --------
    >>> math_eval("2^6")
    4
    >>> math_eval("2**6")
    64
    >>> math_eval("1 + 2*3**(4^5) / (6 + -7)")
    -5.0
    """
    return _eval(ast.parse(expr, mode="eval").body)


def _eval(node):
    if isinstance(node, ast.Constant):  # <number>
        return node.value
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        return _operators[type(node.op)](_eval(node.left), _eval(node.right))
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return _operators[type(node.op)](_eval(node.operand))
    else:
        raise TypeError(node)


def merge_instances(*args, **kwargs):
    """Merges an arbitrary number of instances together.

    Parameters
    ----------
    *args : obj
       all objects dictionaries gets appended to a new class
       which is returned.
    name : str, optional
       name of class to merge, default to ``"MergedClass"``
    """
    name = kwargs.get("name", "MergedClass")
    # We must make a new-type class
    cls = type(name, (object,), {})
    # Create holder of class
    # We could have
    m = cls()
    for arg in args:
        m.__dict__.update(arg.__dict__)
    return m


def size_to_num(size: Union[int, float, str], unit: str = "MB") -> float:
    """Convert a size-specification to a size in a specific `unit`

    Converts the input value (`size`) into a number corresponding to
    the `size` converted to the specified `unit`.

    If `size` is passed as an integer (or float), it will be interpreted
    as a size in MB. Otherwise the string may contain the size specification.
    """
    if isinstance(size, (float, int)):
        return size

    # Now parse the size from a string
    match = re.match(r"(\d+[.\d]*)(\D*)", size)
    size = float(match[1].strip())
    unit_in = match[2].strip()

    # Now parse things
    # We expect data to be in MB (default unit)
    # Then we can always convert
    conv = {
        "b": 1 / (1024 * 1024),
        "B": 1 / (1024 * 1024),
        "k": 1 / 1024,
        "kb": 1 / 1024,
        "kB": 1 / 1024,
        "mb": 1,
        "M": 1,
        "MB": 1,
        "G": 1024,
        "GB": 1024,
        "T": 1024 * 1024,
        "TB": 1024 * 1024,
    }

    if unit_in:
        unit_in = conv[unit_in]
    else:
        unit_in = 1

    # Convert the requested unit
    unit = conv[unit]

    return size * unit_in / unit


def size_to_elements(size: Union[int, str], byte_per_elem: int = 8) -> int:
    """Calculate the number of elements such that they occupy `size` memory

    Parameters
    ----------
    size :
        a size specification, either by an integer, or a str. If an integer it is
        assumed to be in MB, otherwise the str can hold a unit specification.
    byte_per_elem
        number of bytes per element when doing the conversion
    """
    size = size_to_num(size, unit="B")

    return int(size // byte_per_elem)


def iter_shape(shape):
    """Generator for iterating a shape by returning consecutive slices

    Parameters
    ----------
    shape : array_like
      the shape of the iterator

    Yields
    ------
    tuple of int
       a tuple of the same length as the input shape. The iterator
       is using the C-indexing.

    Examples
    --------
    >>> for slc in iter_shape([2, 1, 3]):
    ...    print(slc)
    [0, 0, 0]
    [0, 0, 1]
    [0, 0, 2]
    [1, 0, 0]
    [1, 0, 1]
    [1, 0, 2]
    """
    shape1 = [i - 1 for i in shape]
    ns = len(shape)
    ns1 = ns - 1
    # Create list for iterating
    # we require a list because tuple's are immutable
    slc = [0] * ns

    while slc[0] < shape[0]:
        for i in range(shape[ns1]):
            slc[ns1] = i
            yield slc

        # Increment the previous shape indices
        for i in range(ns1, 0, -1):
            if slc[i] >= shape1[i]:
                slc[i] = 0
                if i > 0:
                    slc[i - 1] += 1


def str_spec(name) -> Tuple[str, Union[None, str]]:
    """Split into a tuple of name and specifier, delimited by ``{...}``.

    Parameters
    ----------
    name : str
       string to split

    Returns
    -------
    tuple of str
       returns the name and the specifier (without delimiter) in a tuple

    Examples
    --------
    >>> str_spec("hello")
    ("hello", None)
    >>> str_spec("hello{TEST}")
    ("hello", "TEST")
    """
    if not name.endswith("}"):
        return name, None

    lname = name[:-1].split("{")
    return "{".join(lname[:-1]), lname[-1]


IterableAny = Iterator[Any]
IterableInstantiation = Callable[..., IterableAny]


class Listify:
    """Convert arguments to an iterable-like (any iterable, default to `list`)

    It provides also an easy mechanism for "piping" content to
    a function call (for better readability).

    Parameters
    ----------
    cls:
        the default list-like object to cast it to

    Examples
    --------
    >>> listify = Listify()
    >>> 1 | listify
    [1]
    >>> Listify(tuple)(1)
    (1,)

    It can greatly improve readability with `map` constructs:
    >>> map(lambda x, [1]) | listify
    [1]

    Notes
    -----
    If using this to convert to `tuple` instances ``Listify(tuple)``,
    please do note the problems of using a tuple as indices for
    `numpy.ndarray` objects.

    This is partly inspired by `pip <https://pypi.org/project/pipe/>`_.
    """

    __slots__ = ("_cls",)

    # Ensures that numpy function calls won't happen!
    # Just a higher priority than *any* numpy arrays and subclasses.
    __array_priority__ = 1000000

    def __init__(self, cls: IterableInstantiation = list):
        self._cls = cls

    def __call__(
        self, arg: Any, cls: Optional[IterableInstantiation] = None
    ) -> IterableAny:
        if cls is None:
            cls = self._cls
        if isinstance(arg, Iterable):
            if isinstance(arg, cls):
                return arg
            return cls(arg)
        return cls([arg])

    def __ror__(self, arg: Any) -> IterableAny:
        """Allow piping of function calls (on the right side)"""
        return self(arg)


listify = Listify()


# Transform a string to a Cartesian direction
def direction(d: Union[int, str], abc=None, xyz=None) -> Union[int, Any]:
    """Index coordinate transformation from int/str to an integer

    Parameters
    ----------
    d : {0, "x", "a", 1, "y", "b", 2, "z", "c"}
       returns the integer that corresponds to the coordinate index (strings are
       lower-cased).
    abc : (3, 3), optional
       for ``"abc"`` inputs the returned value will be the vector ``abc[direction(d)]``

    Returns
    -------
    index : int
       index of the Cartesian coordinate system, only if both `abc` and `xyz` are none
       or if the requested direction is not present, only returned if the corresponding direction
       is none
    vector : (3,)
       the vector corresponding to the value gotten from `abc` or `xyz`, only returned
       if the corresponding direction is not none

    Examples
    --------
    >>> direction(0)
    0
    >>> direction("Y")
    1
    >>> direction("z")
    2
    >>> direction("2")
    2
    >>> direction(" 2")
    2
    >>> direction("b")
    1
    >>> direction("b", abc=np.diag([1, 2, 3])
    [0, 2, 0]
    >>> direction("x", abc=np.diag([1, 2, 3])
    0
    >>> direction(1, abc=np.diag([1, 2, 3])
    [0, 2, 0]
    >>> direction(1, abc=np.diag([1, 2, 3], xyz=np.diag([4, 5, 6])
    [0, 2, 0]
    """
    if isinstance(d, Integral):
        # pass through to find it
        d = str(d)

    # We take it as a string
    d = d.lower().strip()

    if not abc is None and d in "abc012":
        return abc["a0b1c2".index(d) // 2]
    elif not xyz is None and d in "xyz012":
        return xyz["x0y1z2".index(d) // 2]
    else:
        if d in ("x", "y", "z", "a", "b", "c", "0", "1", "2"):
            return "xa0yb1zc2".index(d) // 3
    raise ValueError(
        "direction: Input direction is not an integer, nor a string in 'xyz/abc/012'"
    )


# Transform an input to an angle
def angle(s: str, rad: bool = True, in_rad: bool = True) -> float:
    """Convert the input string to an angle, either radians or degrees.

    Parameters
    ----------
    s :
       If `s` starts with 'r' it is interpreted as radians ``[0:2pi]``.
       If `s` starts with 'a' it is interpreted as a regular angle ``[0:360]``.
       If `s` ends with 'r' it returns in radians.
       If `s` ends with 'a' it returns in regular angle.

       `s` may be any mathematical equation which can be
       intercepted through ``eval``.
    rad :
       Whether the returned angle is in radians.
       Note than an 'r' at the end of `s` has precedence.
    in_rad :
       Whether the calculated angle is in radians.
       Note than an 'r' at the beginning of `s` has precedence.

    Returns
    -------
    float
       the angle in the requested unit
    """
    s = s.lower()

    if s.startswith("r"):
        in_rad = True
    elif s.startswith("a"):
        in_rad = False
    if s.endswith("r"):
        rad = True
    elif s.endswith("a"):
        rad = False

    # Remove all r/a's and remove white-space
    s = s.replace("r", "").replace("a", "").replace(" ", "")

    # Figure out if Pi is circumfered by */+-
    spi = s.split("pi")
    nspi = len(spi)
    if nspi > 1:
        # We have pi at least in one place.
        for i, si in enumerate(spi):
            # In case the last element is a pi
            if len(si) == 0:
                continue
            if i < nspi - 1:
                if not si.endswith(("*", "/", "+", "-")):
                    # it *MUST* be "*"
                    spi[i] = spi[i] + "*"
            if 0 < i:
                if not si.startswith(("*", "/", "+", "-")):
                    # it *MUST* be "*"
                    spi[i] = "*" + spi[i]

        # Now insert Pi dependent on the input type
        if in_rad:
            Pi = pi
        else:
            Pi = 180.0

        s = (f"{Pi}").join(spi)

    # We have now transformed all values
    # to the correct numerical values and we calculate
    # the expression
    ra = math_eval(s)
    if rad and not in_rad:
        return ra / 180.0 * pi
    if not rad and in_rad:
        return ra / pi * 180.0

    # Both radians and in_radians are equivalent
    # so return as-is
    return ra


def allow_kwargs(*args):
    """Decoractor for forcing `func` to have the named arguments as listed in `args`

    This decorator merely removes any keyword argument from the called function
    which is in the list of `args` in case the function does not have the arguments
    or a ``**kwargs`` equivalent.

    Parameters
    ----------
    *args : str
       required arguments in `func`, if already present nothing will be done.
    """

    def deco(func):
        if func is None:
            return None

        # Build list of arguments and keyword arguments
        sig = inspect.signature(func)
        arg_names = []
        kwargs_name = None
        for name, p in sig.parameters.items():
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                arg_names.append(name)
            elif p.kind == p.VAR_KEYWORD:
                kwargs_name = name

        if not kwargs_name is None:
            return func

        # First we figure out which arguments are already in the lists
        args_ = [arg for arg in args if not arg in arg_names]

        # Now we have the compressed lists
        # If there are no arguments required to be added, simply return the function
        if len(args_) == 0:
            return func

        # Basically any function that does not have a named argument
        # cannot use it. So we simply need to create a function which by-passes
        # the named arguments.
        @functools.wraps(func)
        def dec_func(*args, **kwargs):
            # Simply remove all the arguments that cannot be passed to the function
            for arg in args_:
                del kwargs[arg]
            return func(*args, **kwargs)

        return dec_func

    return deco


def import_attr(attr_path):
    """Returns an attribute from a full module path

    Examples
    --------
    >>> func = import_attr("sisl.utils.import_attr")
    >>> assert func is import_attr

    Parameters
    -----------
    attr_path: str
        the module path to the attribute
    """
    module, variable = attr_path.rsplit(".", 1)

    module = importlib.import_module(module)
    return getattr(module, variable)


def lazy_import(name, package=None):
    """Lazily import a module or submodule

    Parameters
    ----------
    name : str
       module name to load, optionally a sub-package from `package`
    package : str, optional
       whether `name` is a sub-package

    Examples
    --------
    >>> hello = lazy_import("hello")
    >>> hello.mod_func

    This will only load the module upon method inspection in
    the module.

    >>> hello = lazy_import(".hello", "package")
    >>> hello.mod_func

    The dot is required to indicate it being a name within package.

    NOTE currently this is not working due to id's changing upon
    actual loading. I have yet to figure out why...
    """
    util = importlib.util

    abs_name = util.resolve_name(name, package)
    if abs_name in sys.modules:
        return sys.modules[abs_name]

    # Create module specification
    # Find specifications for module
    spec = util.find_spec(abs_name)
    module = util.module_from_spec(spec)

    # Make module with proper locking and get it inserted into sys.modules.
    util.LazyLoader(spec.loader).exec_module(module)

    return module


# This class is very much like the addict type
# However, a much reduced usage


class PropertyDict(dict):
    """Simple dictionary which may access items as properties as well"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return list(self.keys())


class NotNonePropertyDict(PropertyDict):
    def __setitem__(self, key, value):
        if value is None:
            return
        super().__setitem__(key, value)
