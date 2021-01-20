import sys
import inspect
import functools
import ast
import operator as op
from numbers import Integral
from math import pi
import importlib


__all__ = ["merge_instances", "str_spec", "direction", "angle"]
__all__ += ["iter_shape", "math_eval", "allow_kwargs"]
__all__ += ["import_attr", "lazy_import"]
__all__ += ["PropertyDict"]


# supported operators
_operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
              ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
              ast.USub: op.neg}


def math_eval(expr):
    """ Evaluate a mathematical expression using a safe evaluation method

    Parameters
    ----------
    expr : str
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
    if isinstance(node, ast.Num): # <number>
        return node.n
    elif isinstance(node, ast.BinOp): # <left> <operator> <right>
        return _operators[type(node.op)](_eval(node.left), _eval(node.right))
    elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
        return _operators[type(node.op)](_eval(node.operand))
    else:
        raise TypeError(node)


def merge_instances(*args, **kwargs):
    """ Merges an arbitrary number of instances together.

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


def iter_shape(shape):
    """ Generator for iterating a shape by returning consecutive slices

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
    shape1 = [i-1 for i in shape]
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
                    slc[i-1] += 1


def str_spec(name):
    """ Split into a tuple of name and specifier, delimited by ``{...}``.

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


# Transform a string to a Cartesian direction
def direction(d):
    """ Index coordinate corresponding to the Cartesian coordinate system.

    Parameters
    ----------
    d : {0, "x", "a", 1, "y", "b", 2, "z", "c"}
       returns the integer that corresponds to the coordinate index (strings are
       lower-cased).

    Returns
    -------
    int
       The index of the Cartesian coordinate system.

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
    """
    if isinstance(d, Integral):
        # pass through to find it
        d = str(d)

    # We take it as a string
    d = d.lower().strip()
    # We must use an array to not allow 'xy' input
    if d in ("x", "y", "z", "a", "b", "c", "0", "1", "2"):
        return "xa0yb1zc2".index(d) // 3

    raise ValueError("direction: Input direction is not an integer, nor a string in 'xyz/abc/012'")


# Transform an input to an angle
def angle(s, rad=True, in_rad=True):
    """ Convert the input string to an angle, either radians or degrees.

    Parameters
    ----------
    s : str
       If `s` starts with 'r' it is interpreted as radians ``[0:2pi]``.
       If `s` starts with 'a' it is interpreted as a regular angle ``[0:360]``.
       If `s` ends with 'r' it returns in radians.
       If `s` ends with 'a' it returns in regular angle.

       `s` may be any mathematical equation which can be
       intercepted through ``eval``.
    rad : bool, optional
       Whether the returned angle is in radians.
       Note than an 'r' at the end of `s` has precedence.
    in_rad : bool, optional
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
            Pi = 180.

        s = (f"{Pi}").join(spi)

    # We have now transformed all values
    # to the correct numerical values and we calculate
    # the expression
    ra = math_eval(s)
    if rad and not in_rad:
        return ra / 180. * pi
    if not rad and in_rad:
        return ra / pi * 180.

    # Both radians and in_radians are equivalent
    # so return as-is
    return ra


def allow_kwargs(*args):
    """ Decoractor for forcing `func` to have the named arguments as listed in `args`

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
            if p.kind == p.POSITIONAL_ONLY or p.kind == p.POSITIONAL_OR_KEYWORD \
               or p.kind == p.KEYWORD_ONLY:
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
    """ Returns an attribute from a full module path

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
    """ Lazily import a module or submodule

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


class PropertyDict(dict):
    """ Simple dictionary which may access items as properties as well """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return list(self.keys())
