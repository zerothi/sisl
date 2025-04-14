# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import argparse
import inspect
import typing
from typing import Any, Callable, Optional, Union

from sisl._lib._docscrape import FunctionDoc

try:
    from rich_argparse import RawTextRichHelpFormatter

    SislHelpFormatter = RawTextRichHelpFormatter
except ImportError:
    SislHelpFormatter = argparse.RawDescriptionHelpFormatter


def is_optional(field):
    """Check whether the annotation for a parameter is an Optional type."""
    return typing.get_origin(field) is Union and type(None) in typing.get_args(field)


def get_optional_arg(field):
    """Get the type of an optional argument from the typehint.

    It only works if the annotation only has one type.

    E.g.: Optional[int] -> int
    E.g.: Optional[Union[int, str]] -> raises ValueError
    """
    if not is_optional(field):
        return field

    args = typing.get_args(field)
    if len(args) > 2:
        raise ValueError("Optional type must have at most 2 arguments")
    for arg in args:
        if arg is not type(None):
            return arg

    raise ValueError("No non-None type found in Union")


class NotPassedArg:
    """Placeholder to use for arguments that have not been passed.

    By setting this as the default value for an argument, we can
    later check if the argument was passed through the CLI or not.
    """

    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return repr(self.val)

    def __str__(self):
        return str(self.val)

    def __eq__(self, other):
        return self.val == other

    def __getattr__(self, name):
        return getattr(self.val, name)


def get_runner(func):
    """Wraps a function to receive the args parsed from argparse"""

    def _runner(args):

        # Get the config argument. If present, load the arguments
        # from the config (yaml) file.
        config = getattr(args, "config", None)
        config_args = {}
        if config is not None:
            import yaml

            with open(args.config, "r") as f:
                config_args = yaml.safe_load(f)

        # Build the final arguments dictionary, using the arguments of the
        # config file as defaults.
        final_args = {}
        for k, v in vars(args).items():
            if k in ("runner", "config"):
                continue
            elif isinstance(v, NotPassedArg):
                final_args[k] = config_args.get(k, v.val)
            else:
                final_args[k] = v

        # And call the function
        return func(**final_args)

    return _runner


def get_argparse_parser(
    func: Callable,
    name: Optional[str] = None,
    subp=None,
    parser_kwargs: dict[str, Any] = {},
    arg_aliases: dict[str, str] = {},
    defaults: dict[str, Any] = {},
    add_config: bool = True,
) -> argparse.ArgumentParser:
    """Creates an argument parser from a function's signature and docstring.

    The created argument parser just mimics the function. It is a CLI version
    of the function.

    Parameters
    ----------
    func :
        The function to create the parser for.
    name :
        The name of the parser. If None, the function's name is used.
    subp :
        The subparser to add the parser to. If None, a new isolated
        parser is created.
    parser_kwargs :
        Additional arguments to pass to the parser.
    arg_aliases :
        Dictionary holding aliases (shortcuts) for the arguments. The keys
        of this dictionary are the argument names, and the values are the
        aliases. For example, if the function has an argument called
        `--size`, and you want to add a shortcut `-s`, you can pass
        `arg_aliases={"size": "s"}`.
    defaults :
        Dictionary holding default values for the arguments. The keys
        of this dictionary are the argument names, and the values are
        the default values. The defaults are taken from the function's
        signature if not specified.
    add_config :
        If True, adds a `--config` argument to the parser. This
        argument accepts a path to a YAML file containing the
        arguments for the function.
    """

    # Check if the function needs to be added as a subparser
    is_sub = not subp is None

    # Get the function's information
    fdoc = FunctionDoc(func)
    signature = inspect.signature(func)

    # Initialize parser
    title = "".join(fdoc["Summary"])
    if is_sub:
        p = subp.add_parser(
            name or func.__name__.replace("_", "-"),
            description=title,
            help=title,
            **parser_kwargs,
        )
    else:
        p = argparse.ArgumentParser(title, **parser_kwargs)

    # Add the config argument to load the arguments from a YAML file
    if add_config:
        p.add_argument(
            "--config",
            "-c",
            type=str,
            default=None,
            help="Path to a YAML file containing the arguments for the command",
        )

    group = p.add_argument_group("Function arguments")

    # Add all the function's parameters to the parser
    parameters_help = {p.name: p.desc for p in fdoc["Parameters"]}
    for param in signature.parameters.values():

        arg_names = [f"--{param.name.replace('_', '-')}"]
        if param.name in arg_aliases:
            arg_names.append(f"-{arg_aliases[param.name]}")

        annotation = param.annotation
        if is_optional(annotation):
            annotation = get_optional_arg(annotation)

        group.add_argument(
            *arg_names,
            type=annotation,
            default=NotPassedArg(param.default),
            action=argparse.BooleanOptionalAction if annotation is bool else None,
            required=param.default is inspect._empty,
            help="\n".join(parameters_help.get(param.name, [])),
        )

    if is_sub:
        defaults = {"runner": get_runner(func), **defaults}

    p.set_defaults(**defaults)

    return p
