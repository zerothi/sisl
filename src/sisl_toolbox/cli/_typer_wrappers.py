import typing
from typing_extensions import Annotated

from enum import Enum

import inspect
from copy import copy
import yaml

import typer

from ._cli_arguments import CLIArgument, CLIOption, get_params_help

def get_dict_param_kwargs(dict_annotation_args):

    def yaml_dict(d: str):

        if isinstance(d, dict):
            return d
        
        return yaml.safe_load(d)
    
    argument_kwargs = {"parser": yaml_dict}

    if len(dict_annotation_args) == 2:
        try:
            argument_kwargs["metavar"] = f"YAML_DICT[{dict_annotation_args[0].__name__}: {dict_annotation_args[1].__name__}]"
        except:
            argument_kwargs["metavar"] = f"YAML_DICT[{dict_annotation_args[0]}: {dict_annotation_args[1]}]"
    
    return argument_kwargs

# This dictionary keeps the kwargs that should be passed to typer arguments/options
# for a given type. This is for example to be used for types that typer does not
# have built in support for.
_CUSTOM_TYPE_KWARGS = {
    dict: get_dict_param_kwargs,
}

def _get_custom_type_kwargs(type_):

    if hasattr(type_, "__metadata__"):
        type_ = type_.__origin__

    if typing.get_origin(type_) is not None:
        args = typing.get_args(type_)
        type_ = typing.get_origin(type_)
    else:
        args = ()

    try:
        argument_kwargs = _CUSTOM_TYPE_KWARGS.get(type_, {})
        if callable(argument_kwargs):
            argument_kwargs = argument_kwargs(args)
    except:
        argument_kwargs = {}

    return argument_kwargs


def annotate_typer(func):
    """Annotates a function for a typer app.
    
    It returns a new function, the original function is not modified.
    """
    # Get the help message for all parameters found at the docstring
    params_help = get_params_help(func)

    # Get the original signature of the function
    sig = inspect.signature(func)

    # Loop over parameters in the signature, modifying them to include the
    # typer info.
    new_parameters = []
    for param in sig.parameters.values():

        argument_kwargs = _get_custom_type_kwargs(param.annotation)

        default = param.default
        if isinstance(param.default, Enum):
            default = default.value

        typer_arg_cls = typer.Argument if param.default == inspect.Parameter.empty else typer.Option
        if hasattr(param.annotation, "__metadata__"):
            for meta in param.annotation.__metadata__:
                if isinstance(meta, CLIArgument):
                    typer_arg_cls = typer.Argument
                    argument_kwargs.update(meta.kwargs)
                elif isinstance(meta, CLIOption):
                    typer_arg_cls = typer.Option
                    argument_kwargs.update(meta.kwargs)

        if "param_decls" in argument_kwargs:
            argument_args = argument_kwargs.pop("param_decls")
        else:
            argument_args = []

        new_parameters.append(
            param.replace(
                default=default,
                annotation=Annotated[param.annotation, typer_arg_cls(*argument_args, help=params_help.get(param.name), **argument_kwargs)]
            )
        )

    # Create a copy of the function and update it with the modified signature.
    # Also remove parameters documentation from the docstring.
    annotated_func = copy(func)

    annotated_func.__signature__ = sig.replace(parameters=new_parameters)
    annotated_func.__doc__ = func.__doc__[:func.__doc__.find("Parameters\n")]

    return annotated_func