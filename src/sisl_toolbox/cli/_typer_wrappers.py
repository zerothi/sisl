from typing_extensions import Annotated

import inspect
from copy import copy

import typer

from ._cli_arguments import CLIArgument, CLIOption, get_params_help

# This dictionary keeps the kwargs that should be passed to typer arguments/options
# for a given type. This is for example to be used for types that typer does not
# have built in support for.
_CUSTOM_TYPE_KWARGS = {}

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

        try:
            argument_kwargs = _CUSTOM_TYPE_KWARGS.get(param.annotation, {})
        except:
            argument_kwargs = {}

        typer_arg_cls = typer.Argument if param.default == inspect.Parameter.empty else typer.Option
        if hasattr(param.annotation, "__metadata__"):
            for meta in param.annotation.__metadata__:
                if isinstance(meta, CLIArgument):
                    typer_arg_cls = typer.Argument
                    argument_kwargs.update(meta.kwargs)
                elif isinstance(meta, CLIOption):
                    typer_arg_cls = typer.Option
                    argument_kwargs.update(meta.kwargs)
                else:
                    continue
                break

        if "param_decls" in argument_kwargs:
            argument_args = argument_kwargs.pop("param_decls")
        else:
            argument_args = []

        new_parameters.append(
            param.replace(annotation=Annotated[param.annotation, typer_arg_cls(*argument_args, help=params_help.get(param.name), **argument_kwargs)])
        )

    # Create a copy of the function and update it with the modified signature.
    # Also remove parameters documentation from the docstring.
    annotated_func = copy(func)

    annotated_func.__signature__ = sig.replace(parameters=new_parameters)
    annotated_func.__doc__ = func.__doc__[:func.__doc__.find("Parameters\n")]

    return annotated_func