from __future__ import annotations
from dataclasses import is_dataclass
from collections.abc import Sequence as collections_Sequence
import inspect
import typing

from .. import types
from .. import input_fields as fields

_TYPE_TO_FIELD_MAPPING: typing.Dict[type, type[fields.InputField]] = {
    bool: fields.BoolInput,
    int: fields.IntegerInput,
    float: fields.FloatInput,
    str: fields.TextInput,
    list: fields.ListInput,
    dict: fields.DictInput,
    
    typing.Literal: fields.OptionsInput,
    typing.Sequence: fields.ListInput,

    # Plotting related input fields
    types.Color: fields.ColorInput,

    # Scientifically meaningful input fields
    types.Axes: fields.GeomAxis,
    #types.AtomSpec: fields.AtomSelect,
    types.SpeciesSpec: fields.SpeciesSelect,
    types.SpinIndex: fields.SpinIndexSelect,
    types.OrbitalsNames: fields.OrbitalsNameSelect,
    types.OrbitalQueries: fields.OrbitalQueries,
    types.OrbitalStyleQueries: fields.OrbitalQueries,
}

@typing.overload
def get_field_from_type(type_, init: typing.Literal[False]) -> type[fields.InputField]:
    ...

@typing.overload
def get_field_from_type(type_, init: typing.Literal[True] = ...) -> fields.InputField:
    ...

@typing.overload
def get_field_from_type(type_, init: bool = True) -> fields.InputField | type[fields.InputField]:
    ...

def get_field_from_type(type_, init: bool = True) -> fields.InputField | type[fields.InputField]:
    """Given a type annotation, returns the corresponding input field.

    If there is no annotation or the annotation couldn't be matched to any input field,
    the base InputField is returned.

    Parameters
    -----------
    type_: type
        The type annotation.
    init: bool, optional
        Whether to initialize the input field.
    """
    field = None
    # Quick return if there was no annotation
    if type_ is inspect._empty:
        field = fields.InputField

    # Detect if this is an Optional[something] type
    if field is None:
        if typing.get_origin(type_) == typing.Union:
            args = typing.get_args(type_)
            if len(args) == 2 and args[1] is type(None):
                # Then, it was an optional type. Use the type inside it. We should
                # probably acknowledge the fact that is optional by passing something
                # like allow_none=True to the input field. (all fields accept None for now)
                type_ = args[0]

    if field is None:
        # If it is a dataclass, then we return a dictionary input
        if is_dataclass(type_):
            field = fields.DictInput

    if field is None:
        if typing.get_origin(type_) == typing.Literal:
            field = fields.OptionsInput

    if field is None:
        # Direct search
        field = _TYPE_TO_FIELD_MAPPING.get(type_)

    if field is None:
        # If the type is a generic type, the search is a bit harder, but we can still
        # implement the most common ones here.
        origin_type = typing.get_origin(type_)
        if origin_type in (typing.Sequence, collections_Sequence, list):
            arg = typing.get_args(type_)[0]
            try:
                if typing.get_origin(arg) == typing.Literal:
                    field = fields.OptionsInput
                else:
                    field = fields.ListInput
            except:
                field = fields.ListInput
                
        elif origin_type in (typing.Dict, dict):
            field = fields.DictInput

    # If we didn't find a specific field, use the blank one.
    if field is None:
        field = fields.InputField
    
    # If the field is to be initialized, then initialize it, otherwise
    # we will just return the class
    if init:
        field = field.from_typehint(type_)

    return field


def _get_fields_help_from_docs(docs: str) -> dict[str, str]:
    """Parses a docstring following sisl's conventions into a dictionary of help strings.
    
    THIS FUNCTION IS EXPERIMENTAL.
    """

    params_i = docs.find("Parameters\n")
    # Check number of spaces used for tabulation
    n_spaces = docs[params_i + 11:].find("-")

    params_docs = {}
    arg_key = None
    key_msg = ""
    for line in docs[params_i:].split("\n")[2:]:
        if len(line) <= n_spaces:
            break
            
        if line[n_spaces] != " ":
            if ":" not in line:
                break
            if arg_key is not None:
                params_docs[arg_key] = key_msg
            key_msg = ""
            arg_key = line.split(":")[0].lstrip()
        else:
            key_msg += line.lstrip()
    if arg_key is not None:
        params_docs[arg_key] = key_msg

    return params_docs


def get_function_fields(fn: typing.Callable, use_docs: bool = False) -> dict[str, fields.InputField]:
    """For a given function, gets the input fields inferred from its type annotations

    Parameters
    ----------
    fn: typing.Callable
        The function for which you want the input fields.
    use_docs: bool, optional
        Whether to use the docstring of the function to try to infer
        docs for each field.

    Returns
    -------
    dict[str, InputField | None]
        The corresponding fields.
    """
    # Get the signature of the function, we use it because get_type_hints doesn't
    # give you the default values of the arguments.
    parameters = inspect.signature(fn).parameters

    # If this is a class, what we want is the __init__ method. inspect.signature
    # gives us that already from a class, but typing.get_type_hints gives as the 
    # type hints of the class variables, which is not what we want.
    if isinstance(fn, type):
        fn = fn.__init__

    # Get the type annotations
    type_hints = typing.get_type_hints(fn)
    
    # If requested, try to get the documentation for each parameter.
    params_docs = {}
    if use_docs and fn.__doc__:
        try:
            params_docs = _get_fields_help_from_docs(fn.__doc__)
        except:
            use_docs = False

    # Loop through all the parameters and try to find a field for each of them
    fields = {}
    for k, parameter in parameters.items():

        # Get the field from the type annotation
        field = get_field_from_type(type_hints.get(k, inspect._empty), init=True)

        # If we found a field, add extra information
        if field is not None:
            field.params.key = k
            field.params.name = k

            if parameter.default is not inspect._empty:
                field.params.default = parameter.default

            if use_docs:
                field.params.doc = params_docs.get(k)

        # Add the new field to the dictionary
        fields[k] = field

    return fields