import pytest

import typing

from sisl.viz.input_fields._typetofield import get_field_from_type, get_function_fields, _get_fields_help_from_docs 
import sisl.viz.input_fields as fields

def test_get_field_from_type():
    assert get_field_from_type(bool).__class__ is fields.BoolInput
    assert get_field_from_type(bool, init=False) is fields.BoolInput

type_field_params = pytest.mark.parametrize(
    ["type_", "field"],
    [(bool, fields.BoolInput), (int, fields.IntegerInput), (float, fields.FloatInput), 
    (str, fields.TextInput), (list, fields.ListInput), (dict, fields.DictInput),
    (typing.Literal["x", "y"], fields.OptionsInput), (typing.Sequence[typing.Literal['x', 'y']], fields.OptionsInput),
    ]
)

@type_field_params
def test_fields_from_types(type_, field):
    assert get_field_from_type(type_, init=False) is field

@type_field_params
def test_fields_from_optional_types(type_, field):
    assert get_field_from_type(typing.Optional[type_], init=False) is field

# Here we create a function and a class that should be equally parsed to fields.
def f(a: int, b: float, c: str, d) -> float:
    """Function that does something
        
    Further explanation

    Parameters
    ----------
    a: int
        A docs
    b: float
        B 
        docs
    """
    return a + b

class F:
    def __init__(self, a: int, b: float, c: str, d):
        """Function that does something
            
        Further explanation

        Parameters
        ----------
        a: int
            A docs
        b: float
            B 
            docs
        """
        return a + b

callable_params = pytest.mark.parametrize(
    ["fn"], [(f,), (F,)]
)

def test_fields_from_function():

    def f(a: int, b: float, c: str, d) -> float:
        return a + b

    expected_fields = {
        "a": fields.IntegerInput,
        "b": fields.FloatInput,
        "c": fields.TextInput,
        "d": fields.InputField
    }
    input_fields = get_function_fields(f)

    assert "self" not in input_fields

    for k, field in expected_fields.items():
        assert k in input_fields
        assert input_fields[k].key == k
        assert input_fields[k].__class__ is field
        assert input_fields[k].doc is None

def test_fields_with_docs():

    def f(a: int, b: float) -> float:
        """Function that does something
        
        Further explanation

        Parameters
        ----------
        a: int
            A docs
        b: float
            B 
            docs
        """
        return a + b

    expected_fields = {
        "a": fields.IntegerInput,
        "b": fields.FloatInput,
    }

    # The expected docstrings (with no whitespace)
    expected_docs = {"a": "Adocs", "b": "Bdocs"}
    input_fields = get_function_fields(f, use_docs=True)

    assert "self" not in input_fields

    for k, field in expected_fields.items():
        assert k in input_fields
        assert input_fields[k].key == k
        assert input_fields[k].__class__ is field
        assert input_fields[k].doc.replace(" ","") == expected_docs[k]