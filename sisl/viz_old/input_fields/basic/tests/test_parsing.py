# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ast import parse
import numpy as np
import pytest

from sisl.viz.input_fields import (
    DictInput, TextInput, IntegerInput, FloatInput,
    BoolInput, ListInput
)


def test_text_input_parse():
    input_field = TextInput(key="test", name="Test")

    assert input_field.parse("Some test input") == "Some test input"


def test_integer_input_parse():
    input_field = IntegerInput(key="test", name="Test")

    assert input_field.parse(3) == 3
    assert input_field.parse("3") == 3
    assert input_field.parse(3.2) == 3

    with pytest.raises(ValueError):
        input_field.parse("Some non-integer")

    assert input_field.parse(None) is None

    # Test that it can also accept arrays of integers and
    # it parses them to numpy arrays.
    for val in [3, 3.2]:
        parsed_array = input_field.parse([val])
        assert isinstance(parsed_array, np.ndarray)
        assert np.all(parsed_array == [3])


def test_float_input_parse():
    input_field = FloatInput(key="test", name="Test")

    assert input_field.parse(3.2) == 3.2

    with pytest.raises(ValueError):
        input_field.parse("Some non-float")

    assert input_field.parse(None) is None

    # Test that it can also accept arrays of floats and
    # it parses them to numpy arrays.
    parsed_array = input_field.parse([3.2])
    assert isinstance(parsed_array, np.ndarray)
    assert np.all(parsed_array == [3.2])


def test_bool_input_parse():
    input_field = BoolInput(key="test", name="Test")

    assert input_field.parse(True) == True
    assert input_field.parse(False) == False

    with pytest.raises(ValueError):
        input_field.parse("Some non-boolean string")

    assert input_field.parse("true") == True
    assert input_field.parse("True") == True
    assert input_field.parse("t") == True
    assert input_field.parse("T") == True

    assert input_field.parse("false") == False
    assert input_field.parse("False") == False
    assert input_field.parse("f") == False
    assert input_field.parse("F") == False

    with pytest.raises(TypeError):
        input_field.parse([])

    assert input_field.parse(None) is None


def test_dict_input_parse():
    input_field = DictInput(
        key="test", name="Test",
        fields=[
            TextInput(key="a", name="A"),
            IntegerInput(key="b", name="B"),
        ]
    )

    assert input_field.parse({"a": "S", "b": 3}) == {"a": "S", "b": 3}
    assert input_field.parse({"a": "S", "b": 3.2}) == {"a": "S", "b": 3}

    with pytest.raises(ValueError):
        input_field.parse({"a": "S", "b": "Some non-integer"})

    assert input_field.parse(None) == {}
    assert input_field.parse({}) == {}

    with pytest.raises(TypeError):
        input_field.parse(3)


def test_list_input_parse():

    input_field = ListInput(key="test", name="Test",
        params={"itemInput": IntegerInput(key="_", name="_")}
    )

    assert input_field.parse([]) == []
    assert input_field.parse([3]) == [3]
    assert input_field.parse([3.2]) == [3]
    assert input_field.parse((3.2,)) == [3]
    assert input_field.parse(np.array([3.2])) == [3]

    with pytest.raises(TypeError):
        input_field.parse(3)
    with pytest.raises(ValueError):
        input_field.parse(["Some non-integer"])

    assert input_field.parse(None) is None
