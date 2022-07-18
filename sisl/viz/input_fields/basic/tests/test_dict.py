from dataclasses import dataclass, is_dataclass
from sisl.viz.input_fields.basic import DictInput

def test_empty_dict_input():
    field = DictInput()

    assert field.complete_dict({}) == {}

def test_complete_dict():
    @dataclass
    class A:
        a: int

    field = DictInput.from_typehint(A)

    assert is_dataclass(field.complete_dict({}))
    complete_dict = field.complete_dict({}, as_dict=True)
    assert isinstance(complete_dict, dict)
    assert complete_dict == {"a": None}

    assert field.complete_dict({"a": 3}, as_dict=True) == {"a": 3}

def test_dict_fields():

    @dataclass
    class A:
        a: int
    
    field = DictInput.from_typehint(A)

    assert len(field.params.fields) == 1
    assert "a" in field.params.fields
    assert field.params.fields['a'].key == "a"
    assert field.complete_dict({}, as_dict=True) == {"a": None}

def test_dict_fields_with_defaults():

    @dataclass
    class A:
        a: int = 2
    
    field = DictInput.from_typehint(A)

    assert len(field.params.fields) == 1
    assert "a" in field.params.fields
    assert field.params.fields['a'].key == "a"
    assert field.params.fields['a'].default == 2
    assert field.complete_dict({}, as_dict=True) == {"a": 2}
    assert field.complete_dict({"a": 3}, as_dict=True) == {"a": 3}