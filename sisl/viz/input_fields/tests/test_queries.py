from dataclasses import dataclass, is_dataclass
from typing import Sequence
from sisl.viz.input_fields import QueriesInput, ListInput
from sisl.viz.input_fields.basic.dict import DictInput

def test_queries_input_init():
    field = QueriesInput()

def test_queries_input_fields():

    @dataclass
    class A:
        a: int = 2

    field = QueriesInput.from_typehint(Sequence[A])

    item_input = field.params.item_input
    assert isinstance(item_input, DictInput)
    assert len(item_input.params.fields) == 1
    assert "a" in item_input.params.fields
    assert item_input.params.fields['a'].key == "a"
    assert item_input.params.fields['a'].default == 2

def test_complete_query():

    @dataclass
    class A:
        a: int = 2

    field = QueriesInput.from_typehint(Sequence[A])

    assert is_dataclass(field.complete_query({}))
    assert field.complete_query({}, as_dict=True) == {"a": 2}
    assert field.complete_query({"a": 3}, as_dict=True) == {"a": 3}
