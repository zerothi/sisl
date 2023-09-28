import pytest

from sisl.viz.processors.logic import matches, swap, switch


def test_swap():

    assert swap(1, (1, 2)) == 2
    assert swap(2, (1, 2)) == 1

    with pytest.raises(ValueError):
        swap(3, (1, 2))

def test_matches():

    assert matches(1, 1) == True
    assert matches(1, 2) == False

    assert matches(1, 1, "a", "b") == "a"
    assert matches(1, 2, "a", "b") == "b"

    assert matches(1, 1, "a") == "a"
    assert matches(1, 2, "a") == False

    assert matches(1, 1, ret_false="b") == True
    assert matches(1, 2, ret_false="b") == "b"

def test_switch():

    assert switch(True, "a", "b") == "a"
    assert switch(False, "a", "b") == "b"

    