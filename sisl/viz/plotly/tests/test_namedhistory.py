import pytest

import numpy as np

from sisl.viz.plotly.configurable import NamedHistory


pytestmark = [pytest.mark.viz, pytest.mark.plotly]


def test_named_history():

    test_keys = ["hey", "nope"]
    val_key, def_key = test_keys

    s = NamedHistory({val_key: 2}, defaults={def_key: 5})

    #Check that all keys have been incorporated
    assert np.all([key in s for key in test_keys])
    assert np.all([len(s._vals[key]) == 1 for key in test_keys])
    assert s.current[val_key] == 2

    # Check that the history updates succesfully
    s.update(**{val_key: 3})
    assert s.last_updated == [val_key]
    assert s.diff_keys(1, 0) == [val_key]
    assert s.last_update_for(val_key) == 1
    assert len(s._vals[val_key]) == 2
    assert s.current[val_key] == 3
    assert val_key in s.last_delta
    assert s.last_delta[val_key]["before"] == 2

    # Check that it can correctly undo settings
    s.undo()
    assert len(s._vals[val_key]) == 2
    assert s._vals[val_key][1] is None
    assert s.current[val_key] == 2

    # One last check with multiple updates
    s.update(**{val_key: 5})
    s.update(**{val_key: 6})
    assert len(s._vals[val_key]) == 4


def test_history_item_getting():

    test_keys = ["hey", "nope"]
    val_key, def_key = test_keys

    s = NamedHistory({val_key: 2}, defaults={def_key: 5})
    s.update(**{val_key: 6})

    assert s[-1][val_key] == 6

    assert len(s[[-1, -2]]) == 2
    assert isinstance(s[:], dict)
    assert len(s[0:1][val_key]) == 1

    assert s[val_key] == [2, 6]
    assert isinstance(s[test_keys], dict)
    assert len(s[test_keys][val_key]) == len(s)


def test_update_array():

    # Here we are just checking that we can update with numpy arrays
    # without an error. This is because comparing two numpy arrays
    # raises an Exception, so we need to make sure this doesn't happen

    test_keys = ["hey", "nope"]
    val_key, def_key = test_keys

    s = NamedHistory({val_key: 2}, defaults={def_key: 5})

    s.update(**{val_key: np.array([1, 2, 3])})
    s.update(**{val_key: np.array([1, 2, 3, 4])})
