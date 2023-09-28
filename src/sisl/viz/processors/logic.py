from typing import Any, Tuple, TypeVar, Union

T1 = TypeVar("T1")
T2 = TypeVar("T2")

def swap(val: Union[T1, T2], vals: Tuple[T1, T2]) -> Union[T1, T2]:
    """Given two values, returns the one that is not the input value."""
    if val == vals[0]:
        return vals[1]
    elif val == vals[1]:
        return vals[0]
    else:
        raise ValueError(f"Value {val} not in {vals}")

def matches(first: Any, second: Any, ret_true: T1 = True, ret_false: T2 = False) -> Union[T1, T2]:
    """If first matches second, return ret_true, else return ret_false."""
    return ret_true if first == second else ret_false
    
def switch(obj: Any, ret_true: T1, ret_false: T2) -> Union[T1, T2]:
    """If obj is True, return ret_true, else return ret_false."""
    return ret_true if obj else ret_false