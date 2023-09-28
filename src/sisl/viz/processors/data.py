from typing import Type, TypeVar

from ..data import Data

DataInstance = TypeVar("DataInstance", bound=Data)

def accept_data(data: DataInstance, cls: Type[Data], check: bool = True) -> DataInstance:

    if not isinstance(data, cls):
        raise TypeError(f"Data must be of type {cls.__name__} and was {type(data).__name__}")
    
    if check:
        data.sanity_check()

    return data

def extract_data(data: Data, cls: Type[Data], check: bool = True):

    if not isinstance(data, cls):
        raise TypeError(f"Data must be of type {cls.__name__} and was {type(data).__name__}")
    
    if check:
        data.sanity_check()

    return data._data