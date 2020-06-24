import numpy as np

from .._input_field import InputField
from .text import TextInput


class ListInput(InputField):

    dtype = "array-like"

    _type = 'list'

    _default = {
        "width": "s100% l50%",
        "params": {"itemInput": TextInput("-", "-"), "sortable": True}
    }
