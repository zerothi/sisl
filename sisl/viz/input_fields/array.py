from .._input_field import InputField

class ArrayInput(InputField):

    dtype = "array-like"
    
    _type = 'array'

    _default = {
        "width": "s100% l50%"
    }

class Array1dInput(ArrayInput):

    _type = 'vector'

class Array2dInput(ArrayInput):

    _type = "matrix"
    