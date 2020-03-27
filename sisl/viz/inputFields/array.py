from ..inputField import InputField

class ArrayInput(InputField):
    
    _type = 'array'

    _default = {
        "width": "s100% l50%"
    }

class Array1dInput(ArrayInput):

    _type = 'vector'

class Array2dInput(ArrayInput):

    _type = "matrix"