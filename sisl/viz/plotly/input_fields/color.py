from .._input_field import InputField


class ColorPicker(InputField):

    dtype = str

    _type = 'color'

    _default = {
        "width": "s50% m30% l15%"
    }
