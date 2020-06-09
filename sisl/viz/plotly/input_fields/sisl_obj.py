'''
This input field is prepared to receive sisl objects that are plotables
'''

from .._input_field import InputField


class SislObjectInput(InputField):

    _type = "sisl_object"

class PlotableInput(SislObjectInput):

    _type = "plotable"
