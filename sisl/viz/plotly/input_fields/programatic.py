from .._input_field import InputField


class ProgramaticInput(InputField):

    _type="programatic"

    def __init__(self, *args, help="", **kwargs):

        #help = f"only meant to be provided programatically. {help}"

        super().__init__(*args, help=help, **kwargs)


class FunctionInput(ProgramaticInput):
    """
    This input will be used for those settings that are expecting functions.

    Parameters
    ---------
    positional: array-like of str, optional
        The names of the positional arguments that this function should expect.
    keyword: array-like of str, optional
        The names of the keyword arguments that this function should expect.
    returns: array-like of type
        The datatypes that the function is expected to return.
    """

    _type="function"

    def __init__(self, *args, positional=None, keyword=None, returns=None, **kwargs):
        super().__init__(*args, **kwargs)
