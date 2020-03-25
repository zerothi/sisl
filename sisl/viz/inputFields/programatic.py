from ..inputField import InputField

class ProgramaticInput(InputField):

    _type="programatic"

    def __init__(self, *args, help="", **kwargs):

        help = f"only meant to be provided programatically. {help}"

        super().__init__(*args, help=help, **kwargs)