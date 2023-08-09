# Classes that hold information regarding how a given parameter should behave in a CLI
# They are meant to be used as metadata for the type annotations. That is, passing them
# to Annotated. E.g.: Annotated[int, CLIArgument(option="some_option")]. Even if they
# are empty, they indicate whether to treat the parameter as an argument or an option.
class CLIArgument:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class CLIOption:
    def __init__(self, *param_decls: str, **kwargs):
        if len(param_decls) > 0:
            kwargs["param_decls"] = param_decls
        self.kwargs = kwargs

def get_params_help(func) -> dict:
    """Gets the text help of parameters from the docstring"""
    params_help = {}

    in_parameters = False
    read_key = None
    arg_content = ""

    for line in func.__doc__.split("\n"):
        if "Parameters" in line:
            in_parameters = True
            space = line.find("Parameters")
            continue
            
        if in_parameters:
            if len(line) < space + 1:
                continue
            if len(line) > 1 and line[0] != " ":
                break

            if line[space] not in (" ", "-"):
                if read_key is not None:
                    params_help[read_key] = arg_content
                
                read_key = line.split(":")[0].strip()
                arg_content = ""
            else:
                if arg_content == "":
                    arg_content = line.strip()
                    arg_content = arg_content[0].upper() + arg_content[1:]
                else:
                    arg_content += " " + line.strip()

        if line.startswith("------"):
            break
                
    if read_key is not None:
        params_help[read_key] = arg_content

    return params_help