from .._input_field import InputField

class DropdownInput(InputField):
    
    _type = 'dropdown'
    
    _default = {
        "width": "s100% m50% l33%",
        "params": {
            "placeholder": "Choose an option...",
            "options": [
            ],
            "isMulti": False,
            "isClearable": True,
            "isSearchable": True,
        }
    }