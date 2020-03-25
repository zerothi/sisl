from ..inputField import InputField

class RangeSlider(InputField):

    _type = 'rangeslider'

    _default = {
        "width": "s100%",
        "params": {
            "min": -10,
            "max": 10,
            "step": 0.1,
            "marks": { i: str(i) for i in range(-10,11) },
        }
    }