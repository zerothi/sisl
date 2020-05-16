from functools import partial
import numpy as np

class ShortCutable:

    def __init__(self, *args, **kwargs):

        self.shortcuts = {}

        super().__init__(*args, **kwargs)
    
    def shortcut(self, keys):

        return self.shortcuts.get(keys, None)

    def add_shortcut(self, _keys, _name, func, *args, _description=None, **kwargs):

        self.shortcuts[_keys] = {
            "name": _name,
            "description": _description,
            "action": partial(func, *args, **kwargs)
        }
    
    def remove_shortcut(self, keys):

        if keys in self.shortcuts:
            del self.shortcuts[keys]
            
    def call_shortcut(self, keys, *args, **kwargs):

        self.shortcuts[keys]["action"](*args, **kwargs)

        return self
    
    def has_shortcut(self, keys):

        return keys in self.shortcuts
    
    @property
    def shortcuts_for_json(self):
        '''
        Returns a jsonifiable object with information of the shortcuts

        This is meant to be passed to the GUI, so that it knows which shortcuts are available.
        '''

        #Basically we are going to remove the action
        return {key: {key:val for key,val in info.items() if key != 'action'} for key, info in self.shortcuts.items()}

    def shortcuts_summary(self, format="str"):
        '''
        Gets a formatted summary of the shortcuts.
        '''

        if format == "str": 
            return "\n".join([ f'{key}: {shortcut["name"]}' for key, shortcut in self.shortcuts.items()])
        elif format == "html":
            summ = "<span style='font-weight:bold'>Available keyboard shortcuts:</span><br>"

            def get_shortcut_div(key, shortcut):
                
                key_span = "".join([f'<span style="background: #ccc; padding: 5px 7px; border-radius: 2px; margin-right: 10px">{key}</span>' for key in key.split()])
                
                name_span = f'<span style="font-weight: bold">{shortcut["name"]}</span>'
                
                description_div = f'<div style="padding-left: 40px"><i>{shortcut["description"] or ""}</i></div>'
                
                return f'<div style="background:aliceblue; border-left: solid 1px; padding: 10px; margin: 10px 0px; border-radius: 3px">{key_span}{name_span}{description_div}</div>'
                
            summ += "".join([ get_shortcut_div(key, shortcut) for key, shortcut in self.shortcuts.items()])

            return f'<div style="background-color:whitesmoke; padding: 10px 20px; border-radius: 5px">{summ}</div>'



