from functools import partial
import numpy as np

__all__ = ["ShortCutable"]


class ShortCutable:
    """
    Class that adds hot key functionality to those that inherit from it.

    Shortcuts help quickly executing common actions without needing to click
    or write a line of code.

    They are supported both in the GUI and in the jupyter notebook.

    """

    def __init__(self, *args, **kwargs):
        self._shortcuts = {}

        super().__init__(*args, **kwargs)

    def shortcut(self, keys):
        """
        Gets the dict that represents a shortcut.

        Parameters
        -----------
        keys: str
            the sequence of keys that trigger the shortcut.
        """
        return self._shortcuts.get(keys, None)

    def add_shortcut(self, _keys, _name, func, *args, _description=None, **kwargs):
        """
        Makes a new shortcut available to the instance.

        You will see that argument names here are marked with "_", in an
        attempt to avoid interfering with the action function's arguments.

        Parameters
        -----------
        _keys: str
            the sequence of keys that trigger the shortcut (e.g.: ctrl+e).
        _name: str
            a short name for the shortcut that should give a first idea of what the shortcut does.
        func: function
            the function to execute when the shortcut is called.
        *args:
            positional arguments that go to the function's execution.
        _description: str, optional
            a longer description of what the shortcut does, maybe including some tips and gotcha's.
        **kwargs:
            keyword arguments that go to the function's execution.
        """
        self._shortcuts[_keys] = {
            "name": _name,
            "description": _description,
            "action": partial(func, *args, **kwargs)
        }

    def remove_shortcut(self, keys):
        """
        Unregisters a given shortcut.

        Parameters
        ------------
        keys: str
            the sequence of keys that trigger the shortcut.
        """
        if keys in self._shortcuts:
            del self._shortcuts[keys]

    def call_shortcut(self, keys, *args, **kwargs):
        """
        Programatic way to call a shortcut.

        In fact, this is the method that is executed when a keypress is detected
        in the GUI or the jupyter notebook.

        Parameters
        -----------
        keys: str
            the sequence of keys that trigger the shortcut.
        *args and **kwargs:
            extra arguments that you pass to the function call.
        """
        self._shortcuts[keys]["action"](*args, **kwargs)

        return self

    def has_shortcut(self, keys):
        """
        Checks if a shortcut is already registered.

        Parameters
        -----------
        keys: str
            the sequence of keys that trigger the shortcut.
        """
        return keys in self._shortcuts

    @property
    def shortcuts_for_json(self):
        """
        Returns a jsonifiable object with information of the shortcuts

        This is meant to be passed to the GUI, so that it knows which shortcuts are available.
        """
        #Basically we are going to remove the action
        return {key: {key: val for key, val in info.items() if key != 'action'} for key, info in self._shortcuts.items()}

    def shortcuts_summary(self, format="str"):
        """
        Gets a formatted summary of the shortcuts.
        """
        if format == "str":
            return "\n".join([f'{key}: {shortcut["name"]}' for key, shortcut in self._shortcuts.items()])
        elif format == "html":
            summ = "<span style='font-weight:bold'>Available keyboard shortcuts:</span><br>"

            def get_shortcut_div(key, shortcut):

                key_span = "".join([f'<span style="background: #ccc; padding: 5px 7px; border-radius: 2px; margin-right: 10px">{key}</span>' for key in key.split()])

                name_span = f'<span style="font-weight: bold">{shortcut["name"]}</span>'

                description_div = f'<div style="padding-left: 40px"><i>{shortcut["description"] or ""}</i></div>'

                return f'<div style="background:aliceblue; border-left: solid 1px; padding: 10px; margin: 10px 0px; border-radius: 3px">{key_span}{name_span}{description_div}</div>'

            summ += "".join([get_shortcut_div(key, shortcut) for key, shortcut in self._shortcuts.items()])

            return f'<div style="background-color:whitesmoke; padding: 10px 20px; border-radius: 5px">{summ}</div>'
