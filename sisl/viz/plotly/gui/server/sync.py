""" In this file we build the patches that will make the session
capable of updating the GUI automatically when an action is performed
on it """

from functools import wraps

from sisl._dispatcher import AbstractDispatch
from sisl.viz.plotly.plotutils import call_method_if_present

from .emiters import emit_object

__all__ = ["AutoSync", "Connected"]


class AutoSync(AbstractDispatch):
    """
    Takes care of emiting all changes automatically to the GUI.

    You probably don't need to use it directly. Inheriting from the `Connected`
    class supercharges the object by providing an autosync attribute that returns
    an instance of this class.

    AutoSync works by wrapping all methods of the object, in a way that after
    their execution, the updated session is sent to the front end of the
    graphical interface. 

    For this purpose, the wrapped object needs to have an emit method, as this is
    what AutoSync triggers after the method call. Again, the Connected class 
    automatically provides an emit method that is suitable for Session and Plot
    objects (but can be extended to others at any point).

    Note that one can avoid the behavior of AutoSync in two different ways:
        - Setting the autosync_enabled attribute to False. 
        - Passing emit=False to the executed method

    Parameters
    ----------
    obj: any
        the object that we want to autosync by using its emit method.

    Example
    ----------
    ```
    from sisl.viz import gui

    # Launch the graphical interface
    GUI.launch()

    # Now, the GUI has an associated session
    session = GUI.session
    # Since Session inherits from Connected you have an AutoSync object
    # available on its 'autosync' attribute
    session = session.autosync

    # So running any method on this session will update the GUI automatically
    session.add_tab("Tab that it's automatically seen by the GUI")

    # However, if we want to try some things before sending them, we can do
    session.autosync_enabled = False
    .
    .   Everything we do here will not be sent automatically to the GUI
    .
    session.autosync_enabled = True # And back to auto-sync

    # If you just want to avoid it for a single method, just pass emit=False
    session.add_tab("Not sent automatically", emit=False)

    ```
    """

    def __init__(self, obj):

        self.autosync_enabled = True

        super().__init__(obj)

    def dispatch(self, method):

        # We won't act if it's a private method or autosync is turned off
        if method.__name__[0] == "_" or not self.autosync_enabled:
            return method

        @wraps(method)
        def with_changes_emitted(*args, emit=True, **kwargs):

            ret = method(*args, **kwargs)

            if emit:
                self._obj.emit()

            return ret

        return with_changes_emitted

class Connected:
    """
    Helps connecting objects to the graphical interface.

    Objects that inherit from this class have the possibility
    to be synced automatically through a socketio channel.

    Parameters
    ----------
    socketio: socketio
        The socketIO channel that the object will use for transmission.

    Attributes
    ----------
    socketio: socketio
        The socketIO channel that the object will use for transmission.

    Usage
    ----------
    ```
    session.autosync.add_tab("Just a new tab")
    # This change will be automatically transmitted to all socket listeners.
    ```
    """

    def get_socketio(self):
        return self._socketio

    def set_socketio(self, new_socketio):

        self._socketio = new_socketio

        call_method_if_present(self, '_on_socketio_change')

    socketio = property(fget=get_socketio, fset=set_socketio)

    @property
    def autosync(self):
        """
        A super-powered version of the object that syncs automatically through
        the socketio channel after each method call.

        Usage
        ----------
        ```
        session.autosync.add_tab("Just a new tab")
        # This change will be automatically transmitted to all socket listeners.
        ```
        """
        return AutoSync(self)

    def emit(self, socketio=None):
        """
        Emits the object through a socketio channel.

        Parameters
        -----------
        socketio: socketio, optional
            The socketio channel through which the object must be emitted.

            NOTE: Most certainly YOU DON'T NEED TO PROVIDE A SOCKETIO, if your object
            has a socketio already associated to it, the object will be sent through 
            that channel.
        """
        if socketio is None:
            socketio = self.socketio

        emit_object(self, socketio=socketio)

        return self
