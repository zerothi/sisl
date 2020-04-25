''' In this file we build the patches that will make the session
capable of updating the GUI automatically when an action is performed
on it '''

from functools import wraps

from sisl.viz.plotutils import call_method_if_present
from ._dispatcher import AbstractDispatch
from .emiters import emit_object

class AutoSync(AbstractDispatch):
    '''
    Takes care of emiting all changes to the GUI.

    It works by wrapping all methods of the session, in a way that after
    their execution, the updated session is sent to the front end of the
    graphical interface.

    Note that one can avoid this behavior in two different ways:
        - Setting the autosync attribute to False.
        - Passing emit=False to the executed method

    Parameters
    ----------
    session: Session or AutoSyncSession
        the session object that we want to track.
    socketio:
        the socket channel that is serving as a backend for the GUI.

    Example
    ----------
    ```
    from sisl.viz import GUI

    # Launch the graphical interface
    GUI.launch()

    # Now, the GUI has an associated session that already uses AutoSyncSession
    session = GUI.session

    # So running any method on this session will update the GUI automatically
    session.addTab("Tab that it's automatically seen by the GUI")

    # However, if we want to try some things before sending them, we can do
    session.autosync = False
    .
    .   Everything we do here will not be sent automatically to the GUI
    .
    session.autosync = True # And back to auto-sync

    # If you just want to avoid it for a single method, just pass emit=False
    session.addTab("Not sent automatically", emit=False)

    ```
    '''
    def __init__(self, obj):

        self.autosync = True

        super().__init__(obj)

    def dispatch(self, method):

        # We won't act if it's a private method or autosync is turned off
        if method.__name__[0] == "_" or not self.autosync:
            return method

        @wraps(method)
        def with_changes_emitted(*args, emit=True, **kwargs):

            ret = method(*args, **kwargs)

            if emit:
                emit_object(self._obj, socketio=self._obj.socketio)

            return ret
        
        return with_changes_emitted

class Connected:
    '''
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
    '''

    def __init__(self, *args, socketio=None, **kwargs):

        self._socketio = socketio

        super().__init__(*args, **kwargs)

    def get_socketio(self):
        return self._socketio

    def set_socketio(self, new_socketio):

        self._socketio = new_socketio

        call_method_if_present(self, '_on_socketio_change')

    socketio = property(fget=get_socketio, fset=set_socketio)

    @property
    def autosync(self):
        '''
        A super-powered version of the object that syncs automatically through
        the socketio channel after each method call.

        Usage
        ----------
        ```
        session.autosync.add_tab("Just a new tab")
        # This change will be automatically transmitted to all socket listeners.
        ```
        '''
        return AutoSync(self)
    
    def emit(self, socketio=None):
        '''
        Emits the object through a socketio channel.

        Parameters
        -----------
        socketio: socketio, optional
            The socketio channel through which the object must be emitted.

            NOTE: Most certainly YOU DON'T NEED TO PROVIDE A SOCKETIO, if your object
            has a socketio already associated to it, the object will be sent through 
            that channel.
        '''

        if socketio is None:
            socketio = self.socketio
        
        emit_object(self, socketio=socketio)

        return self

