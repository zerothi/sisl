import logging
import sys

from sisl._environ import register_environ_variable, get_environ_variable

from . import api

SESSION = None

SERVER_HOST = None
SERVER_PORT = None

# Register the environment variables that the user can tweak
register_environ_variable("SISL_PLOTLY_API_HOST", "localhost",
                          "The host where the GUI will run when self-hosted by the user.",
                          process=str)
register_environ_variable("SISL_PLOTLY_API_PORT", 4000,
                          "The port where the GUI will run when self-hosted by the user.",
                          process=int)


def set_session(new_session, ret_old=False):
    """
    Binds a new session to the GUI.

    The old session is unbound from the GUI. If you want to keep it, store
    it in a variable (or save it to disk with `save`)before setting the new session.
    Note that, otherwise, THE OLD SESSION WILL BE LOST. You can also set "ret_old"
    to True to get a tuple of (new_session, old_session) from this method

    Parameters
    ----------
    new_session: Session
        the new session to be used by the API.
    ret_old: bool, optional
        whether the old session should also be returned.

    Returns
    ----------
    Session
        the new session, bound to the GUI. Note that if you keep using old references
        the GUI will not be in sync automatically. You need to store the returned session
        and perform all the actions on it.
    Session
        the old session, only returned if `ret_old` is `True`
    """
    global SESSION

    socketio = None

    if SESSION is not None:
        old_session = SESSION

        socketio = old_session.socketio

        old_session.socketio = None

    SESSION = new_session

    if socketio is not None:
        SESSION.socketio = socketio

    if ret_old:
        return SESSION, old_session
    return SESSION


def get_session():
    return SESSION


def get_server_address():
    return f"http://{SERVER_HOST}:{SERVER_PORT}"


def run(host=None, port=None, debug=False, app=None, socketio=None, prelaunch=None):

    global SERVER_PORT
    global SERVER_HOST

    if app is None:
        app, socketio = api.create_app(get_session, set_session)

    # Disable all kinds of logging
    if not debug:
        app.logger.disabled = True
        log = logging.getLogger("werkzeug")
        log.disabled = True
        cli = sys.modules["flask.cli"]
        cli.show_server_banner = lambda *x: None

    if prelaunch is not None:
        prelaunch(get_session, set_session)

    if host is None:
        host = get_environ_variable("SISL_PLOTLY_API_HOST")
    if port is None:
        port = get_environ_variable("SISL_PLOTLY_API_PORT")

    SERVER_HOST = host
    SERVER_PORT = port

    print(
        f"\nApi running on {get_server_address()}...\nconnect the GUI to this address or send it to someone for sharing.")

    socketio.run(app, debug=debug, host=host, port=port)
