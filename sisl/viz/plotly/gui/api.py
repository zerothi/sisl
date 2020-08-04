import shutil
import pathlib
import traceback

from plotly.graph_objects import Figure
import numpy as np
import simplejson
from simplejson.encoder import JSONEncoder
from functools import partial

import flask
from flask import Flask, request, jsonify, make_response
from flask_socketio import SocketIO, join_room

from sisl.viz import BlankSession, Plot
from sisl._environ import register_environ_variable, get_environ_variable
from ..plotutils import load
from .api_utils import with_user_management, if_user_can, listen_to_users, \
    emit_plot, emit_session, emit_error, emit_loading_plot, emit

__all__ = ["APP", "SESSION", "SOCKETIO", "set_session", "create_app"]

# Register the environment variables that the user can tweak
register_environ_variable("SISL_PLOTLY_API_HOST", "localhost",
                          "The host where the GUI will run when self-hosted by the user.",
                          process=str)
register_environ_variable("SISL_PLOTLY_API_PORT", 4000,
                          "The port where the GUI will run when self-hosted by the user.",
                          process=int)

__DEBUG = False

APP = None
SESSION = None
SOCKETIO = None


def create_app():

    global APP
    global SESSION
    global SOCKETIO

    APP = Flask("SISL GUI API")

    __all__ = ["SESSION", "set_session"]

    class CustomJSONEncoder(JSONEncoder):

        def default(self, obj):

            if isinstance(obj, Figure):
                return obj.to_plotly_json()
            elif hasattr(obj, "to_json"):
                return obj.to_json()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, pathlib.Path):
                return str(obj)

            return super().default(obj)

    # We need to use simplejson because built-in json happily parses nan to NaN
    # and then javascript does not understand it
    simplejson.dumps = partial(simplejson.dumps, ignore_nan=True, cls=CustomJSONEncoder)

    # No user management yet
    if False:
        with_user_management(APP)

    SOCKETIO = SocketIO(APP, cors_allowed_origins="*",
                        json=simplejson, manage_session=True)
                        # async_mode="threading" (this option can not use websockets, less communication performance)
    on = SOCKETIO.on

    if False:
        listen_to_users(on, emit_session)

    # This will be the global session
    SESSION = BlankSession(socketio=SOCKETIO)

    @SOCKETIO.on_error()
    def send_error(err):
        emit_error(err)
        raise err

    @on("request_session")
    @if_user_can("see")
    def send_session(path = None):
        global SESSION

        if path is not None:
            SESSION = load(path)

        emit_session(SESSION, broadcast = False)

    @on("apply_method_on_session")
    @if_user_can("edit")
    def apply_method(method_name, kwargs = {}, *args):

        if __DEBUG:
            print(f"Applying {method_name}. Args: {args}. Kwargs: {kwargs}")

        if kwargs is None:
            # This is because the GUI might send None
            kwargs = {}

        # Remember that if the method is not found an error will be raised
        # but it will be handled socketio.on_error (used above)
        method = getattr(SESSION.autosync, method_name)

        # Since the session is bound to the APP, this will automatically emit the
        # session
        returns = method(*args, **kwargs)

        if kwargs.get("get_returns", None):
            # Let's send the returned values if the user asked for it
            event_name = kwargs.get("returns_as", "call_returns")
            emit(event_name, returns, {"method_name": method_name}, broadcast=False)

    @on("get_plot")
    @if_user_can("see")
    def retrieve_plot(plotID):
        if __DEBUG:
            print(f"Asking for plot: {plotID}")

        emit_plot(plotID, SESSION, broadcast=False)

    @on("upload_file")
    @if_user_can("edit")
    def plot_uploaded_file(file_bytes, name):

        dirname = SESSION.setting("file_storage_dir")
        if not dirname.exists():
            dirname.mkdir()

        file_name = dirname / name
        with open(file_name, "wb") as fh:
            fh.write(file_bytes)

        # file_name = name
        # file_contents = {name: file_bytes}

        plot = Plot(file_name)#, attrs_for_plot={"_file_contents": file_contents}, _debug=True) #
        SESSION.autosync.add_plot(plot, SESSION.tabs[0]["id"])

        if not SESSION.setting("keep_uploaded"):
            shutil.rmtree(str(dirname))

    return APP


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
        whether the old session should also be returned

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

    old_session = SESSION

    SESSION.socketio = None

    SESSION = new_session

    SESSION.socketio = SOCKETIO

    if ret_old:
        return SESSION, old_session
    return SESSION


def run(host=None, port=None, debug=False):

    if APP is None:
        create_app()

    if host is None:
        host = get_environ_variable("SISL_PLOTLY_API_HOST")
    if port is None:
        port = get_environ_variable("SISL_PLOTLY_API_PORT")

    print(
        f"\nApi running on http://{host}:{port}...\nconnect the GUI to this address or send it to someone for sharing.")

    SOCKETIO.run(APP, debug=debug, host=host, port=port)

if __name__ == "__main__":

    run()
