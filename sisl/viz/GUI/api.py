import os
import shutil
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
from sisl.viz.plotutils import load
from sisl.viz.GUI.api_utils import with_user_management, if_user_can, listen_to_users, \
    emit_plot, emit_session, emit_error, emit_loading_plot, emit

__DEBUG = False

app = Flask("SISL GUI API")

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
        
        return super().default(obj)

# We need to use simplejson because built-in json happily parses nan to NaN
# and then javascript does not understand it
simplejson.dumps = partial(simplejson.dumps, ignore_nan=True, cls=CustomJSONEncoder)

if False:
    with_user_management(app)

socketio = SocketIO(app, cors_allowed_origins="*",
                    json=simplejson, manage_session=True, async_mode='threading')
on = socketio.on

if False:
    listen_to_users(on, emit_session)

# This will be the global session
session = BlankSession(socketio=socketio)

@socketio.on_error()
def send_error(err):
    emit_error(err)
    raise err

@on('request_session')
@if_user_can('see')
def send_session(path = None):
    global session

    if path is not None:
        session = load(path)

    emit_session(session, broadcast = False)

@on('apply_method_on_session')
@if_user_can("edit")
def apply_method(method_name, kwargs = {}, *args):

    if __DEBUG:
        print(f"Applying {method_name}. Args: {args}. Kwargs: {kwargs}")


    if kwargs is None:
        # This is because the GUI might send None
        kwargs = {}
    
    # Remember that if the method is not found an error will be raised
    # but it will be handled socketio.on_error (used above)
    method = getattr(session.autosync, method_name)
    
    # Since the session is bound to the app, this will automatically emit the
    # session
    returns = method(*args, **kwargs)

    if kwargs.get("get_returns", None):
        # Let's send the returned values if the user asked for it
        event_name = kwargs.get("returns_as", "call_returns")
        emit(event_name, returns, {"method_name": method_name}, broadcast=False)

@on('get_plot')
@if_user_can("see")
def retrieve_plot(plotID):
    if __DEBUG:
        print(f"Asking for plot: {plotID}")

    emit_plot(plotID, session, broadcast=False)

@on("upload_file")
@if_user_can("edit")
def plot_uploaded_file(file_bytes, name):

    dirname = session.setting("file_storage_dir")
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    file_name = os.path.join(dirname, name)
    with open(file_name, "wb") as fh:
        fh.write(file_bytes)

    # file_name = name
    # file_contents = {name: file_bytes}

    plot = Plot(file_name)#, attrs_for_plot={"_file_contents": file_contents}, _debug=True) # 
    session.autosync.add_plot(plot, session.tabs[0]["id"])

    if not session.setting("keep_uploaded"):
        shutil.rmtree(dirname)

def set_session(new_session):
    '''
    Binds a new session to the GUI.

    The old session is unbound from the GUI. If you want to keep it, store
    it in a variable (or save it to disk with `save`)before setting the new session.
    Note that, otherwise, THE OLD SESSION WILL BE LOST. 

    Returns
    ----------
    Session
        the new session, bound to the GUI. Note that if you keep using old references
        the GUI will not be in sync automatically. You need to store the returned session
        and perform all the actions on it.
    '''
    global session

    session.socketio = None

    session = new_session

    session.socketio = socketio

    return session

def run(host="localhost", port=4000, debug=False):

    print(
        f"\nApi running on http://{host}:{port}...\nconnect the GUI to this address or send it to someone for sharing.")
    
    socketio.run(app, debug=debug, host=host, port=port)

if __name__ == '__main__':
    run()
