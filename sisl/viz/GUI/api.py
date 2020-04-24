import os
import traceback

from plotly.graph_objects import Figure
import numpy as np
import simplejson
from simplejson.encoder import JSONEncoder
from functools import partial

import flask
from flask import Flask, request, jsonify, make_response
from flask_socketio import SocketIO, join_room

from sisl.viz import BlankSession
from sisl.viz.plotutils import load
from sisl.viz.GUI.api_utils import with_user_management, if_user_can, listen_to_users
from sisl.viz.GUI.api_utils.emiters import emit_plot, emit_session, emit_error, emit_loading_plot, emit_outside_flask, emit

__DEBUG = True

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

def patch_session(session):
	'''
	Makes the session trigger events when manipulating plots
	'''
	session.before_plot_change = lambda plot: emit_loading_plot(plot)
	session.on_plot_change = lambda plot: emit_plot(plot)
	session.on_plot_change_error = lambda plot, err: emit_error(err)

	return session

if False:
	with_user_management(app)

socketio = SocketIO(app, cors_allowed_origins="*",
                    json=simplejson, manage_session=True)
on = socketio.on

if False:
	listen_to_users(on, emit_session)

# This will be the global session
session = BlankSession()
patch_session(session)

@socketio.on_error()
def send_error(err):
	emiters.emit_error(err)

@on('request_session')
@if_user_can('see')
def send_session(path = None):
	global session

	session = None
	if path is not None:
		session = load(path)

	emiters.emit_session(session, broadcast = False)

@on('apply_method_on_session')
@if_user_can("edit")
def apply_method(method_name, kwargs = {}, *args):

	if __DEBUG:
		print(f"Applying {method_name}. Args: {args}. Kwargs: {kwargs}")


	if kwargs is None:
		# This is because the GUI might send None
		kwargs = {}
	
	# Remember that if the method is not found an error will be raised
	# but it will be handled by our super nice error handler wrapper
	method = getattr(session, method_name)
		
	returns = method(*args, **kwargs)

	if kwargs.get("get_returns", None):
		# Let's send the returned values if the user asked for it
		event_name = kwargs.get("returns_as", "call_returns")
		emit(event_name, returns, {"method_name": method_name}, broadcast=False)

	emiters.emit_session(session)

@on('get_plot')
@if_user_can("see")
def retrieve_plot(plotID):
	if __DEBUG:
		print(f"Asking for plot: {plotID}")

	emiters.emit_plot(plotID, broadcast=False)

def set_session(new_session):
	global session
	session = new_session
	patch_session(session)

def run(host="localhost", port=4000, debug=False):

	print(
		f"\nApi running on http://{host}:{port}...\nconnect the GUI to this address or send it to someone for sharing.")
	
	socketio.run(app, debug=debug, host=host, port=port)

if __name__ == '__main__':
	run()
