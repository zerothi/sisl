import os
import traceback

from plotly.graph_objects import Figure
import numpy as np
import simplejson
from simplejson.encoder import JSONEncoder
from functools import partial

import flask
from flask import Flask, request, jsonify, make_response
from flask_socketio import SocketIO, join_room, emit, send

from sisl.viz import BlankSession
from sisl.viz.plotutils import load

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

socketio = SocketIO(app, cors_allowed_origins="*", json=simplejson)

def emit_session(session_to_emit = None, broadcast=True, **kwargs):
	'''
	Emits a session through the socket connection

	Parameters
	-----------
	session_to_emit: Session, optional
		The session you want to emit. If not provided, the current session will be used.
	'''
		
	if session_to_emit is None:
		session_to_emit = session
	
	return emit("current_session", session._getJsonifiableInfo(), broadcast=broadcast, **kwargs)

def emit_plot(plot, broadcast=True, **kwargs):
	'''
	Emits a plot through the socket connection

	Parameters
	-----------
	plot: str or Plot
		The plot that you want to send. 
		
		It can be either its ID, to search for the plot in the current session, or
		an actual plot instance.
	'''

	if isinstance(plot, str):
		plot = session.plot(plot)


	emit("plot", plot._getDictForGUI(), broadcast=broadcast, **kwargs)

def emit_loading_plot(plot, broadcast=True):
	'''
	Emits a message to inform that an action is being performed on this plot.

	This is useful to make the client know that their request is on its way to be fulfilled.

	Parameters
	-----------
	plot: str or Plot
		It can be either a plot ID or an actual plot instance.
	'''

	return emit("loading_plot", {"plot_id": plot if isinstance(plot, str) else plot.id}, broadcast=broadcast)

def emit_error(err):

	emit("error", str(err), broadcast=False)

def with_error_handling(func):
	def handle_errors(*args, **kwargs):
		try:
			func(*args, **kwargs)
		except Exception as e:
			emit_error(e)
	return handle_errors

def on(*args, **kwargs):
	'''
	Wraps socketio.on so that we have error handling.
	(or other functionalities) by default
	'''
	def wrapper(message_handler):

		mh = with_error_handling(message_handler)
		return socketio.on(*args, **kwargs)(mh)
	
	return wrapper

def patch_session(session):
	'''
	Makes the session trigger events when manipulating plots
	'''
	session.before_plot_change = lambda plot: emit_loading_plot(plot)
	session.on_plot_change = lambda plot: emit_plot(plot)
	session.on_plot_change_error = lambda plot, err: emit_error(err)

	return session

session = BlankSession()
patch_session(session)

@on('request_session')
def send_session(path = None):
	
	session = None
	if path is not None:
		session = load(path)
	
	emit_session(session, broadcast = False)

@on('apply_method_on_session')
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

	emit_session()

@on('get_plot')
def retrieve_plot(plotID):
	if __DEBUG:
		print(f"Asking for plot: {plotID}")

	emit_plot(plotID, broadcast=False)

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
