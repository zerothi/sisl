from flask_socketio import SocketIO, emit
from functools import wraps

def emit_session(session_to_emit=None, broadcast=True, **kwargs):
	'''
	Emits a session through the socket connection

	Parameters
	-----------
	session_to_emit: Session, optional
		The session you want to emit. If none is provided, it will try to find the global session
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

def emit_outside_flask(emiter, app):

    @wraps(emiter)
    def emit_from_outside(*args, **kwargs):

        with app.app_context():
            return emiter(*args, **kwargs)
