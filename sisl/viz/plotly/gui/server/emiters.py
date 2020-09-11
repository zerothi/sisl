from flask_socketio import SocketIO, emit as real_emit
from functools import wraps

__all__ = ["emit", "emit_session", "emit_plot", "emit_loading_plot",
                "emit_loading_plot", "emit_error", "emit_object"]


def emit(*args, socketio=None, **kwargs):

    if socketio is not None:
        socketio.emit(*args, **kwargs)
    else:
        real_emit(*args, **kwargs)


def emit_session(session_to_emit=None, broadcast=True, **kwargs):
    """
    Emits a session through the socket connection

    Parameters
    -----------
    session_to_emit: Session, optional
        The session you want to emit. If none is provided, it will try to find the global session
    """
    if session_to_emit is None:
        session_to_emit = session

    return emit("current_session", session_to_emit._get_dict_for_GUI(), broadcast=broadcast, **kwargs)


def emit_plot(plot, session=None, broadcast=True, **kwargs):
    """
    Emits a plot through the socket connection

    Parameters
    -----------
    plot: str or Plot
            The plot that you want to send. 

            It can be either its ID, to search for the plot in the current session, or
            an actual plot instance.
    """
    if isinstance(plot, str):
        plot = session.plot(plot)

    emit("plot", plot._get_dict_for_GUI(), broadcast=broadcast, **kwargs)


def emit_loading_plot(plot, broadcast=True, **kwargs):
    """
    Emits a message to inform that an action is being performed on this plot.

    This is useful to make the client know that their request is on its way to be fulfilled.

    Parameters
    -----------
    plot: str or Plot
            It can be either a plot ID or an actual plot instance.
    """
    return emit("loading_plot", {"plot_id": plot if isinstance(plot, str) else plot.id}, broadcast=broadcast, **kwargs)


def emit_error(err, **kwargs):
    emit("server_error", str(err), broadcast=False, **kwargs)


def emit_object(obj, *args, **kwargs):
    """
    Emits a sisl object (plot or session) to the GUI.

    Parameters
    ----------
    obj: Plot or Session
            the object that you want to emit
    *args:
            passed directly to the corresponding emiter.
    **kwargs:
            passed directly to the corresponding emiter.
    """
    from sisl.viz import Plot, Session

    if isinstance(obj, Plot):
        emiter = emit_plot
    elif isinstance(obj, Session):
        emiter = emit_session

    emiter(obj, *args, **kwargs)
