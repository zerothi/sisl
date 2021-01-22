from sisl.viz.plotly import Session, Plot

from .launch import launch, open_gui
from .server import *
from .server.sync import Connected

__all__ = ["launch", "open_gui"]

__all__ += ["get_server_address", "set_session", "get_session"]

# Patch the Session and Plot classes so that they work with the GUI
# This is a bit ugly but it works for now
def connect_class(cls):
    cls.socketio = Connected.socketio
    cls._socketio = None
    cls.autosync = Connected.autosync
    cls.emit = Connected.emit

connect_class(Session)
connect_class(Plot)

# The Session class needs some extra patches to make sure the socket
# connection is transmitted to plots
def _transmit_socket_to_plot(session, plot, tabID=None):
    plot.socketio = session.socketio

Session._on_plot_added = _transmit_socket_to_plot

def _remove_socket_from_plot(session, plot):
    plot.socketio = None

Session._on_plot_removed = _transmit_socket_to_plot

def _transmit_socket_change(session):
    """ Transmit the socketio change to all the plots """
    for plot in session.plots.values():
        _transmit_socket_to_plot(session, plot)

Session._on_socketio_change = _transmit_socket_change

