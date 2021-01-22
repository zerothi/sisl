"""
This file makes sure that patching has been successful, and therefore connection with the GUI will work
"""
# This import is just so that the gui module is initialized and therefore
# the Session and Plot classes get patched.
import sisl.viz.plotly.gui

from sisl.viz import Plot, Session

def test_plot_connected():

    plot = Plot()

    # Check that the plot has a socketio attribute and that it can be changed
    # Seems dumb, but socketio is really a property that uses functions to set the
    # real attribute, so they may be broken by something
    assert hasattr(plot, 'socketio'), f"Socketio connectivity is not initialized correctly in {plot.__class__}"
    assert plot.socketio is None
    plot.socketio = 2
    assert plot.socketio == 2, f'Problems setting a new socketio for {plot.__class__}'

def test_session_connected():

    session = Session()

    # Check that the session has a socketio attribute and that it can be changed
    # Seems dumb, but socketio is really a property that uses functions to set the
    # real attribute, so they may be broken by something
    assert hasattr(session, 'socketio')
    assert session.socketio is None
    session.socketio = 2
    assert session.socketio == 2

    # Check that if we add a plot to the session, their socketio will be tracked by
    # the session
    new_plot = Plot()
    session.add_tab('Test tab')
    session.add_plot(new_plot, 'Test tab')
    assert new_plot.socketio == 2, f'Socketio not transfered from {session.__class__} to plot on add_plot'
    # Fake a disconnection of the session and see if the plot follows
    session.socketio = None
    assert new_plot.socketio is None, f'Socketio change in {session.__class__} not transmitted to plots'