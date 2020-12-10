import os

from ..session import Session


class BlankSession(Session):
    """
    The most basic session one could have, really.

    Parameters
    ------------
    root_dir: str, optional

    file_storage_dir: str, optional
        Directory where files uploaded in the GUI will be stored
    keep_uploaded: bool, optional
        Whether uploaded files should be kept in disk or directly removed
        after plotting them.
    searchDepth: array-like of shape (2,), optional
        Determines the depth limits of the search for structures (from the
        root directory).
    showTooltips: bool, optional
        Tooltips help you understand how something works or what something
        will do.If you are already familiar with the interface, you can
        turn this off.
    listenForUpdates: bool, optional
        Determines whether the session updates plots when files change 
        This is very useful to track progress. It is only meaningful in the
        GUI.
    updateInterval: int, optional
        The time in ms between consecutive checks for updates.
    plotDims: array-like, optional
        The initial width and height of a new plot.  Width is in columns
        (out of a total of 12). For height, you really should try what works
        best for you
    plot_preset: str, optional
        Preset that is passed directly to each plot initialization
    plotly_template: str, optional
        Plotly template that should be used as the default for this session
    """

    _sessionName = "Blank session"

    _description = "The most basic session one could have, really."

    def _after_init(self):
        # Add a first tab so that the user can see something :)
        self.add_tab("First tab")
