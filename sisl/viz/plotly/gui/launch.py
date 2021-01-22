from functools import partial
from pathlib import Path
from threading import Thread
import time
import webbrowser

from sisl.viz.plotly.plotutils import get_session_classes

from . import server


def open_gui():
    """
    Opens the graphical interface
    """
    webbrowser.open(str(Path(__file__).parent / "build" / "index.html"))


def launch(no_api=False, only_api=False, server_kwargs={}, load_session=None, session_cls=None, session_settings={}, interactive=False):
    """
    Launches the graphical interface.

    Parameters
    -----------
    no_api: bool, optional
        if `True`, avoids initializing the API. All it does then is to open the GUI.
    only_api: bool, optional
        whether only the api should be run. 
        You will probably want this if you already have the GUI open or are planning
        to use the online version of the GUI (https://sisl-siesta.xyz)
    server_kwargs: dict, optional
        keyword arguments that go into `sisl.viz.plotly.gui.server.run`. These are {"host", "port", "debug"}
    load_session: str or Session, optional
        the session to set.
        If it is a string, it will be interpreted as the path were the session is.

        If it is a Session, it will be used directly.

        If it is not provided, the default one will be used.
    session_cls: str or child of Session, optional
        If load_session is not provided, you can pass a class and a new session will be initialized from it.

        If it's a string it should be the name of the class.
    session_settings: dict, optional
        settings that will be applied to the new session, regardless of how it has been obtained (default, from load_session
        or from session_cls.)
    interactive: bool, optional
        whether an interactive console should be started. Probably you will never need to set this to true.
        It is only meant to open a python console in the terminal and it is used by sgui.
    """

    if no_api:
        return open_gui()

    global threads

    def prelaunch(get_session, set_session, load_session=None, session_settings={}, session_cls=None):

        session = get_session()

        if load_session is not None:
            if isinstance(load_session, str):
                load_session = load(load_session)
            set_session(load_session)
        elif session_cls is not None:
            if isinstance(session_cls, str):
                session_cls = get_session_classes().get(session_cls)
            set_session(session_cls(**session_settings))

        if session_settings:
            session.update_settings(**session_settings)

    server_kwargs = {**server_kwargs, "prelaunch": partial(prelaunch, session_settings=session_settings, load_session=load_session, session_cls=session_cls)}

    threads = [Thread(target=server.run, kwargs=server_kwargs)]

    if interactive:
        from code import interact
        #To launch an interactive console (not needed from jupyter)
        threads.append(Thread(target=interact, kwargs={'local': vars(server)}))

    if not only_api:
        threads.append(Thread(target=open_gui))

    for t in threads:
        t.start()

    print("\nThe session has started succesfully. Happy visualization!\n")

    if interactive:
        try:
            while 1:
                time.sleep(.1)
        except KeyboardInterrupt:
            print("Please use Ctrl+D to kill the interactive console first")

if __name__ == "__main__":
    run()
