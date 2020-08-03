import argparse
import time
import os
from ..plotutils import load, get_session_classes
from .._user_customs import SESSION_FILE

SESSION = None

__all__ = ["SESSION", "launch"]


def launch(only_api=False, api_kwargs=None, server_kwargs=None, load_session=None, session_cls=None, session_settings={}, interactive=False):
    '''
    Launches the graphical interface.

    Parameters
    -----------
    only_api: bool, optional
        whether only the api should be run. 
        You will probably want this if you already have an instance of the GUI running and/or are planning
        to use the online version of the GUI (https://sisl-siesta.xyz)
    api_kwargs: dict, optional
        keyword arguments that go into `api.run`. These are {"host", "port", "debug"}
    server_kwargs: dict, optional
        keyword arguments that go into `server.run`. These are {"host", "port", "debug"}.
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
    '''

    from code import interact
    from threading import Thread, Semaphore

    from . import server
    from . import api

    global threads
    global SESSION
    global set_session

    app = api.create_app()

    SESSION = api.SESSION
    set_session = api.set_session

    if load_session is not None:
        if isinstance(load_session, str):
            load_session = load(load_session)
        set_session(load_session)
    elif session_cls is not None:
        if isinstance(session_cls, str):
            session_cls = get_session_classes().get(session_cls)
        set_session(session_cls(**session_settings))

    if session_settings:
        SESSION.update_settings(**session_settings)

    threads = [Thread(target=api.run, kwargs=api_kwargs), Thread(target=server.run, kwargs=server_kwargs)]

    if only_api:
        threads = [threads[0]]

    if interactive:
        #To launch an interactive console (not needed from jupyter)
        threads.append(Thread(target=interact, kwargs={'local': globals()}))
    try:
        for t in threads:
            t.start()
        print("\nThe session has started succesfully. Happy visualization!\n")
    except Exception as e:
        print(e)

    if interactive:
        try:
            while 1:
                time.sleep(.1)
        except KeyboardInterrupt:
            print("Please use Ctrl+D to kill the interactive console first")


def general_arguments(parser):

    parser.add_argument('--only-api', dest='only_api', action="store_true",
        help="Pass this flag if all you want to do is initialize the API, but not the GUI. You can do this if you plan"+
        "to access the graphical interface by some other means (e.g. https://sisl-siesta.xyz)"
    )


def sgui():
    '''
    Command line interface for launching GUI related stuff
    '''
    from sisl.viz import Session

    avail_session_classes = get_session_classes()

    parser = argparse.ArgumentParser(prog='sgui',
                                     description="Command line utility to launch sisl's graphical interface.")

    general_arguments(parser)

    parser.add_argument('--load', '-l', type=str, nargs="?", default=None,
                        help='The path to the session that you want to open in the GUI. If not provided, a fresh new session will be used.')

    for param in Session._get_class_params()[0]:
        if param.dtype is not None and not isinstance(param.dtype, str):
            parser.add_argument(f'--{param.key}', type=param.parse, required=False, help=getattr(param, "help", ""))

    subparsers = parser.add_subparsers(
        help="YOU DON'T NEED TO PASS A SESSION CLASS. You can provide a session file to load a saved session (see the --load flag)."+
        " However, if you want to start a new session and the default one (BlankSession) is not good for you"+
        " you can pass a session class. By doing so, you will also get access to session-specific settings. Try sgui BlankSession -h, for example." +
        " Note that you can also build your own sessions that will be automatically available here." +
        f" Sisl is looking to import plots defined in {SESSION_FILE}",
        dest="session_class"
    )

    for name, SessionClass in avail_session_classes.items():
        doc = SessionClass.__doc__ or ""
        specific_parser = subparsers.add_parser(name, help=doc.split(".")[0])
        general_arguments(specific_parser)
        for param in SessionClass._get_class_params()[0]:
            if param.dtype is not None and not isinstance(param.dtype, str):
                specific_parser.add_argument(f'--{param.key}', type=param.parse, required=False, help=getattr(param, "help", ""))

    args = parser.parse_args()

    # Note that it doesn't matter if we include invalid settings. Configurable will just ignore them
    settings = {key: val for key, val in vars(args).items() if val is not None}

    launch(interactive=True, load_session=args.load, session_settings=settings, session_cls=args.session_class, only_api=args.only_api)
