import argparse
import time
from ..plotutils import load, get_session_classes

session = None

def launch(inconsole=False, only_api=False, api_kwargs=None, load_session=None, session_settings={}, session_cls=None):

    from code import interact
    from threading import Thread, Semaphore

    from .server import app as server, run as run_server
    from .api import session as api_session, set_session as set_api_session, run as run_api, app as app
    
    global threads
    global session
    global set_session
    
    session = api_session
    set_session = set_api_session

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

    threads = [Thread(target=run_api, kwargs=api_kwargs), Thread(target=run_server)]

    if only_api:
        threads = [threads[0]]

    if inconsole:
        #To launch an interactive console (not needed from jupyter) 
        threads.append(Thread(target=interact, kwargs={'local': globals()}))
    try:
        for t in threads:
            t.start()
        print("\nThe session has started succesfully. Happy visualization!\n")
    except Exception as e:
        print(e)

    if inconsole:
        try:
            while 1:
                time.sleep(.1)
        except KeyboardInterrupt:
            print("Please use Ctrl+D to kill the interactive console first")
    
def sgui():
    '''
    Command line interface for launching GUI related stuff
    '''
    from sisl.viz import Session

    parser = argparse.ArgumentParser(prog='sgui', 
                                     description="Command line utility to launch sisl's graphical interface.") 
  
    parser.add_argument('session', type=str, nargs="?", default=None,
                        help='The session that you want to open in the GUI. If not provided, a fresh new session will be used.')

    parser.add_argument('--session-cls', '-c', required=False,
                        help='If a new session is started, the class that should be used.')

    for param in Session._parameters:
        if param.dtype is not None and not isinstance(param.dtype, str):
            parser.add_argument(f'--{param.key}', type=param._parse, required=False, help=getattr(param, "help", ""))
  
    args = parser.parse_args()

    settings = { param.key: getattr(args, param.key) for param in Session._parameters if getattr(args, param.key, None) is not None}

    launch(inconsole=True, load_session=args.session, session_settings=settings, session_cls=args.session_cls)
