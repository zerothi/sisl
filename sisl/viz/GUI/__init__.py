from code import interact

import webbrowser
from threading import Thread, Semaphore

import logging
import sys

from .server import app as server
from .api import session, set_session, run as run_api

__all__ = ['session', 'set_session', 'run_api', 'launch', 'SERVER_ADRESS']

# Disable all flask logging
server.logger.disabled = True
log = logging.getLogger('werkzeug')
log.disabled = True
cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None

SERVER_HOST = "localhost"
SERVER_PORT = 7001
SERVER_ADRESS = f'http://{SERVER_HOST}:{SERVER_PORT}'

def runserver():
    host = "localhost"
    port = 7001

    webbrowser.open(SERVER_ADRESS, new=2)
    server.run(host=host, debug=False, use_reloader=False, port=port, threaded=True)

def launch(inconsole=False, only_api=False, api_kwargs=None):
    global threads

    threads = [Thread(target=run_api, kwargs=api_kwargs), Thread(target=runserver)]

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
    


