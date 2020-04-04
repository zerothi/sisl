from code import interact

import webbrowser
from threading import Thread, Semaphore

import logging
import sys

from .server import app as server
from .api import app, session, set_session

__all__ = ['session', 'set_session', 'launch', 'SERVER_ADRESS']

# Disable all flask logging
app.logger.disabled = True
log = logging.getLogger('werkzeug')
log.disabled = True
cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None

def runapi():
    app.run(debug=False, port = 4000)

SERVER_HOST = "localhost"
SERVER_PORT = 7001
SERVER_ADRESS = f'http://{SERVER_HOST}:{SERVER_PORT}'

def runserver():
    host = "localhost"
    port = 7001

    webbrowser.open(SERVER_ADRESS, new=2)
    server.run(host=host, debug=False, use_reloader=False, port=port, threaded=True)

threads = [Thread(target=runapi), Thread(target=runserver)] 

def launch(inconsole=False):

    if inconsole:
        #To launch an interactive console (not needed from jupyter) 
        threads.append(Thread(target=interact, kwargs={'local': globals()}))
    try:
        for t in threads:
            t.start()
        print("\nThe session has started succesfully. Happy visualization!\n")
    except Exception as e:
        print(e)
    


