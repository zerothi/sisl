
session = None

def launch(inconsole=False, only_api=False, api_kwargs=None):

    from code import interact
    from threading import Thread, Semaphore

    from .server import app as server, run as run_server
    from .api import session as api_session, set_session as set_api_session, run as run_api, app as app
    
    global threads
    global session
    global set_session
    
    session = api_session
    set_session = set_api_session

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
    


