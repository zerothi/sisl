import os
import webbrowser
import logging
import sys

from flask import Flask, send_from_directory

scriptDir = os.path.dirname(os.path.realpath(__file__))

app = Flask("SISL GUI", static_folder=os.path.join(scriptDir, 'build'))

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# Disable all flask logging
app.logger.disabled = True
log = logging.getLogger('werkzeug')
log.disabled = True
cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None

SERVER_HOST = "localhost"
SERVER_PORT = 7001
SERVER_ADRESS = f'http://{SERVER_HOST}:{SERVER_PORT}'


def run():
    host = SERVER_HOST
    port = SERVER_PORT

    webbrowser.open(SERVER_ADRESS, new=2)
    app.run(host=host, debug=False, use_reloader=False,
               port=port, threaded=True)

if __name__ == "__main__":
    run()
