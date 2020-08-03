from pathlib import Path
import webbrowser
import logging
import sys

from flask import Flask, send_from_directory

__all__ = ["SERVER_HOST", "SERVER_PORT", "SERVER_ADRESS", "SERVER_APP"]

SERVER_APP = None
SERVER_HOST = None
SERVER_PORT = None
SERVER_ADRESS = None

def create_app():

    global SERVER_APP

    script_dir = Path(__file__).resolve().parent

    SERVER_APP = Flask("SISL GUI", static_folder=str(script_dir / "build"))

    # Serve React App
    @SERVER_APP.route("/", defaults={"path": ""})
    @SERVER_APP.route("/<path:path>")
    def serve(path):
        if path != "" and (Path(SERVER_APP.static_folder) / path).exists():
            return send_from_directory(SERVER_APP.static_folder, path)
        else:
            return send_from_directory(SERVER_APP.static_folder, "index.html")

    # Disable all flask logging
    SERVER_APP.logger.disabled = True
    log = logging.getLogger("werkzeug")
    log.disabled = True
    cli = sys.modules["flask.cli"]
    cli.show_server_banner = lambda *x: None


def run(host="localhost", port=7001, debug=False):

    global SERVER_HOST, SERVER_PORT, SERVER_ADRESS

    if SERVER_APP is None:
        create_app()

    SERVER_HOST = host
    SERVER_PORT = port

    SERVER_ADRESS = f"http://{SERVER_HOST}:{SERVER_PORT}"

    webbrowser.open(SERVER_ADRESS, new=2)
    SERVER_APP.run(host=host, debug=debug, use_reloader=False,
               port=port, threaded=True)

if __name__ == "__main__":
    run()
