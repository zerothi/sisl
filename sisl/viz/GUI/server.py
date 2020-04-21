import os
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

if __name__ == "__main__":
    import webbrowser
    import time

    host = "localhost"
    port = 7001

    #Wait for the api
    time.sleep(1)
    webbrowser.open(f'http://{host}:{port}', new=2)
    app.run(host=host, use_reloader=True, port=port, threaded=True)