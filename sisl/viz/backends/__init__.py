import importlib

def load_backends():
    for backend in ("plotly", "matplotlib"):
        try:
            importlib.import_module(f".{backend}", __name__)
        except ModuleNotFoundError:
            pass