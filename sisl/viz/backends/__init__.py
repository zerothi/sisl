import importlib

__all__ = ["load_backends"]

def load_backends():
    for backend in ("templates", "plotly", "matplotlib"):
        try:
            importlib.import_module(f".{backend}", __name__)
        except ModuleNotFoundError:
            pass