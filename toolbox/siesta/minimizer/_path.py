from pathlib import Path


def path_rel_or_abs(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return "./" / path


def path_abs(path, base=None):
    path = Path(path)
    if path.is_absolute():
        return path
    if base is None:
        base = Path.cwd()
    return base / path
