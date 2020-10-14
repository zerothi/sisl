import requests
import tarfile
import urllib
import shutil
from pathlib import Path

def update_gui():
    """Installs the latest version of the GUI"""
    gui_root = Path(__file__).parent
    # Request releases info
    releases = requests.get("https://api.github.com/repos/pfebrer96/sislGUIpublic/releases").json()

    # Get latest release
    release = releases[0]

    # Get the downloadable assets
    assets = release["assets"]

    # Find the file that we want to upload
    for asset in assets:
        if "build" in asset["name"] and asset["name"].endswith("tar.gz"):
            file_url = asset["browser_download_url"]
            break

    response = urllib.request.urlopen(file_url)
    tar_file = tarfile.open(fileobj=response, mode="r|gz")

    tar_file.extractall(gui_root)

if __name__ == "__main__":
    update_gui()