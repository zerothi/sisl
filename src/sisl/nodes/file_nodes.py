# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from pathlib import Path

try:
    import watchdog.events
    import watchdog.observers

    WATCHDOG_IMPORTED = True
except ImportError:
    WATCHDOG_IMPORTED = False

from sisl.messages import warn

from .node import Node

if WATCHDOG_IMPORTED:

    class CallbackHandler(watchdog.events.PatternMatchingEventHandler):
        def __init__(self, callback_obj, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.callback_obj = callback_obj

        def _run_method(self, method_name, event):
            if not hasattr(self.callback_obj, "matches_path"):
                return
            if not self.callback_obj.matches_path(event.src_path):
                return

            func = getattr(self.callback_obj, method_name, None)
            if callable(func):
                return func(event)

        def on_modified(self, event):
            self._run_method("on_file_modified", event)

        def on_created(self, event):
            self._run_method("on_file_created", event)

        def on_moved(self, event):
            self._run_method("on_file_moved", event)

        def on_deleted(self, event):
            self._run_method("on_file_deleted", event)


class FileNode(Node):
    """Node that listens to changes to a given file.

    If the file changes, the node will be marked as outdated and the signal
    will be propagated to all downstream nodes.

    Therefore, if autoupdate is enabled at some point down the tree, an
    update will be triggered.

    This node requires `watchdog` to be installed in order to be fully
    functional. Otherwise, it will not fail but it won't do anything.

    Parameters
    ----------
    path :
        Path to the file to watch.

    Examples
    --------

    .. code-block:: python
        from sisl.nodes import Node, FileNode
        import time

        # Write something to file
        with open("testfile", "w") as f:
            f.write("HELLO")

        # Create a FileNode
        n = FileNode("testfile")

        # Define a file reader node that prints the contents of the file
        @Node.from_func
        def print_contents(path):
            print("printing contents...")
            with open(path, "r") as f:
                print(f.read())

        # And initialize it by passing the FileNode as an input
        printer = print_contents(path=n)

        print("---RUNNING NODE")

        # Run the printer node
        printer.get()

        print("--- SET AUTOMATIC UPDATING")

        # Now set it to automatically update on changes to upstream inputs
        printer.context.update(lazy=False)

        print("--- APPENDING TO FILE")

        # Append to the file which will trigger the update.
        with open("testfile", "a") as f:
            f.write("\nBYE")

        # Give some time for the printer to react before exiting
        time.sleep(1)

    This should give the following output:

    .. code-block::

        ---RUNNING NODE
        printing contents...
        HELLO
        --- SET AUTOMATIC UPDATING
        --- APPENDING TO FILE
        printing contents...
        HELLO
        BYE

    """

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        self._setup_observer()

    def _setup_observer(self):
        """Sets up the watchdog observer."""
        if not WATCHDOG_IMPORTED:
            warn(
                "Watchdog is not installed. {self.__class__.__name__} will not be able to detect changes in files."
            )
        else:
            # Watchdog watches directories so we should watch the parent
            parent_path = Path(self.inputs["path"]).parent

            self.observer = watchdog.observers.Observer()
            handler = CallbackHandler(self)
            self.observer.schedule(handler, parent_path)
            self.observer.start()

    def _update_observer(self):
        """Updates the observer to watch the (possibly) new path."""
        if not WATCHDOG_IMPORTED:
            return

        self.observer.unschedule_all()
        parent_path = Path(self.inputs["path"]).parent
        self.observer.schedule(CallbackHandler(self), parent_path)

    # Methods to interact with the observer
    def matches_path(self, path: str):
        return Path(path) == Path(self.inputs["path"])

    def on_file_modified(self, event):
        self.on_file_change(event)

    def on_file_created(self, event):
        self.on_file_change(event)

    def on_file_moved(self, event):
        self.on_file_change(event)

    def on_file_deleted(self, event):
        self.on_file_change(event)

    def on_file_change(self, event):
        self._receive_outdated()

    def update_inputs(self, **inputs):
        # We wrap the update_inputs method to update the observer if the path
        # changes.
        super().update_inputs(**inputs)
        if "path" in inputs:
            self._update_observer()

    @staticmethod
    def function(path: str) -> Path:
        return Path(path)
