# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import nbformat


def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
    :returns (parsed nb object, execution errors)
    """
    dirname, __ = os.path.split(path)
    os.chdir(dirname)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=60",
            "--output",
            fout.name,
            path,
        ]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [
        output
        for cell in nb.cells
        if "outputs" in cell
        for output in cell["outputs"]
        if output.output_type == "error"
    ]

    return nb, errors


class NotebookTester:
    path = ""
    generated = []

    def test_ipynb(self):
        # Check that the notebook has ran without errors
        nb, errors = _notebook_run(self.path)
        assert errors == []

        # Remove all generated files
        for path in self.generated:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)


tut_root = Path(__file__) / "basic-tutorials"


class TestDemo(NotebookTester):
    path = os.path.join(tut_root, "Demo.ipynb")

    generated = [
        os.path.join(tut_root, file_name)
        for file_name in ("From_animated.plot", "From_animation.plot")
    ]

    def test_ipynb(self):
        super().test_ipynb()


class TestDIY(NotebookTester):
    path = os.path.join(tut_root, "DIY.ipynb")


class TestGUISession(NotebookTester):
    path = os.path.join(tut_root, "GUI with Python Demo.ipynb")
