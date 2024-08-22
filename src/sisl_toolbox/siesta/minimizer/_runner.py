# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from sisl._internal import set_module
from sisl.io.siesta import fdfSileSiesta

from ._path import path_abs, path_rel_or_abs

__all__ = [
    "AbstractRunner",
    "AndRunner",
    "PathRunner",
    "CleanRunner",
    "CopyRunner",
    "CommandRunner",
    "AtomRunner",
    "SiestaRunner",
    "FunctionRunner",
]


_log = logging.getLogger(__name__)


def commonprefix(*paths):
    common = os.path.commonprefix(paths)
    return common, [Path(path).relative_to(common) for path in paths]


@set_module("sisl_toolbox.siesta.minimizer")
class AbstractRunner(ABC):
    """Define a runner"""

    def __iter__(self):
        yield self

    def __and__(self, other):
        return AndRunner(self, other)

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run this runner"""


@set_module("sisl_toolbox.siesta.minimizer")
class AndRunner(AbstractRunner):
    """Placeholder for two runners"""

    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __iter__(self):
        # in correct sequence (as they are runned)
        yield from self.A
        yield from self.B

    def run(self, A=None, B=None, **kwargs):
        """Run `self.A` first, then `self.B`

        Both runners get ``kwargs`` as arguments, and `A`
        only gets passed to `self.A`.

        Parameters
        ----------
        A : None or dict, optional
           a dictionary that gets passed to ``self.A(**A, **kwargs)``
        B : None or dict, optional
           a dictionary that gets passed to ``self.B(**B, **kwargs)``

        Returns
        -------
        tuple of return from `self.A` and `self.B`
        """
        # print("running A")
        if A is None:
            A = self.A.run(**kwargs)
        else:
            kw = kwargs.copy()
            kw.update(**A)
            A = self.A.run(**kw)
        if not isinstance(self.A, AndRunner):
            A = (A,)
        # print("running B")
        if B is None:
            B = self.B.run(**kwargs)
        else:
            kw = kwargs.copy()
            kw.update(**B)
            B = self.B.run(**kw)
        if not isinstance(self.B, AndRunner):
            B = (B,)
        # Return merged tuple
        return A + B


@set_module("sisl_toolbox.siesta.minimizer")
class PathRunner(AbstractRunner):
    """Define a runner"""

    def __init__(self, path):
        self.path = path_abs(path)

    def absattr(self, attr):
        out = getattr(self, attr)
        if out.is_absolute():
            return out
        return self.path / out

    def __iter__(self):
        cwd = Path.cwd()
        # we know that `self.path` is absolute, so we can freely chdir
        os.chdir(self.path)
        yield self
        os.chdir(cwd)

    def clean(self, *files):
        def _parse_glob(f):
            if isinstance(f, Path):
                return [f]
            return self.path.glob(f)

        for f in files:
            if isinstance(f, (tuple, list)):
                self.clean(*f)
                continue

            for fs in _parse_glob(f):
                if fs.is_file():
                    os.remove(fs)
                elif fs.is_dir():
                    shutil.rmtree(fs)


@set_module("sisl_toolbox.siesta.minimizer")
class CleanRunner(PathRunner):
    def __init__(self, path, *files):
        super().__init__(path)
        self.files = files

    def run(self):
        _log.debug(f"running cleaner")
        self.clean(*self.files)


@set_module("sisl_toolbox.siesta.minimizer")
class CopyRunner(PathRunner):
    def __init__(self, from_path, to_path, *files, **rename):
        # we store .path as *from_path*
        super().__init__(from_path)
        self.to = path_abs(to_path)
        self.files = files
        self.rename = rename
        if not self.path.is_dir():
            raise ValueError(
                f"{self.__class__.__name__} path={self.path} must be a directory"
            )
        if not self.to.is_dir():
            raise ValueError(
                f"{self.__class__.__name__} path={self.to} must be a directory"
            )

    def run(self):
        copy = []
        rem = []
        for f in self.files:
            fin = self.path / f
            fout = self.to / f
            if not fin.is_file():
                copy.append(f"Path(.) {fin.relative_to('.')}->{fout.relative_to('.')}")
            elif fin.is_file():
                copy.append(str(f))
                shutil.copyfile(fin, fout)
            elif fout.is_file():
                rem.append(str(f))
                os.remove(fout)
        for fin, fout in self.rename.items():
            f_in = self.path / fin
            f_out = self.to / fout
            common, (f_in_rel, f_out_rel) = commonprefix(f_in, f_out)
            if not f_in.is_file():
                _log.warning(f"file {f_in} not found for copying")
            elif f_in.is_file():
                copy.append(f"Path({common}) {f_in_rel}->{f_out_rel}")
                shutil.copyfile(f_in, f_out)
            elif f_out.is_file():
                rem.append(f"[{common}] rm {fout}")
                os.remove(f_out)
        _log.debug(f"copying {copy}; removing {rem}")


@set_module("sisl_toolbox.siesta.minimizer")
class CommandRunner(PathRunner):
    def __init__(
        self, path, cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, hook=None
    ):
        super().__init__(path)
        abs_cmd = path_abs(cmd, self.path)
        if abs_cmd.is_file():
            self.cmd = [abs_cmd]
            if not os.access(self.cmd, os.X_OK):
                raise ValueError(
                    f"{self.__class__.__name__} shell script {self.cmd.relative_to(self.path.cwd())} not executable"
                )
        else:
            self.cmd = cmd.split()

        if isinstance(stdout, type(subprocess.PIPE)):
            self.stdout = stdout
        else:
            self.stdout = path_rel_or_abs(stdout, self.path)
        if isinstance(stderr, type(subprocess.PIPE)):
            self.stderr = stderr
        else:
            self.stderr = path_rel_or_abs(stderr, self.path)

        if hook is None:

            def hook(subprocess_output):
                return subprocess_output

        assert callable(hook)
        self.hook = hook

    def _get_standard(self):
        out = self.stdout
        if isinstance(out, (Path, str)):
            out = open(out, "w")
        err = self.stderr
        if isinstance(err, (Path, str)):
            err = open(err, "w")
        return out, err

    def run(self):
        cmd = [str(cmd) for cmd in self.cmd]
        _log.debug(f"running command: {' '.join(cmd)}")
        # atom generates lots of files
        # We need to clean the directory so that subsequent VPSFMT users don't
        # accidentially use a prior output
        stdout, stderr = self._get_standard()
        return self.hook(
            subprocess.run(
                cmd,
                cwd=self.path,
                encoding="utf-8",
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        )


@set_module("sisl_toolbox.siesta.minimizer")
class AtomRunner(CommandRunner):
    """Run a command with atom-input file as first argument and output file as second argument

    This is tailored for atom in the sense of arguments for this class, but not
    restricted in any way.
    Note that ``atom`` requires the input file to be ``INP`` and thus the
    script should move stuff in `input` is not ``INP``.

    It is best explained through an example:

    >>> run = AtomRunner("atom", "run.sh", "INP", "OUT")
    >>> run.run()
    ... # will run these shell commands:
    ... # cd atom
    ... # ./run.sh INP OUT
    ... # cd ..
    """

    def __init__(
        self,
        path,
        cmd="atom",
        input="INP",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        hook=None,
    ):
        super().__init__(path, cmd, stdout, stderr, hook)
        self.input = path_rel_or_abs(input, self.path)

    def run(self):
        cmd = [str(cmd) for cmd in self.cmd + [self.input]]
        _log.debug(f"running atom using command: {' '.join(cmd)}")
        # atom generates lots of files
        # We need to clean the directory so that subsequent VPSFMT users don't
        # accidentially use a prior output
        self.clean("RHO", "OUT", "PS*", "AE*", "CHARGE", "COREQ", "FOURIER*", "VPS*")
        stdout, stderr = self._get_standard()
        return self.hook(
            subprocess.run(
                cmd,
                cwd=self.path,
                encoding="utf-8",
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        )


@set_module("sisl_toolbox.siesta.minimizer")
class SiestaRunner(CommandRunner):
    """Run a script/cmd with fdf as first argument and output file as second argument

    This is tailored for Siesta in the sense of arguments for this class, but not
    restricted in any way.

    It is best explained through an example:

    >>> run = SiestaRunner("siesta", "run.sh", "RUN.fdf", "RUN.out")
    >>> run.run()
    ... # will run these shell commands:
    ... # cd siesta
    ... # ./run.sh RUN.fdf RUN.out
    ... # cd ..
    """

    def __init__(
        self,
        path,
        cmd="siesta",
        fdf="RUN.fdf",
        stdout="RUN.out",
        stderr=subprocess.PIPE,
        hook=None,
    ):
        super().__init__(path, cmd, stdout, stderr, hook)
        self.fdf = path_rel_or_abs(fdf, self.path)

        fdf = self.absattr("fdf")
        self.systemlabel = fdfSileSiesta(fdf, base=self.path).get(
            "SystemLabel", "siesta"
        )

    def run(self):
        pipe = ""
        stdout, stderr = self._get_standard()
        for pre, f in [(">", stdout), ("2>", stderr)]:
            try:
                pipe += f"{pre} {f.name}"
            except Exception:
                pass
        cmd = [str(cmd) for cmd in self.cmd + [self.fdf]]
        _log.debug(f"running Siesta using command[{self.path}]: {' '.join(cmd)} {pipe}")
        # Remove stuff to ensure that we don't read information from prior calculations
        self.clean("*.ion*", "fdf-*.log", f"{self.systemlabel}.*")
        return self.hook(
            subprocess.run(
                cmd,
                cwd=self.path,
                encoding="utf-8",
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        )


@set_module("sisl_toolbox.siesta.minimizer")
class FunctionRunner(AbstractRunner):
    """Run a method `func` with specified arguments and kwargs"""

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        _log.debug(f"running function")
        return self.func(*self.args, **self.kwargs)
