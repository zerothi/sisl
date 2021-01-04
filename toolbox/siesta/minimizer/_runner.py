from abc import abstractmethod, ABC
import subprocess
from pathlib import Path
import glob
import shutil
import os
import logging

from sisl.io.siesta import fdfSileSiesta
from ._path import path_abs, path_rel_or_abs


__all__ = ["AbstractRunner", "AndRunner", "PathRunner",
           "CleanRunner", "CopyRunner", "ScriptRunner",
           "AtomRunner", "SiestaRunner", "FunctionRunner"]


_log = logging.getLogger("sisl_toolbox.siesta.pseudo")


class AbstractRunner(ABC):
    """ Define a runner """

    def __iter__(self):
        yield self

    def __and__(self, other):
        return AndRunner(self, other)

    @abstractmethod
    def run(self, *args, **kwargs):
        """ Run this runner """


class AndRunner(AbstractRunner):
    """ Placeholder for two runners """

    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __iter__(self):
        # in correct sequence (as they are runned)
        yield from self.A
        yield from self.B

    def run(self, A=None, B=None, **kwargs):
        """ Run `self.A` first, then `self.B`

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
        #print("running A")
        if A is None:
            A = self.A.run(**kwargs)
        else:
            kw = kwargs.copy()
            kw.update(**A)
            A = self.A.run(**kw)
        if not isinstance(self.A, AndRunner):
            A = (A,)
        #print("running B")
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


class PathRunner(AbstractRunner):
    """ Define a runner """

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


class CleanRunner(PathRunner):
    def __init__(self, path, *files):
        super().__init__(path)
        self.files = files

    def run(self):
        _log.debug(f"running cleaner")
        self.clean(*self.files)


class CopyRunner(PathRunner):
    def __init__(self, from_path, to_path, *files, **rename):
        # we store .path as *from_path*
        super().__init__(from_path)
        self.to = path_abs(to_path)
        self.files = files
        self.rename = rename
        if not self.path.is_dir():
            raise ValueError(f"script {self.path} must be a directory")
        if not self.to.is_dir():
            raise ValueError(f"script {self.to} must be a directory")

    def run(self):
        copy = []
        rem = []
        for f in self.files:
            fin = self.path / f
            fout = self.to / f
            if fin.is_file():
                copy.append(str(f))
                shutil.copyfile(fin, fout)
            elif fout.is_file():
                rem.append(str(f))
                os.remove(fout)
        for fin, fout in self.rename.items():
            f_in = self.path / fin
            f_out = self.to / fout
            if f_in.is_file():
                copy.append("{}->{}".format(str(fin), str(fout)))
                shutil.copyfile(f_in, f_out)
            elif f_out.is_file():
                rem.append("->{}".format(str(fout)))
                os.remove(f_out)
        _log.debug(f"copying {copy}; removing {rem}")


class ScriptRunner(PathRunner):
    def __init__(self, path, script="run.sh"):
        super().__init__(path)
        self.script = path_abs(script, self.path)

        if not self.script.is_file():
            raise ValueError(f"script {self.script.relative_to(self.path.cwd())} must be a file")
        # ensure scirpt is executable
        if not os.access(self.script, os.X_OK):
            raise ValueError(f"Script {self.script.relative_to(self.path.cwd())} not executable")


class AtomRunner(ScriptRunner):
    """ Run a script with atom-input file as first argument and output file as second argument

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

    def __init__(self, path, script="run.sh", input="INP", out="OUT"):
        super().__init__(path, script)
        self.input = path_rel_or_abs(input)
        self.out = path_rel_or_abs(out)

    def run(self):
        _log.debug(f"running atom using script: {self.script}")
        # atom generates lots of files
        # We need to clean the directory so that subsequent VPSFMT users don't
        # accidentially use a prior output
        self.clean("RHO", "OUT", "PS*", "AE*", "CHARGE", "COREQ", "FOURIER*", "VPS*")
        return subprocess.run([self.script, self.input, self.out], cwd=self.path,
                              encoding='utf-8',
                              capture_output=True, check=False)


class SiestaRunner(ScriptRunner):
    """ Run a script with fdf as first argument and output file as second argument

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

    def __init__(self, path, script="run.sh", fdf="RUN.fdf", out="RUN.out"):
        super().__init__(path, script)
        self.fdf = path_rel_or_abs(fdf)
        self.out = path_rel_or_abs(out)

        fdf = self.absattr("fdf")
        self.systemlabel = fdfSileSiesta(fdf, base=self.path).get("SystemLabel", "siesta")

    def run(self):
        _log.debug(f"running siesta using script: {self.script}")
        # Remove stuff to ensure that we don't read information from prior calculations
        self.clean("*.ion*", "fdf-*.log", f"{self.systemlabel}.*", self.out)
        return subprocess.run([self.script, self.fdf, self.out], cwd=self.path,
                              encoding='utf-8',
                              capture_output=True, check=False)


class FunctionRunner(AbstractRunner):
    """ Run a method `func` with specified arguments and kwargs """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        _log.debug(f"running function")
        return self.func(*self.args, **self.kwargs)
