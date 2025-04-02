# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

r""" Atom input/output writer

Developer: Nick Papior
Contact: nickpapior <at> gmail.com
sisl-version: >0.10.0

This enables reading input files for atom and also helps in parsing output files.

To read in an input file one can do:

>>> atom = AtomInput.from_input("INP")
>>> atom.pg("OUT")

which will read and write the same file.
One can also plot output from `atom` using:

>>> atom.plot()

which will show 4 plots for different sections. A command-line tool is also
made available through the `stoolbox`.
"""
import sys
from functools import reduce
from pathlib import Path

import numpy as np

try:
    from scipy.integrate import trapezoid
except ImportError:
    from scipy.integrate import trapz as trapezoid
from scipy.interpolate import interp1d

import sisl as si
from sisl._internal import set_module
from sisl.utils import NotNonePropertyDict, PropertyDict

__all__ = ["AtomInput", "atom_plot_cli"]


_script = Path(sys.argv[0]).name

# We don't use the Siesta units here...
_Ang2Bohr = si.units.convert("Ang", "Bohr")

# Atom/Siesta uses this order of occupation for
# core electrons
# idx | shell | electrons | cum. electrons | noble gas
#  1  |  1s   |    2      |       2        |   [He]
#  2  |  2s   |    2      |       4        |   [He]2s2
#  3  |  2p   |    6      |      10        |   [Ne]
#  4  |  3s   |    2      |      12        |   [Ne]3s
#  5  |  3p   |    6      |      18        |   [Ar]
#  6  |  3d   |   10      |      28        |   [Ar]3d10
#  7  |  4s   |    2      |      30        |   [Ar]3d10 2s2
#  8  |  4p   |    6      |      36        |   [Kr]
#  9  |  4d   |   10      |      46        |   [Kr]4d10
# 10  |  5s   |    2      |      48        |   [Kr]4d10 5s2
# 11  |  5p   |    6      |      54        |   [Xe]
# 12  |  4f   |   14      |      68        |   [Xe]4f14
# 13  |  5d   |   10      |      78        |   [Xe]4f14 5d10
# 14  |  6s   |    2      |      80        |   [Xe]4f14 5d10 6s2
# 15  |  6p   |    6      |      86        |   [Rn]
# 16  |  7s   |    2      |      88        |   [Rn]7s2
# 17  |  5f   |   14      |     102        |   [Rn]7s2 5f14
# 18  |  6d   |   10      |     112        |   [Rn]7s2 5f14 6d10
# 19  |  7p   |    6      |     118        |   [Og]
_shell_order = [
    # Occupation shell order
    "1s",  # [He]
    "2s",
    "2p",  # [Ne]
    "3s",
    "3p",  # [Ar]
    "3d",
    "4s",
    "4p",  # [Kr]
    "4d",
    "5s",
    "5p",  # [Xe]
    "4f",
    "5d",
    "6s",
    "6p",  # [Rn]
    "7s",
    "5f",
    "6d",
    "7p",  # [Og]
]
_spdfgh = "spdfgh"


@set_module("sisl_toolbox.siesta.atom")
class AtomInput:
    """Input for the ``atom`` program see [AtomLicense]_

    This class enables the construction of the ``INP`` file to be fed to ``atom``.

    ::

        # Example input for ATOM
        #
        #  Comments allowed here
        #
        #   ae Si ground state all-electron
        #   Si   car
        #       0.0
        #    3    2
        #    3    0      2.00      0.00
        #    3    1      2.00      0.00
        #
        # Comments allowed here
        #
        #2345678901234567890123456789012345678901234567890      Ruler

    References
    ----------
    .. [AtomLicense] https://siesta.icmab.es/SIESTA_MATERIAL/Pseudos/atom_licence.html
    """

    def __init__(
        self, atom, define=("NEW_CC", "FREE_FORMAT_RC_INPUT", "NO_PS_CUTOFFS"), **opts
    ):
        # opts = {
        #   "flavor": "tm2",
        #   "xc": "pb",
        #  optionally libxc
        #   "equation": "r",
        #   "logr": 2.
        #   "cc": False,
        #   "rcore": 2.
        # }

        self.atom = atom
        assert isinstance(atom, si.Atom)
        if "." in self.atom.tag:
            raise ValueError("The atom 'tag' must not contain a '.'!")

        # We need to check that atom has 4 orbitals, with increasing l
        # We don't care about n or any other stuff, so these could be
        # SphericalOrbital, for that matter
        l = 0
        for orb in self.atom:
            if orb.l != l:
                raise ValueError(
                    f"{self.__class__.__name__} atom argument does not have "
                    f"increasing l quantum number index {l} has l={orb.l}"
                )
            l += 1
        if l != 4:
            raise ValueError(
                f"{self.__class__.__name__} atom argument must have 4 orbitals. "
                f"One for each s-p-d-f shell"
            )

        self.opts = PropertyDict(**opts)

        # Check options passed and define defaults

        self.opts.setdefault("equation", "r")
        if self.opts.equation not in " rs":
            # ' ' == non-polarized
            # s == polarized
            # r == relativistic
            raise ValueError(
                f"{self.__class__.__name__} failed to initialize; opts{'equation': <v>} has wrong value, should be [ rs]."
            )
        if self.opts.equation == "s":
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement spin-polarized option (use relativistic)"
            )

        self.opts.setdefault("flavor", "tm2")
        if self.opts.flavor not in ("hsc", "ker", "tm2"):
            # hsc == Hamann-Schluter-Chiang
            # ker == Kerker
            # tm2 == Troullier-Martins
            raise ValueError(
                f"{self.__class__.__name__} failed to initialize; opts{'flavor': <v>} has wrong value, should be [hsc|ker|tm2]."
            )

        self.opts.setdefault("logr", 2.0)

        # default to true if set
        self.opts.setdefault("cc", "rcore" in self.opts)
        # rcore only used if cc is True
        self.opts.setdefault("rcore", 0.0)
        self.opts.setdefault("xc", "pb")

        # Read in the core valence shells for this atom
        # figure out what the default value is.
        # We do this my finding the minimum index of valence shells
        # in the _shell_order list, then we use that as the default number
        # of core-shells occpupied
        # e.g if the minimum valence shell is 2p, it would mean that
        #  _shell_order.index("2p") == 2
        # which has 1s and 2s occupied.
        try:
            core = reduce(
                min,
                (_shell_order.index(f"{orb.n}{_spdfgh[orb.l]}") for orb in atom),
                len(_shell_order),
            )
        except Exception:
            core = -1

        self.opts.setdefault("core", core)
        if self.opts.core == -1:
            raise ValueError(
                f"Default value for {self.atom.symbol} not added, please add core= at instantiation"
            )

        # Store the defined names
        if define is None:
            define = []
        elif isinstance(define, str):
            define = [define]
        # store
        self.define = define

    @classmethod
    def from_input(cls, inp):
        """Return atom object respecting the input

        Parameters
        ----------
        inp : list or str
           create `AtomInput` from the content of `inp`
        """

        def _get_content(f):
            if f.is_file():
                return open(f, "r").readlines()
            return None

        if isinstance(inp, (tuple, list)):
            # it is already in correct format
            pass
        elif isinstance(inp, (str, Path)):
            # convert to path
            inp = Path(inp)

            # Check if it is a path or an input
            content = _get_content(inp)
            if content is None:
                content = _get_content(inp / "INP")
            if content is None:
                raise ValueError(
                    f"Could not find any input file in {str(inp)} or {str(inp / 'INP')}"
                )
            inp = content

        else:
            raise ValueError(f"Unknown input format inp={inp}?")

        # Now read lines
        defines = []
        opts = PropertyDict()

        def bypass_comments(inp):
            if inp[0].startswith("#"):
                inp.pop(0)
                bypass_comments(inp)

        def bypass(inp, defines):
            bypass_comments(inp)
            if inp[0].startswith("%define"):
                line = inp.pop(0)
                defines.append(line.split()[1].strip())
                bypass(inp, defines)

        bypass(inp, defines)

        # Now prepare reading
        # First line has to contain the *type* of calculation
        # pg|pe|ae|pt <comment>
        line = inp.pop(0).strip()
        if line.startswith("pg"):
            opts.cc = False
        elif line.startswith("pe"):
            opts.cc = True

        # <flavor> logr?
        line = inp.pop(0).strip().split()
        opts.flavor = line[0]
        if len(line) >= 2:
            opts.logr = float(line[1]) / _Ang2Bohr

        # <element> <xc>' rs'?
        line = inp.pop(0)
        symbol = line.split()[0]
        # now get xc equation
        if len(line) >= 11:
            opts.equation = line[10:10]
        opts.xc = line[:10].split()[1]
        line = line.split()
        if len(line) >= 3:
            opts.libxc = int(line[2])

        # currently not used line
        inp.pop(0)

        # core, valence
        core, valence = inp.pop(0).split()
        opts.core = int(core)
        valence = int(valence)

        orbs = []
        for _ in range(valence):
            n, l, *occ = inp.pop(0).split()
            orb = PropertyDict()
            orb.n = int(n)
            orb.l = int(l)
            # currently we don't distinguish between up/down
            orb.q0 = sum(map(float, occ))
            orbs.append(orb)

        # now we read the line with rc's and core-correction
        rcs = inp.pop(0).split()
        if len(rcs) >= 6:
            # core-correction
            opts.rcore = float(rcs[5]) / _Ang2Bohr

        for orb in orbs:
            orb.R = float(rcs[orb.l]) / _Ang2Bohr

        # Now create orbitals
        orbs = [si.AtomicOrbital(**orb, m=0, zeta=1) for orb in orbs]
        # now re-arrange ensuring we have correct order of l shells
        orbs = sorted(orbs, key=lambda orb: orb.l)
        atom = si.Atom(symbol, orbs)
        return cls(atom, defines, **opts)

    @classmethod
    def from_yaml(cls, file, nodes=()):
        """Parse the yaml file"""
        from sisl_toolbox.siesta.minimizer._yaml_reader import parse_variable, read_yaml

        dic = read_yaml(file, nodes)

        element = dic["element"]
        tag = dic.get("tag")
        mass = dic.get("mass", None)

        # get default options for pseudo
        opts = NotNonePropertyDict()
        pseudo = dic["pseudo"]
        opts["logr"] = parse_variable(pseudo.get("log-radius"), unit="Ang").value
        opts["rcore"] = parse_variable(pseudo.get("core-correction"), unit="Ang").value
        opts["xc"] = pseudo.get("xc")
        opts["equation"] = pseudo.get("equation")
        opts["flavor"] = pseudo.get("flavor")
        define = pseudo.get(
            "define", ("NEW_CC", "FREE_FORMAT_RC_INPUT", "NO_PS_CUTOFFS")
        )

        # Now on to parsing the valence shells
        orbs = []
        for key in dic:
            if key not in _shell_order:
                continue

            # Now we know the occupation is a shell
            pseudo = dic[key].get("pseudo", {})
            cutoff = parse_variable(pseudo.get("cutoff"), 2.1, "Ang").value
            charge = parse_variable(pseudo.get("charge"), 0.0).value
            orbs.append(si.AtomicOrbital(key, m=0, R=cutoff, q0=charge))

        atom = si.Atom(element, orbs, mass=mass, tag=tag)
        return cls(atom, define, **opts)

    def write_header(self, f):
        f.write("# This file is generated by sisl pseudo\n")
        # Define all names
        for define in self.define:
            f.write(f"%define {define.upper()}\n")

    def write_generation(self, f):
        if self.opts.cc:
            # use core corrections
            pg = "pe"
        else:
            # do not use core corrections
            pg = "pg"
        logr = self.opts.logr * _Ang2Bohr
        f.write(f"   {pg:2s} {self.atom.symbol} pseudo potential\n")
        if logr < 0.0:
            f.write(f"        {self.opts.flavor:3s}\n")
        else:
            f.write(f"        {self.opts.flavor:3s}{logr:9.3f}\n")

        xc = self.opts.xc
        equation = self.opts.equation
        rcore = self.opts.rcore * _Ang2Bohr
        f.write(f"   {self.atom.symbol:2s}   {xc:2s}{equation:1s}")
        if "libxc" in self.opts:
            f.write(f" {self.opts.libxc:8d}")
        f.write(f"\n  {0.0:5.1f}\n")
        # now extract the charges for each orbital
        atom = self.atom
        core = self.opts.core
        valence = len(atom)

        f.write(f"{core:5d}{valence:5d}\n")

        Rs = [0.0] * 4  # always 4: s, p, d, f
        for orb in sorted(atom.orbitals, key=lambda x: x.l):
            # Write the configuration of this orbital
            n, l = orb.n, orb.l
            f.write(f"{n:5d}{l:5d}{orb.q0:10.3f}{0.0:10.3f}\n")
            Rs[l] = orb.R * _Ang2Bohr
        f.write(
            f"{Rs[0]:10.7f} {Rs[1]:10.7f} {Rs[2]:10.7f} {Rs[3]:10.7f} {0.0:10.7f} {rcore:10.7f}\n"
        )

    def write_all_electron(self, f, charges=(0.0,)):
        q0 = self.atom.q0.sum()
        xc = self.opts.xc
        equation = self.opts.equation
        core = self.opts.core
        for charge in charges:
            if isinstance(charge, dict):
                out = self.excite(**charge)
            else:
                out = self.excite(charge)
            q = out.atom.q0.sum()
            f.write(f"   ae {out.atom.symbol} # all-electron q={q0 - q}\n")
            f.write(f"   {out.atom.symbol:2s}   {xc:2s}{equation:1s}")
            if "libxc" in out.opts:
                f.write(f" {out.opts.libxc:8d}")
            f.write(f"\n  {0.0:5.1f}\n")
            # now extract the charges for each orbital
            atom = out.atom
            valence = len(atom)

            f.write(f"{core:5d}{valence:5d}\n")
            for orb in sorted(atom.orbitals, key=lambda x: x.l):
                n, l = orb.n, orb.l
                f.write(f"{n:5d}{l:5d}{orb.q0:10.3f}{0.0:10.3f}\n")

    def write_test(self, f, charges=(0,)):
        q0 = self.atom.q0.sum()
        xc = self.opts.xc
        equation = self.opts.equation
        core = self.opts.core
        for charge in charges:
            if isinstance(charge, dict):
                out = self.excite(**charge)
            else:
                out = self.excite(charge)
            q = out.atom.q0.sum()
            f.write(f"   pt {out.atom.symbol} # pseudo test q={q0 - q}\n")
            f.write(f"   {out.atom.symbol:2s}   {xc:2s}{equation:1s}")
            if "libxc" in out.opts:
                f.write(f" {out.opts.libxc:8d}")
            f.write(f"\n  {0.0:5.1f}\n")
            # now extract the charges for each orbital
            atom = out.atom
            valence = len(atom)

            f.write(f"{core:5d}{valence:5d}\n")
            for orb in sorted(atom.orbitals, key=lambda x: x.l):
                n, l = orb.n, orb.l
                f.write(f"{n:5d}{l:5d}{orb.q0:10.3f}{0.0:10.3f}\n")

    def _get_out(self, path, filename):
        if path is None:
            return Path(filename)
        return Path(path) / Path(filename)

    def __call__(self, filename="INP", path=None):
        """Open a file and return self"""
        out = self._get_out(path, filename)
        self._enter = open(out, "w"), self.atom
        return self

    def __enter__(self):
        if not hasattr(self, "_enter"):
            self()
        return self._enter[0]

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, "_enter"):
            self.atom = self._enter[1]
            del self._enter

    def ae(self, filename="INP", path=None):
        with self(filename, path) as f:
            self.write_header(f)
            self.write_all_electron(f)

    def pg(self, filename="INP", path=None):
        with self(filename, path) as f:
            self.write_header(f)
            self.write_generation(f)

    def excite(self, *charge, **lq):
        """Excite contained atom to another charge

        Notes
        -----
        This is charge, *not* electrons.
        """
        if len(charge) > 1:
            raise ValueError(
                f"{self.__class__.__name__}.excite takes only "
                "a single argument or [spdf]=charge arguments"
            )
        elif len(charge) == 1:
            charge = charge[0]
            if "charge" in lq:
                raise ValueError(
                    f"{self.__class__.__name__}.excite does not accept "
                    "both charge as argument and keyword argument"
                )
        else:
            charge = 0
        charge = lq.pop("charge", charge)
        # get indices of the orders
        shell_idx = [
            _shell_order.index(f"{orb.n}{_spdfgh[orb.l]}") for orb in self.atom
        ]

        def _charge(idx):
            # while the g and h shells are never used, we just have them...
            return {"s": 2, "p": 6, "d": 10, "f": 14, "g": 18, "h": 22}[
                _shell_order[idx][1]
            ]

        orig_charge = [_charge(idx) for idx in shell_idx]
        atom = self.atom.copy()
        # find order of orbitals (from highest index to lowest)
        # depending on the sign of charge
        shell_sort = shell_idx[:]
        shell_sort.sort(reverse=charge > 0)

        for l, q in lq.items():
            l = _spdfgh.index(l)
            for orb in atom:
                if orb.l == l:
                    orb._q0 -= q
                    assert orb.q0 >= 0.0

        # now finalize the charge
        while abs(charge) > 1e-9:
            for idx_sort in shell_sort:
                idx = shell_idx.index(idx_sort)
                orb = atom[idx]
                if charge > 0 and orb.q0 > 0:
                    dq = min(orb.q0, charge)
                elif charge < 0 and orb.q0 < orig_charge[idx]:
                    dq = max(orb.q0 - orig_charge[idx], charge)
                else:
                    dq = 0.0
                orb._q0 -= dq
                charge -= dq
        return self.__class__(atom, self.define, **self.opts)

    def plot(
        self,
        path=None,
        plot=("wavefunction", "charge", "log", "potential"),
        l="spdf",
        show=True,
    ):
        """Plot everything related to this psf file

        Parameters
        ----------
        path : str or pathlib.Path, optional
           from which directory should files be read
        plot : list-like of str, optional
           which data to plot
        l : list-like, optional
           which l-shells to plot (for those that have l-shell decompositions)
        show : bool, optional
           call `matplotlib.pyplot.show()` at the end

        Returns
        -------
        fig : figure for axes
        axs : axes used for plotting
        """
        import matplotlib.pyplot as plt

        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)

        def get_xy(f, yfactors=None):
            """Return x, y data from file `f` with y being calculated as the factors between the columns"""
            nonlocal path
            f = path / f
            if not f.is_file():
                print(f"Could not find file: {str(f)}")
                return None, None

            data = np.loadtxt(f)
            ncol = data.shape[1]

            if yfactors is None:
                yfactors = [0, 1]
            yfactors = np.pad(yfactors, (0, ncol - len(yfactors)), constant_values=0.0)
            x = data[:, 0]
            y = (data * yfactors.reshape(1, -1)).sum(1)
            return x, y

        l2i = {
            "s": 0,
            0: 0,
            "p": 1,
            1: 1,
            "d": 2,
            2: 2,
            "f": 3,
            3: 3,
            "g": 4,
            4: 4,  # never used
        }

        # Get this atoms default calculated binding length
        # We use this one since there are many missing elements
        # in vdw table.
        # And convert to Bohr
        atom_r = self.atom.radius("calc") * _Ang2Bohr

        def plot_wavefunction(ax):
            # somewhat similar to ae.gplot
            ax.set_title("Wavefunction")
            ax.set_xlabel("Radius [Bohr]")

            for shell in l:
                il = l2i[shell]
                orb = self.atom.orbitals[il]

                r, w = get_xy(f"AEWFNR{il}")
                if not r is None:
                    p = ax.plot(r, w, label=f"AE {_spdfgh[il]}")
                    color = p[0].get_color()
                    ax.axvline(orb.R * _Ang2Bohr, color=color, alpha=0.5)

                r, w = get_xy(f"PSWFNR{il}")
                if not r is None:
                    ax.plot(r, w, "--", label=f"PS {_spdfgh[il]}")

            ax.set_xlim(0, atom_r * 5)
            ax.autoscale(enable=True, axis="y", tight=True)
            ax.legend()

        def plot_charge(ax):
            ax.set_title("Charge")
            ax.set_xlabel("Radius [Bohr]")
            ax.set_ylabel("(4.pi.r^2) Charge [electrons/Bohr]")

            # Get current core-correction length
            ae_r, ae_cc = get_xy("AECHARGE", [0, 0, 0, 1])
            _, ae_vc = get_xy("AECHARGE", [0, 1, 1, -1])

            if not ae_cc is None:
                p = ax.plot(ae_r, ae_cc, label=f"AE core")
                color = p[0].get_color()
                if self.opts.get("cc", False):
                    ax.axvline(self.opts.rcore * _Ang2Bohr, color=color, alpha=0.5)
                ax.plot(ae_r, ae_vc, "--", label=f"AE valence")

            ps_r, ps_cc = get_xy("PSCHARGE", [0, 0, 0, 1])
            _, ps_vc = get_xy("PSCHARGE", [0, 1, 1])

            if not ps_r is None:
                ax.plot(ps_r, ps_cc, "--", label=f"PS core")
                ax.plot(ps_r, ps_vc, ":", label=f"PS valence")

            # Now determine the overlap between all-electron core-charge
            # and the pseudopotential valence charge
            if np.allclose(ae_r, ps_r):
                # Determine dR
                # dr = ae_r[1] - ae_r[0]

                # Integrate number of core-electrons and valence electrons
                core_c = trapezoid(ae_cc, ae_r)
                valence_c = trapezoid(ps_vc, ps_r)
                print(f"Total charge in atom: {core_c + valence_c:.5f}")
                overlap_c = trapezoid(np.minimum(ae_cc, ps_vc), ae_r)
                ax.set_title(f"Charge: int(min(AE_cc, PS_vc)) = {overlap_c:.3f} e")

                # We will try and *guess-stimate* a good position for rc for core-corrections
                # Javier Junquera's document says:
                # r_pc has to be chosen such that the valence charge density is negligeable compared to
                # the core one for r < r_pc.
                # Tests show that it might be located where the core charge density is from 1 to 2 times
                # larger than the valence charge density
                with np.errstate(divide="ignore", invalid="ignore"):
                    fraction = ae_cc / ps_vc
                    np.nan_to_num(fraction, copy=False)

                ax2 = (
                    ax.twinx()
                )  # instantiate a second axes that shares the same x-axis
                ax2.plot(ae_r, fraction, "k", alpha=0.5, label="c/v")

                marks = np.array([0.5, 1.0, 1.5])
                min_x = (ae_r > 0.2).nonzero()[0].min()
                max_x = (fraction[min_x:] > 0.0).nonzero()[0].max() + min_x

                r_marks = interp1d(fraction[min_x:max_x], ae_r[min_x:max_x])(marks)
                ax2.scatter(r_marks, marks, alpha=0.5)
                ax2.set_ylim(0, 3)
                print(f"Core-correction r_pc {marks}: {r_marks} Bohr")

            ax.set_xlim(0, atom_r)
            ax.set_ylim(0)
            ax.legend()

        def plot_log(ax):
            ax.set_title("d-log of wavefunction")
            ax.set_xlabel("Energy [Ry]")
            ax.set_ylabel("Derivative of wavefunction")

            for shell in l:
                il = l2i[shell]

                e, log = get_xy(f"AELOGD{il}")
                emark = np.loadtxt(path / f"AEEV{il}")
                if emark.ndim == 1:
                    emark.shape = (1, -1)
                emark = emark[:, 0]
                if not e is None:
                    p = ax.plot(e, log, label=f"AE {_spdfgh[il]}")

                    idx_mark = np.fabs(e.reshape(-1, 1) - emark.reshape(1, -1)).argmin(
                        axis=0
                    )
                    ax.scatter(emark, log[idx_mark], color=p[0].get_color(), alpha=0.5)

                # And now PS
                e, log = get_xy(f"PSLOGD{il}")
                emark = np.loadtxt(path / f"PSEV{il}")
                if emark.ndim == 1:
                    emark.shape = (1, -1)
                emark = emark[:, 0]
                if not e is None:
                    p = ax.plot(e, log, ":", label=f"PS {_spdfgh[il]}")

                    idx_mark = np.fabs(e.reshape(-1, 1) - emark.reshape(1, -1)).argmin(
                        axis=0
                    )
                    ax.scatter(emark, log[idx_mark], color=p[0].get_color(), alpha=0.5)

            ax.legend()

        def plot_potential(ax):
            ax.set_title("Pseudopotential")
            ax.set_xlabel("Radius [Bohr]")
            ax.set_ylabel("Potential [Ry]")

            for shell in l:
                il = l2i[shell]
                orb = self.atom.orbitals[il]

                r, V = get_xy(f"PSPOTR{il}")
                if not r is None:
                    p = ax.plot(r, V, label=f"PS {_spdfgh[il]}")
                    color = p[0].get_color()
                    ax.axvline(orb.R * _Ang2Bohr, color=color, alpha=0.5)

            ax.set_xlim(0, atom_r * 3)
            ax.legend()

        nrows = len(l) // 2
        ncols = len(l) // nrows
        if nrows * ncols < len(l):
            ncols += 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(11, 10))

        def next_rc(ir, ic, nrows, ncols):
            ic = ic + 1
            if ic == ncols:
                ic = 0
                ir = ir + 1
            return ir, ic

        ir, ic = 0, 0
        for this_plot in map(lambda x: x.lower(), plot):
            if this_plot == "wavefunction":
                plot_wavefunction(axs[ir][ic])
            elif this_plot == "log":
                plot_log(axs[ir][ic])
            elif this_plot == "charge":
                plot_charge(axs[ir][ic])
            elif this_plot == "potential":
                plot_potential(axs[ir][ic])

            ir, ic = next_rc(ir, ic, nrows, ncols)

        if show:
            plt.show()
        return fig, axs


def atom_plot_cli(subp=None, parser_kwargs={}):
    """Run plotting command for the output of atom"""

    is_sub = not subp is None

    title = "Plotting facility for atom output (run in the atom output directory)"
    if is_sub:
        global _script
        _script = f"{_script} atom-plot"
        p = subp.add_parser("atom-plot", description=title, help=title, **parser_kwargs)
    else:
        import argparse

        p = argparse.ArgumentParser(title, **parser_kwargs)

    p.add_argument(
        "--plot",
        "-P",
        action="append",
        type=str,
        choices=("wavefunction", "charge", "log", "potential"),
        help="""Determine what to plot""",
    )

    p.add_argument("-l", default="spdf", type=str, help="""Which l shells to plot""")

    p.add_argument("--save", "-S", default=None, help="""Save output plots to file.""")

    p.add_argument(
        "--show",
        default=False,
        action="store_true",
        help="""Force showing the plot (only if --save is specified)""",
    )

    p.add_argument(
        "input", type=str, default="INP", help="""Input file name (default INP)"""
    )

    if is_sub:
        p.set_defaults(runner=atom_plot)
    else:
        atom_plot(p.parse_args())


# Import object holding all the CLI
from sisl_toolbox.cli import register_toolbox_cli

register_toolbox_cli(atom_plot_cli)


@set_module("sisl_toolbox.siesta.atom")
def atom_plot(args):
    import matplotlib.pyplot as plt

    input = Path(args.input)
    atom = AtomInput.from_input(input)

    # If the specified input is a file, use the parent
    # Otherwise use the input *as is*.
    if input.is_file():
        path = input.parent
    else:
        path = input

    # if users have not specified what to plot, we plot everything
    if args.plot is None:
        args.plot = ("wavefunction", "charge", "log", "potential")
    fig = atom.plot(path, plot=args.plot, l=args.l, show=False)[0]

    if args.save is None:
        plt.show()
    else:
        fig.savefig(args.save)
        if args.show:
            plt.show()
