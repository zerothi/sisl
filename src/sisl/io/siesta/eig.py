# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

from sisl._internal import set_module
from sisl.physics import get_distribution
from sisl.unit import units
from sisl.utils import strmap
from sisl.utils.cmd import default_ArgumentParser, default_namespace

from ..sile import SileError, add_sile, sile_fh_open
from .kp import kpSileSiesta
from .sile import SileSiesta

__all__ = ["eigSileSiesta"]


@set_module("sisl.io.siesta")
class eigSileSiesta(SileSiesta):
    """Eigenvalues as calculated in the SCF loop, easy plots using `sdata`

    The .EIG file from Siesta contains the eigenvalues for k-points used during the SCF.
    Using the command-line utility `sdata` one may plot the eigenvalue spectrum to visualize the
    spread of eigenvalues.

    .. code:: bash

        # Plot the eigenspectrum
        sdata siesta.EIG --plot
        # or to save to png file
        sdata siesta.EIG --plot eig_spread.png

    One may also extract/plot the DOS using the eigenvalues and the k-point weights:

    .. code:: bash

        # plot the DOS using default values
        sdata siesta.EIG --dos --plot
        # change the energy spacing and plot it
        sdata siesta.EIG --dos dE=1meV --plot
        # change the energy spacing;
        # plot two different temperatures
        sdata siesta.EIG --dos dE=1meV kT=5K --dos kT=25K --plot
        # store the data in a dos.dat (3 columns, E, kT=5, kT=25K)
        sdata siesta.EIG --dos dE=1meV kT=5K --dos kT=25K --out dos.dat

    This will default to plot the DOS using these parameters:
    dE = 5 meV, kT = 300 K (25 meV), Gaussian distribution and in the full energy-range of the eigenvalue spectrum.
    By default the k-point weights will be read in the ``*.KP`` file, however if the file does not
    exist one may use the option ``--kp-file FILE`` to read in the weights from ``FILE``.

    To limit the shown energy region, simply use:

    .. code:: bash

        sdata siesta.EIG -E -10:10 --dos

    to reduce to the -10 eV to 10 eV energy range.

    One may optionally choose the temperature smearing and the used distribution function:

    .. code:: bash

        # position dependent values
        sdata siesta.EIG -E -10:10 --dos 0.01 0.1 lorentzian
        # key based values
        sdata siesta.EIG -E -10:10 --dos dE=0.01 kT=0.1 dist=lorentzian

    which will calculate the DOS in steps of 10 meV, the temperature smearing is 0.1 eV and
    the used distribution is a Lorentzian.
    Several invocations of ``--dos`` will collect the data until either a ``--plot`` or ``--out``
    is found on the command line.
    """

    @sile_fh_open(True)
    def read_fermi_level(self) -> float:
        r"""Query the Fermi-level contained in the file

        Returns
        -------
        Ef : fermi-level of the system
        """
        return float(self.readline())

    @sile_fh_open()
    def read_data(self) -> np.ndarray:
        r"""Read eigenvalues, as calculated and written by Siesta

        Returns
        -------
        numpy.ndarray : all eigenvalues, shifted to :math:`E_F = 0`, shape ``(ns, nk, nb)``
                        where ``ns`` number of spin-components, ``nk`` number of k-points and
                        ``nb`` number of bands.
        """
        Ef = self.read_fermi_level()

        # Read the total length of the path
        nb, ns, nk = map(int, self.readline().split())
        if ns > 2:
            # This is simply a NC/SOC calculation which is irrelevant in
            # regards of the eigenvalues.
            ns = 1

        # Allocate
        eigs = np.empty([ns, nk, nb], np.float64)

        readline = self.readline

        def iterE(size):
            ne = 0
            out = readline().split()[1:]
            ne += len(out)
            yield from out
            while ne < size:
                out = readline().split()
                ne += len(out)
                yield from out

        for ik in range(nk):
            # The first line is special
            E_list = np.fromiter(iterE(ns * nb), dtype=np.float64)
            eigs[:, ik, :] = E_list.reshape(ns, nb)

        return eigs - Ef

    @default_ArgumentParser(description="Manipulate Siesta EIG file.")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """Returns the arguments that is available for this Sile"""
        # limit_args = kwargs.get("limit_arguments", True)
        # short = kwargs.get("short", False)

        # We limit the import to occur here
        import argparse

        # The first thing we do is adding the geometry to the NameSpace of the
        # parser.
        # This will enable custom actions to interact with the geometry in a
        # straight forward manner.
        d = {
            "_eigs": self.read_data(),
            "_Emap": None,
            "_data": [],
            "_data_header": [],
            "_dos_args": [
                # default dE 5 meV
                0.005,
                # default T = 300 k
                units("K", "eV") * 300,
                # distribution type
                "gaussian",
            ],
        }
        try:
            d["_weights"] = kpSileSiesta(
                str(self.file).replace("EIG", "KP")
            ).read_data()[1]
        except Exception:
            d["_weights"] = None
        namespace = default_namespace(**d)

        # Energy grabs
        class ERange(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                ns._Emap = strmap(float, value)[0]

        p.add_argument(
            "--energy",
            "-E",
            action=ERange,
            help="Denote the sub-section of energies that are plotted: '-1:0,1:2' [eV]",
        )

        # k-point weights
        class KP(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                ns._weights = kpSileSiesta(value[0]).read_data()[1]

        p.add_argument(
            "--kp-file",
            "-kp",
            nargs=1,
            metavar="FILE",
            action=KP,
            help="The k-point file from which to read the band-weights (only applicable to --dos option)",
        )

        # Energy grabs
        class DOS(argparse.Action):
            def __call__(
                dos_self, parser, ns, values, option_string=None
            ):  # pylint: disable=E0213
                if getattr(ns, "_weights", None) is None:
                    if ns._eigs.shape[1] > 1:
                        raise ValueError(
                            "Can not calculate DOS when k-point weights are unknown, please pass -kp before this command"
                        )

                if len(ns._weights) != ns._eigs.shape[1]:
                    raise SileError(
                        f"{self!s} --dos the number of k-points for the eigenvalues and k-point weights "
                        "are different, please use -kp before --dos."
                    )

                # Specify default settings
                dE = ns._dos_args[0]
                kT = ns._dos_args[1]
                distribution = ns._dos_args[2]

                # check that all or non have "="
                n_eq = sum(("=" in v for v in values))
                if n_eq == 0:
                    for i, value in enumerate(values):
                        if i == 0:
                            dE = value
                        elif i == 1:
                            kT = value
                        elif i == 2:
                            distribution = value
                        else:
                            raise ValueError(
                                f"Too many values passed? Unknown value {value}?"
                            )

                elif n_eq == len(values):
                    for key, val in map(lambda x: x.split("="), values):
                        # it is a direct parseable thing
                        if key.lower() in ("de", "e"):
                            dE = val
                        elif key.lower() in ("kt", "t"):
                            kT = val
                        elif key.lower().startswith("dist"):
                            distribution = val
                        else:
                            raise ValueError(
                                f"Unknown key: {key}, should be one of [dE, kT, dist]"
                            )

                else:
                    raise ValueError(
                        "Mixing position arguments and keyword arguments is not allowed, either key=val or val, only"
                    )

                try:
                    dE = units(dE, "eV")
                except Exception:
                    dE = float(dE)
                try:
                    kT = units(kT, "eV")
                except Exception:
                    kT = float(kT)

                # store for next invocation
                ns._dos_args[0] = dE
                ns._dos_args[1] = kT
                ns._dos_args[2] = distribution

                # Now create the final distribution
                distribution = get_distribution(distribution, smearing=kT)

                if ns._Emap is None:
                    # We will plot the DOS in the entire energy window
                    ns._Emap = [ns._eigs.min() - kT * 4, ns._eigs.max() + kT * 4]

                # Now we are ready to process
                E = np.arange(ns._Emap[0], ns._Emap[1], dE)
                n_data = len(ns._data_header)
                same_E = False
                if n_data > 0:
                    for i in range(n_data):
                        header = ns._data_header[-i]
                        if header == "Energy":
                            try:
                                same_E = np.allclose(E, ns._data[-i])
                            except Exception:
                                same_E = False

                if not same_E:
                    ns._data_header.append("Energy")
                    ns._data.append(E)

                def calc_dos(E, eig, w):
                    nonlocal distribution
                    DOS = np.zeros(len(E))
                    for w_band, e_band in zip(w, eig):
                        for e in e_band:
                            DOS += distribution(E - e) * w_band
                    return DOS

                T = kT * units(f"eV", "K")
                str_T = f"{T:.2f}K"

                if ns._eigs.shape[0] == 2:
                    for eigs, ud in zip(ns._eigs, ("up", "down")):
                        ns._data_header.append(f"DOS-{ud} T={str_T}")
                        ns._data.append(calc_dos(E, eigs, ns._weights))
                else:
                    ns._data_header.append(f"DOS T={str_T}")
                    ns._data.append(calc_dos(E, ns._eigs[0, :, :], ns._weights))

        p.add_argument(
            "--dos",
            action=DOS,
            nargs="*",
            metavar="dE, kT, DIST",
            help="Calculate (and internally store) the density of states from the .EIG file, "
            "dE = energy separation (5 meV), kT = smearing (300 K), DIST = distribution function (Gaussian). "
            "The arguments will be the new defaults for later --dos calls.",
        )

        class Plot(argparse.Action):
            def _plot_data(self, parser, ns, value, option_string=None):
                """Plot data as contained in ns._data"""
                from matplotlib import pyplot as plt

                plt.figure()

                E = None
                is_DOS = True
                for header, data in zip(ns._data_header, ns._data):
                    if header == "Energy":
                        E = data
                    elif "DOS" in header:
                        plt.plot(E, data, label=header)
                    else:
                        # currently, we can only do DOS, so we shouldn't worry
                        is_DOS = False

                Emap = ns._Emap
                if Emap is not None:
                    plt.xlim(Emap[0], Emap[1])

                if is_DOS:
                    plt.ylabel("DOS [1/eV]")
                plt.legend()

                if value is None:
                    plt.show()
                else:
                    plt.savefig(value)

            def _plot_eig(self, parser, ns, value, option_string=None):
                import matplotlib.pyplot as plt

                E = ns._eigs
                # Emin = np.min(E)
                # Emax = np.max(E)
                # n = E.shape[1]
                # We need to setup a relatively good size of the scatter
                # plots
                s = 10  # 20. / max(Emax - Emin, n)

                def myplot(ax, title, y, E, s):
                    ax.set_title(title)
                    ik = np.arange(y.shape[0])
                    for ib in range(y.shape[1]):
                        ax.scatter(ik, y[:, ib], s=s)
                    ax.set_xlabel("k-index")
                    ax.set_xlim(-0.5, len(y) + 0.5)
                    if not E is None:
                        ax.set_ylim(E[0], E[1])

                if E.shape[0] == 2:
                    _, axs = plt.subplots(1, 2)
                    axs[0].set_ylabel("E-Ef [eV]")

                    # We must plot spin-up/down separately
                    for ax, eig, ud in zip(axs, E, ("UP", "DOWN")):
                        myplot(ax, f"Eigenspectrum {ud}", eig, ns._Emap, s)
                else:
                    plt.figure()
                    ax = plt.gca()
                    ax.set_ylabel("E-Ef [eV]")
                    myplot(ax, "Eigenspectrum", E[0, :, :], ns._Emap, s)

                if value is None:
                    plt.show()
                else:
                    plt.savefig(value)

            def __call__(self, parser, ns, value, option_string=None):
                if len(ns._data) == 0:
                    self._plot_eig(parser, ns, value, option_string)
                else:
                    self._plot_data(parser, ns, value, option_string)

        p.add_argument(
            "--plot",
            "-p",
            action=Plot,
            nargs="?",
            metavar="FILE",
            help="Plot the currently collected information (at its current invocation), or the eigenspectrum",
        )

        class Out(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                out = value[0]

                if len(ns._data) == 0:
                    # do nothing if data has not been collected
                    raise ValueError(
                        "No data has been collected in the arguments, nothing will be written, have you forgotten arguments?"
                    )

                if sum((h == "Energy" for h in ns._data_header)) > 1:
                    raise ValueError(
                        "There are multiple non-commensurate energy-grids, saving data requires a single energy-grid."
                    )

                from sisl.io import tableSile

                tableSile(out, mode="w").write(*ns._data, header=ns._data_header)

        p.add_argument(
            "--out",
            "-o",
            nargs=1,
            action=Out,
            help="Store currently collected information (at its current invocation) to the out file.",
        )

        return p, namespace


add_sile("EIG", eigSileSiesta, gzip=True)
