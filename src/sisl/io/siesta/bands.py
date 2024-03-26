# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import sisl._array as _a
from sisl._internal import set_module
from sisl.utils import strmap
from sisl.utils.cmd import default_ArgumentParser, default_namespace

from ..sile import add_sile, sile_fh_open
from .sile import SileSiesta

__all__ = ["bandsSileSiesta"]


@set_module("sisl.io.siesta")
class bandsSileSiesta(SileSiesta):
    """Bandstructure information"""

    @sile_fh_open(True)
    def read_fermi_level(self) -> float:
        """Returns the Fermi level in the bands file"""
        # Luckily the data is in eV
        return float(self.readline())

    @sile_fh_open()
    def read_data(self, as_dataarray: bool = False):
        """Returns data associated with the bands file

        The energy levels are shifted with respect to the Fermi-level.

        Parameters
        --------
        as_dataarray: boolean, optional
            if `True`, the information is returned as an `xarray.DataArray`
            Ticks (if read) are stored as an attribute of the DataArray
            (under `array.ticks` and `array.ticklabels`)
        """
        band_lines = False

        # Luckily the data is in eV
        Ef = self.read_fermi_level()

        # Read the total length of the path (not used)
        _, _ = map(float, self.readline().split())
        l = self.readline()
        try:
            _, _ = map(float, l.split())
            band_lines = True
        except Exception:
            # We are dealing with a band-points file
            pass

        # orbitals, n-spin, n-k
        nk2 = 3
        if band_lines:
            l = self.readline()
            nk2 = 1
        no, ns, nk = map(int, l.split())

        # Create the data to contain all band points
        eb = _a.emptyd([nk, ns, no])
        k = _a.emptyd([nk, nk2])
        for ik in range(nk):
            l = [float(x) for x in self.readline().split()]
            for i in range(nk2):
                k[ik, i] = l[0]
                del l[0]

            # Now populate the eigenvalues
            while len(l) < ns * no:
                l.extend([float(x) for x in self.readline().split()])
            l = _a.arrayd(l)
            eb[ik, :, :] = l.reshape(ns, no) - Ef

        vals = (k, eb)

        # for band-lines
        if band_lines:
            k.shape = (-1,)

            # Now we need to read the labels for the points
            xlabels = []
            labels = []
            nl = int(self.readline())
            for _ in range(nl):
                l = self.readline().split()
                xlabels.append(float(l[0]))
                labels.append((" ".join(l[1:])).replace("'", ""))
            vals = (xlabels, labels), *vals

        if as_dataarray:
            from xarray import DataArray

            ticks = {"ticks": xlabels, "ticklabels": labels} if band_lines else {}

            vals = DataArray(
                eb,
                name="Energy",
                attrs=ticks,
                coords=[
                    ("k", k),
                    ("spin", _a.arangei(0, eb.shape[1])),
                    ("band", _a.arangei(0, eb.shape[2])),
                ],
            )

        return vals

    @default_ArgumentParser(description="Manipulate bands file in sisl.")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """Returns the arguments that is available for this Sile"""
        # limit_args = kwargs.get("limit_arguments", True)
        short = kwargs.get("short", False)

        def opts(*args):
            if short:
                return args
            return [args[0]]

        # We limit the import to occur here
        import argparse

        # The first thing we do is adding the geometry to the NameSpace of the
        # parser.
        # This will enable custom actions to interact with the geometry in a
        # straight forward manner.
        namespace = default_namespace(
            _bands=self.read_data(),
            _Emap=None,
        )

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

        class BandsPlot(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                import matplotlib.pyplot as plt

                # Decide whether this is BandLines or BandPoints
                if len(ns._bands) == 2:
                    # We do not plot "points"
                    raise ValueError(
                        "The bands file only contains points in the BZ, not a bandstructure."
                    )
                lbls, k, b = ns._bands
                b = b.T
                # Extract to tick-marks and names
                xlbls, lbls = lbls

                def myplot(ax, title, x, y, E):
                    ax.set_title(title)
                    for ib in range(y.shape[0]):
                        ax.plot(x, y[ib, :])
                    ax.set_ylabel("E-Ef [eV]")
                    ax.set_xlim(x.min(), x.max())
                    if not E is None:
                        ax.set_ylim(E[0], E[1])

                if b.shape[1] == 2:
                    _, ax = plt.subplots(2, 1)
                    ax[0].set_xticks(xlbls)
                    ax[0].set_xticklabels([""] * len(xlbls))
                    ax[1].set_xticks(xlbls)
                    ax[1].set_xticklabels(lbls, rotation=45)
                    # We must plot spin-up/down separately
                    for i, ud in enumerate(["UP", "DOWN"]):
                        myplot(
                            ax[i], f"Bandstructure SPIN-{ud}", k, b[:, i, :], ns._Emap
                        )
                else:
                    plt.figure()
                    ax = plt.gca()
                    ax.set_xticks(xlbls)
                    ax.set_xticklabels(lbls, rotation=45)
                    myplot(ax, "Bandstructure", k, b[:, 0, :], ns._Emap)
                if value is None:
                    plt.show()
                else:
                    plt.savefig(value)

        p.add_argument(
            *opts("--plot", "-p"),
            action=BandsPlot,
            nargs="?",
            metavar="FILE",
            help="Plot the bandstructure from the .bands file, possibly saving to a file.",
        )

        return p, namespace


add_sile("bands", bandsSileSiesta, gzip=True)
