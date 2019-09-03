from __future__ import print_function

import numpy as np

from sisl.utils import strmap
from sisl.utils.cmd import default_ArgumentParser, default_namespace
from ..sile import add_sile, sile_fh_open
from .sile import *


__all__ = ['bandsSileSiesta']


class bandsSileSiesta(SileSiesta):
    """ Bandstructure information """

    @sile_fh_open()
    def read_data(self):
        """ Returns data associated with the bands file """
        band_lines = False

        # Luckily the data is in eV
        Ef = float(self.readline())
        # Read the total length of the path (not used)
        _, _ = map(float, self.readline().split())
        l = self.readline()
        try:
            _, _ = map(float, l.split())
            band_lines = True
        except:
            # We are dealing with a band-points file
            pass

        # orbitals, n-spin, n-k
        if band_lines:
            l = self.readline()
        no, ns, nk = map(int, l.split())

        # Create the data to contain all band points
        b = np.empty([nk, ns, no], np.float64)

        # for band-lines
        if band_lines:
            k = np.empty([nk], np.float64)
            for ik in range(nk):
                l = [float(x) for x in self.readline().split()]
                k[ik] = l[0]
                del l[0]
                # Now populate the eigenvalues
                while len(l) < ns * no:
                    l.extend([float(x) for x in self.readline().split()])
                l = np.array(l, np.float64)
                l.shape = (ns, no)
                b[ik, :, :] = l[:, :] - Ef
            # Now we need to read the labels for the points
            xlabels = []
            labels = []
            nl = int(self.readline())
            for _ in range(nl):
                l = self.readline().split()
                xlabels.append(float(l[0]))
                labels.append((' '.join(l[1:])).replace("'", ''))
            vals = (xlabels, labels), k, b

        else:
            k = np.empty([nk, 3], np.float64)
            for ik in range(nk):
                l = [float(x) for x in self.readline().split()]
                k[ik, :] = l[0:3]
                del l[2]
                del l[1]
                del l[0]
                # Now populate the eigenvalues
                while len(l) < ns * no:
                    l.extend([float(x) for x in self.readline().split()])
                l = np.array(l, np.float64)
                l.shape = (ns, no)
                b[ik, :, :] = l[:, :] - Ef
            vals = k, b
        return vals

    @default_ArgumentParser(description="Manipulate bands file in sisl.")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        #limit_args = kwargs.get('limit_arguments', True)
        short = kwargs.get('short', False)

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
        d = {
            "_bands": self.read_data(),
            "_Emap": None,
        }
        namespace = default_namespace(**d)

        # Energy grabs
        class ERange(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                ns._Emap = strmap(float, value)[0]
        p.add_argument('--energy', '-E',
                       action=ERange,
                       help='Denote the sub-section of energies that are plotted: "-1:0,1:2" [eV]')

        class BandsPlot(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                import matplotlib.pyplot as plt
                # Decide whether this is BandLines or BandPoints
                if len(ns._bands) == 2:
                    # We do not plot "points"
                    raise ValueError("The bands file only contains points in the BZ, not a bandstructure.")
                lbls, k, b = ns._bands
                b = b.T
                # Extract to tick-marks and names
                xlbls, lbls = lbls

                def myplot(ax, title, x, y, E):
                    ax.set_title(title)
                    for ib in range(y.shape[0]):
                        ax.plot(x, y[ib, :])
                    ax.set_ylabel('E-Ef [eV]')
                    ax.set_xlim(x.min(), x.max())
                    if not E is None:
                        ax.set_ylim(E[0], E[1])

                if b.shape[1] == 2:
                    _, ax = plt.subplots(2, 1)
                    ax[0].set_xticks(xlbls)
                    ax[0].set_xticklabels([''] * len(xlbls))
                    ax[1].set_xticks(xlbls)
                    ax[1].set_xticklabels(lbls, rotation=45)
                    # We must plot spin-up/down separately
                    for i, ud in enumerate(['UP', 'DOWN']):
                        myplot(ax[i], 'Bandstructure SPIN-'+ud, k, b[:, i, :], ns._Emap)
                else:
                    plt.figure()
                    ax = plt.gca()
                    ax.set_xticks(xlbls)
                    ax.set_xticklabels(lbls, rotation=45)
                    myplot(ax, 'Bandstructure', k, b[:, 0, :], ns._Emap)
                if value is None:
                    plt.show()
                else:
                    plt.savefig(value)
        p.add_argument(*opts('--plot', '-p'), action=BandsPlot, nargs='?', metavar='FILE',
                       help='Plot the bandstructure from the .bands file, possibly saving to a file.')

        return p, namespace


add_sile('bands', bandsSileSiesta, case=False, gzip=True)
