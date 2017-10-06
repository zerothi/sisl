from __future__ import print_function

import numpy as np

# Import sile objects
from sisl.utils import strmap
from sisl.utils.cmd import default_ArgumentParser, default_namespace
from ..sile import add_sile, Sile_fh_open
from .sile import *


__all__ = ['eigSileSiesta']


class eigSileSiesta(SileSiesta):
    """ EIG Siesta file object """

    @Sile_fh_open
    def read_data(self):
        """ Returns data associated with the EIG file """

        # Luckily the data is in eV
        Ef = float(self.readline())

        # Read the total length of the path
        no, ns, nk = map(int, self.readline().split())
        # Now we may read the eigenvalues

        # Allocate
        eigs = np.empty([ns, nk, no], np.float32)

        for ik in range(nk):
            # The first line is special
            E = map(float, self.readline().split()[1:])
            s = 0
            e = len(E)
            tmp_E = np.empty([ns*no], np.float32)
            tmp_E[s:e] = E
            for _ in range(ns):
                while e < ns*no:
                    E = map(float, self.readline().split())
                    s = e
                    e += len(E)
                    tmp_E[s:e] = E
            tmp_E.shape = (ns, no)
            eigs[:, ik, :] = tmp_E
        return eigs - Ef

    @default_ArgumentParser(description="Manipulate EIG file in sisl.")
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
            "_eigs": self.read_data(),
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

        class EIGPlot(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                import matplotlib.pyplot as plt
                E = ns._eigs
                #Emin = np.min(E)
                #Emax = np.max(E)
                #n = E.shape[1]
                # We need to setup a relatively good size of the scatter
                # plots
                s = 10 #20. / max(Emax - Emin, n)

                def myplot(ax, title, y, E, s):
                    ax.set_title(title)
                    for ib in range(y.shape[0]):
                        ax.scatter(np.ones(y.shape[1])*ib, y[ib, :], s=s)
                    ax.set_xlabel('k-index')
                    ax.set_xlim(-0.5, len(y) + 0.5)
                    if not E is None:
                        ax.set_ylim(E[0], E[1])

                if E.shape[0] == 2:
                    _, ax = plt.subplots(1, 2)
                    ax[0].set_ylabel('E-Ef [eV]')

                    # We must plot spin-up/down separately
                    for i, ud in enumerate(['UP', 'DOWN']):
                        myplot(ax[i], 'Eigenspectrum SPIN-'+ud, E[i, :, :], ns._Emap, s)
                else:
                    plt.figure()
                    ax = plt.gca()
                    ax.set_ylabel('E-Ef [eV]')
                    myplot(ax, 'Eigenspectrum', E[0, :, :], ns._Emap, s)
                if value is None:
                    plt.show()
                else:
                    plt.savefig(value)
        p.add_argument(*opts('--plot', '-p'), action=EIGPlot, nargs='?', metavar='FILE',
                       help='Plot the eigenvalues from the .EIG file, possibly saving plot to a file.')

        return p, namespace


add_sile('EIG', eigSileSiesta, case=False, gzip=True)
