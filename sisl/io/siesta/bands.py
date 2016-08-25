"""
Sile object for reading/writing SIESTA bands files
"""

from __future__ import print_function

import numpy as np

# Import sile objects
from ..sile import add_sile, Sile_fh_open
from .sile import *


__all__ = ['BandsSIESTASile']


class BandsSIESTASile(SileSIESTA):
    """ bands SIESTA file object """

    def _setup(self):
        """ Setup the `BandsSIEASTASile` after initialization """
        self._comment = []

    @Sile_fh_open
    def read_data(self):
        """ Returns data associated with the bands file """
        band_lines = False

        # Luckily the data is in eV
        Ef = float(self.readline())
        # Read the total length of the path
        minl, maxl = map(float, self.readline().split())
        l = self.readline()
        try:
            minE, maxE = map(float, l.split())
            band_lines = True
        except:
            # We are dealing with a band-points file
            minE, maxE = minl, maxl

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
                l = map(float, self.readline().split())
                k[ik] = l[0]
                del l[0]
                # Now populate the eigenvalues
                while len(l) < ns * no:
                    l.extend(map(float, self.readline().split()))
                l = np.array(l, np.float64)
                l.shape = (ns, no)
                b[ik,:,:] = l[:,:]
            # Now we need to read the labels for the points
            xlabels = []
            labels = []
            nl = int(self.readline())
            for il in range(nl):
                l = self.readline().split()
                xlabels.append(float(l[0]))
                labels.append(' '.join(l[1:]))
            vals = (xlabels, labels), k, b
        else:
            k = np.empty([nk, 3], np.float64)
            for ik in range(nk):
                l = map(float, self.readline().split())
                k[ik,:] = l[0:2]
                del l[2]
                del l[1]
                del l[0]
                # Now populate the eigenvalues
                while len(l) < ns * no:
                    l.extend(map(float, self.readline().split()))
                l = np.array(l, np.float64)
                l.shape = (ns, no)
                b[ik,:,:] = l[:,:]
            vals = k, b
        return vals

    def ArgumentParser(self, parser=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        limit_args = kwargs.get('limit_arguments', True)
        short = kwargs.get('short', False)

        def opts(*args):
            if short:
                return args
            return [args[0]]
        
        # We limit the import to occur here
        import argparse as arg

        if parser is None:
            p = arg.ArgumentParser("Manipulate a bands file in sisl.")
        else:
            p = parser

        # The first thing we do is adding the geometry to the NameSpace of the
        # parser.
        # This will enable custom actions to interact with the geometry in a
        # straight forward manner.
        class CustomNamespace(object):
            pass
        namespace = CustomNamespace()
        namespace._data = self.read_data()

        # Create actions
        class BandsPlot(arg.Action):
            def __call__(self, parser, ns, value, option_string=None):
                import matplotlib.pyplot as plt
                # Decide whether this is BandLines or BandPoints
                if len(ns._data) == 2:
                    # We do not plot "points"
                    raise ValueError("The bands file only contains points in the BZ, not a bandstructure.")
                lbls, k, b = ns._data
                b = b.T
                def myplot(title, x, y):
                    plt.figure()
                    plt.title(title)
                    for ib in range(y.shape[0]):
                        plt.plot(x, y[ib,:])
                    plt.xlabel('k-path [1/Bohr]')
                    plt.ylabel('Energy [eV]')
                    plt.xticks(xlbls, lbls, rotation=45)
                    plt.xlim(x.min(), x.max())
                    
                xlbls, lbls = lbls
                if b.shape[1] == 2:
                    # We must plot spin-up/down separately
                    for i, ud in [(0, 'UP'), (1, 'DOWN')]:
                        myplot('Bandstructure SPIN-'+ud,
                               k, b[:,i,:])
                else:
                    myplot('Bandstructure', 
                           k, b[:,0,:])
                plt.show()
        p.add_argument(*opts('--plot','-p'), action=BandsPlot, nargs=0,
                   help='Plot the bandstructure from the .bands file.')

        return p, namespace


add_sile('bands', BandsSIESTASile, case=False, gzip=True)
