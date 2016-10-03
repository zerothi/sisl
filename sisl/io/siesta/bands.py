"""
Sile object for reading/writing SIESTA bands files
"""

from __future__ import print_function

import numpy as np

# Import sile objects
from sisl.utils import strmap
from sisl.utils.cmd import *
from ..sile import add_sile, Sile_fh_open
from .sile import *


__all__ = ['bandsSileSiesta']


class bandsSileSiesta(SileSiesta):
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
                l = [float(x) for x in self.readline().split()]
                k[ik] = l[0]
                del l[0]
                # Now populate the eigenvalues
                while len(l) < ns * no:
                    l.extend([float(x) for x in self.readline().split()])
                l = np.array(l, np.float64)
                l.shape = (ns, no)
                b[ik,:,:] = l[:,:] - Ef
            # Now we need to read the labels for the points
            xlabels = []
            labels = []
            nl = int(self.readline())
            for il in range(nl):
                l = self.readline().split()
                xlabels.append(float(l[0]))
                labels.append((' '.join(l[1:])).replace("'",''))
            vals = (xlabels, labels), k, b
            
        else:
            k = np.empty([nk, 3], np.float64)
            for ik in range(nk):
                l = [float(x) for x in self.readline().split()]
                k[ik,:] = l[0:2]
                del l[2]
                del l[1]
                del l[0]
                # Now populate the eigenvalues
                while len(l) < ns * no:
                    l.extend([float(x) for x in self.readline().split()])
                l = np.array(l, np.float64)
                l.shape = (ns, no)
                b[ik,:,:] = l[:,:] - Ef
            vals = k, b
        return vals

    @dec_default_AP("Manipulate bands file in sisl.")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        limit_args = kwargs.get('limit_arguments', True)
        short = kwargs.get('short', False)

        def opts(*args):
            if short:
                return args
            return [args[0]]
        
        # We limit the import to occur here
        import argparse as arg

        # The first thing we do is adding the geometry to the NameSpace of the
        # parser.
        # This will enable custom actions to interact with the geometry in a
        # straight forward manner.
        d = {
            "_bands": self.read_data(),
            "_Emap" : None,
        }
        namespace = default_namespace(**d)

        # Ensure the namespace is populated
        ensure_namespace(p, namespace)

        # Energy grabs
        class ERange(arg.Action):
            def __call__(self, parser, ns, value, option_string=None):
                ns._Emap = strmap(float, value, recursive=False, sep=':')[0]
        p.add_argument('--energy', '-E', 
                       action=ERange,
                       help='Denote the sub-section of energies that are plotted: "-1:0,1:2" [eV]')
        
        class BandsPlot(arg.Action):
            def __call__(self, parser, ns, value, option_string=None):
                import matplotlib.pyplot as plt
                # Decide whether this is BandLines or BandPoints
                if len(ns._bands) == 2:
                    # We do not plot "points"
                    raise ValueError("The bands file only contains points in the BZ, not a bandstructure.")
                lbls, k, b = ns._bands
                b = b.T
                def myplot(title, x, y, E):
                    plt.figure()
                    plt.title(title)
                    for ib in range(y.shape[0]):
                        plt.plot(x, y[ib,:])
                    plt.xlabel('k-path [1/Bohr]')
                    plt.ylabel('E-Ef [eV]')
                    plt.xticks(xlbls, lbls, rotation=45)
                    plt.xlim(x.min(), x.max())
                    if not E is None:
                        plt.ylim(E[0], E[1])
                    
                xlbls, lbls = lbls
                if b.shape[1] == 2:
                    # We must plot spin-up/down separately
                    for i, ud in [(0, 'UP'), (1, 'DOWN')]:
                        myplot('Bandstructure SPIN-'+ud, k, b[:,i,:], ns._Emap)
                else:
                    myplot('Bandstructure', k, b[:,0,:], ns._Emap)
                if value is None:
                    plt.show()
                else:
                    plt.savefig(value)
        p.add_argument(*opts('--plot','-p'), action=BandsPlot, nargs='?', metavar='FILE',
                       help='Plot the bandstructure from the .bands file, possibly saving to a file.')

        return p, namespace


add_sile('bands', bandsSileSiesta, case=False, gzip=True)
