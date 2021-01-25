import numpy as np

from ..sile import add_sile, sile_fh_open, SileError
from .sile import SileSiesta

from sisl._array import arrayd
from sisl._internal import set_module
from sisl.physics import get_distribution
from sisl.utils import strmap
from sisl.utils.cmd import default_ArgumentParser, default_namespace
from sisl.unit.siesta import units
from .kp import kpSileSiesta


__all__ = ['eigSileSiesta']


@set_module("sisl.io.siesta")
class eigSileSiesta(SileSiesta):
    """ Eigenvalues as calculated in the SCF loop, easy plots using `sdata`

    The .EIG file from Siesta contains the eigenvalues for k-points used during the SCF.
    Using the command-line utility `sdata` one may plot the eigenvalue spectrum to visualize the
    spread of eigenvalues.

    .. code:: bash

        sdata siesta.EIG --plot
        # or to save to png file
        sdata siesta.EIG --plot eig_spread.png


    One may also plot the DOS using the eigenvalues and the k-point weights:

    .. code:: bash

        sdata siesta.EIG --dos
        # or to save to png file
        sdata siesta.EIG --dos dos.png

    This will default to plot the DOS using these parameters:
    dE = 5 meV, kT = 300 K (25 meV), Gaussian distribution and in the full energy-range of the eigenvalue spectrum.
    By default the k-point weights will be read in the ``siesta.KP`` file, however if the file does not
    exist one may use the option ``--kp-file FILE`` to read in the weights from ``FILE``.

    To limit the shown energy region, simply use:

    .. code:: bash

        sdata siesta.EIG -E -10:10 --dos

    to reduce to the -10 eV to 10 eV energy range.

    One may optionally choose the temperature smearing and the used distribution function:

    .. code:: bash

        sdata siesta.EIG -E -10:10 --dos 0.01 0.1 lorentzian

    which will calculate the DOS in steps of 10 meV, the temperature smearing is 0.1 eV and
    the used distribution is a Lorentzian.
    """

    @sile_fh_open(True)
    def read_fermi_level(self):
        r""" Query the Fermi-level contained in the file

        Returns
        -------
        Ef : fermi-level of the system
        """
        return float(self.readline())

    @sile_fh_open()
    def read_data(self):
        r""" Read eigenvalues, as calculated and written by Siesta

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

        # k-point weights
        class KP(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                ns._weights = kpSileSiesta(value[0]).read_data()[1]
        p.add_argument('--kp-file', '-kp', nargs=1, metavar='FILE', action=KP,
                       help='The k-point file from which to read the band-weights (only applicable to --dos option)')

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
                    ik = np.arange(y.shape[0])
                    for ib in range(y.shape[1]):
                        ax.scatter(ik, y[:, ib], s=s)
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

        # Energy grabs
        class DOSPlot(argparse.Action):
            def __call__(dos_self, parser, ns, value, option_string=None):
                import matplotlib.pyplot as plt
                if not hasattr(ns, '_weight'):
                    # Try and read in the k-point-weights
                    ns._weight = kpSileSiesta(str(self.file).replace('EIG', 'KP')).read_data()[1]

                if ns._Emap is None:
                    # We will plot the DOS in the entire energy window
                    ns._Emap = [ns._eigs.min(), ns._eigs.max()]

                if len(ns._weight) != ns._eigs.shape[1]:
                    raise SileError(str(self) + ' --dos the number of k-points for the eigenvalues and k-point weights '
                                    'are different, please use --weight correctly.')

                # Specify default settings
                dE = 0.005 # 5 meV
                kT = units('K', 'eV') * 300
                distr = get_distribution('gaussian', smearing=kT)
                out = None
                if len(value) > 0:
                    i = 0
                    try:
                        dE = float(value[i])
                        i += 1
                    except: pass
                    try:
                        kT = float(value[i])
                        i += 1
                    except: pass
                    try:
                        distr = get_distribution(value[i], smearing=kT)
                        i += 1
                    except: pass
                    try:
                        out = value[i]
                    except: pass

                # Now we are ready to process
                E = np.arange(ns._Emap[0] - kT * 4, ns._Emap[1] + kT * 4, dE)

                def myplot(ax, legend, E, eig, w):
                    DOS = np.zeros(len(E))
                    for ib in range(eig.shape[0]):
                        for e in eig[ib, :]:
                            DOS += distr(E - e) * w[ib]
                    ax.plot(E, DOS, label=legend)
                    ax.set_ylim(0, None)

                plt.figure()
                ax = plt.gca()
                ax.set_title('DOS kT={:.1f} K'.format(kT * units('eV', 'K')))
                ax.set_xlabel('E - Ef [eV]')
                ax.set_xlim(E.min(), E.max())
                ax.set_ylabel('DOS [1/eV]')
                if ns._eigs.shape[0] == 2:
                    for i, ud in enumerate(['up', 'down']):
                        myplot(ax, ud, E, ns._eigs[i, :, :], ns._weight)
                    plt.legend()
                else:
                    myplot(ax, '', E, ns._eigs[0, :, :], ns._weight)
                if out is None:
                    plt.show()
                else:
                    plt.savefig(out)
        p.add_argument('--dos', action=DOSPlot, nargs='*', metavar='dE,kT,DIST,FILE',
                       help='Plot the density of states from the .EIG file, '
                       'dE = energy separation, kT = smearing, DIST = distribution function (Gaussian) possibly saving plot to a file.')

        return p, namespace


add_sile('EIG', eigSileSiesta, gzip=True)
