# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

try:
    from StringIO import StringIO
except Exception:
    from io import StringIO

import numpy as np

from sisl._internal import set_module

# Import the geometry object
from sisl.unit.siesta import unit_convert
from sisl.utils import default_ArgumentParser, default_namespace

from ..sile import add_sile
from ._cdf import _devncSileTBtrans

__all__ = ["tbtsencSileTBtrans", "phtsencSilePHtrans"]


Bohr2Ang = unit_convert("Bohr", "Ang")
Ry2eV = unit_convert("Ry", "eV")


@set_module("sisl.io.tbtrans")
class tbtsencSileTBtrans(_devncSileTBtrans):
    r"""TBtrans self-energy file object with downfolded self-energies to the device region

    The :math:`\boldsymbol\Sigma` object contains all self-energies on the specified k- and energy grid projected
    into the device region.

    This is mainly an output file object from TBtrans and can be used as a post-processing utility for
    testing various things in Python.

    Note that *anything* returned from this object are the self-energies in eV.

    Examples
    --------
    >>> H = Hamiltonian(device)
    >>> se = tbtsencSileTBtrans(...)
    >>> # Return the self-energy for the left electrode (unsorted)
    >>> se_unsorted = se.self_energy('Left', 0.1, [0, 0, 0])
    >>> # Return the self-energy for the left electrode (sorted)
    >>> se_sorted = se.self_energy('Left', 0.1, [0, 0, 0], sort=True)
    >>> # Query the indices in the full Hamiltonian
    >>> pvt_unsorted = se.pivot('Left').reshape(-1, 1)
    >>> pvt_sorted = se.pivot('Left', sort=True).reshape(-1, 1)
    >>> # The following two lines are equivalent
    >>> Hfull1[pvt_unsorted, pvt_unsorted.T] -= se_unsorted[:, :]
    >>> Hfull2[pvt_sorted, pvt_sorted.T] -= se_sorted[:, :]
    >>> np.allclose(Hfull1, Hfull2)
    True
    >>> # Query the indices in the device Hamiltonian
    >>> dev_pvt = se.pivot('Left', in_device=True).reshape(-1, 1)
    >>> dev_unpvt = se.pivot('Left', in_device=True, sort=True).reshape(-1, 1)
    >>> Hdev_pvt[dev_pvt, dev_pvt.T] -= se_unsorted[:, :]
    >>> Hdev[dpvt_sorted, dpvt_sorted.T] -= se_sorted[:, :]
    >>> pvt_dev = se.pivot(in_device=True).reshape(-1, 1)
    >>> np.allclose(Hdev_pvt, Hdev[pvt_dev, pvt_dev.T])
    True
    """

    _trans_type = "TBT"
    _E2eV = Ry2eV

    def self_energy(self, elec, E, k=0, sort=False):
        """Return the self-energy from the electrode `elec`

        Parameters
        ----------
        elec : str or int
           the corresponding electrode to return the self-energy from
        E : float or int
           energy to retrieve the self-energy at, if a floating point the closest
           energy value will be found and returned, if an integer it will correspond
           to the exact index
        k : array_like or int
           k-point to retrieve, if an integer it is the k-index in the file
        sort : bool, optional
           if ``True`` the returned self-energy will be sorted according to the order of
           the orbitals in the non-pivoted geometry, otherwise the self-energy will
           be returned according to the pivoted orbitals in the device region.
        """
        tree = self._elec(elec)
        ik = self.kindex(k)
        iE = self.Eindex(E)

        # When storing fortran arrays in C-type files reading it in
        # C-codes will transpose the data.
        # So we have to transpose back to get the correct order
        re = self._variable("ReSelfEnergy", tree=tree)[ik, iE].T
        im = self._variable("ImSelfEnergy", tree=tree)[ik, iE].T

        SE = self._E2eV * re + (1j * self._E2eV) * im
        if sort:
            pvt = self.pivot(elec)
            idx = np.argsort(pvt).reshape(-1, 1)

            # pivot for sorted device region
            return SE[idx, idx.T]

        return SE

    def broadening_matrix(self, elec, E, k=0, sort=False):
        r"""Return the broadening matrix from the electrode `elec`

        The broadening matrix is calculated as:

        .. math::
            \boldsymbol \Gamma(E) = i [\boldsymbol\Sigma(E) - \boldsymbol\Sigma^\dagger(E)]

        Parameters
        ----------
        elec : str or int
           the corresponding electrode to return the broadening matrix from
        E : float or int
           energy to retrieve the broadening matrix at, if a floating point the closest
           energy value will be found and returned, if an integer it will correspond
           to the exact index
        k : array_like or int
           k-point to retrieve, if an integer it is the k-index in the file
        sort : bool, optional
           if ``True`` the returned broadening matrix will be sorted according to the order of
           the orbitals in the non-pivoted geometry, otherwise the broadening matrix will
           be returned according to the pivoted orbitals in the device region.
        """
        tree = self._elec(elec)
        ik = self.kindex(k)
        iE = self.Eindex(E)

        # When storing fortran arrays in C-type files reading it in
        # C-codes will transpose the data.
        # So we have to transpose back to get the correct order
        re = self._variable("ReSelfEnergy", tree=tree)[ik, iE].T
        im = self._variable("ImSelfEnergy", tree=tree)[ik, iE].T

        G = -self._E2eV * (im + im.T) + (1j * self._E2eV) * (re - re.T)
        if sort:
            pvt = self.pivot(elec)
            idx = np.argsort(pvt)
            idx.shape = (-1, 1)

            # pivot for sorted device region
            return G[idx, idx.T]

        return G

    def self_energy_average(self, elec, E, sort=False):
        """Return the k-averaged average self-energy from the electrode `elec`

        Parameters
        ----------
        elec : str or int
           the corresponding electrode to return the self-energy from
        E : float or int
           energy to retrieve the self-energy at, if a floating point the closest
           energy value will be found and returned, if an integer it will correspond
           to the exact index
        sort : bool, optional
           if ``True`` the returned self-energy will be sorted according to the order of
           the orbitals in the non-pivoted geometry, otherwise the self-energy will
           be returned according to the pivoted orbitals in the device region.
        """
        tree = self._elec(elec)
        iE = self.Eindex(E)

        # When storing fortran arrays in C-type files reading it in
        # C-codes will transpose the data.
        # So we have to transpose back to get the correct order
        re = self._variable("ReSelfEnergyMean", tree=tree)[iE].T
        im = self._variable("ImSelfEnergyMean", tree=tree)[iE].T

        SE = self._E2eV * re + (1j * self._E2eV) * im
        if sort:
            pvt = self.pivot(elec)
            idx = np.argsort(pvt)
            idx.shape = (-1, 1)

            # pivot for sorted device region
            return SE[idx, idx.T]

        return SE

    def info(self, elec=None):
        """Information about the self-energy file available for extracting in this file

        Parameters
        ----------
        elec : str or int
           the electrode to request information from
        """
        if not elec is None:
            elec = self._elec(elec)

        # Create a StringIO object to retain the information
        out = StringIO()

        # Create wrapper function
        def prnt(*args, **kwargs):
            option = kwargs.pop("option", None)
            if option is None:
                print(*args, file=out)
            else:
                print("{:60s}[{}]".format(" ".join(args), ", ".join(option)), file=out)

        # Retrieve the device atoms
        prnt("Device information:")
        # Print out some more information related to the
        # k-point sampling.
        # However, we still do not know whether TRS is
        # applied.
        kpt = self.k
        nA = len(np.unique(kpt[:, 0]))
        nB = len(np.unique(kpt[:, 1]))
        nC = len(np.unique(kpt[:, 2]))
        prnt(
            (
                "  - number of kpoints: {} <- "
                "[ A = {} , B = {} , C = {} ] (time-reversal unknown)"
            ).format(self.nk, nA, nB, nC)
        )
        prnt("  - energy range:")
        E = self.E
        Em, EM = np.amin(E), np.amax(E)
        dE = np.diff(E)
        dEm, dEM = np.amin(dE) * 1000, np.amax(dE) * 1000  # convert to meV
        if (dEM - dEm) < 1e-3:  # 0.001 meV
            prnt(f"     {Em:.5f} -- {EM:.5f} eV  [{dEm:.3f} meV]")
        else:
            prnt(f"     {Em:.5f} -- {EM:.5f} eV  [{dEm:.3f} -- {dEM:.3f} meV]")
        prnt("  - imaginary part (eta): {:.4f} meV".format(self.eta() * 1e3))
        prnt("  - atoms with DOS (fortran indices):")
        prnt("     " + list2str(self.a_dev + 1))
        prnt("  - number of BTD blocks: {}".format(self.n_btd()))
        if elec is None:
            elecs = self.elecs
        else:
            elecs = [elec]

        # Print out information for each electrode
        for elec in elecs:
            if not elec in self.groups:
                prnt("  * no information available")
                continue

            try:
                bloch = self.bloch(elec)
            except Exception:
                bloch = [1] * 3
            try:
                n_btd = self.n_btd(elec)
            except Exception:
                n_btd = "unknown"
            prnt()
            prnt(f"Electrode: {elec}")
            prnt(f"  - number of BTD blocks: {n_btd}")
            prnt("  - Bloch: [{}, {}, {}]".format(*bloch))
            if "TBT" in self._trans_type:
                prnt(
                    "  - chemical potential: {:.4f} eV".format(
                        self.chemical_potential(elec)
                    )
                )
                prnt(
                    "  - electron temperature: {:.2f} K".format(
                        self.electron_temperature(elec)
                    )
                )
            else:
                prnt(
                    "  - phonon temperature: {:.4f} K".format(
                        self.phonon_temperature(elec)
                    )
                )
            prnt("  - imaginary part (eta): {:.4f} meV".format(self.eta(elec) * 1e3))
            prnt("  - atoms in down-folding region (not in device):")
            prnt("     " + list2str(self.a_down(elec) + 1))
            prnt("  - orbitals in down-folded device region:")
            prnt("     " + list2str(np.sort(self.pivot(elec)) + 1))

        s = out.getvalue()
        out.close()
        return s

    @default_ArgumentParser(
        description="Show information about data in a TBT.SE.nc file"
    )
    def ArgumentParser(self, p=None, *args, **kwargs):
        """Returns the arguments that is available for this Sile"""

        # We limit the import to occur here
        import argparse

        namespace = default_namespace(_tbtse=self, _geometry=self.geom)

        class Info(argparse.Action):
            """Action to print information contained in the TBT.SE.nc file, helpful before performing actions"""

            def __call__(self, parser, ns, value, option_string=None):
                # First short-hand the file
                print(ns._tbtse.info(value))

        p.add_argument(
            "--info",
            "-i",
            action=Info,
            nargs="?",
            metavar="ELEC",
            help="Print out what information is contained in the TBT.SE.nc file, optionally only for one of the electrodes.",
        )

        return p, namespace


add_sile("TBT.SE.nc", tbtsencSileTBtrans)
# Add spin-dependent files
add_sile("TBT_UP.SE.nc", tbtsencSileTBtrans)
add_sile("TBT_DN.SE.nc", tbtsencSileTBtrans)


@set_module("sisl.io.phtrans")
class phtsencSilePHtrans(tbtsencSileTBtrans):
    """PHtrans file object"""

    _trans_type = "PHT"
    _E2eV = Ry2eV**2


add_sile("PHT.SE.nc", phtsencSilePHtrans)
